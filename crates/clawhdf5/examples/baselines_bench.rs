//! P1.2b comparable-primitive baseline benchmark harness.
//!
//! Drives the three `clawhdf5_format::baselines` backends (flat SHA-256,
//! per-chunk HMAC-SHA-256, per-chunk Ed25519) through the same metrics
//! measured for the Merkle tree (§6.6: storage overhead, commit time,
//! single-chunk/whole-dataset verification latency, append/update cost,
//! and per-chunk-sig subset-proof size), so Phase 1 results can be reported
//! relative to measured baselines rather than asserted in the abstract.
//!
//! Run with:
//!   cargo run -p clawhdf5 --release --example baselines_bench --features parallel
//!     -- [SYNTHETIC_H5_PATH] [TRIALS] [NOAA_PATH] [NOAA_DATASET]
//!
//! SYNTHETIC_H5_PATH should point at the P1.1 10 GB synthetic file (see
//! `generate_10gb.rs`); its three datasets (`dataset_64kb`/`dataset_256kb`/
//! `dataset_1mb`) are read for real per-chunk bytes, tagged `source=synthetic`.
//! If omitted, the harness falls back to a small in-memory fabricated dataset
//! (256 chunks/size) for quick local smoke-testing, tagged
//! `source=synthetic_fabricated` so it can't be mistaken for the spec-required
//! P1.1 run. TRIALS defaults to 5; the spec requires 30 for publishable numbers.
//!
//! Output: crates/clawhdf5-format/benches/results/baselines-$(hostname).csv
//! and a companion `baselines-capability-matrix-$(hostname).md`
//! (both gitignored by `benches/results/` — commit deliberately with
//! `git add -f` if intended as a P1.2b artifact).
//!
//! NOAA_PATH/NOAA_DATASET are optional: if given, the harness additionally
//! extracts real on-disk chunk bytes from that dataset (filtered/compressed
//! chunks are hashed as-is, same as the existing filter-agnostic Merkle
//! code) and reruns the same suite tagged `source=NOAA`. If extraction
//! fails or the chunk-index type is unsupported, that source is skipped
//! with a printed warning rather than aborting the synthetic run. Defaults
//! to the `Rad` dataset (GOES-18 ABI L1b radiance) when present.
//!
//! Also emits, per real chunk size measured: an N=10^7 Ed25519 projection
//! (storage and single-core signing cost) per §4.3, and explicit capability
//! rows for `sig_mldsa` (ML-DSA-65 hybrid, deferred to Phase 2 — not
//! implemented, so every metric is recorded as unsupported rather than
//! silently omitted from the matrix).

use clawhdf5::File;
use clawhdf5_format::baselines::{
    BaselineError, FlatHashBackend, PerChunkMacBackend, PerChunkSigBackend,
};
use clawhdf5_format::chunked_read::collect_chunk_info;
use clawhdf5_format::data_layout::DataLayout;
use clawhdf5_format::extensible_array::{ExtensibleArrayHeader, read_extensible_array_chunks};
use clawhdf5_format::filter_pipeline::FilterPipeline;
use clawhdf5_format::fixed_array::{FixedArrayHeader, read_fixed_array_chunks};
use clawhdf5_format::group_v2;
use clawhdf5_format::message_type::MessageType;
use clawhdf5_format::object_header::ObjectHeader;
use clawhdf5_format::signature;
use clawhdf5_format::superblock::Superblock;

use std::fs;
use std::time::Instant;

const CHUNK_SIZES_KB: &[usize] = &[64, 256, 1024];

struct Row {
    backend: &'static str,
    metric: &'static str,
    n_chunks: usize,
    chunk_size_kb: usize,
    value: f64,
    unit: &'static str,
    supported: bool,
    source: &'static str,
    trial: usize,
}

fn csv_field(s: &str) -> String {
    if s.contains(',') || s.contains('"') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn cpu_model() -> String {
    fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn hostname() -> String {
    std::process::Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Deterministic, non-trivially-compressible chunk data — same pattern as
/// `hash_bench.rs::make_chunk`, salted by chunk index so chunks differ.
fn make_chunk(size: usize, idx: usize) -> Vec<u8> {
    (0..size)
        .map(|i| ((i ^ (i >> 8) ^ (i >> 16)).wrapping_add(idx)) as u8)
        .collect()
}

/// Run the full metric battery for one (chunk_size, n_chunks, source) cell
/// against all three backends, appending rows to `rows`.
fn bench_cell(
    rows: &mut Vec<Row>,
    chunks: &[&[u8]],
    chunk_size_kb: usize,
    trials: usize,
    source: &'static str,
) {
    let n_chunks = chunks.len();
    if n_chunks == 0 {
        return;
    }

    // --- flat hash ---
    for trial in 1..=trials {
        let start = Instant::now();
        let backend = FlatHashBackend::commit(chunks);
        let commit_ns = start.elapsed().as_nanos() as f64;
        rows.push(Row {
            backend: "flat",
            metric: "commit_time",
            n_chunks,
            chunk_size_kb,
            value: commit_ns,
            unit: "ns",
            supported: true,
            source,
            trial,
        });

        let start = Instant::now();
        let ok = backend.verify_dataset(chunks);
        let verify_ns = start.elapsed().as_nanos() as f64;
        assert!(ok, "flat hash failed to verify its own commitment");
        rows.push(Row {
            backend: "flat",
            metric: "verify_dataset_latency",
            n_chunks,
            chunk_size_kb,
            value: verify_ns,
            unit: "ns",
            supported: true,
            source,
            trial,
        });

        // append/update are both O(N) re-commits for a flat hash; that cost
        // *is* the capability gap being measured.
        let mut all: Vec<&[u8]> = chunks.to_vec();
        let extra = make_chunk(chunk_size_kb * 1024, n_chunks);
        all.push(&extra);
        let mut mutable_backend = backend;
        let start = Instant::now();
        mutable_backend.append(&all);
        rows.push(Row {
            backend: "flat",
            metric: "append_time",
            n_chunks,
            chunk_size_kb,
            value: start.elapsed().as_nanos() as f64,
            unit: "ns",
            supported: true,
            source,
            trial,
        });

        let start = Instant::now();
        mutable_backend.update(&all);
        rows.push(Row {
            backend: "flat",
            metric: "update_time",
            n_chunks,
            chunk_size_kb,
            value: start.elapsed().as_nanos() as f64,
            unit: "ns",
            supported: true,
            source,
            trial,
        });
    }
    let flat = FlatHashBackend::commit(chunks);
    rows.push(Row {
        backend: "flat",
        metric: "metadata_bytes",
        n_chunks,
        chunk_size_kb,
        value: flat.metadata_bytes() as f64,
        unit: "bytes",
        supported: true,
        source,
        trial: 1,
    });
    match flat.verify_chunk(0, chunks[0]) {
        Ok(_) => unreachable!("flat hash must not support verify_chunk"),
        Err(BaselineError::Unsupported { .. }) => rows.push(Row {
            backend: "flat",
            metric: "verify_chunk_latency",
            n_chunks,
            chunk_size_kb,
            value: 0.0,
            unit: "ns",
            supported: false,
            source,
            trial: 1,
        }),
        Err(e) => panic!("unexpected flat hash error: {e}"),
    }
    rows.push(Row {
        backend: "flat",
        metric: "public_verification",
        n_chunks,
        chunk_size_kb,
        value: 1.0,
        unit: "bool",
        supported: true,
        source,
        trial: 1,
    });

    // --- per-chunk MAC ---
    for trial in 1..=trials {
        let start = Instant::now();
        let mut backend = PerChunkMacBackend::commit(chunks);
        rows.push(Row {
            backend: "mac",
            metric: "commit_time",
            n_chunks,
            chunk_size_kb,
            value: start.elapsed().as_nanos() as f64,
            unit: "ns",
            supported: true,
            source,
            trial,
        });

        let start = Instant::now();
        let ok = backend.verify_chunk(0, chunks[0]);
        rows.push(Row {
            backend: "mac",
            metric: "verify_chunk_latency",
            n_chunks,
            chunk_size_kb,
            value: start.elapsed().as_nanos() as f64,
            unit: "ns",
            supported: true,
            source,
            trial,
        });
        assert_eq!(ok, Ok(true));

        let start = Instant::now();
        let ok = backend.verify_dataset(chunks);
        rows.push(Row {
            backend: "mac",
            metric: "verify_dataset_latency",
            n_chunks,
            chunk_size_kb,
            value: start.elapsed().as_nanos() as f64,
            unit: "ns",
            supported: true,
            source,
            trial,
        });
        assert!(ok);

        let extra = make_chunk(chunk_size_kb * 1024, n_chunks);
        let start = Instant::now();
        backend.append(&extra);
        rows.push(Row {
            backend: "mac",
            metric: "append_time",
            n_chunks,
            chunk_size_kb,
            value: start.elapsed().as_nanos() as f64,
            unit: "ns",
            supported: true,
            source,
            trial,
        });

        let updated = make_chunk(chunk_size_kb * 1024, 0xDEAD);
        let start = Instant::now();
        backend.update(0, &updated).expect("chunk 0 was committed");
        rows.push(Row {
            backend: "mac",
            metric: "update_time",
            n_chunks,
            chunk_size_kb,
            value: start.elapsed().as_nanos() as f64,
            unit: "ns",
            supported: true,
            source,
            trial,
        });
    }
    let mac = PerChunkMacBackend::commit(chunks);
    rows.push(Row {
        backend: "mac",
        metric: "metadata_bytes",
        n_chunks,
        chunk_size_kb,
        value: mac.metadata_bytes() as f64,
        unit: "bytes",
        supported: true,
        source,
        trial: 1,
    });
    rows.push(Row {
        backend: "mac",
        metric: "public_verification",
        n_chunks,
        chunk_size_kb,
        value: 0.0,
        unit: "bool",
        supported: false,
        source,
        trial: 1,
    });

    // --- per-chunk Ed25519 ---
    for trial in 1..=trials {
        let start = Instant::now();
        let mut backend = PerChunkSigBackend::commit(chunks);
        rows.push(Row {
            backend: "sig_ed25519",
            metric: "commit_time",
            n_chunks,
            chunk_size_kb,
            value: start.elapsed().as_nanos() as f64,
            unit: "ns",
            supported: true,
            source,
            trial,
        });

        let start = Instant::now();
        let ok = backend.verify_chunk(0, chunks[0]);
        rows.push(Row {
            backend: "sig_ed25519",
            metric: "verify_chunk_latency",
            n_chunks,
            chunk_size_kb,
            value: start.elapsed().as_nanos() as f64,
            unit: "ns",
            supported: true,
            source,
            trial,
        });
        assert_eq!(ok, Ok(true));

        let start = Instant::now();
        let ok = backend.verify_dataset(chunks);
        rows.push(Row {
            backend: "sig_ed25519",
            metric: "verify_dataset_latency",
            n_chunks,
            chunk_size_kb,
            value: start.elapsed().as_nanos() as f64,
            unit: "ns",
            supported: true,
            source,
            trial,
        });
        assert!(ok);

        let extra = make_chunk(chunk_size_kb * 1024, n_chunks);
        let start = Instant::now();
        backend.append(&extra);
        rows.push(Row {
            backend: "sig_ed25519",
            metric: "append_time",
            n_chunks,
            chunk_size_kb,
            value: start.elapsed().as_nanos() as f64,
            unit: "ns",
            supported: true,
            source,
            trial,
        });

        let updated = make_chunk(chunk_size_kb * 1024, 0xDEAD);
        let start = Instant::now();
        backend.update(0, &updated).expect("chunk 0 was committed");
        rows.push(Row {
            backend: "sig_ed25519",
            metric: "update_time",
            n_chunks,
            chunk_size_kb,
            value: start.elapsed().as_nanos() as f64,
            unit: "ns",
            supported: true,
            source,
            trial,
        });
    }
    let sig = PerChunkSigBackend::commit(chunks);
    rows.push(Row {
        backend: "sig_ed25519",
        metric: "metadata_bytes",
        n_chunks,
        chunk_size_kb,
        value: sig.metadata_bytes() as f64,
        unit: "bytes",
        supported: true,
        source,
        trial: 1,
    });
    rows.push(Row {
        backend: "sig_ed25519",
        metric: "public_verification",
        n_chunks,
        chunk_size_kb,
        value: 1.0,
        unit: "bool",
        supported: true,
        source,
        trial: 1,
    });
    let k = (n_chunks / 10).max(1);
    rows.push(Row {
        backend: "sig_ed25519",
        metric: "subset_proof_bytes",
        n_chunks,
        chunk_size_kb,
        value: PerChunkSigBackend::subset_proof_bytes(k) as f64,
        unit: "bytes",
        supported: true,
        source,
        trial: 1,
    });
}

/// Best-effort extraction of raw on-disk chunk bytes from a real HDF5/
/// NetCDF-4 file (NOAA product or the P1.1 synthetic generator's output —
/// the parsing is generic). Returns `None` (with a printed warning) rather
/// than erroring out, since callers treat this leg as optional.
///
/// The bytes returned are whatever is physically stored per chunk —
/// filtered/compressed if the dataset uses a filter pipeline. The baseline
/// backends (like `merkle.rs`) are filter-agnostic: they hash/sign whatever
/// byte slice they're given, so this matches how the existing Merkle code
/// treats chunks and lets the harness run against real (compressed) NOAA
/// products instead of only fabricated, uncompressed data.
fn extract_chunks_from_file(
    path: &str,
    dataset_override: Option<&str>,
) -> Option<(Vec<Vec<u8>>, usize)> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("warning: failed to open {path}: {e}");
            return None;
        }
    };
    let datasets = file.root().datasets().ok()?;
    let test_dataset = dataset_override.map(|s| s.to_string()).or_else(|| {
        if datasets.contains(&"Rad".to_string()) {
            Some("Rad".to_string())
        } else {
            datasets.first().cloned()
        }
    })?;

    let file_data = fs::read(path).ok()?;
    let sig_offset = signature::find_signature(&file_data).ok()?;
    let superblock = Superblock::parse(&file_data, sig_offset).ok()?;
    let addr = group_v2::resolve_path_any(&file_data, &superblock, &test_dataset).ok()?;
    let header = ObjectHeader::parse(
        &file_data,
        addr as usize,
        superblock.offset_size,
        superblock.length_size,
    )
    .ok()?;

    let pipeline = header
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::FilterPipeline)
        .and_then(|msg| FilterPipeline::parse(&msg.data).ok());
    if let Some(p) = &pipeline {
        println!(
            "  note: dataset '{test_dataset}' has {} filter(s) applied; hashing on-disk (filtered) chunk bytes",
            p.filters.len()
        );
    }

    let layout_msg = header
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::DataLayout)?;
    let layout = DataLayout::parse(
        &layout_msg.data,
        superblock.offset_size,
        superblock.length_size,
    )
    .ok()?;
    let shape = file.dataset(&test_dataset).ok()?.shape().ok()?;

    let (chunks, chunk_dims, element_size) = match &layout {
        DataLayout::Chunked {
            chunk_dimensions,
            btree_address,
            version,
            chunk_index_type,
            ..
        } => {
            let addr = (*btree_address)?;
            let rank = shape.len();
            let spatial_dims: Vec<u32> = chunk_dimensions[..rank].to_vec();
            let elem_size = *chunk_dimensions.last().unwrap_or(&1) as usize;
            let chunks = match (*version, *chunk_index_type) {
                (3, _) => collect_chunk_info(
                    &file_data,
                    addr,
                    rank + 1,
                    superblock.offset_size,
                    superblock.length_size,
                )
                .ok()?,
                (4, Some(3)) => {
                    let fa = FixedArrayHeader::parse(
                        &file_data,
                        addr as usize,
                        superblock.offset_size,
                        superblock.length_size,
                    )
                    .ok()?;
                    read_fixed_array_chunks(
                        &file_data,
                        &fa,
                        &shape,
                        &spatial_dims,
                        elem_size as u32,
                        superblock.offset_size,
                        superblock.length_size,
                    )
                    .ok()?
                }
                (4, Some(4)) => {
                    let ea = ExtensibleArrayHeader::parse(
                        &file_data,
                        addr as usize,
                        superblock.offset_size,
                        superblock.length_size,
                    )
                    .ok()?;
                    read_extensible_array_chunks(
                        &file_data,
                        &ea,
                        &shape,
                        &spatial_dims,
                        elem_size as u32,
                        superblock.offset_size,
                        superblock.length_size,
                    )
                    .ok()?
                }
                _ => {
                    eprintln!(
                        "warning: unsupported chunk index for '{test_dataset}'; skipping this source"
                    );
                    return None;
                }
            };
            (chunks, spatial_dims, elem_size)
        }
        _ => {
            eprintln!("warning: dataset '{test_dataset}' is not chunked; skipping this source");
            return None;
        }
    };

    let chunk_elements: usize = chunk_dims.iter().map(|&d| d as usize).product();
    let nominal_size = chunk_elements * element_size;

    let mut out = Vec::with_capacity(chunks.len());
    for c in &chunks {
        let start = c.address as usize;
        let end = start + c.chunk_size as usize;
        if end > file_data.len() {
            eprintln!(
                "warning: chunk at 0x{:x} extends past EOF; skipping this source",
                c.address
            );
            return None;
        }
        out.push(file_data[start..end].to_vec());
    }
    if out.is_empty() {
        return None;
    }
    Some((out, nominal_size.max(1) / 1024))
}

/// N=10^7 Ed25519 projection per §4.3: takes the measured per-trial
/// commit_time and metadata_bytes for one (chunk_size, source) cell already
/// in `rows`, derives per-chunk sign cost and per-chunk storage from the
/// real measurement, and projects both to N=10^7 chunks. Appends two rows
/// (`projected_storage_bytes_1e7`, `projected_signing_cpu_hours_1e7`) tagged
/// with the same chunk_size_kb/source so they can be cross-referenced.
fn add_ed25519_projection(rows: &mut Vec<Row>, chunk_size_kb: usize, source: &'static str) {
    const PROJECTED_N: f64 = 10_000_000.0;

    let commit_samples: Vec<f64> = rows
        .iter()
        .filter(|r| {
            r.backend == "sig_ed25519"
                && r.metric == "commit_time"
                && r.chunk_size_kb == chunk_size_kb
                && r.source == source
        })
        .map(|r| r.value)
        .collect();
    let n_chunks = rows
        .iter()
        .find(|r| {
            r.backend == "sig_ed25519"
                && r.metric == "commit_time"
                && r.chunk_size_kb == chunk_size_kb
                && r.source == source
        })
        .map(|r| r.n_chunks);
    let metadata_bytes = rows.iter().find(|r| {
        r.backend == "sig_ed25519"
            && r.metric == "metadata_bytes"
            && r.chunk_size_kb == chunk_size_kb
            && r.source == source
    });
    let (Some(n_chunks), Some(metadata_bytes)) = (n_chunks, metadata_bytes) else {
        return;
    };
    if commit_samples.is_empty() || n_chunks == 0 {
        return;
    }

    let mean_commit_ns = commit_samples.iter().sum::<f64>() / commit_samples.len() as f64;
    let per_chunk_sign_ns = mean_commit_ns / n_chunks as f64;
    let per_chunk_storage_bytes = metadata_bytes.value / n_chunks as f64;

    let projected_storage_bytes = PROJECTED_N * per_chunk_storage_bytes;
    let projected_signing_cpu_hours = (PROJECTED_N * per_chunk_sign_ns) / 1e9 / 3600.0;

    rows.push(Row {
        backend: "sig_ed25519",
        metric: "projected_storage_bytes_1e7",
        n_chunks: PROJECTED_N as usize,
        chunk_size_kb,
        value: projected_storage_bytes,
        unit: "bytes",
        supported: true,
        source,
        trial: 1,
    });
    rows.push(Row {
        backend: "sig_ed25519",
        metric: "projected_signing_cpu_hours_1e7",
        n_chunks: PROJECTED_N as usize,
        chunk_size_kb,
        value: projected_signing_cpu_hours,
        unit: "cpu_hours",
        supported: true,
        source,
        trial: 1,
    });
    println!(
        "  N=1e7 Ed25519 projection ({chunk_size_kb} KB chunks, {source}): \
         storage={:.1} MB, signing={:.3} CPU-hours (single core)",
        projected_storage_bytes / 1e6,
        projected_signing_cpu_hours
    );
}

/// `sig_mldsa` (ML-DSA-65 hybrid) is explicitly deferred to Phase 2 and is
/// not implemented in `clawhdf5_format::baselines`. Record every metric as
/// an unsupported capability gap so the matrix reports a known absence
/// rather than silently omitting the backend.
fn add_mldsa_capability_gap(rows: &mut Vec<Row>) {
    const METRICS: &[(&str, &str)] = &[
        ("commit_time", "ns"),
        ("verify_chunk_latency", "ns"),
        ("verify_dataset_latency", "ns"),
        ("append_time", "ns"),
        ("update_time", "ns"),
        ("metadata_bytes", "bytes"),
        ("public_verification", "bool"),
        ("subset_proof_bytes", "bytes"),
    ];
    for &(metric, unit) in METRICS {
        rows.push(Row {
            backend: "sig_mldsa",
            metric,
            n_chunks: 0,
            chunk_size_kb: 0,
            value: 0.0,
            unit,
            supported: false,
            source: "N/A_not_implemented_phase2",
            trial: 1,
        });
    }
}

/// Write a human-readable backend x metric capability matrix, sourced from
/// the `chunk_size_kb` cell's rows (averaging time metrics over trials).
fn write_capability_matrix(
    rows: &[Row],
    out_path: &str,
    chunk_size_kb: usize,
    source: &'static str,
) -> std::io::Result<()> {
    const BACKENDS: &[&str] = &["flat", "mac", "sig_ed25519", "sig_mldsa"];
    const METRICS: &[&str] = &[
        "commit_time",
        "verify_chunk_latency",
        "verify_dataset_latency",
        "append_time",
        "update_time",
        "metadata_bytes",
        "public_verification",
        "subset_proof_bytes",
    ];

    let mut md = String::new();
    md.push_str("# P1.2b Capability Matrix\n\n");
    md.push_str(&format!(
        "Representative cell: chunk_size={chunk_size_kb} KB, source={source} \
         (time metrics averaged over all trials).\n\n"
    ));
    md.push_str(
        "| metric | flat (SHA-256) | mac (HMAC-SHA-256) | sig_ed25519 | sig_mldsa (ML-DSA-65) |\n",
    );
    md.push_str("|---|---|---|---|---|\n");

    for &metric in METRICS {
        md.push_str(&format!("| {metric} |"));
        for &backend in BACKENDS {
            let matching: Vec<&Row> = rows
                .iter()
                .filter(|r| {
                    r.backend == backend
                        && r.metric == metric
                        && (backend == "sig_mldsa"
                            || (r.chunk_size_kb == chunk_size_kb && r.source == source))
                })
                .collect();
            if matching.is_empty() {
                md.push_str(" N/A |");
                continue;
            }
            let supported = matching[0].supported;
            if !supported {
                let why = match (backend, metric) {
                    ("flat", "verify_chunk_latency") => "N/A — no partial access",
                    ("mac", "public_verification") => "N/A — no public key",
                    ("sig_mldsa", _) => "not implemented (Phase 2)",
                    _ => "unsupported",
                };
                md.push_str(&format!(" {why} |"));
                continue;
            }
            let mean = matching.iter().map(|r| r.value).sum::<f64>() / matching.len() as f64;
            let unit = matching[0].unit;
            let formatted = match unit {
                "ns" => format!("{:.0} ns", mean),
                "bool" => (if mean >= 1.0 { "Yes" } else { "No" }).to_string(),
                _ => format!("{:.0} {unit}", mean),
            };
            md.push_str(&format!(" {formatted} |"));
        }
        md.push('\n');
    }

    fs::write(out_path, md)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let synthetic_path = args.get(1).filter(|s| !s.is_empty()).cloned();
    let trials: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let noaa_path = args.get(3).cloned();
    let noaa_dataset = args.get(4).cloned();

    println!("=== P1.2b baseline benchmark harness ===");
    println!("trials={trials}");

    let host = hostname();
    let cpu = cpu_model();
    let mut rows: Vec<Row> = Vec::new();

    if let Some(path) = &synthetic_path {
        const P1_1_DATASETS: &[(&str, usize)] = &[
            ("dataset_64kb", 64),
            ("dataset_256kb", 256),
            ("dataset_1mb", 1024),
        ];
        for &(dataset, nominal_kb) in P1_1_DATASETS {
            println!("--- synthetic (P1.1 real file): {dataset} ---");
            match extract_chunks_from_file(path, Some(dataset)) {
                Some((owned, kb)) => {
                    let refs: Vec<&[u8]> = owned.iter().map(|c| c.as_slice()).collect();
                    println!("  extracted {} chunks (~{kb} KB each)", refs.len());
                    bench_cell(&mut rows, &refs, nominal_kb, trials, "synthetic");
                    add_ed25519_projection(&mut rows, nominal_kb, "synthetic");
                }
                None => {
                    eprintln!("warning: failed to extract '{dataset}' from {path}; skipping");
                }
            }
        }
    } else {
        println!("(no SYNTHETIC_H5_PATH given — using fabricated in-memory chunks)");
        for &kb in CHUNK_SIZES_KB {
            println!("--- synthetic_fabricated: {kb} KB chunks ---");
            let owned: Vec<Vec<u8>> = (0..256).map(|i| make_chunk(kb * 1024, i)).collect();
            let refs: Vec<&[u8]> = owned.iter().map(|c| c.as_slice()).collect();
            bench_cell(&mut rows, &refs, kb, trials, "synthetic_fabricated");
            add_ed25519_projection(&mut rows, kb, "synthetic_fabricated");
        }
    }

    if let Some(path) = noaa_path {
        println!("--- NOAA: {path} ---");
        if let Some((owned, kb)) = extract_chunks_from_file(&path, noaa_dataset.as_deref()) {
            let refs: Vec<&[u8]> = owned.iter().map(|c| c.as_slice()).collect();
            println!("  extracted {} chunks (~{kb} KB each)", refs.len());
            bench_cell(&mut rows, &refs, kb, trials, "NOAA");
            add_ed25519_projection(&mut rows, kb, "NOAA");
        }
    }

    add_mldsa_capability_gap(&mut rows);

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let results_dir = format!("{manifest_dir}/../clawhdf5-format/benches/results");
    fs::create_dir_all(&results_dir)?;
    let out_path = format!("{results_dir}/baselines-{host}.csv");

    let mut csv = String::from(
        "backend,metric,n_chunks,chunk_size_kb,value,unit,supported,source,trial,hostname,cpu_model\n",
    );
    for r in &rows {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{}\n",
            csv_field(r.backend),
            csv_field(r.metric),
            r.n_chunks,
            r.chunk_size_kb,
            r.value,
            csv_field(r.unit),
            r.supported,
            csv_field(r.source),
            r.trial,
            csv_field(&host),
            csv_field(&cpu),
        ));
    }
    fs::write(&out_path, csv)?;

    let matrix_source = if synthetic_path.is_some() {
        "synthetic"
    } else {
        "synthetic_fabricated"
    };
    let matrix_path = format!("{results_dir}/baselines-capability-matrix-{host}.md");
    write_capability_matrix(&rows, &matrix_path, 256, matrix_source)?;

    println!();
    println!("Wrote {} rows to {out_path}", rows.len());
    println!("Wrote capability matrix to {matrix_path}");
    println!(
        "Note: benches/results/ is gitignored — `git add -f` these files if you intend to commit them as P1.2b artifacts."
    );

    Ok(())
}
