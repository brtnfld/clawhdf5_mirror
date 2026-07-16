//! P1.6 Phase 1 benchmarks — storage overhead, partial-verification latency,
//! incremental update cost (RQ1–RQ3).
//!
//! Runs all scenarios for the Merkle tree and the P1.2b baselines against
//! both a synthetic 10 GB file (P1.1) and a real NOAA dataset, then writes
//! `benches/results/phase1-$(hostname).csv` in the format required by §7.
//!
//! Usage:
//!   cargo run --example phase1_bench --release \
//!     --features "parallel" \
//!     -- <synthetic_10gb.h5> <noaa_sample.nc>
//!
//! The `baselines` feature on `clawhdf5-format` must be active (it is set in
//! the dev-dependency in Cargo.toml).

use clawhdf5::File;
use clawhdf5_format::baselines::FlatHashBackend;
use clawhdf5_format::chunked_read::{ChunkInfo, collect_chunk_info};
use clawhdf5_format::data_layout::DataLayout;
use clawhdf5_format::dataspace::Dataspace;
use clawhdf5_format::extensible_array::{ExtensibleArrayHeader, read_extensible_array_chunks};
use clawhdf5_format::file_writer::FileWriter;
use clawhdf5_format::filter_pipeline::FilterPipeline;
use clawhdf5_format::fixed_array::{FixedArrayHeader, read_fixed_array_chunks};
use clawhdf5_format::group_v2::resolve_path_any;
use clawhdf5_format::merkle::{
    HashAlg, MerkleCompanionResult, MerkleTree, hash_chunk, write_merkle_companion,
};
use clawhdf5_format::message_type::MessageType;
use clawhdf5_format::object_header::ObjectHeader;
use clawhdf5_format::signature::find_signature;
use clawhdf5_format::superblock::Superblock;

use std::io::Write as IoWrite;
use std::time::Instant;

const TRIALS: usize = 30;
/// Discarded warmup iterations run (and not recorded) before the `TRIALS`
/// measured iterations, matching `hash_bench_harness.rs`/`baselines_bench.rs`
/// and the Statistical Protocol ("minimum 30 trials after 5 discarded
/// warmups"). Without this, cold-start effects (page faults, allocator
/// growth, frequency-scaling ramp) leak into trial 1 of every cell.
const WARMUP_TRIALS: usize = 5;
const COMPANION_SERIALIZATION_NS: &[usize] = &[16, 256, 1_024, 65_536];

// ---------------------------------------------------------------------------
// RSS measurement
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
fn peak_rss_mb() -> f64 {
    // VmHWM ("high water mark") is the actual peak RSS the process has ever
    // reached; VmRSS is only the current RSS at the time of the read, which
    // would understate the peak if memory was freed before this was called.
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<f64>().ok())
        })
        .map(|kb| kb / 1024.0)
        .unwrap_or(0.0)
}

#[cfg(target_os = "macos")]
fn peak_rss_mb() -> f64 {
    // getrusage(RUSAGE_SELF) — ru_maxrss is bytes on macOS
    use std::mem::MaybeUninit;
    #[repr(C)]
    struct Timeval {
        tv_sec: i64,
        tv_usec: i32,
        _pad: i32,
    }
    #[repr(C)]
    struct Rusage {
        ru_utime: Timeval,
        ru_stime: Timeval,
        ru_maxrss: i64,
        _rest: [i64; 14],
    }
    unsafe extern "C" {
        fn getrusage(who: i32, usage: *mut Rusage) -> i32;
    }
    let mut u = MaybeUninit::<Rusage>::zeroed();
    unsafe {
        getrusage(0, u.as_mut_ptr());
        u.assume_init().ru_maxrss as f64 / 1_048_576.0
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn peak_rss_mb() -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// Disk type detection
// ---------------------------------------------------------------------------

fn detect_disk_type() -> String {
    #[cfg(target_os = "macos")]
    {
        if let Ok(out) = std::process::Command::new("diskutil")
            .args(["info", "/"])
            .output()
        {
            let s = String::from_utf8_lossy(&out.stdout);
            if s.contains("Solid State") {
                return "ssd".into();
            }
            if s.contains("Rotational") {
                return "hdd".into();
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        // /sys/block/sdX/queue/rotational: 0 = SSD, 1 = HDD
        if let Ok(entries) = std::fs::read_dir("/sys/block") {
            for entry in entries.flatten() {
                let rot = entry.path().join("queue/rotational");
                if let Ok(s) = std::fs::read_to_string(rot) {
                    return if s.trim() == "0" {
                        "ssd".into()
                    } else {
                        "hdd".into()
                    };
                }
            }
        }
    }
    "unknown".into()
}

// ---------------------------------------------------------------------------
// HDF5 chunk loader
// ---------------------------------------------------------------------------

struct LoadedFile {
    chunks: Vec<Vec<u8>>, // raw on-disk chunk bytes (may be compressed)
    chunk_size_kb: f64,
    n_chunks: usize,
}

/// Load `dataset_name`'s chunks if given; otherwise fall back to whichever
/// known dataset name appears first, or the first dataset in the file.
fn load_chunks(
    path: &str,
    dataset_name: Option<&str>,
) -> Result<LoadedFile, Box<dyn std::error::Error>> {
    // Use high-level API to list datasets, low-level API to read raw chunk bytes.
    let hf = File::open(path)?;
    let root = hf.root();
    let names = root.datasets()?;

    let dataset_name = match dataset_name {
        Some(name) => name.to_string(),
        None => names
            .iter()
            .find(|n| {
                matches!(
                    n.as_str(),
                    "Rad" | "dataset_1mb" | "dataset_64kb" | "dataset_256kb"
                )
            })
            .or_else(|| names.first())
            .cloned()
            .ok_or("no datasets found in root group")?,
    };

    let file_data = std::fs::read(path)?;
    let sig = find_signature(&file_data)?;
    let sb = Superblock::parse(&file_data, sig)?;

    let addr = resolve_path_any(&file_data, &sb, &dataset_name)?;
    let hdr = ObjectHeader::parse(&file_data, addr as usize, sb.offset_size, sb.length_size)?;

    let layout_msg = hdr
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::DataLayout)
        .ok_or("no DataLayout message")?;
    let layout = DataLayout::parse(&layout_msg.data, sb.offset_size, sb.length_size)?;

    let _pipeline = hdr
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::FilterPipeline)
        .and_then(|m| FilterPipeline::parse(&m.data).ok());

    let ds_msg = hdr
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::Dataspace)
        .ok_or("no Dataspace message")?;
    let ds = Dataspace::parse(&ds_msg.data, sb.length_size)?;

    let chunk_infos: Vec<ChunkInfo> = match &layout {
        DataLayout::Chunked {
            chunk_dimensions,
            btree_address,
            version,
            chunk_index_type,
            ..
        } => {
            let baddr = btree_address.ok_or("no btree address")?;
            let rank = ds.dimensions.len();
            let elem_size = *chunk_dimensions.last().unwrap_or(&1) as usize;
            let spatial: Vec<u32> = chunk_dimensions[..rank].to_vec();

            match (*version, *chunk_index_type) {
                (3, _) => {
                    collect_chunk_info(&file_data, baddr, rank + 1, sb.offset_size, sb.length_size)?
                }
                (4, Some(1)) => vec![ChunkInfo {
                    chunk_size: spatial.iter().map(|&d| d as u64).product::<u64>() as u32
                        * elem_size as u32,
                    filter_mask: 0,
                    offsets: vec![0; rank],
                    address: baddr,
                }],
                (4, Some(3)) => {
                    let fh = FixedArrayHeader::parse(
                        &file_data,
                        baddr as usize,
                        sb.offset_size,
                        sb.length_size,
                    )?;
                    read_fixed_array_chunks(
                        &file_data,
                        &fh,
                        &ds.dimensions,
                        &spatial,
                        elem_size as u32,
                        sb.offset_size,
                        sb.length_size,
                    )?
                }
                (4, Some(6)) => {
                    let ea = ExtensibleArrayHeader::parse(
                        &file_data,
                        baddr as usize,
                        sb.offset_size,
                        sb.length_size,
                    )?;
                    read_extensible_array_chunks(
                        &file_data,
                        &ea,
                        &ds.dimensions,
                        &spatial,
                        elem_size as u32,
                        sb.offset_size,
                        sb.length_size,
                    )?
                }
                _ => {
                    return Err(format!(
                        "unsupported layout v{} index {:?}",
                        version, chunk_index_type
                    )
                    .into());
                }
            }
        }
        _ => return Err("dataset is not chunked".into()),
    };

    // Read raw (possibly compressed) bytes for every allocated chunk.
    let mut chunks = Vec::with_capacity(chunk_infos.len());
    let mut total_bytes: u64 = 0;
    for ci in &chunk_infos {
        let start = ci.address as usize;
        let end = start + ci.chunk_size as usize;
        if end > file_data.len() {
            continue;
        }
        chunks.push(file_data[start..end].to_vec());
        total_bytes += ci.chunk_size as u64;
    }

    let n = chunks.len();
    let avg_kb = if n > 0 {
        (total_bytes as f64 / n as f64) / 1024.0
    } else {
        0.0
    };

    Ok(LoadedFile {
        chunks,
        chunk_size_kb: avg_kb,
        n_chunks: n,
    })
}

// ---------------------------------------------------------------------------
// CSV writer
// ---------------------------------------------------------------------------

struct CsvRow {
    source: &'static str,
    scenario: &'static str,
    n_chunks: usize,
    chunk_size_kb: f64,
    wall_time_ms: f64,
    peak_rss_mb: f64,
    companion_bytes: u64,
    trial: usize,
    hostname: String,
    disk_type: String,
}

fn write_header(w: &mut impl IoWrite, hostname: &str) -> std::io::Result<()> {
    let cpu = cpu_model();
    let ram = ram_gb();
    let date = {
        // Simple RFC-3339 date via `date` command
        std::process::Command::new("date")
            .arg("+%Y-%m-%dT%H:%M:%SZ")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .unwrap_or_default()
            .trim()
            .to_string()
    };
    writeln!(
        w,
        "# hostname={hostname} cpu_model=\"{cpu}\" ram_gb={ram:.1} date={date}"
    )?;
    writeln!(
        w,
        "source,scenario,n_chunks,chunk_size_kb,wall_time_ms,peak_rss_mb,\
         companion_bytes,trial,hostname,disk_type"
    )
}

fn write_row(w: &mut impl IoWrite, r: &CsvRow) -> std::io::Result<()> {
    writeln!(
        w,
        "{},{},{},{:.1},{:.3},{:.1},{},{},{},{}",
        r.source,
        r.scenario,
        r.n_chunks,
        r.chunk_size_kb,
        r.wall_time_ms,
        r.peak_rss_mb,
        r.companion_bytes,
        r.trial,
        r.hostname,
        r.disk_type
    )
}

// ---------------------------------------------------------------------------
// System info helpers
// ---------------------------------------------------------------------------

fn hostname() -> String {
    std::process::Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn cpu_model() -> String {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .unwrap_or_default()
            .trim()
            .to_string()
    }
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("model name"))
                    .and_then(|l| l.split(':').nth(1))
                    .map(|v| v.trim().to_string())
            })
            .unwrap_or_default()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        "unknown".into()
    }
}

fn ram_gb() -> f64 {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse::<u64>().ok())
            .map(|b| b as f64 / 1_073_741_824.0)
            .unwrap_or(0.0)
    }
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("MemTotal:"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|v| v.parse::<f64>().ok())
            })
            .map(|kb| kb / 1_048_576.0)
            .unwrap_or(0.0)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Scenario runners
// ---------------------------------------------------------------------------

/// Run `WARMUP_TRIALS` discarded iterations, then time `TRIALS` measured
/// iterations, returning per-trial ms values for the measured set only.
fn time_trials<F: FnMut()>(mut f: F) -> Vec<f64> {
    for _ in 0..WARMUP_TRIALS {
        f();
    }
    (0..TRIALS)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1000.0
        })
        .collect()
}

fn run_source(
    source: &'static str,
    lf: &LoadedFile,
    hostname: &str,
    disk_type: &str,
    rows: &mut Vec<CsvRow>,
) {
    let chunks_ref: Vec<&[u8]> = lf.chunks.iter().map(|c| c.as_slice()).collect();
    let n = lf.n_chunks;
    let kb = lf.chunk_size_kb;

    // Pre-build tree once for scenarios that only need it as a reference.
    let alg = HashAlg::Blake3;
    let tree = MerkleTree::from_chunks_owned(&lf.chunks, alg);

    // ── verify_dataset ──────────────────────────────────────────────────────
    // Full Merkle re-hash: rebuild tree from all chunks, compare roots.
    let reference_root = *tree.root();
    for (trial, ms) in time_trials(|| {
        let t = MerkleTree::from_chunks_owned(&lf.chunks, alg);
        assert_eq!(t.root(), &reference_root);
    })
    .into_iter()
    .enumerate()
    {
        rows.push(CsvRow {
            source,
            scenario: "verify_dataset",
            n_chunks: n,
            chunk_size_kb: kb,
            wall_time_ms: ms,
            peak_rss_mb: peak_rss_mb(),
            companion_bytes: 0,
            trial: trial + 1,
            hostname: hostname.into(),
            disk_type: disk_type.into(),
        });
    }

    // ── verify_chunk ────────────────────────────────────────────────────────
    // Re-hash the chunk and compare against the leaf hash already stored in
    // this tree's own node array — an O(1) lookup against a tree the
    // verifier already holds in full, not a sibling-proof walk to the root.
    // See `verify_proof` below for the O(chunk_size + log N) proof-path
    // scenario this one does NOT measure.
    for (trial, ms) in time_trials(|| {
        let ok = tree.verify_chunk(0, &lf.chunks[0]);
        assert!(ok);
    })
    .into_iter()
    .enumerate()
    {
        rows.push(CsvRow {
            source,
            scenario: "verify_chunk",
            n_chunks: n,
            chunk_size_kb: kb,
            wall_time_ms: ms,
            peak_rss_mb: peak_rss_mb(),
            companion_bytes: 0,
            trial: trial + 1,
            hostname: hostname.into(),
            disk_type: disk_type.into(),
        });
    }

    // ── verify_proof ────────────────────────────────────────────────────────
    // The actual O(chunk_size + log N) proof-path verification: generate an
    // inclusion proof (sibling hashes, leaf to root) and verify it against
    // only the root hash, the scenario for a verifier that does NOT hold the
    // full tree (e.g. checking a chunk received in transit) — distinct from
    // verify_chunk's O(1) lookup against a locally held tree above.
    let proof_root = *tree.root();
    for (trial, ms) in time_trials(|| {
        let proof = tree.proof(0).unwrap();
        let ok = MerkleTree::verify_proof_standalone(&proof_root, 0, &lf.chunks[0], &proof);
        assert!(ok);
    })
    .into_iter()
    .enumerate()
    {
        rows.push(CsvRow {
            source,
            scenario: "verify_proof",
            n_chunks: n,
            chunk_size_kb: kb,
            wall_time_ms: ms,
            peak_rss_mb: peak_rss_mb(),
            companion_bytes: 0,
            trial: trial + 1,
            hostname: hostname.into(),
            disk_type: disk_type.into(),
        });
    }

    // ── flat_verify ─────────────────────────────────────────────────────────
    // FlatHashBackend: SHA-256 over all chunk bytes (status quo).
    let flat = FlatHashBackend::commit(&chunks_ref);
    for (trial, ms) in time_trials(|| {
        let ok = flat.verify_dataset(&chunks_ref);
        assert!(ok);
    })
    .into_iter()
    .enumerate()
    {
        rows.push(CsvRow {
            source,
            scenario: "flat_verify",
            n_chunks: n,
            chunk_size_kb: kb,
            wall_time_ms: ms,
            peak_rss_mb: peak_rss_mb(),
            companion_bytes: 0,
            trial: trial + 1,
            hostname: hostname.into(),
            disk_type: disk_type.into(),
        });
    }

    // ── extend_merkle ───────────────────────────────────────────────────────
    // O(log N) path recomputation — new chunk appended at leaf 0 (same cost).
    let new_hash = hash_chunk(b"new_chunk_data", alg);
    for (trial, ms) in time_trials(|| {
        let mut t = tree.clone();
        t.update_leaf(0, new_hash).unwrap();
    })
    .into_iter()
    .enumerate()
    {
        rows.push(CsvRow {
            source,
            scenario: "extend_merkle",
            n_chunks: n,
            chunk_size_kb: kb,
            wall_time_ms: ms,
            peak_rss_mb: peak_rss_mb(),
            companion_bytes: 0,
            trial: trial + 1,
            hostname: hostname.into(),
            disk_type: disk_type.into(),
        });
    }

    // ── update_merkle ───────────────────────────────────────────────────────
    // O(log N) in-place overwrite of leaf 0.
    let updated_hash = hash_chunk(b"updated_chunk_data", alg);
    for (trial, ms) in time_trials(|| {
        let mut t = tree.clone();
        t.update_leaf(0, updated_hash).unwrap();
    })
    .into_iter()
    .enumerate()
    {
        rows.push(CsvRow {
            source,
            scenario: "update_merkle",
            n_chunks: n,
            chunk_size_kb: kb,
            wall_time_ms: ms,
            peak_rss_mb: peak_rss_mb(),
            companion_bytes: 0,
            trial: trial + 1,
            hostname: hostname.into(),
            disk_type: disk_type.into(),
        });
    }

    // ── update_leaf_inplace ─────────────────────────────────────────────────
    // Isolates update_leaf's true O(log N) cost from the O(N) tree.clone()
    // that dominates extend_merkle/update_merkle above (see Anomaly 1 in the
    // explanatory note): clone the tree once, outside the timed region, then
    // repeatedly call update_leaf in place on that same tree.
    let mut inplace_tree = tree.clone();
    for (trial, ms) in time_trials(|| {
        inplace_tree.update_leaf(0, updated_hash).unwrap();
    })
    .into_iter()
    .enumerate()
    {
        rows.push(CsvRow {
            source,
            scenario: "update_leaf_inplace",
            n_chunks: n,
            chunk_size_kb: kb,
            wall_time_ms: ms,
            peak_rss_mb: peak_rss_mb(),
            companion_bytes: 0,
            trial: trial + 1,
            hostname: hostname.into(),
            disk_type: disk_type.into(),
        });
    }

    // ── full_rebuild ─────────────────────────────────────────────────────────
    // Build the complete tree from scratch (same operation as verify_dataset
    // but labelled separately for Figure 5 / RQ3 line chart).
    for (trial, ms) in time_trials(|| {
        let _ = MerkleTree::from_chunks_owned(&lf.chunks, alg);
    })
    .into_iter()
    .enumerate()
    {
        rows.push(CsvRow {
            source,
            scenario: "full_rebuild",
            n_chunks: n,
            chunk_size_kb: kb,
            wall_time_ms: ms,
            peak_rss_mb: peak_rss_mb(),
            companion_bytes: 0,
            trial: trial + 1,
            hostname: hostname.into(),
            disk_type: disk_type.into(),
        });
    }
}

// ── companion_serialization ─────────────────────────────────────────────────
// Measure write_merkle_companion at N ∈ {16, 256, 1024, 65536}. Uses synthetic
// 64-byte chunks so the result is independent of the source file's own chunk
// size; only the companion write cost varies with N. Run once per top-level
// `source` (not once per synthetic chunk-size dataset) since it doesn't
// depend on which P1.1 dataset is being swept in `run_source`.
//
// Named "serialization", not "io": `write_merkle_companion` is called with
// a fresh in-memory `FileWriter` (no path is ever passed to it here), so
// this measures the cost of building the inline attribute or
// companion-dataset structure in memory, never a real disk write or fsync.
fn run_companion_serialization(
    source: &'static str,
    hostname: &str,
    disk_type: &str,
    rows: &mut Vec<CsvRow>,
) {
    let alg = HashAlg::Blake3;
    for &nc in COMPANION_SERIALIZATION_NS {
        let syn_chunks: Vec<Vec<u8>> = (0..nc).map(|i| vec![(i & 0xFF) as u8; 64]).collect();
        let t = MerkleTree::from_chunks_owned(&syn_chunks, alg);

        // Measure bytes once (deterministic).
        let companion_bytes = {
            let mut fw = FileWriter::new();
            match write_merkle_companion(&mut fw, "bench", &t) {
                Ok(MerkleCompanionResult::Dataset { .. }) => t.nodes().len() as u64 * 32,
                Ok(MerkleCompanionResult::Inline { ref nodes, .. }) => nodes.len() as u64,
                Err(_) => 0,
            }
        };

        for (trial, ms) in time_trials(|| {
            let mut fw = FileWriter::new();
            let _ = write_merkle_companion(&mut fw, "bench", &t);
        })
        .into_iter()
        .enumerate()
        {
            rows.push(CsvRow {
                source,
                scenario: "companion_serialization",
                n_chunks: nc,
                chunk_size_kb: 0.0625, // 64-byte synthetic chunks
                wall_time_ms: ms,
                peak_rss_mb: peak_rss_mb(),
                companion_bytes,
                trial: trial + 1,
                hostname: hostname.into(),
                disk_type: disk_type.into(),
            });
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: phase1_bench <synthetic_10gb.h5> <noaa_sample.nc>");
        std::process::exit(1);
    }
    let synthetic_path = &args[1];
    let noaa_path = &args[2];

    let host = hostname();
    let disk = detect_disk_type();

    println!("P1.6 Phase 1 benchmarks");
    println!("  hostname  : {host}");
    println!("  cpu       : {}", cpu_model());
    println!("  ram       : {:.1} GB", ram_gb());
    println!("  disk      : {disk}");
    println!("  trials    : {TRIALS}");
    println!();

    let mut rows: Vec<CsvRow> = Vec::new();

    // Synthetic file: sweep all three P1.1 chunk-size datasets, matching the
    // P1.2b baseline harness (`baselines_bench.rs`'s `P1_1_DATASETS`), so the
    // Merkle numbers here are comparable chunk-size-by-chunk-size against the
    // already-committed `baselines-$(hostname).csv`.
    const P1_1_DATASETS: &[&str] = &["dataset_64kb", "dataset_256kb", "dataset_1mb"];
    for &dataset in P1_1_DATASETS {
        println!("Loading synthetic file: {synthetic_path} [{dataset}]");
        let syn = load_chunks(synthetic_path, Some(dataset))?;
        println!(
            "  {} chunks, avg {:.0} KB/chunk",
            syn.n_chunks, syn.chunk_size_kb
        );
        let prev = rows.len();
        run_source("synthetic", &syn, &host, &disk, &mut rows);
        println!("  {dataset} done ({} rows)", rows.len() - prev);
    }
    run_companion_serialization("synthetic", &host, &disk, &mut rows);

    // NOAA file
    println!("Loading NOAA file: {noaa_path}");
    let noaa = load_chunks(noaa_path, None)?;
    println!(
        "  {} chunks, avg {:.0} KB/chunk",
        noaa.n_chunks, noaa.chunk_size_kb
    );
    let prev = rows.len();
    run_source("NOAA", &noaa, &host, &disk, &mut rows);
    run_companion_serialization("NOAA", &host, &disk, &mut rows);
    println!("  NOAA done ({} rows)", rows.len() - prev);

    // Write CSV
    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/clawhdf5-format/benches/results");
    std::fs::create_dir_all(&out_dir)?;
    let csv_path = out_dir.join(format!("phase1-{host}.csv"));
    println!("\nWriting {}", csv_path.display());

    let mut f = std::fs::File::create(&csv_path)?;
    write_header(&mut f, &host)?;
    for r in &rows {
        write_row(&mut f, r)?;
    }

    println!("Done — {} rows written.", rows.len());
    println!(
        "\nTo run:\n  cargo run --example phase1_bench --release -- \\\n    \
         {} {}",
        synthetic_path, noaa_path
    );
    Ok(())
}
