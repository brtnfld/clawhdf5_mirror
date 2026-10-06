//! Four-way integrity-backend comparison: Merkle tree vs. flat SHA-256 vs.
//! per-chunk HMAC-SHA-256 vs. per-chunk Ed25519, in a single process, on the
//! same chunk sets, with one shared definition of every operation.
//!
//! This closes two gaps between P1.6 (`phase1_bench.rs`, Merkle + flat only)
//! and P1.2b (`baselines_bench.rs`, flat/MAC/Ed25519 only): the backends were
//! never run side by side, and P1.6's Merkle tree used BLAKE3 while every
//! baseline uses SHA-256. Here the Merkle tree is run with *both* SHA-256
//! (hash-matched to the baselines) and BLAKE3 (the crate default).
//!
//! Operation definitions (identical for every backend that supports them):
//!
//! - `commit`         — build all integrity metadata from every chunk.
//! - `verify_dataset` — check every chunk against the committed metadata.
//! - `verify_chunk`   — a verifier checks one chunk (index N/2) holding only
//!   the public/trusted commitment plus that chunk's own metadata: for the
//!   Merkle tree that is the root + an inclusion proof
//!   (`verify_proof_standalone`, proof generated outside the timed region);
//!   for MAC the key + tag; for Ed25519 the public key + signature. Flat
//!   SHA-256 cannot do this and is recorded as unsupported.
//! - `update`         — overwrite chunk N/2 with new same-size data and bring
//!   the metadata up to date (Merkle: leaf hash + O(log N) path; flat: full
//!   rehash). In place, no per-trial clone.
//! - `append`         — add one new chunk (same size as chunk 0). Merkle
//!   writes the new leaf into the first padding slot and recomputes its
//!   path; every N in this benchmark is below its next power of two, so no
//!   tree resize is ever needed (a resize would be an O(N) internal-node
//!   rebuild, not a data rehash).
//!
//! All backends run single-threaded. 30 measured trials after 5 discarded
//! warmups per cell.
//!
//! Usage:
//!   cargo run -p clawhdf5 --release --example fourway_bench \
//!     [--features bench-ed25519-fast] -- \
//!     <synthetic_10gb.h5> <noaa_sample.nc> [--backends a,b,...] [--tag name]
//!
//! Backends: merkle_sha256, merkle_blake3, flat_sha256, hmac_sha256, ed25519.
//! Output: `benches/results/fourway[-<tag>].csv` (the host is recorded in
//! the CSV header line).

use clawhdf5::File;
use clawhdf5_format::baselines::{FlatHashBackend, PerChunkMacBackend, PerChunkSigBackend};
use clawhdf5_format::chunked_read::{ChunkInfo, collect_chunk_info};
use clawhdf5_format::data_layout::DataLayout;
use clawhdf5_format::dataspace::Dataspace;
use clawhdf5_format::extensible_array::{ExtensibleArrayHeader, read_extensible_array_chunks};
use clawhdf5_format::fixed_array::{FixedArrayHeader, read_fixed_array_chunks};
use clawhdf5_format::group_v2::resolve_path_any;
use clawhdf5_format::merkle::{HashAlg, MerkleTree};
use clawhdf5_format::message_type::MessageType;
use clawhdf5_format::object_header::ObjectHeader;
use clawhdf5_format::signature::find_signature;
use clawhdf5_format::superblock::Superblock;

use std::hint::black_box;
use std::io::Write as IoWrite;
use std::time::Instant;

const TRIALS: usize = 30;
const WARMUP_TRIALS: usize = 5;
const ALL_BACKENDS: &[&str] = &[
    "merkle_sha256",
    "merkle_blake3",
    "flat_sha256",
    "hmac_sha256",
    "ed25519",
];

// ---------------------------------------------------------------------------
// Chunk loader (same logic as phase1_bench.rs::load_chunks)
// ---------------------------------------------------------------------------

struct LoadedFile {
    chunks: Vec<Vec<u8>>,
    chunk_size_kb: f64,
}

fn load_chunks(
    path: &str,
    dataset_name: Option<&str>,
) -> Result<LoadedFile, Box<dyn std::error::Error>> {
    let hf = File::open(path)?;
    let names = hf.root().datasets()?;
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
    let chunk_size_kb = if n > 0 {
        (total_bytes as f64 / n as f64) / 1024.0
    } else {
        0.0
    };
    Ok(LoadedFile {
        chunks,
        chunk_size_kb,
    })
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

/// Discard `WARMUP_TRIALS` iterations, then return `TRIALS` timings in ns.
fn time_trials<F: FnMut()>(mut f: F) -> Vec<u128> {
    for _ in 0..WARMUP_TRIALS {
        f();
    }
    (0..TRIALS)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_nanos()
        })
        .collect()
}

struct Ctx<'a> {
    source: &'a str,
    dataset: &'a str,
    n_chunks: usize,
    chunk_size_kb: f64,
    out: &'a mut Vec<String>,
}

impl Ctx<'_> {
    fn timed(&mut self, backend: &str, op: &str, ns: Vec<u128>) {
        let med = {
            let mut s = ns.clone();
            s.sort_unstable();
            s[s.len() / 2]
        };
        println!(
            "    {backend:<14} {op:<15} median {:>14.3} µs",
            med as f64 / 1000.0
        );
        for (i, v) in ns.into_iter().enumerate() {
            self.row(backend, op, &v.to_string(), "ns", i + 1);
        }
    }

    fn fixed(&mut self, backend: &str, metric: &str, value: &str, unit: &str) {
        self.row(backend, metric, value, unit, 0);
    }

    fn row(&mut self, backend: &str, metric: &str, value: &str, unit: &str, trial: usize) {
        self.out.push(format!(
            "{},{},{},{},{:.3},{},{},{},{}",
            self.source,
            self.dataset,
            backend,
            metric,
            self.chunk_size_kb,
            self.n_chunks,
            value,
            unit,
            trial
        ));
    }
}

/// Deterministic same-size replacement data for update/append.
fn perturbed(chunk: &[u8]) -> Vec<u8> {
    chunk.iter().map(|b| b ^ 0xA5).collect()
}

fn run_merkle(ctx: &mut Ctx, name: &str, alg: HashAlg, chunks: &[Vec<u8>]) {
    let n = chunks.len();
    let mid = n / 2;

    ctx.timed(
        name,
        "commit",
        time_trials(|| {
            black_box(MerkleTree::from_chunks_owned(black_box(chunks), alg));
        }),
    );

    let tree = MerkleTree::from_chunks_owned(chunks, alg);
    let root = *tree.root();
    ctx.timed(
        name,
        "verify_dataset",
        time_trials(|| {
            let t = MerkleTree::from_chunks_owned(black_box(chunks), alg);
            assert_eq!(t.root(), &root);
        }),
    );

    let proof = tree.proof(mid).expect("mid in range");
    ctx.timed(
        name,
        "verify_chunk",
        time_trials(|| {
            assert!(MerkleTree::verify_proof_standalone(
                &root,
                mid,
                black_box(&chunks[mid]),
                &proof
            ));
        }),
    );

    let new_mid = perturbed(&chunks[mid]);
    let mut t = tree.clone();
    ctx.timed(
        name,
        "update",
        time_trials(|| {
            let h = alg.hash_leaf(black_box(&new_mid));
            t.update_leaf(mid, h).unwrap();
            black_box(t.root());
        }),
    );

    assert!(n < tree.padded_leaf_count(), "append needs a free padding slot");
    let new_tail = perturbed(&chunks[0]);
    let mut t = tree.clone();
    ctx.timed(
        name,
        "append",
        time_trials(|| {
            let h = alg.hash_leaf(black_box(&new_tail));
            t.update_leaf(n, h).unwrap();
            black_box(t.root());
        }),
    );

    // Full companion (all nodes) is what write_merkle_companion persists.
    ctx.fixed(
        name,
        "metadata_bytes",
        &(tree.nodes().len() * 32).to_string(),
        "bytes",
    );
    ctx.fixed(
        name,
        "chunk_proof_bytes",
        &proof.size_bytes().to_string(),
        "bytes",
    );
    ctx.fixed(name, "public_verification", "yes", "bool");
}

fn run_flat(ctx: &mut Ctx, chunks: &[Vec<u8>]) {
    let name = "flat_sha256";
    let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
    let mid = chunks.len() / 2;

    ctx.timed(
        name,
        "commit",
        time_trials(|| {
            black_box(FlatHashBackend::commit(black_box(&refs)));
        }),
    );
    let flat = FlatHashBackend::commit(&refs);
    ctx.timed(
        name,
        "verify_dataset",
        time_trials(|| assert!(flat.verify_dataset(black_box(&refs)))),
    );
    assert!(flat.verify_chunk(mid, &chunks[mid]).is_err());
    ctx.fixed(name, "verify_chunk", "unsupported", "na");

    let new_mid = perturbed(&chunks[mid]);
    let mut updated = refs.clone();
    updated[mid] = &new_mid;
    let mut f = flat;
    ctx.timed(
        name,
        "update",
        time_trials(|| {
            f.update(black_box(&updated));
            black_box(&f);
        }),
    );

    let new_tail = perturbed(&chunks[0]);
    let mut appended = refs.clone();
    appended.push(&new_tail);
    let mut f = flat;
    ctx.timed(
        name,
        "append",
        time_trials(|| {
            f.append(black_box(&appended));
            black_box(&f);
        }),
    );

    ctx.fixed(name, "metadata_bytes", &flat.metadata_bytes().to_string(), "bytes");
    ctx.fixed(name, "chunk_proof_bytes", "unsupported", "na");
    ctx.fixed(name, "public_verification", "yes", "bool");
}

fn run_mac(ctx: &mut Ctx, chunks: &[Vec<u8>]) {
    let name = "hmac_sha256";
    let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
    let mid = chunks.len() / 2;

    ctx.timed(
        name,
        "commit",
        time_trials(|| {
            black_box(PerChunkMacBackend::commit(black_box(&refs)));
        }),
    );
    let mut mac = PerChunkMacBackend::commit(&refs);
    ctx.timed(
        name,
        "verify_dataset",
        time_trials(|| assert!(mac.verify_dataset(black_box(&refs)))),
    );
    ctx.timed(
        name,
        "verify_chunk",
        time_trials(|| assert_eq!(mac.verify_chunk(mid, black_box(&chunks[mid])), Ok(true))),
    );
    let new_mid = perturbed(&chunks[mid]);
    ctx.timed(
        name,
        "update",
        time_trials(|| mac.update(mid, black_box(&new_mid)).unwrap()),
    );
    let metadata = mac.metadata_bytes();
    let new_tail = perturbed(&chunks[0]);
    ctx.timed(
        name,
        "append",
        time_trials(|| mac.append(black_box(&new_tail))),
    );
    ctx.fixed(name, "metadata_bytes", &metadata.to_string(), "bytes");
    ctx.fixed(name, "chunk_proof_bytes", "32", "bytes");
    ctx.fixed(name, "public_verification", "no", "bool");
}

fn run_sig(ctx: &mut Ctx, chunks: &[Vec<u8>]) {
    let name = "ed25519";
    let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
    let mid = chunks.len() / 2;

    ctx.timed(
        name,
        "commit",
        time_trials(|| {
            black_box(PerChunkSigBackend::commit(black_box(&refs)));
        }),
    );
    let mut sig = PerChunkSigBackend::commit(&refs);
    ctx.timed(
        name,
        "verify_dataset",
        time_trials(|| assert!(sig.verify_dataset(black_box(&refs)))),
    );
    ctx.timed(
        name,
        "verify_chunk",
        time_trials(|| assert_eq!(sig.verify_chunk(mid, black_box(&chunks[mid])), Ok(true))),
    );
    let new_mid = perturbed(&chunks[mid]);
    ctx.timed(
        name,
        "update",
        time_trials(|| sig.update(mid, black_box(&new_mid)).unwrap()),
    );
    let metadata = sig.metadata_bytes();
    let new_tail = perturbed(&chunks[0]);
    ctx.timed(
        name,
        "append",
        time_trials(|| sig.append(black_box(&new_tail))),
    );
    ctx.fixed(name, "metadata_bytes", &metadata.to_string(), "bytes");
    ctx.fixed(
        name,
        "chunk_proof_bytes",
        &PerChunkSigBackend::subset_proof_bytes(1).to_string(),
        "bytes",
    );
    ctx.fixed(name, "public_verification", "yes", "bool");
}

fn run_all(ctx: &mut Ctx, backends: &[String], chunks: &[Vec<u8>]) {
    for b in backends {
        match b.as_str() {
            "merkle_sha256" => run_merkle(ctx, b, HashAlg::Sha256, chunks),
            "merkle_blake3" => run_merkle(ctx, b, HashAlg::Blake3, chunks),
            "flat_sha256" => run_flat(ctx, chunks),
            "hmac_sha256" => run_mac(ctx, chunks),
            "ed25519" => run_sig(ctx, chunks),
            other => panic!("unknown backend {other}"),
        }
    }
}

// ---------------------------------------------------------------------------
// System info
// ---------------------------------------------------------------------------

fn cmd_out(cmd: &str, args: &[&str]) -> String {
    std::process::Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|v| v.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into())
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: fourway_bench <synthetic_10gb.h5> <noaa_sample.nc> \
             [--backends a,b,...] [--tag name]"
        );
        std::process::exit(1);
    }
    let synthetic_path = &args[1];
    let noaa_path = &args[2];
    let mut backends: Vec<String> = ALL_BACKENDS.iter().map(|s| s.to_string()).collect();
    let mut tag: Option<String> = None;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--backends" => {
                backends = args[i + 1].split(',').map(str::to_string).collect();
                i += 2;
            }
            "--tag" => {
                tag = Some(args[i + 1].clone());
                i += 2;
            }
            other => return Err(format!("unknown argument {other}").into()),
        }
    }

    let host = cmd_out("hostname", &[]);
    let ed_fast = cfg!(feature = "bench-ed25519-fast");
    println!("Four-way integrity benchmark");
    println!("  host      : {host}");
    println!("  cpu       : {}", cpu_model());
    println!("  backends  : {}", backends.join(","));
    println!("  ed25519   : fast(precomputed tables)={ed_fast}");
    println!("  trials    : {TRIALS} (+{WARMUP_TRIALS} warmup)\n");

    let mut out: Vec<String> = Vec::new();
    let sources: [(&str, &str, Option<&str>); 4] = [
        ("synthetic", synthetic_path, Some("dataset_64kb")),
        ("synthetic", synthetic_path, Some("dataset_256kb")),
        ("synthetic", synthetic_path, Some("dataset_1mb")),
        ("NOAA", noaa_path, None),
    ];
    for (source, path, ds) in sources {
        let dataset = ds.unwrap_or("Rad");
        println!("Loading {source} {path} [{dataset}]");
        let lf = load_chunks(path, ds)?;
        println!(
            "  {} chunks, avg {:.1} KB/chunk",
            lf.chunks.len(),
            lf.chunk_size_kb
        );
        let mut ctx = Ctx {
            source,
            dataset,
            n_chunks: lf.chunks.len(),
            chunk_size_kb: lf.chunk_size_kb,
            out: &mut out,
        };
        run_all(&mut ctx, &backends, &lf.chunks);
    }

    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/clawhdf5-format/benches/results");
    std::fs::create_dir_all(&out_dir)?;
    let file_name = match &tag {
        Some(t) => format!("fourway-{t}.csv"),
        None => "fourway.csv".to_string(),
    };
    let csv_path = out_dir.join(file_name);
    let mut f = std::fs::File::create(&csv_path)?;
    writeln!(
        f,
        "# hostname={host} cpu_model=\"{}\" date={} trials={TRIALS} warmup={WARMUP_TRIALS} \
         ed25519_fast={ed_fast} threads=1",
        cpu_model(),
        cmd_out("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"])
    )?;
    writeln!(
        f,
        "source,dataset,backend,metric,chunk_size_kb,n_chunks,value,unit,trial"
    )?;
    for r in &out {
        writeln!(f, "{r}")?;
    }
    println!("\nWrote {} rows to {}", out.len(), csv_path.display());
    Ok(())
}
