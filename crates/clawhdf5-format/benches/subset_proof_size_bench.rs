//! P1.5 Part 4 & 6: Measure subset proof size |π| for three hyperslab shapes.
//!
//! This benchmark measures the serialized size of `SubsetProof` for the three
//! shapes required by S2-D2-Yr2 P1.5:
//! 1. Contiguous rectangular slab (best case, many shared Merkle nodes)
//! 2. Strided selection (medium case, fewer shared nodes)
//! 3. Random sparse selection (near-worst case, minimal sharing)
//!
//! `proof_time_ms`/`verify_time_ms` follow the Statistical Protocol (S2-D2
//! spec, p.52: minimum 30 trials after 5 discarded warmups, median + 95%
//! bootstrap CI), matching `hash_bench_harness.rs`/`baselines_bench.rs`.
//! `proof_size_bytes` itself is deterministic for a given (shape, k, N,
//! leaf order) and needs no trial statistics.
//!
//! Results are saved to `benches/results/subset-proof-size.csv` (summary,
//! one row per condition) and `benches/results/subset-proof-size-raw-trials.csv`
//! (all 30 raw per-trial timings per condition).
//! Test vectors are saved to `../../test-vectors/subset-vectors.json` (repo root).

use std::fs::{self, File};
use std::io::Write;
use std::time::Instant;

use clawhdf5_format::merkle::{HashAlg, MerkleTree};
use clawhdf5_format::selection::Selection;
use clawhdf5_format::subset_proof::{
    ChunkData, ChunkGridParams, LeafOrder, SubsetProof, extract_subset, verify_subset,
};

/// Total chunks in the test dataset (2^16 = 65,536).
const N: usize = 65_536;

/// Hash size in bytes.
const HASH_SIZE: usize = 32;

/// Discarded warmup iterations run (and not recorded) before the measured
/// ones, per the Statistical Protocol (S2-D2 spec, p.52: "minimum 30 trials
/// after 5 discarded warmups").
const WARMUP_TRIALS: usize = 5;

/// Measured trials per (hyperslab_type, k) cell, per the same protocol.
const TRIALS: usize = 30;

/// Bootstrap resamples used to compute the 95% CI on the median.
const BOOTSTRAP_ITERATIONS: usize = 2000;

/// Minimal xorshift64* PRNG so bootstrap resampling needs no extra
/// dependency; not cryptographic, just a deterministic resampler.
/// (Duplicated from `hash_bench_harness.rs`/`baselines_bench.rs` rather than
/// shared, since these are standalone harnesses.)
struct Xorshift64(u64);

impl Xorshift64 {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn next_index(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

fn median(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    let mut s = samples.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = (p * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Median plus 95% bootstrap confidence interval, per the Statistical
/// Protocol (S2-D2 spec, p.52). `seed` only needs to differ across call
/// sites to avoid correlated resampling; it is not a security boundary.
fn bootstrap_median_ci(samples: &[f64], seed: u64) -> (f64, f64, f64) {
    if samples.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    if samples.len() == 1 {
        return (samples[0], samples[0], samples[0]);
    }
    let mut rng = Xorshift64(seed | 1);
    let n = samples.len();
    let mut medians: Vec<f64> = Vec::with_capacity(BOOTSTRAP_ITERATIONS);
    for _ in 0..BOOTSTRAP_ITERATIONS {
        let resample: Vec<f64> = (0..n).map(|_| samples[rng.next_index(n)]).collect();
        medians.push(median(&resample));
    }
    medians.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (
        median(samples),
        percentile(&medians, 0.025),
        percentile(&medians, 0.975),
    )
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

fn ram_gb() -> f64 {
    fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|l| l.starts_with("MemTotal:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|s| s.parse::<f64>().ok())
        })
        .map(|kb| kb / 1024.0 / 1024.0)
        .unwrap_or(f64::NAN)
}

fn now_utc_iso() -> String {
    std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Compute the serialized size of a SubsetProof in bytes.
///
/// This is the wire format size that would be transmitted to a verifier:
/// - chunk_indices: k * 8 bytes (u64 per index)
/// - leaf_hashes: k * 32 bytes
/// - proof_nodes: n * (8 + 32) bytes (u64 key + 32-byte hash per node)
/// - grid_params.dims: d * 8 bytes
/// - grid_params.chunk_shape: d * 8 bytes
/// - grid_params.grid_hash: 32 bytes
/// - coverage_cert: 32 bytes
fn proof_size_bytes(proof: &SubsetProof) -> usize {
    let k = proof.chunk_indices.len();
    let n_proof_nodes = proof.proof_nodes.len();
    let d = proof.grid_params.dims.len();

    // chunk_indices: k * 8 (as u64)
    let indices_size = k * 8;
    // leaf_hashes: k * 32
    let leaf_hashes_size = k * HASH_SIZE;
    // proof_nodes: n * (8 + 32) for BTreeMap<u64, [u8; 32]>
    let proof_nodes_size = n_proof_nodes * (8 + HASH_SIZE);
    // grid_params: dims (d * 8) + chunk_shape (d * 8) + grid_hash (32)
    let grid_params_size = d * 8 + d * 8 + HASH_SIZE;
    // coverage_cert: 32
    let coverage_cert_size = HASH_SIZE;

    indices_size + leaf_hashes_size + proof_nodes_size + grid_params_size + coverage_cert_size
}

/// Theoretical bound: O(k * log N) bytes for k chunks in N-chunk tree.
/// Each chunk needs ~log2(N) sibling hashes, but contiguous chunks share many.
fn theoretical_worst_case_bytes(k: usize, n: usize) -> usize {
    let log_n = (n as f64).log2().ceil() as usize;
    // k * log(N) proof nodes * 40 bytes each (8 + 32)
    // Plus k * 40 bytes for indices + leaf hashes
    // Plus fixed overhead (~100 bytes for grid params)
    k * log_n * 40 + k * 40 + 100
}

/// Create synthetic chunk data for testing.
fn make_chunks(n: usize) -> Vec<Vec<u8>> {
    (0..n)
        .map(|i| format!("chunk-{i:06}").into_bytes())
        .collect()
}

/// Build a MerkleTree and ChunkGridParams for a 1D dataset.
fn make_tree_1d(chunks: &[Vec<u8>]) -> (MerkleTree, ChunkGridParams) {
    let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
    let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);
    let grid = ChunkGridParams::new(vec![chunks.len() as u64], vec![1], HashAlg::Blake3);
    (tree, grid)
}

#[derive(Debug)]
struct BenchResult {
    hyperslab_type: String,
    k_chunks: usize,    // Number of chunks selected
    proof_size: usize,  // Actual proof size in bytes (deterministic)
    theoretical: usize, // Theoretical worst-case bound
    proof_time_ms: f64, // Median over TRIALS measured trials
    proof_time_ci_low: f64,
    proof_time_ci_high: f64,
    verify_time_ms: f64, // Median over TRIALS measured trials
    verify_time_ci_low: f64,
    verify_time_ci_high: f64,
    /// All TRIALS raw per-trial timings, written to the raw-trials sidecar
    /// CSV alongside the summary row.
    raw_proof_times_ms: Vec<f64>,
    raw_verify_times_ms: Vec<f64>,
}

/// Run `WARMUP_TRIALS` discarded iterations, then time `TRIALS` measured
/// iterations of `extract_subset` followed by `verify_subset`, returning the
/// per-trial proof-generation and verification times in milliseconds.
fn time_extract_and_verify(
    tree: &MerkleTree,
    grid: &ChunkGridParams,
    sel: &Selection,
    chunks: &[Vec<u8>],
) -> (SubsetProof, Vec<f64>, Vec<f64>) {
    for _ in 0..WARMUP_TRIALS {
        let proof = extract_subset(tree, grid, sel, LeafOrder::RowMajor).unwrap();
        let delivered: Vec<ChunkData<'_>> = proof
            .chunk_indices
            .iter()
            .map(|&idx| ChunkData {
                index: idx,
                data: &chunks[idx],
            })
            .collect();
        let _ = verify_subset(
            tree.root(),
            HashAlg::Blake3,
            &delivered,
            &proof,
            grid,
            sel,
            LeafOrder::RowMajor,
        );
    }

    let mut proof_times_ms = Vec::with_capacity(TRIALS);
    let mut verify_times_ms = Vec::with_capacity(TRIALS);
    let mut last_proof = None;
    for _ in 0..TRIALS {
        let proof_start = Instant::now();
        let proof = extract_subset(tree, grid, sel, LeafOrder::RowMajor).unwrap();
        proof_times_ms.push(proof_start.elapsed().as_secs_f64() * 1000.0);

        let delivered: Vec<ChunkData<'_>> = proof
            .chunk_indices
            .iter()
            .map(|&idx| ChunkData {
                index: idx,
                data: &chunks[idx],
            })
            .collect();

        let verify_start = Instant::now();
        let ok = verify_subset(
            tree.root(),
            HashAlg::Blake3,
            &delivered,
            &proof,
            grid,
            sel,
            LeafOrder::RowMajor,
        );
        verify_times_ms.push(verify_start.elapsed().as_secs_f64() * 1000.0);
        assert!(matches!(ok, Ok(true)));

        last_proof = Some(proof);
    }

    (last_proof.unwrap(), proof_times_ms, verify_times_ms)
}

/// Build the selection for one of the three required hyperslab shapes.
fn selection_for_shape(shape: &str, k: usize) -> Selection {
    match shape {
        "contiguous" => {
            let start = (N / 4) as u64;
            let end = start + k as u64;
            Selection::slice(&[start..end])
        }
        "strided" => {
            let stride = N / k;
            let points: Vec<Vec<u64>> = (0..k).map(|i| vec![(i * stride) as u64]).collect();
            Selection::Points(points)
        }
        "random" => {
            let mut indices: Vec<u64> = Vec::with_capacity(k);
            let mut seed = 12345u64;
            while indices.len() < k {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let idx = (seed >> 33) % (N as u64);
                if !indices.contains(&idx) {
                    indices.push(idx);
                }
            }
            indices.sort();
            Selection::Points(indices.iter().map(|&i| vec![i]).collect())
        }
        other => panic!("unknown hyperslab shape: {other}"),
    }
}

fn run_benchmark(chunks: &[Vec<u8>]) -> Vec<BenchResult> {
    let (tree, grid) = make_tree_1d(chunks);

    let mut results = Vec::new();
    let k_values = [64, 256, 1024, 4096];
    let shapes = ["contiguous", "strided", "random"];

    // Distinct seed per (shape, k) cell so bootstrap resamples aren't
    // correlated across cells, matching hash_bench_harness.rs.
    let mut seed: u64 = 0x5EED_5170;

    for &k in &k_values {
        println!("\n=== k = {} chunks ===", k);

        for &shape in &shapes {
            let sel = selection_for_shape(shape, k);
            let (proof, proof_times_ms, verify_times_ms) =
                time_extract_and_verify(&tree, &grid, &sel, chunks);

            let (proof_time_ms, proof_time_ci_low, proof_time_ci_high) =
                bootstrap_median_ci(&proof_times_ms, seed);
            seed = seed.wrapping_add(1);
            let (verify_time_ms, verify_time_ci_low, verify_time_ci_high) =
                bootstrap_median_ci(&verify_times_ms, seed);
            seed = seed.wrapping_add(1);

            let size = proof_size_bytes(&proof);
            let theoretical = theoretical_worst_case_bytes(k, N);

            println!(
                "{shape}: {size} bytes, {} proof nodes, ratio = {:.3}, proof={proof_time_ms:.4}ms [{proof_time_ci_low:.4}, {proof_time_ci_high:.4}], verify={verify_time_ms:.4}ms [{verify_time_ci_low:.4}, {verify_time_ci_high:.4}]",
                proof.proof_nodes.len(),
                size as f64 / theoretical as f64,
            );

            results.push(BenchResult {
                hyperslab_type: shape.to_string(),
                k_chunks: k,
                proof_size: size,
                theoretical,
                proof_time_ms,
                proof_time_ci_low,
                proof_time_ci_high,
                verify_time_ms,
                verify_time_ci_low,
                verify_time_ci_high,
                raw_proof_times_ms: proof_times_ms,
                raw_verify_times_ms: verify_times_ms,
            });
        }
    }

    results
}

fn write_csv(results: &[BenchResult], path: &str) -> std::io::Result<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }

    let mut file = File::create(path)?;

    // Hardware/provenance line, matching hash-bench-*.csv/phase1-*.csv so the
    // benchmark is reproducible per the Statistical Protocol (S2-D2 spec,
    // p.52) without needing a separate sidecar for hardware info.
    writeln!(
        file,
        "# hostname={} cpu_model=\"{}\" ram_gb={:.1} date={}",
        hostname(),
        cpu_model(),
        ram_gb(),
        now_utc_iso()
    )?;

    // Required columns per P1.5 Part 6 spec, plus ci95_low/high for the two
    // timing columns (proof_size_bytes/theoretical_bound_bytes are
    // deterministic and need no CI; proof_time_ms/verify_time_ms are now the
    // median over TRIALS=30 measured trials after WARMUP_TRIALS=5 discarded
    // warmups, per the Statistical Protocol).
    writeln!(
        file,
        "hyperslab_type,k_chunks,n_total,proof_size_bytes,proof_time_ms,proof_time_ci95_low,proof_time_ci95_high,verify_time_ms,verify_time_ci95_low,verify_time_ci95_high,theoretical_bound_bytes"
    )?;

    for r in results {
        writeln!(
            file,
            "{},{},{},{},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{}",
            r.hyperslab_type,
            r.k_chunks,
            N,
            r.proof_size,
            r.proof_time_ms,
            r.proof_time_ci_low,
            r.proof_time_ci_high,
            r.verify_time_ms,
            r.verify_time_ci_low,
            r.verify_time_ci_high,
            r.theoretical,
        )?;
    }

    Ok(())
}

/// Sidecar CSV with all `TRIALS` raw per-trial timings per condition, for
/// full reproducibility/auditability behind the summary CSV's medians+CIs —
/// matching `hash-bench-raw-trials-*.csv`.
fn write_raw_trials_csv(results: &[BenchResult], path: &str) -> std::io::Result<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }

    let mut file = File::create(path)?;
    writeln!(
        file,
        "# hostname={} cpu_model=\"{}\" ram_gb={:.1} date={}",
        hostname(),
        cpu_model(),
        ram_gb(),
        now_utc_iso()
    )?;
    writeln!(
        file,
        "hyperslab_type,k_chunks,trial,proof_time_ms,verify_time_ms"
    )?;

    for r in results {
        for (i, (pt, vt)) in r
            .raw_proof_times_ms
            .iter()
            .zip(r.raw_verify_times_ms.iter())
            .enumerate()
        {
            writeln!(
                file,
                "{},{},{},{:.5},{:.5}",
                r.hyperslab_type,
                r.k_chunks,
                i + 1,
                pt,
                vt
            )?;
        }
    }

    Ok(())
}

/// Generate test vectors for the three hyperslab types (contiguous, strided, random).
/// Uses k=64 as a representative sample size.
fn generate_test_vectors(chunks: &[Vec<u8>]) -> std::io::Result<()> {
    let (tree, grid) = make_tree_1d(chunks);
    let root = tree.root();

    let k = 64;
    let mut vectors = Vec::new();
    for &shape in &["contiguous", "strided", "random"] {
        let sel = selection_for_shape(shape, k);
        let proof = extract_subset(&tree, &grid, &sel, LeafOrder::RowMajor).unwrap();
        vectors.push(proof_to_json(shape, root, &proof));
    }

    // Write JSON to root test-vectors/ directory (alongside morton-vectors.json)
    let json_path = "../../test-vectors/subset-vectors.json";
    if let Some(parent) = std::path::Path::new(json_path).parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(json_path)?;
    writeln!(file, "[")?;
    for (i, v) in vectors.iter().enumerate() {
        if i > 0 {
            writeln!(file, ",")?;
        }
        write!(file, "{}", v)?;
    }
    writeln!(file, "\n]")?;

    println!("Test vectors saved to {}", json_path);
    Ok(())
}

fn proof_to_json(hyperslab_type: &str, root: &[u8; 32], proof: &SubsetProof) -> String {
    let root_hex = hex_encode(root);
    let coverage_cert_hex = hex_encode(&proof.coverage_cert);
    let grid_hash_hex = hex_encode(&proof.grid_params.grid_hash);

    let leaf_hashes: Vec<String> = proof.leaf_hashes.iter().map(hex_encode).collect();
    let proof_nodes: Vec<String> = proof
        .proof_nodes
        .iter()
        .map(|(k, v)| format!("{{\"index\":{},\"hash\":\"{}\"}}", k, hex_encode(v)))
        .collect();

    format!(
        r#"  {{
    "hyperslab_type": "{}",
    "n_total": {},
    "k_chunks": {},
    "expected_root": "{}",
    "chunk_indices": {:?},
    "leaf_hashes": {:?},
    "proof_nodes": [{}],
    "grid_params": {{
      "dims": {:?},
      "chunk_shape": {:?},
      "grid_hash": "{}"
    }},
    "coverage_cert": "{}"
  }}"#,
        hyperslab_type,
        N,
        proof.chunk_indices.len(),
        root_hex,
        proof.chunk_indices,
        leaf_hashes,
        proof_nodes.join(", "),
        proof.grid_params.dims,
        proof.grid_params.chunk_shape,
        grid_hash_hex,
        coverage_cert_hex,
    )
}

fn hex_encode(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn main() {
    println!("P1.5 Part 4 & 6: Subset Proof Size Benchmark");
    println!("=============================================");
    println!("N = {} chunks (2^{})", N, (N as f64).log2() as usize);
    println!(
        "Theoretical bound: O(k * log N) = O(k * {})",
        (N as f64).log2() as usize
    );
    println!("warmup={WARMUP_TRIALS} trials={TRIALS} bootstrap_iterations={BOOTSTRAP_ITERATIONS}");
    println!();

    println!("Building {} chunks...", N);
    let chunks = make_chunks(N);

    let results = run_benchmark(&chunks);

    // Print summary table
    println!("\n\n=== Summary (median [95% CI]) ===");
    println!(
        "{:<12} {:>8} {:>12} {:>12} {:>22} {:>22}",
        "Type", "k", "Size (B)", "Theoretical", "Proof ms", "Verify ms"
    );
    println!("{}", "-".repeat(100));

    for r in &results {
        println!(
            "{:<12} {:>8} {:>12} {:>12} {:>10.4} [{:>6.4},{:>6.4}] {:>10.4} [{:>6.4},{:>6.4}]",
            r.hyperslab_type,
            r.k_chunks,
            r.proof_size,
            r.theoretical,
            r.proof_time_ms,
            r.proof_time_ci_low,
            r.proof_time_ci_high,
            r.verify_time_ms,
            r.verify_time_ci_low,
            r.verify_time_ci_high,
        );
    }

    // Write summary CSV (name fixed by the P1.5 spec) and the raw-trials
    // sidecar behind it.
    let csv_path = "benches/results/subset-proof-size.csv";
    match write_csv(&results, csv_path) {
        Ok(()) => println!("\nResults saved to {}", csv_path),
        Err(e) => eprintln!("\nFailed to write CSV: {}", e),
    }

    let raw_csv_path = "benches/results/subset-proof-size-raw-trials.csv";
    match write_raw_trials_csv(&results, raw_csv_path) {
        Ok(()) => println!("Raw per-trial timings saved to {}", raw_csv_path),
        Err(e) => eprintln!("Failed to write raw-trials CSV: {}", e),
    }

    // Generate test vectors
    match generate_test_vectors(&chunks) {
        Ok(()) => {}
        Err(e) => eprintln!("Failed to write test vectors: {}", e),
    }
}
