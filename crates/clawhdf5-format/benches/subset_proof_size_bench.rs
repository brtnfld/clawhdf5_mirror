//! P1.5 Part 4 & 6: Measure subset proof size |π| for three hyperslab shapes.
//!
//! This benchmark measures the serialized size of `SubsetProof` for the three
//! shapes required by S2-D2-Yr2 P1.5:
//! 1. Contiguous rectangular slab (best case, many shared Merkle nodes)
//! 2. Strided selection (medium case, fewer shared nodes)
//! 3. Random sparse selection (near-worst case, minimal sharing)
//!
//! Results are saved to `benches/results/subset-proof-size.csv`.
//! Test vectors are saved to `../../test-vectors/subset-vectors.json` (repo root).

use std::fs::{self, File};
use std::io::Write;
use std::time::Instant;

use clawhdf5_format::merkle::{HashAlg, MerkleTree};
use clawhdf5_format::selection::Selection;
use clawhdf5_format::subset_proof::{
    extract_subset, verify_subset, ChunkData, ChunkGridParams, LeafOrder, SubsetProof,
};

/// Total chunks in the test dataset (2^16 = 65,536).
const N: usize = 65_536;

/// Hash size in bytes.
const HASH_SIZE: usize = 32;

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
    k_chunks: usize,       // Number of chunks selected
    proof_size: usize,     // Actual proof size in bytes
    proof_nodes: usize,    // Number of deduplicated proof nodes
    theoretical: usize,    // Theoretical worst-case bound
    ratio: f64,            // proof_size / theoretical
    proof_time_ms: f64,    // Time to generate proof
    verify_time_ms: f64,   // Time to verify proof
}

/// Number of timing iterations for averaging.
const TIMING_ITERATIONS: usize = 10;

fn run_benchmark(chunks: &[Vec<u8>]) -> Vec<BenchResult> {
    let (tree, grid) = make_tree_1d(chunks);

    let mut results = Vec::new();

    // Test various k values
    let k_values = [64, 256, 1024, 4096];

    for &k in &k_values {
        println!("\n=== k = {} chunks ===", k);

        // 1. Contiguous 1D slab (best case)
        {
            let start = (N / 4) as u64;
            let end = start + k as u64;
            let sel = Selection::slice(&[start..end]);

            // Measure proof generation time
            let proof_start = Instant::now();
            let mut proof = None;
            for _ in 0..TIMING_ITERATIONS {
                proof = Some(extract_subset(&tree, &grid, &sel, LeafOrder::RowMajor).unwrap());
            }
            let proof_time_ms = proof_start.elapsed().as_secs_f64() * 1000.0 / TIMING_ITERATIONS as f64;
            let proof = proof.unwrap();

            // Prepare chunks for verification
            let delivered: Vec<ChunkData<'_>> = proof
                .chunk_indices
                .iter()
                .map(|&idx| ChunkData { index: idx, data: &chunks[idx] })
                .collect();

            // Measure verification time
            let verify_start = Instant::now();
            for _ in 0..TIMING_ITERATIONS {
                let _ = verify_subset(tree.root(), HashAlg::Blake3, &delivered, &proof, &grid, &sel, LeafOrder::RowMajor);
            }
            let verify_time_ms = verify_start.elapsed().as_secs_f64() * 1000.0 / TIMING_ITERATIONS as f64;

            let size = proof_size_bytes(&proof);
            let theoretical = theoretical_worst_case_bytes(k, N);

            println!(
                "Contiguous 1D: {} bytes, {} proof nodes, ratio = {:.3}, proof={:.3}ms, verify={:.3}ms",
                size, proof.proof_nodes.len(), size as f64 / theoretical as f64, proof_time_ms, verify_time_ms
            );

            results.push(BenchResult {
                hyperslab_type: "contiguous".to_string(),
                k_chunks: k,
                proof_size: size,
                proof_nodes: proof.proof_nodes.len(),
                theoretical,
                ratio: size as f64 / theoretical as f64,
                proof_time_ms,
                verify_time_ms,
            });
        }

        // 2. Strided selection (medium case)
        {
            let stride = N / k;
            let points: Vec<Vec<u64>> = (0..k).map(|i| vec![(i * stride) as u64]).collect();
            let sel = Selection::Points(points);

            let proof_start = Instant::now();
            let mut proof = None;
            for _ in 0..TIMING_ITERATIONS {
                proof = Some(extract_subset(&tree, &grid, &sel, LeafOrder::RowMajor).unwrap());
            }
            let proof_time_ms = proof_start.elapsed().as_secs_f64() * 1000.0 / TIMING_ITERATIONS as f64;
            let proof = proof.unwrap();

            let delivered: Vec<ChunkData<'_>> = proof
                .chunk_indices
                .iter()
                .map(|&idx| ChunkData { index: idx, data: &chunks[idx] })
                .collect();

            let verify_start = Instant::now();
            for _ in 0..TIMING_ITERATIONS {
                let _ = verify_subset(tree.root(), HashAlg::Blake3, &delivered, &proof, &grid, &sel, LeafOrder::RowMajor);
            }
            let verify_time_ms = verify_start.elapsed().as_secs_f64() * 1000.0 / TIMING_ITERATIONS as f64;

            let size = proof_size_bytes(&proof);
            let theoretical = theoretical_worst_case_bytes(k, N);

            println!(
                "Strided 1D (stride={}): {} bytes, {} proof nodes, ratio = {:.3}, proof={:.3}ms, verify={:.3}ms",
                stride, size, proof.proof_nodes.len(), size as f64 / theoretical as f64, proof_time_ms, verify_time_ms
            );

            results.push(BenchResult {
                hyperslab_type: "strided".to_string(),
                k_chunks: k,
                proof_size: size,
                proof_nodes: proof.proof_nodes.len(),
                theoretical,
                ratio: size as f64 / theoretical as f64,
                proof_time_ms,
                verify_time_ms,
            });
        }

        // 3. Random sparse selection (near-worst case)
        {
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

            let points: Vec<Vec<u64>> = indices.iter().map(|&i| vec![i]).collect();
            let sel = Selection::Points(points);

            let proof_start = Instant::now();
            let mut proof = None;
            for _ in 0..TIMING_ITERATIONS {
                proof = Some(extract_subset(&tree, &grid, &sel, LeafOrder::RowMajor).unwrap());
            }
            let proof_time_ms = proof_start.elapsed().as_secs_f64() * 1000.0 / TIMING_ITERATIONS as f64;
            let proof = proof.unwrap();

            let delivered: Vec<ChunkData<'_>> = proof
                .chunk_indices
                .iter()
                .map(|&idx| ChunkData { index: idx, data: &chunks[idx] })
                .collect();

            let verify_start = Instant::now();
            for _ in 0..TIMING_ITERATIONS {
                let _ = verify_subset(tree.root(), HashAlg::Blake3, &delivered, &proof, &grid, &sel, LeafOrder::RowMajor);
            }
            let verify_time_ms = verify_start.elapsed().as_secs_f64() * 1000.0 / TIMING_ITERATIONS as f64;

            let size = proof_size_bytes(&proof);
            let theoretical = theoretical_worst_case_bytes(k, N);

            println!(
                "Random sparse: {} bytes, {} proof nodes, ratio = {:.3}, proof={:.3}ms, verify={:.3}ms",
                size, proof.proof_nodes.len(), size as f64 / theoretical as f64, proof_time_ms, verify_time_ms
            );

            results.push(BenchResult {
                hyperslab_type: "random".to_string(),
                k_chunks: k,
                proof_size: size,
                proof_nodes: proof.proof_nodes.len(),
                theoretical,
                ratio: size as f64 / theoretical as f64,
                proof_time_ms,
                verify_time_ms,
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

    // Header per P1.5 Part 6 spec
    writeln!(
        file,
        "hyperslab_type,k_chunks,n_total,proof_size_bytes,proof_time_ms,verify_time_ms,theoretical_bound_bytes"
    )?;

    for r in results {
        writeln!(
            file,
            "{},{},{},{},{:.4},{:.4},{}",
            r.hyperslab_type,
            r.k_chunks,
            N,
            r.proof_size,
            r.proof_time_ms,
            r.verify_time_ms,
            r.theoretical,
        )?;
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

    // 1. Contiguous
    {
        let start = (N / 4) as u64;
        let end = start + k as u64;
        let sel = Selection::slice(&[start..end]);
        let proof = extract_subset(&tree, &grid, &sel, LeafOrder::RowMajor).unwrap();
        vectors.push(proof_to_json("contiguous", root, &proof));
    }

    // 2. Strided
    {
        let stride = N / k;
        let points: Vec<Vec<u64>> = (0..k).map(|i| vec![(i * stride) as u64]).collect();
        let sel = Selection::Points(points);
        let proof = extract_subset(&tree, &grid, &sel, LeafOrder::RowMajor).unwrap();
        vectors.push(proof_to_json("strided", root, &proof));
    }

    // 3. Random sparse
    {
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
        let points: Vec<Vec<u64>> = indices.iter().map(|&i| vec![i]).collect();
        let sel = Selection::Points(points);
        let proof = extract_subset(&tree, &grid, &sel, LeafOrder::RowMajor).unwrap();
        vectors.push(proof_to_json("random", root, &proof));
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
    println!("Theoretical bound: O(k * log N) = O(k * {})", (N as f64).log2() as usize);
    println!();

    println!("Building {} chunks...", N);
    let chunks = make_chunks(N);

    let results = run_benchmark(&chunks);

    // Print summary table
    println!("\n\n=== Summary ===");
    println!(
        "{:<12} {:>8} {:>12} {:>12} {:>10} {:>10}",
        "Type", "k", "Size (B)", "Theoretical", "Proof ms", "Verify ms"
    );
    println!("{}", "-".repeat(70));

    for r in &results {
        println!(
            "{:<12} {:>8} {:>12} {:>12} {:>10.3} {:>10.3}",
            r.hyperslab_type, r.k_chunks, r.proof_size, r.theoretical, r.proof_time_ms, r.verify_time_ms
        );
    }

    // Write CSV
    let csv_path = "benches/results/subset-proof-size.csv";
    match write_csv(&results, csv_path) {
        Ok(()) => println!("\nResults saved to {}", csv_path),
        Err(e) => eprintln!("\nFailed to write CSV: {}", e),
    }

    // Generate test vectors
    match generate_test_vectors(&chunks) {
        Ok(()) => {}
        Err(e) => eprintln!("Failed to write test vectors: {}", e),
    }
}
