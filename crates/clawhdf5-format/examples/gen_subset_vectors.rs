//! Generate subset-proof test vectors JSON (P1.5 artifact).
//!
//! Run with: cargo run --features merkle --example gen_subset_vectors

use clawhdf5_format::merkle::{HashAlg, MerkleTree};
use clawhdf5_format::selection::Selection;
use clawhdf5_format::subset_proof::{
    ChunkData, ChunkGridParams, LeafOrder, SubsetProof, extract_subset, verify_subset,
};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// 8x8 grid of single-element chunks (64 chunks total), row-major leaf order.
fn build_grid() -> (MerkleTree, ChunkGridParams, Vec<Vec<u8>>) {
    let dims = vec![8u64, 8u64];
    let chunk_shape = vec![1u64, 1u64];
    let grid = ChunkGridParams::new(dims, chunk_shape, HashAlg::Blake3);

    let chunks: Vec<Vec<u8>> = (0..64).map(|i| format!("chunk-{i}").into_bytes()).collect();
    let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
    let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);

    (tree, grid, chunks)
}

fn print_proof(label: &str, proof: &SubsetProof, root: &[u8; 32], last: bool) {
    println!("  \"{label}\": {{");
    println!(
        "    \"chunk_indices\": [{}],",
        proof
            .chunk_indices
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("    \"leaf_hashes\": [");
    for (i, h) in proof.leaf_hashes.iter().enumerate() {
        let comma = if i + 1 < proof.leaf_hashes.len() {
            ","
        } else {
            ""
        };
        println!("      \"{}\"{}", hex(h), comma);
    }
    println!("    ],");
    println!("    \"proof_nodes\": {{");
    let n = proof.proof_nodes.len();
    for (i, (k, v)) in proof.proof_nodes.iter().enumerate() {
        let comma = if i + 1 < n { "," } else { "" };
        println!("      \"{}\": \"{}\"{}", k, hex(v), comma);
    }
    println!("    }},");
    println!("    \"grid_params\": {{");
    println!(
        "      \"dims\": [{}],",
        proof
            .grid_params
            .dims
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "      \"chunk_shape\": [{}],",
        proof
            .grid_params
            .chunk_shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "      \"grid_hash\": \"{}\"",
        hex(&proof.grid_params.grid_hash)
    );
    println!("    }},");
    println!("    \"coverage_cert\": \"{}\",", hex(&proof.coverage_cert));
    println!("    \"expected_root\": \"{}\",", hex(root));
    println!("    \"proof_size_bytes\": {}", proof_size_bytes(proof));
    println!("  }}{}", if last { "" } else { "," });
}

fn proof_size_bytes(proof: &SubsetProof) -> usize {
    proof.chunk_indices.len() * 8
        + proof.leaf_hashes.len() * 32
        + proof.proof_nodes.len() * (8 + 32)
        + proof.grid_params.dims.len() * 8
        + proof.grid_params.chunk_shape.len() * 8
        + 32 // grid_hash
        + 32 // coverage_cert
}

fn main() {
    let (tree, grid, chunks) = build_grid();
    let root = *tree.root();

    println!("{{");
    println!("  \"description\": \"Verifiable subset extraction test vectors (P1.5)\",");
    println!("  \"specification\": \"S2-D2-Yr2/Merkle-tree-HDF5.tex sec:subset\",");
    println!(
        "  \"grid\": {{ \"dims\": [8, 8], \"chunk_shape\": [1, 1], \"leaf_order\": \"row_major\" }},"
    );
    println!("  \"root\": \"{}\",", hex(&root));

    // (1) Contiguous rectangular slab: rows 2..5, cols 2..5 (3x3 block).
    let contiguous = Selection::slice(&[2..5, 2..5]);
    let proof_c = extract_subset(&tree, &grid, &contiguous, LeafOrder::RowMajor).unwrap();
    let delivered_c: Vec<ChunkData<'_>> = proof_c
        .chunk_indices
        .iter()
        .map(|&i| ChunkData {
            index: i,
            data: &chunks[i],
        })
        .collect();
    assert!(
        verify_subset(
            &root,
            HashAlg::Blake3,
            &delivered_c,
            &proof_c,
            &grid,
            &grid.grid_hash,
            &contiguous,
            LeafOrder::RowMajor
        )
        .unwrap()
    );
    print_proof("contiguous", &proof_c, &root, false);

    // (2) Strided slab: every other row in 0..8, every other column in 0..8.
    let strided = Selection::Hyperslab {
        start: vec![0, 0],
        stride: vec![2, 2],
        count: vec![4, 4],
        block: vec![1, 1],
    };
    let proof_s = extract_subset(&tree, &grid, &strided, LeafOrder::RowMajor).unwrap();
    let delivered_s: Vec<ChunkData<'_>> = proof_s
        .chunk_indices
        .iter()
        .map(|&i| ChunkData {
            index: i,
            data: &chunks[i],
        })
        .collect();
    assert!(
        verify_subset(
            &root,
            HashAlg::Blake3,
            &delivered_s,
            &proof_s,
            &grid,
            &grid.grid_hash,
            &strided,
            LeafOrder::RowMajor
        )
        .unwrap()
    );
    print_proof("strided", &proof_s, &root, false);

    // (3) Random sparse selection of individual points.
    let random_points = Selection::Points(vec![
        vec![0, 0],
        vec![1, 5],
        vec![3, 2],
        vec![6, 7],
        vec![7, 0],
    ]);
    let proof_r = extract_subset(&tree, &grid, &random_points, LeafOrder::RowMajor).unwrap();
    let delivered_r: Vec<ChunkData<'_>> = proof_r
        .chunk_indices
        .iter()
        .map(|&i| ChunkData {
            index: i,
            data: &chunks[i],
        })
        .collect();
    assert!(
        verify_subset(
            &root,
            HashAlg::Blake3,
            &delivered_r,
            &proof_r,
            &grid,
            &grid.grid_hash,
            &random_points,
            LeafOrder::RowMajor
        )
        .unwrap()
    );
    print_proof("random", &proof_r, &root, true);

    println!("}}");

    eprintln!("All assertions passed!");
}
