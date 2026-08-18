//! Generate contiguous-dataset subset-proof test vectors JSON (P2.8b artifact).
//!
//! Emits three serialized `SubsetProof`s over a *contiguous* dataset -- a
//! compact sub-cube, a plane, and a 1-D range -- in the same schema as P1.5's
//! `gen_subset_vectors`, so both files can be fed to a single verifier
//! harness. The only structural difference is `layout_class: "contiguous"`
//! and the `grid_hash` it feeds; that identity is the whole point of P2.8b's
//! reuse claim.
//!
//! Run with:
//! `cargo run --features merkle,blake3 --example gen_contiguous_subset_vectors`

use clawhdf5_format::merkle::HashAlg;
use clawhdf5_format::selection::Selection;
use clawhdf5_format::subset_proof::{
    ChunkData, LeafOrder, SubsetProof, contiguous_tree_and_grid, extract_subset, verify_subset,
};
use clawhdf5_format::verification_grid::{LayoutClass, leaf_byte_range};

/// A 3-D dataset small enough to keep the vector file readable, with a
/// deliberately small target so the grid actually tiles rather than falling
/// back to a single leaf. `target_bytes` is pinned here (never read from the
/// environment) for exactly the reason P2.8a requires it be an explicit
/// parameter: a verifier that re-derived the grid from its own configuration
/// would accept whatever grid its environment happened to produce.
const DIMS: [u64; 3] = [16, 16, 16];
const ELEM_SIZE: u32 = 4;
/// 1 KiB target => grid [1, 16, 16], i.e. 16 leaves of one plane each. Chosen
/// so the three selections below land on genuinely different leaf sets
/// (a coarser target collapses all three onto leaf 0 and the vectors stop
/// distinguishing anything).
const TARGET_BYTES: u64 = 1024;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Deterministic little-endian f32 payload, distinguishable per element.
fn build_data() -> Vec<u8> {
    let n: u64 = DIMS.iter().product();
    (0..n as u32).flat_map(|v| (v as f32).to_le_bytes()).collect()
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

fn list(v: &[u64]) -> String {
    v.iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

fn print_proof(label: &str, proof: &SubsetProof, root: &[u8; 32], last: bool) {
    println!("  \"{label}\": {{");
    println!(
        "    \"chunk_indices\": [{}],",
        proof
            .chunk_indices
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("    \"leaf_hashes\": [");
    for (i, h) in proof.leaf_hashes.iter().enumerate() {
        let comma = if i + 1 < proof.leaf_hashes.len() { "," } else { "" };
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
    println!("      \"dims\": [{}],", list(&proof.grid_params.dims));
    println!(
        "      \"chunk_shape\": [{}],",
        list(&proof.grid_params.chunk_shape)
    );
    println!("      \"elem_size\": {},", proof.grid_params.elem_size);
    println!(
        "      \"layout_class\": \"{}\",",
        match proof.grid_params.layout_class {
            LayoutClass::Chunked => "chunked",
            LayoutClass::Contiguous => "contiguous",
        }
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

fn main() {
    let data = build_data();
    let (tree, grid) =
        contiguous_tree_and_grid(&data, &DIMS, ELEM_SIZE, TARGET_BYTES, HashAlg::Blake3);
    let root = *tree.root();

    let cases: [(&str, Selection); 3] = [
        // Compact sub-cube.
        ("sub_cube", Selection::slice(&[4..8, 4..8, 4..8])),
        // Full plane along the slowest-varying axis.
        ("plane", Selection::slice(&[3..4, 0..16, 0..16])),
        // 1-D range along the fastest-varying axis.
        ("range_1d", Selection::slice(&[0..1, 0..1, 2..14])),
    ];

    println!("{{");
    println!(
        "  \"description\": \"Contiguous-dataset verifiable subset extraction test vectors (P2.8b)\","
    );
    println!(
        "  \"specification\": \"S2-D2-Yr2/Merkle-tree-HDF5.tex sec:contiguous-design\","
    );
    println!(
        "  \"note\": \"Same schema as test-vectors/subset-vectors.json (P1.5); the \
         proofs differ only in layout_class and the grid_hash it feeds, which is \
         the P2.8b reuse claim.\","
    );
    println!("  \"target_bytes\": {TARGET_BYTES},");
    println!(
        "  \"grid\": {{ \"dims\": [{}], \"chunk_shape\": [{}], \"elem_size\": {}, \
         \"layout_class\": \"contiguous\", \"leaf_order\": \"row_major\" }},",
        list(&DIMS),
        list(&grid.chunk_shape),
        ELEM_SIZE
    );
    println!("  \"root\": \"{}\",", hex(&root));

    for (i, (label, sel)) in cases.iter().enumerate() {
        let proof = extract_subset(&tree, &grid, sel, LeafOrder::RowMajor)
            .expect("extract_subset should succeed");

        // Deliver by re-slicing the raw byte stream, exactly as a real
        // contiguous-dataset reader would -- not by reaching into the tree.
        let delivered: Vec<ChunkData<'_>> = proof
            .chunk_indices
            .iter()
            .map(|&idx| {
                let r = leaf_byte_range(&grid.chunk_shape, &DIMS, ELEM_SIZE, idx as u64);
                ChunkData {
                    index: idx,
                    data: &data[r.start as usize..r.end as usize],
                }
            })
            .collect();

        assert!(
            verify_subset(
                &root,
                HashAlg::Blake3,
                &delivered,
                &proof,
                &grid,
                &grid.grid_hash,
                sel,
                LeafOrder::RowMajor,
            )
            .expect("vectors must verify before being emitted"),
            "{label} proof failed to verify"
        );

        print_proof(label, &proof, &root, i + 1 == cases.len());
    }

    println!("}}");
}
