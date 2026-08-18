//! Generate verification-grid shape test vectors JSON (P2.8a artifact).
//!
//! Pins the DAOS-derived grid-selection rule so a later re-implementation
//! stays faithful to it.
//!
//! Run with:
//! `cargo run --features merkle --example gen_verification_grid_vectors`

use clawhdf5_format::verification_grid::verification_grid;

const TARGET: u64 = 1024 * 1024; // 1 MiB

/// `(dims, elem_size, note)` — expected grids are computed, not hardcoded,
/// so this file regenerates faithfully if the rule is ever retuned.
const CASES: &[(&[u64], u32, &str)] = &[
    (&[1000, 1000, 1000], 4, "4000 leaves, depth 12"),
    (&[8760, 720, 1440], 4, "35,040 leaves, depth 16"),
    (&[1 << 30], 8, "1-D, 8192 leaves"),
    (&[1_000_000], 4, "1-D, 4 leaves"),
    (&[100, 100], 8, "below threshold, single leaf"),
    (&[2, 3, 4], 4, "degenerate, single leaf"),
];

fn list(v: &[u64]) -> String {
    v.iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

fn main() {
    println!("{{");
    println!(
        "  \"description\": \"Verification-grid derivation test vectors (P2.8a)\","
    );
    println!(
        "  \"specification\": \"S2-D2-Yr2/Merkle-tree-HDF5.tex sec:contiguous-design\","
    );
    println!("  \"target_bytes\": {TARGET},");
    println!("  \"cases\": [");

    for (i, (dims, elem_size, note)) in CASES.iter().enumerate() {
        let grid = verification_grid(dims, *elem_size, TARGET);
        let grid_json = match &grid {
            Some(g) => format!("[{}]", list(g)),
            None => "null".to_string(),
        };
        let n_leaves = match &grid {
            Some(g) => dims
                .iter()
                .zip(g.iter())
                .map(|(&n, &t)| n.div_ceil(t))
                .product::<u64>(),
            None => 1,
        };
        let comma = if i + 1 < CASES.len() { "," } else { "" };
        println!("    {{");
        println!("      \"dims\": [{}],", list(dims));
        println!("      \"elem_size\": {elem_size},");
        println!("      \"grid\": {grid_json},");
        println!("      \"n_leaves\": {n_leaves},");
        println!("      \"note\": \"{note}\"");
        println!("    }}{comma}");
    }

    println!("  ]");
    println!("}}");
}
