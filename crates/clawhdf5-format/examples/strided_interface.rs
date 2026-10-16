//! Strided selections issued as a hyperslab vs. as a point list.
//!
//! The verifier derives the expected chunk set from its own request. For a
//! hyperslab the per-axis test is overlap with the selection's extent, so a
//! strided hyperslab whose gaps exceed a chunk delivers its whole bounding
//! box; a point list delivers only the chunks holding a selected point. The
//! strided cases of the paper's Fig. 4 are point lists. This reproduces them
//! (cubic grids of single-element chunks, every `s`-th cell along the fastest
//! axis, block side `m`, aligned and offset origins, three leaf orders) and
//! issues each selection both ways, recording chunks delivered, pruned
//! witnesses, and serialized proof bytes under the accounting of
//! `subset_proof_geometry_bench`. Every proof is verified before it is
//! counted. Sizes are computed, not timed.
//!
//! Usage:
//!   cargo run -p clawhdf5-format --release --features merkle,blake3 --example strided_interface
//!
//! Output: benches/results/strided-interface.csv

use clawhdf5_format::merkle::{HashAlg, MerkleTree};

const HASH_SIZE: usize = 32;
use clawhdf5_format::selection::Selection;
use clawhdf5_format::subset_proof::{
    ChunkData, ChunkGridParams, LeafOrder, ProofConstruction, SubsetProof, extract_subset_with,
    leaf_index_for_coord, verify_subset_with,
};
use clawhdf5_format::verification_grid::LayoutClass;
use std::io::Write;

fn proof_size_bytes(proof: &SubsetProof) -> usize {
    let k = proof.chunk_indices.len();
    let d = proof.grid_params.dims.len();
    k * 8 + k * HASH_SIZE + proof.proof_nodes.len() * (8 + HASH_SIZE) + d * 8 + d * 8 + HASH_SIZE + HASH_SIZE
}

/// Cubic `side^rank` grid of single-element chunks under `order`, with the
/// payload indexed by leaf position (as in `subset_proof_geometry_bench`).
fn build(side: u64, rank: usize, order: LeafOrder) -> (MerkleTree, ChunkGridParams, Vec<Vec<u8>>) {
    let grid = ChunkGridParams::new(vec![side; rank], vec![1u64; rank], 4, LayoutClass::Chunked, HashAlg::Blake3);
    let n_per_dim = grid.n_chunks_per_dim();
    let total = grid.total_chunk_count() as usize;
    let mut payload = vec![Vec::new(); total];
    let mut coord = vec![0u64; rank];
    for f in 0..total {
        let mut rem = f as u64;
        for d in (0..rank).rev() {
            coord[d] = rem % side;
            rem /= side;
        }
        payload[leaf_index_for_coord(&coord, &n_per_dim, order) as usize] = format!("c{f}").into_bytes();
    }
    let refs: Vec<&[u8]> = payload.iter().map(Vec::as_slice).collect();
    (MerkleTree::from_chunks(&refs, HashAlg::Blake3), grid, payload)
}

/// Every `stride`-th cell along the fastest axis of an `m`-sided block at
/// `origin`, clipped to the grid: as a point list and as the hyperslab with
/// the same start, stride, count and unit block.
fn strided(side: u64, rank: usize, m: u64, origin: u64, stride: u64) -> (Selection, Selection, usize) {
    let mut points = Vec::new();
    let total = m.pow(rank as u32);
    for f in 0..total {
        let mut rem = f;
        let mut c = vec![0u64; rank];
        for d in (0..rank).rev() {
            c[d] = rem % m;
            rem /= m;
        }
        let mut p: Vec<u64> = (0..rank).map(|d| origin + c[d]).collect();
        p[rank - 1] = origin + c[rank - 1] * stride;
        if p[rank - 1] < side {
            points.push(p);
        }
    }
    let last = (0..m).filter(|&c| origin + c * stride < side).count() as u64;
    let mut count = vec![m; rank];
    count[rank - 1] = last;
    let mut st = vec![1u64; rank];
    st[rank - 1] = stride;
    let hyper = Selection::Hyperslab { start: vec![origin; rank], stride: st, count, block: vec![1u64; rank] };
    let k = points.len();
    (Selection::Points(points), hyper, k)
}

fn checked_proof(
    tree: &MerkleTree,
    grid: &ChunkGridParams,
    payload: &[Vec<u8>],
    sel: &Selection,
    order: LeafOrder,
) -> SubsetProof {
    let c = ProofConstruction::CanonicalPruned;
    let proof = extract_subset_with(tree, grid, sel, order, c).expect("extract");
    let delivered: Vec<ChunkData<'_>> =
        proof.chunk_indices.iter().map(|&i| ChunkData { index: i, data: &payload[i] }).collect();
    assert!(
        verify_subset_with(tree.root(), HashAlg::Blake3, &delivered, &proof, grid, &grid.grid_hash, sel, order, c)
            .expect("verify"),
        "proof failed to verify"
    );
    proof
}

fn main() -> std::io::Result<()> {
    let orders = [("row_major", LeafOrder::RowMajor), ("morton", LeafOrder::Morton), ("hilbert", LeafOrder::Hilbert)];
    // Fig. 4's largest selections: rank 2 on 256^2 (m = 32), rank 3 on 32^3 (m = 16).
    let configs: [(usize, u64, u64); 2] = [(2, 256, 32), (3, 32, 16)];
    let alignments: [(&str, fn(u64, u64) -> u64); 2] =
        [("aligned", |side, m| (side / 4 / m.max(1)) * m), ("offset", |side, m| (side / 4 / m.max(1)) * m + m / 2 + 1)];

    let out_path = concat!(env!("CARGO_MANIFEST_DIR"), "/benches/results/strided-interface.csv");
    let mut out = std::fs::File::create(out_path)?;
    writeln!(out, "rank,side,m,stride,alignment,order,selected_k,interface,chunks_delivered,witnesses,proof_bytes")?;
    println!(
        "{:>4} {:>6} {:>8} {:>10} {:>5} | {:>18} | {:>18} | {:>18}",
        "rank", "stride", "align", "order", "k", "chunks pts/hyp", "witness pts/hyp", "proof B pts/hyp"
    );
    for &(rank, side, m) in &configs {
        for &(oname, order) in &orders {
            let (tree, grid, payload) = build(side, rank, order);
            for stride in [2u64, 4] {
                for &(aname, ofn) in &alignments {
                    let origin = ofn(side, m).min(side - m);
                    let (pts, hyp, k) = strided(side, rank, m, origin, stride);
                    let pp = checked_proof(&tree, &grid, &payload, &pts, order);
                    let hp = checked_proof(&tree, &grid, &payload, &hyp, order);
                    for (iname, p) in [("points", &pp), ("hyperslab", &hp)] {
                        writeln!(
                            out,
                            "{rank},{side},{m},{stride},{aname},{oname},{k},{iname},{},{},{}",
                            p.chunk_indices.len(),
                            p.proof_nodes.len(),
                            proof_size_bytes(p)
                        )?;
                    }
                    println!(
                        "{rank:>4} {stride:>6} {aname:>8} {oname:>10} {k:>5} | {:>8} / {:>7} | {:>8} / {:>7} | {:>8} / {:>7}",
                        pp.chunk_indices.len(),
                        hp.chunk_indices.len(),
                        pp.proof_nodes.len(),
                        hp.proof_nodes.len(),
                        proof_size_bytes(&pp),
                        proof_size_bytes(&hp)
                    );
                }
            }
        }
    }
    println!("wrote {out_path}");
    Ok(())
}
