#![no_main]
use clawhdf5_format::merkle::{HashAlg, MerkleTree};
use clawhdf5_format::selection::Selection;
use clawhdf5_format::subset_proof::{
    ChunkData, ChunkGridParams, LeafOrder, SubsetProof, extract_subset, verify_subset,
};
use libfuzzer_sys::fuzz_target;
use std::collections::BTreeMap;

/// Byte cursor for carving small, bounded values out of the fuzzer's raw
/// input. The goal is panic/hang detection across `subset_proof`'s public
/// API surface (both the "verifier's own trusted grid/selection" inputs and
/// the untrusted, wire-supplied `SubsetProof`/`ChunkData`), not deep semantic
/// coverage — so most derived sizes are kept small enough that an in-bounds
/// `O(total_chunks)` sweep stays cheap, with an occasional huge value to
/// exercise the overflow/depth guards.
struct Cursor<'a>(&'a [u8]);

impl<'a> Cursor<'a> {
    fn byte(&mut self) -> u8 {
        match self.0.split_first() {
            Some((&b, rest)) => {
                self.0 = rest;
                b
            }
            None => 0,
        }
    }

    fn u64(&mut self) -> u64 {
        let mut buf = [0u8; 8];
        for b in &mut buf {
            *b = self.byte();
        }
        u64::from_le_bytes(buf)
    }

    fn hash(&mut self) -> [u8; 32] {
        let mut h = [0u8; 32];
        for b in &mut h {
            *b = self.byte();
        }
        h
    }

    fn bounded(&mut self, max: u64) -> u64 {
        self.u64() % (max + 1)
    }

    fn rest(&mut self) -> &'a [u8] {
        core::mem::take(&mut self.0)
    }
}

fuzz_target!(|data: &[u8]| {
    let mut c = Cursor(data);

    let ndim = 1 + c.bounded(2) as usize; // rank 1..=3
    let mut dims: Vec<u64> = (0..ndim).map(|_| c.bounded(16)).collect();
    let chunk_shape: Vec<u64> = (0..ndim).map(|_| c.bounded(8)).collect();
    if c.byte() % 8 == 0 {
        dims[0] = u64::MAX; // occasionally probe the overflow/depth guards
    }
    let grid = ChunkGridParams::new(dims.clone(), chunk_shape, HashAlg::Blake3);

    let start = c.bounded(16);
    let len = 1 + c.bounded(16);
    let ranges: Vec<_> = (0..ndim)
        .map(|_| start..start.saturating_add(len))
        .collect();
    let sel = Selection::slice(&ranges);
    let order = if c.byte() % 2 == 0 {
        LeafOrder::RowMajor
    } else {
        LeafOrder::Morton
    };

    let tree_chunks: Vec<Vec<u8>> = (0..8u8).map(|i| vec![i; 4]).collect();
    let refs: Vec<&[u8]> = tree_chunks.iter().map(Vec::as_slice).collect();
    let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);

    // `extract_subset` must never panic on a caller-supplied grid/selection,
    // however malformed.
    let extracted = extract_subset(&tree, &grid, &sel, order);

    // `verify_subset` must never panic on an adversarial `proof`/`chunks`
    // pair, regardless of what the (trusted) expected_grid/sel/order say.
    let n_idx = c.bounded(8) as usize;
    let chunk_indices: Vec<usize> = (0..n_idx).map(|_| c.u64() as usize).collect();
    let leaf_hashes: Vec<[u8; 32]> = (0..n_idx).map(|_| c.hash()).collect();
    let mut proof_nodes = BTreeMap::new();
    for _ in 0..c.bounded(8) {
        proof_nodes.insert(c.u64(), c.hash());
    }
    let adversarial_proof = SubsetProof {
        chunk_indices: chunk_indices.clone(),
        leaf_hashes,
        proof_nodes,
        grid_params: grid.clone(),
        coverage_cert: c.hash(),
    };
    // Fuzz both a "trusted" grid hash that matches `grid` and an arbitrary
    // adversarial one, since verify_subset must never panic either way. Read
    // this before `c.rest()` below drains the cursor.
    let use_real_grid_hash = c.byte() % 2 == 0;
    let fuzzed_grid_hash = c.hash();
    let trusted_grid_hash = if use_real_grid_hash {
        grid.grid_hash
    } else {
        fuzzed_grid_hash
    };
    let chunk_data = c.rest();
    let chunks: Vec<ChunkData<'_>> = chunk_indices
        .iter()
        .enumerate()
        .map(|(i, &idx)| {
            let start = (i * 4).min(chunk_data.len());
            let end = (start + 4).min(chunk_data.len());
            ChunkData {
                index: idx,
                data: &chunk_data[start..end],
            }
        })
        .collect();
    let _ = verify_subset(
        tree.root(),
        HashAlg::Blake3,
        &chunks,
        &adversarial_proof,
        &grid,
        &trusted_grid_hash,
        &sel,
        order,
    );

    // And the legitimate round trip — a real proof over real delivered
    // chunks — must also never panic.
    if let Ok(proof) = extracted {
        let delivered: Vec<ChunkData<'_>> = proof
            .chunk_indices
            .iter()
            .map(|&idx| ChunkData {
                index: idx,
                data: tree_chunks.get(idx).map_or(&[][..], Vec::as_slice),
            })
            .collect();
        let _ = verify_subset(
            tree.root(),
            HashAlg::Blake3,
            &delivered,
            &proof,
            &grid,
            &grid.grid_hash,
            &sel,
            order,
        );
    }
});
