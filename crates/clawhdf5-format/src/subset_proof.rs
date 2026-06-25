//! Verifiable subset extraction (P1.5★).
//!
//! Lets a recipient of an arbitrary hyperslab cryptographically prove that
//! the delivered chunks are a *complete and correct* subset of a signed
//! parent dataset: every delivered chunk is authenticated by a Merkle path
//! to the root, and a coverage certificate binds the exact set of chunk
//! indices that were extracted so that a silently omitted or substituted
//! chunk is detectable without the verifier needing to re-derive the
//! original hyperslab.
//!
//! See `S2-D2-Yr2/Merkle-tree-HDF5.tex` §"Verifiable Subset Extraction"
//! for the design rationale and soundness sketch.

#[cfg(not(feature = "std"))]
use alloc::{collections::BTreeMap, vec, vec::Vec};

#[cfg(feature = "std")]
use std::collections::BTreeMap;

use crate::merkle::{HASH_SIZE, HashAlg, MerkleError, MerkleTree, constant_time_eq};
use crate::selection::Selection;

/// Trusted chunk-grid parameters anchored by the Merkle path to the
/// file-level signed root (the "coverage certificate" component (b)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkGridParams {
    /// Dataset dimensions (element counts per axis).
    pub dims: Vec<u64>,
    /// Chunk dimensions (element counts per axis).
    pub chunk_shape: Vec<u64>,
    /// `H(dims || chunk_shape)`, binding the grid parameters.
    pub grid_hash: [u8; HASH_SIZE],
}

impl ChunkGridParams {
    /// Construct grid params, computing `grid_hash` from `dims` and `chunk_shape`.
    #[must_use]
    pub fn new(dims: Vec<u64>, chunk_shape: Vec<u64>, alg: HashAlg) -> Self {
        let grid_hash = compute_grid_hash(&dims, &chunk_shape, alg);
        Self {
            dims,
            chunk_shape,
            grid_hash,
        }
    }

    /// Number of chunks per dimension, `ceil(dims[d] / chunk_shape[d])`.
    ///
    /// A zero entry in `chunk_shape` is structurally invalid (internal callers
    /// reject it via `validate_grid_shape` before ever reaching the
    /// `O(total_chunks)` sweep); rather than divide-by-zero, such an axis
    /// reports `0` chunks here so direct callers of this public method can't
    /// be panicked by a malformed `ChunkGridParams` (its fields are all
    /// `pub`, so construction isn't restricted to [`ChunkGridParams::new`]).
    #[must_use]
    pub fn n_chunks_per_dim(&self) -> Vec<u64> {
        self.dims
            .iter()
            .zip(self.chunk_shape.iter())
            .map(|(&d, &c)| if c == 0 { 0 } else { d.div_ceil(c) })
            .collect()
    }

    /// Total chunk count across all dimensions.
    #[must_use]
    pub fn total_chunk_count(&self) -> u64 {
        self.n_chunks_per_dim().iter().product()
    }
}

fn compute_grid_hash(dims: &[u64], chunk_shape: &[u64], alg: HashAlg) -> [u8; HASH_SIZE] {
    let mut buf = Vec::with_capacity((dims.len() + chunk_shape.len()) * 8);
    for &d in dims {
        buf.extend_from_slice(&d.to_le_bytes());
    }
    for &c in chunk_shape {
        buf.extend_from_slice(&c.to_le_bytes());
    }
    alg.hash_leaf(&buf)
}

/// Leaf-linearization ordering used to map an N-dimensional chunk
/// coordinate to a 1D Merkle-tree leaf index (RQ6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LeafOrder {
    /// 1D index = `sum_i coord[i] * prod_{j>i} n_chunks[j]`.
    #[default]
    RowMajor,
    /// Z-order curve via bit-interleaving of per-axis chunk coordinates.
    Morton,
}

/// A delivered chunk paired with its claimed leaf index, as supplied by the
/// recipient to [`verify_subset`].
#[derive(Debug, Clone, Copy)]
pub struct ChunkData<'a> {
    /// Leaf index (under the proof's [`LeafOrder`]) this chunk corresponds to.
    pub index: usize,
    /// Raw chunk bytes (pre-filter-pipeline, as hashed by the leaf hash).
    pub data: &'a [u8],
}

/// A proof that a set of delivered chunks is a complete, correct subset of
/// a signed parent dataset (§"Verifiable Subset Extraction").
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubsetProof {
    /// Sorted, deduplicated leaf indices covering the requested hyperslab.
    pub chunk_indices: Vec<usize>,
    /// Leaf hashes for `chunk_indices`, in the same order.
    pub leaf_hashes: Vec<[u8; HASH_SIZE]>,
    /// Deduplicated sibling nodes, keyed by level-order tree index (root = 0,
    /// left child of `i` = `2i+1`, right child = `2i+2`).
    pub proof_nodes: BTreeMap<u64, [u8; HASH_SIZE]>,
    /// Coverage certificate component (b): the trusted chunk-grid parameters.
    pub grid_params: ChunkGridParams,
    /// `H(sorted(chunk_indices) || grid_hash)` — component for completeness
    /// binding (component (b)/(c) combined at the index-set level).
    pub coverage_cert: [u8; HASH_SIZE],
}

/// Bit-interleave per-axis chunk coordinates into a single Z-order (Morton)
/// index. Bit `k` of `coords[d]` is placed at output bit `k * coords.len() + d`.
///
/// Bits beyond `64 / coords.len()` per axis are dropped (caller must ensure
/// chunk-grid extents fit in the available bit budget).
#[must_use]
pub fn morton_index(coords: &[u64]) -> u64 {
    let ndim = coords.len();
    if ndim == 0 {
        return 0;
    }
    let bits_per_coord = 64 / ndim;
    let mut result = 0u64;
    for bit in 0..bits_per_coord {
        for (dim, &coord) in coords.iter().enumerate() {
            if (coord >> bit) & 1 == 1 {
                result |= 1u64 << (bit * ndim + dim);
            }
        }
    }
    result
}

fn row_major_index(coord: &[u64], n_per_dim: &[u64]) -> u64 {
    let mut idx = 0u64;
    for d in 0..coord.len() {
        let trailing_product: u64 = n_per_dim[d + 1..].iter().product();
        idx += coord[d] * trailing_product;
    }
    idx
}

fn coord_to_leaf_index(coord: &[u64], n_per_dim: &[u64], order: LeafOrder) -> u64 {
    match order {
        LeafOrder::RowMajor => row_major_index(coord, n_per_dim),
        LeafOrder::Morton => morton_index(coord),
    }
}

/// Inclusive chunk-coordinate bounds `(lo, hi)` per dimension covering every
/// chunk that could possibly overlap `sel`, or `None` if the selection touches
/// no chunk in this grid. Derived from the selection's element-space bounding
/// box, this lets [`chunk_coords_for_selection`] sweep only the enclosing
/// sub-grid instead of the whole grid.
///
/// `chunk_shape` must be non-zero on every axis (the sole caller validates this
/// via [`checked_padded_leaf_count`]); a zero axis is treated as "no chunks" to
/// stay panic-free if this helper is ever reused.
fn selection_chunk_bounds(
    sel: &Selection,
    n_per_dim: &[u64],
    chunk_shape: &[u64],
) -> Option<(Vec<u64>, Vec<u64>)> {
    let ndim = n_per_dim.len();

    // Element-space half-open bounding box `[lo, hi_excl)` per dimension.
    let (el_lo, el_hi_excl): (Vec<u64>, Vec<u64>) = match sel {
        // Whole dataspace: every chunk is in range.
        Selection::All => {
            return Some((vec![0u64; ndim], n_per_dim.iter().map(|&n| n - 1).collect()));
        }
        Selection::None => return None,
        Selection::Hyperslab {
            start,
            stride,
            count,
            block,
        } => {
            let mut lo = Vec::with_capacity(ndim);
            let mut hi = Vec::with_capacity(ndim);
            for d in 0..ndim {
                if count[d] == 0 || block[d] == 0 {
                    return None; // empty on this axis ⇒ empty overall
                }
                lo.push(start[d]);
                hi.push(start[d] + (count[d] - 1) * stride[d] + block[d]);
            }
            (lo, hi)
        }
        Selection::Points(pts) => {
            if pts.is_empty() {
                return None;
            }
            let mut lo = vec![u64::MAX; ndim];
            let mut hi_incl = vec![0u64; ndim];
            for pt in pts {
                for d in 0..ndim {
                    lo[d] = lo[d].min(pt[d]);
                    hi_incl[d] = hi_incl[d].max(pt[d]);
                }
            }
            (lo, hi_incl.iter().map(|&h| h + 1).collect())
        }
    };

    // Map the element box onto inclusive chunk-coordinate bounds, clamped to
    // the grid. Any axis whose box falls entirely outside the grid ⇒ no chunks.
    let mut lo = Vec::with_capacity(ndim);
    let mut hi = Vec::with_capacity(ndim);
    for d in 0..ndim {
        if chunk_shape[d] == 0 || el_hi_excl[d] == 0 || el_lo[d] >= n_per_dim[d] * chunk_shape[d] {
            return None;
        }
        let lo_chunk = el_lo[d] / chunk_shape[d];
        let hi_chunk = ((el_hi_excl[d] - 1) / chunk_shape[d]).min(n_per_dim[d] - 1);
        if lo_chunk > hi_chunk {
            return None;
        }
        lo.push(lo_chunk);
        hi.push(hi_chunk);
    }
    Some((lo, hi))
}

/// Enumerate every chunk coordinate (as an N-dim index tuple) that overlaps
/// `sel`, reusing [`Selection::intersects_chunk`]. Rather than sweeping the
/// full chunk grid, this sweeps only the selection's chunk-space bounding box
/// (see [`selection_chunk_bounds`]), reducing the cost from `O(total_chunks)`
/// to `O(bounding_box_chunks)` — a large win for a small selection in a big
/// grid. The `intersects_chunk` filter is retained inside the box so strided
/// hyperslabs and sparse point sets still exclude non-overlapping chunks. The
/// emitted coordinates and their row-major order are identical to the
/// full-grid sweep.
fn chunk_coords_for_selection(
    sel: &Selection,
    n_per_dim: &[u64],
    chunk_shape: &[u64],
) -> Vec<Vec<u64>> {
    let ndim = n_per_dim.len();
    if ndim == 0 || n_per_dim.contains(&0) {
        return vec![];
    }

    let (lo, hi) = match selection_chunk_bounds(sel, n_per_dim, chunk_shape) {
        Some(bounds) => bounds,
        None => return vec![],
    };

    let mut coords = Vec::new();
    let mut counter = lo.clone();
    loop {
        let chunk_offset: Vec<u64> = counter
            .iter()
            .zip(chunk_shape.iter())
            .map(|(&c, &s)| c * s)
            .collect();
        if sel.intersects_chunk(&chunk_offset, chunk_shape) {
            coords.push(counter.clone());
        }

        // Odometer increment over the inclusive box [lo, hi], last axis first.
        let mut d = ndim;
        loop {
            if d == 0 {
                return coords;
            }
            d -= 1;
            counter[d] += 1;
            if counter[d] <= hi[d] {
                break;
            }
            counter[d] = lo[d];
        }
    }
}

fn coverage_cert(
    sorted_indices: &[usize],
    grid_hash: &[u8; HASH_SIZE],
    alg: HashAlg,
) -> [u8; HASH_SIZE] {
    let mut buf = Vec::with_capacity(sorted_indices.len() * 8 + HASH_SIZE);
    for &idx in sorted_indices {
        buf.extend_from_slice(&(idx as u64).to_le_bytes());
    }
    buf.extend_from_slice(grid_hash);
    alg.hash_leaf(&buf)
}

/// Compute the sorted, deduplicated set of leaf indices that `sel` touches
/// under `grid`'s chunk grid and `order`'s leaf-linearization.
///
/// Shared by [`extract_subset`] (to build a proof) and [`verify_subset`] (to
/// independently recompute the expected chunk set from the verifier's own
/// trusted selection/grid, rather than trusting the prover's claimed
/// `chunk_indices`). Keeping a single implementation prevents the two from
/// silently drifting apart.
///
/// # Errors
///
/// Returns [`MerkleError::HyperslabOutOfBounds`] if `sel`'s rank does not
/// match `grid.dims`. `chunk_coords_for_selection` builds chunk-offset tuples
/// sized to `grid.dims` and hands them to [`Selection::intersects_chunk`],
/// which indexes them per-axis without a bounds check, so a rank mismatch
/// (e.g. a 3D selection against a 2D grid) would otherwise panic there
/// instead of surfacing as an error.
///
/// Also rejects (via [`checked_padded_leaf_count`]) a structurally invalid or
/// adversarially huge `grid` *before* the `O(total_chunks)` sweep below, so a
/// malformed grid fails fast (zero chunk-shape, mismatched axis counts) or
/// bounded (`MerkleError::TreeTooDeep`) instead of dividing by zero or
/// sweeping an astronomical chunk count.
fn compute_expected_chunk_indices(
    grid: &ChunkGridParams,
    sel: &Selection,
    order: LeafOrder,
) -> Result<Vec<usize>, MerkleError> {
    if let Some(rank) = sel.rank().filter(|&r| r != grid.dims.len()) {
        return Err(MerkleError::HyperslabOutOfBounds { idx: rank });
    }
    checked_padded_leaf_count(grid)?;

    let n_per_dim = grid.n_chunks_per_dim();
    let coords = chunk_coords_for_selection(sel, &n_per_dim, &grid.chunk_shape);

    let mut chunk_indices: Vec<usize> = coords
        .iter()
        .map(|c| coord_to_leaf_index(c, &n_per_dim, order) as usize)
        .collect();
    chunk_indices.sort_unstable();
    chunk_indices.dedup();
    Ok(chunk_indices)
}

/// Extract a [`SubsetProof`] covering hyperslab `sel` from `tree`.
///
/// `order` must match the leaf-linearization ordering `tree`'s leaves were
/// built with, or the returned proof will authenticate the wrong chunks.
///
/// # Errors
///
/// Returns [`MerkleError::HyperslabOutOfBounds`] if the selection touches a
/// chunk coordinate outside `grid`'s chunk grid, or if `sel`'s rank does not
/// match `grid.dims`.
pub fn extract_subset(
    tree: &MerkleTree,
    grid: &ChunkGridParams,
    sel: &Selection,
    order: LeafOrder,
) -> Result<SubsetProof, MerkleError> {
    let chunk_indices = compute_expected_chunk_indices(grid, sel, order)?;

    let padded_count = tree.padded_leaf_count();
    let internal_nodes = padded_count - 1;
    let nodes = tree.nodes();

    let mut proof_nodes: BTreeMap<u64, [u8; HASH_SIZE]> = BTreeMap::new();
    let mut leaf_hashes: Vec<[u8; HASH_SIZE]> = Vec::with_capacity(chunk_indices.len());

    for &leaf_idx in &chunk_indices {
        let hash = tree
            .leaf_hash(leaf_idx)
            .ok_or(MerkleError::HyperslabOutOfBounds { idx: leaf_idx })?;
        leaf_hashes.push(*hash);

        let mut node_idx = internal_nodes + leaf_idx;
        while node_idx > 0 {
            let sibling_idx = if node_idx % 2 == 1 {
                node_idx + 1
            } else {
                node_idx - 1
            };
            let sibling_hash = *nodes
                .get(sibling_idx)
                .ok_or(MerkleError::HyperslabOutOfBounds { idx: leaf_idx })?;
            proof_nodes
                .entry(sibling_idx as u64)
                .or_insert(sibling_hash);
            node_idx = (node_idx - 1) / 2;
        }
    }

    let cert = coverage_cert(&chunk_indices, &grid.grid_hash, tree.algorithm());

    Ok(SubsetProof {
        chunk_indices,
        leaf_hashes,
        proof_nodes,
        grid_params: grid.clone(),
        coverage_cert: cert,
    })
}

/// Reject a structurally invalid grid: mismatched axis counts between `dims`
/// and `chunk_shape`, or a zero chunk-shape entry (which would divide-by-zero
/// in [`ChunkGridParams::n_chunks_per_dim`]).
fn validate_grid_shape(grid: &ChunkGridParams) -> Result<(), MerkleError> {
    if grid.dims.len() != grid.chunk_shape.len() {
        return Err(MerkleError::CompanionTampered);
    }
    if grid.chunk_shape.contains(&0) {
        return Err(MerkleError::CompanionTampered);
    }
    Ok(())
}

/// Maximum allowed Merkle-tree depth (and, transitively, the bound enforced
/// on the chunk-grid size that [`compute_expected_chunk_indices`]'s
/// `O(total_chunks)` sweep is allowed to run over).
const MAX_TREE_DEPTH: usize = 40;

/// Compute the padded leaf count implied by `grid`, without panicking or
/// hanging on adversarial input.
///
/// `grid` arrives over the wire as part of an untrusted [`SubsetProof`] (and,
/// via [`compute_expected_chunk_indices`], is also checked before a verifier-
/// supplied `expected_grid` drives an `O(total_chunks)` sweep), so this must
/// reject — rather than divide-by-zero, overflow, panic, or spin forever on —
/// mismatched axis counts, zero chunk-shape entries, chunk counts that don't
/// fit in a `usize` or whose next power of two would overflow, and grids
/// whose implied depth exceeds [`MAX_TREE_DEPTH`].
fn checked_padded_leaf_count(grid: &ChunkGridParams) -> Result<usize, MerkleError> {
    validate_grid_shape(grid)?;
    let mut total: u64 = 1;
    for (&d, &c) in grid.dims.iter().zip(grid.chunk_shape.iter()) {
        total = total
            .checked_mul(d.div_ceil(c))
            .ok_or(MerkleError::TreeTooDeep { depth: usize::MAX })?;
    }
    let padded_count = usize::try_from(total)
        .ok()
        .and_then(|t| t.checked_next_power_of_two())
        .ok_or(MerkleError::TreeTooDeep { depth: usize::MAX })?;
    let depth = padded_count.trailing_zeros() as usize + 1;
    if depth > MAX_TREE_DEPTH {
        return Err(MerkleError::TreeTooDeep { depth });
    }
    Ok(padded_count)
}

/// Verify that `chunks` is a complete, correct, untampered subset of the
/// dataset rooted at `root`, covering exactly the region the verifier
/// requested (`expected_grid`, `sel`, `order`).
///
/// `expected_grid`/`sel`/`order` must come from the verifier's own trusted
/// knowledge of the dataset and request — never from `proof` itself, or a
/// prover holding the real tree could satisfy every other check here while
/// substituting a proof for a different region than was requested.
/// `proof.grid_params` is treated as untrusted and is not used to derive the
/// expected chunk set; it only feeds the (redundant, defense-in-depth)
/// coverage-certificate check below.
///
/// `chunks` must be in the same order as `proof.chunk_indices`, with
/// `chunks[i].index == proof.chunk_indices[i]` — this is what lets the
/// coverage certificate detect a silently omitted chunk: dropping an entry
/// changes the delivered index set's hash, which no longer matches
/// `proof.coverage_cert`.
///
/// # Errors
///
/// - [`MerkleError::CompanionTampered`] if `chunks` doesn't match
///   `proof.chunk_indices` (wrong length, wrong index, or reordered) —
///   this is the "chunk silently omitted/substituted" detection path. The
///   same error is returned for a structurally invalid `proof.grid_params`
///   (mismatched axis counts or a zero chunk-shape entry).
/// - [`MerkleError::SelectionMismatch`] if `proof.chunk_indices` does not
///   equal the chunk set independently recomputed from `expected_grid`/
///   `sel`/`order` — the proof covers the wrong region.
/// - [`MerkleError::HashMismatch`] if a delivered chunk's content doesn't
///   match its claimed leaf hash.
/// - [`MerkleError::TreeTooDeep`] if the grid's implied tree depth exceeds
///   the maximum allowed, including when the implied chunk count overflows.
pub fn verify_subset(
    root: &[u8; HASH_SIZE],
    alg: HashAlg,
    chunks: &[ChunkData<'_>],
    proof: &SubsetProof,
    expected_grid: &ChunkGridParams,
    sel: &Selection,
    order: LeafOrder,
) -> Result<bool, MerkleError> {
    // `proof` is untrusted wire data: chunk_indices/leaf_hashes are walked
    // in lockstep below by index, so their lengths (and the caller-supplied
    // `chunks`) must all agree before any indexing happens.
    if chunks.len() != proof.chunk_indices.len() || chunks.len() != proof.leaf_hashes.len() {
        return Err(MerkleError::CompanionTampered);
    }
    for (chunk, &expected_idx) in chunks.iter().zip(proof.chunk_indices.iter()) {
        if chunk.index != expected_idx {
            return Err(MerkleError::CompanionTampered);
        }
    }

    // Bind the proof to what the verifier actually asked for. Recomputing
    // from `expected_grid`/`sel` (trusted) rather than `proof.grid_params`
    // (untrusted) is what prevents a prover from presenting a proof for a
    // different — but internally self-consistent — region.
    let expected_indices = compute_expected_chunk_indices(expected_grid, sel, order)?;
    if proof.chunk_indices != expected_indices {
        return Err(MerkleError::SelectionMismatch);
    }

    // Recompute the coverage certificate over the delivered index set;
    // an omitted/substituted/reordered chunk changes this hash.
    let mut delivered: Vec<usize> = chunks.iter().map(|c| c.index).collect();
    delivered.sort_unstable();
    let cert = coverage_cert(&delivered, &proof.grid_params.grid_hash, alg);
    if !constant_time_eq(&cert, &proof.coverage_cert) {
        return Err(MerkleError::CompanionTampered);
    }

    let padded_count = checked_padded_leaf_count(expected_grid)?;
    let internal_nodes = padded_count - 1;

    for (i, (chunk, &leaf_idx)) in chunks.iter().zip(proof.chunk_indices.iter()).enumerate() {
        let computed_leaf_hash = alg.hash_leaf(chunk.data);
        if !constant_time_eq(&computed_leaf_hash, &proof.leaf_hashes[i]) {
            return Err(MerkleError::HashMismatch {
                chunk_idx: leaf_idx,
            });
        }

        let mut node_idx = internal_nodes + leaf_idx;
        // Guard against overflow in sibling_idx calculation below. In practice,
        // node_idx cannot approach usize::MAX because internal_nodes + leaf_idx
        // is bounded by 2*n_total - 1 where n_total is the number of chunks,
        // which is itself bounded by practical storage limits.
        debug_assert!(node_idx < usize::MAX, "node_idx overflow guard");
        let mut current = computed_leaf_hash;
        let mut level_leaf_idx = leaf_idx;
        while node_idx > 0 {
            let sibling_idx = if node_idx % 2 == 1 {
                node_idx + 1
            } else {
                node_idx - 1
            };
            let sibling = proof
                .proof_nodes
                .get(&(sibling_idx as u64))
                .copied()
                .ok_or(MerkleError::CompanionTampered)?;
            current = if level_leaf_idx % 2 == 0 {
                alg.hash_pair(&current, &sibling)
            } else {
                alg.hash_pair(&sibling, &current)
            };
            node_idx = (node_idx - 1) / 2;
            level_leaf_idx /= 2;
        }

        if !constant_time_eq(&current, root) {
            return Err(MerkleError::HashMismatch {
                chunk_idx: leaf_idx,
            });
        }
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tree_and_grid(n_chunks: usize) -> (MerkleTree, ChunkGridParams, Vec<Vec<u8>>) {
        let chunks: Vec<Vec<u8>> = (0..n_chunks)
            .map(|i| format!("chunk-{i}").into_bytes())
            .collect();
        let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
        let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);
        // 1D grid: n_chunks chunks of shape [1], dataset dims [n_chunks].
        let grid = ChunkGridParams::new(vec![n_chunks as u64], vec![1], HashAlg::Blake3);
        (tree, grid, chunks)
    }

    #[test]
    fn test_morton_index_3d_reference() {
        // Standard 3D Z-order: bit k of axis d -> output bit k*3 + d.
        // (1,2,3): x=001 y=010 z=011 -> bits set at 0(x0),2(z0),4(y1),5(z1) = 0b110101 = 0x35.
        //
        // NOTE: The S2-D2-Yr2 TeX draft's worked example (line 3052) states
        // MortonIndex(1,2,3) = 0x15, but this is an arithmetic error in the spec.
        // Applying the stated bit-interleaving rule correctly yields 0x35.
        // See test-vectors/morton-vectors.json for full documentation.
        assert_eq!(morton_index(&[1, 2, 3]), 0x35);
        assert_eq!(morton_index(&[0, 0, 0]), 0);
        assert_eq!(morton_index(&[1, 0, 0]), 1);
        assert_eq!(morton_index(&[0, 1, 0]), 2);
        assert_eq!(morton_index(&[0, 0, 1]), 4);
    }

    #[test]
    fn test_row_major_matches_natural_order() {
        let n_per_dim = vec![4u64];
        for i in 0..4u64 {
            assert_eq!(row_major_index(&[i], &n_per_dim), i);
        }
        let n_per_dim_2d = vec![2u64, 3u64];
        // coord (1, 2) in a 2x3 grid -> 1*3 + 2 = 5
        assert_eq!(row_major_index(&[1, 2], &n_per_dim_2d), 5);
    }

    #[test]
    fn test_n_chunks_per_dim_rejects_zero_chunk_shape_without_panicking() {
        // `ChunkGridParams`'s fields are all `pub`, so a caller can construct
        // one with a zero `chunk_shape` entry directly (bypassing every
        // `validate_grid_shape` gate inside this module) and then call the
        // public accessors straight away.
        let grid = ChunkGridParams::new(vec![10, 10], vec![2, 0], HashAlg::Blake3);
        assert_eq!(grid.n_chunks_per_dim(), vec![5, 0]);
        assert_eq!(grid.total_chunk_count(), 0);
    }

    #[test]
    fn test_bounded_sweep_matches_full_grid_sweep() {
        // Reference: the original O(total_chunks) full-grid sweep that the
        // bounding-box optimization replaced. The optimized
        // `chunk_coords_for_selection` must return identical coordinates (same
        // set AND same row-major order) for every selection shape.
        fn brute_force(sel: &Selection, n_per_dim: &[u64], chunk_shape: &[u64]) -> Vec<Vec<u64>> {
            let ndim = n_per_dim.len();
            if ndim == 0 || n_per_dim.contains(&0) {
                return vec![];
            }
            let mut coords = Vec::new();
            let mut counter = vec![0u64; ndim];
            loop {
                let chunk_offset: Vec<u64> = counter
                    .iter()
                    .zip(chunk_shape.iter())
                    .map(|(&c, &s)| c * s)
                    .collect();
                if sel.intersects_chunk(&chunk_offset, chunk_shape) {
                    coords.push(counter.clone());
                }
                let mut d = ndim;
                loop {
                    if d == 0 {
                        return coords;
                    }
                    d -= 1;
                    counter[d] += 1;
                    if counter[d] < n_per_dim[d] {
                        break;
                    }
                    counter[d] = 0;
                }
            }
        }

        // A large-ish 2D grid so a small selection's bounding box is a tiny
        // fraction of the full grid.
        let n_per_dim = [20u64, 16u64];
        let chunk_shape = [4u64, 8u64]; // element extent 80 x 128

        let cases: Vec<Selection> = vec![
            Selection::All,
            Selection::None,
            // Small contiguous slab far from the origin.
            Selection::slice(&[50..60, 70..90]),
            // Strided hyperslab: blocks of 2, stride 10 — leaves gaps so the
            // intersects_chunk filter must still exclude chunks inside the box.
            Selection::Hyperslab {
                start: vec![3, 5],
                stride: vec![10, 20],
                count: vec![4, 3],
                block: vec![2, 4],
            },
            // Sparse points spread across the grid.
            Selection::Points(vec![vec![1, 2], vec![77, 5], vec![40, 120], vec![0, 0]]),
            // Selection partly past the grid edge — clamping must not panic.
            Selection::slice(&[70..100, 120..200]),
            // Selection entirely past the grid edge ⇒ no chunks.
            Selection::slice(&[500..600, 0..8]),
        ];

        for sel in &cases {
            assert_eq!(
                chunk_coords_for_selection(sel, &n_per_dim, &chunk_shape),
                brute_force(sel, &n_per_dim, &chunk_shape),
                "bounded sweep diverged from full-grid sweep for {sel:?}",
            );
        }
    }

    #[test]
    fn test_extract_and_verify_subset_contiguous_slab() {
        let (tree, grid, chunks) = make_tree_and_grid(8);
        let sel = Selection::slice(&[2..5]);

        let proof = extract_subset(&tree, &grid, &sel, LeafOrder::RowMajor).unwrap();
        assert_eq!(proof.chunk_indices, vec![2, 3, 4]);

        let delivered: Vec<ChunkData<'_>> = proof
            .chunk_indices
            .iter()
            .map(|&idx| ChunkData {
                index: idx,
                data: &chunks[idx],
            })
            .collect();

        let ok = verify_subset(
            tree.root(),
            HashAlg::Blake3,
            &delivered,
            &proof,
            &grid,
            &sel,
            LeafOrder::RowMajor,
        )
        .unwrap();
        assert!(ok);
    }

    #[test]
    fn test_verify_subset_detects_modified_chunk() {
        let (tree, grid, chunks) = make_tree_and_grid(8);
        let sel = Selection::slice(&[0..4]);
        let proof = extract_subset(&tree, &grid, &sel, LeafOrder::RowMajor).unwrap();

        let mut delivered: Vec<Vec<u8>> = proof
            .chunk_indices
            .iter()
            .map(|&idx| chunks[idx].clone())
            .collect();
        // Tamper with one delivered chunk's bytes.
        delivered[1][0] ^= 0xFF;

        let chunk_data: Vec<ChunkData<'_>> = proof
            .chunk_indices
            .iter()
            .zip(delivered.iter())
            .map(|(&idx, data)| ChunkData { index: idx, data })
            .collect();

        let err = verify_subset(
            tree.root(),
            HashAlg::Blake3,
            &chunk_data,
            &proof,
            &grid,
            &sel,
            LeafOrder::RowMajor,
        )
        .unwrap_err();
        assert!(matches!(err, MerkleError::HashMismatch { .. }));
    }

    #[test]
    fn test_verify_subset_detects_omitted_chunk() {
        let (tree, grid, chunks) = make_tree_and_grid(8);
        let sel = Selection::slice(&[0..4]);
        let proof = extract_subset(&tree, &grid, &sel, LeafOrder::RowMajor).unwrap();
        assert_eq!(proof.chunk_indices.len(), 4);

        // Deliver only 3 of the 4 requested chunks, silently dropping index 2.
        let delivered: Vec<ChunkData<'_>> = proof
            .chunk_indices
            .iter()
            .filter(|&&idx| idx != 2)
            .map(|&idx| ChunkData {
                index: idx,
                data: &chunks[idx],
            })
            .collect();

        let err = verify_subset(
            tree.root(),
            HashAlg::Blake3,
            &delivered,
            &proof,
            &grid,
            &sel,
            LeafOrder::RowMajor,
        )
        .unwrap_err();
        assert!(matches!(err, MerkleError::CompanionTampered));
    }

    #[test]
    fn test_extract_subset_morton_order_round_trips() {
        // 4x4 grid of single-element chunks, built with Morton leaf ordering.
        let dims = vec![4u64, 4u64];
        let chunk_shape = vec![1u64, 1u64];
        let grid = ChunkGridParams::new(dims, chunk_shape, HashAlg::Blake3);
        let n_per_dim = grid.n_chunks_per_dim();

        let n_chunks = grid.total_chunk_count() as usize;
        let mut chunk_at_leaf = vec![Vec::new(); n_chunks];
        for y in 0..n_per_dim[0] {
            for x in 0..n_per_dim[1] {
                let leaf = morton_index(&[y, x]) as usize;
                chunk_at_leaf[leaf] = format!("chunk-{y}-{x}").into_bytes();
            }
        }
        let refs: Vec<&[u8]> = chunk_at_leaf.iter().map(Vec::as_slice).collect();
        let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);

        let sel = Selection::slice(&[1..3, 1..3]);
        let proof = extract_subset(&tree, &grid, &sel, LeafOrder::Morton).unwrap();

        let delivered: Vec<ChunkData<'_>> = proof
            .chunk_indices
            .iter()
            .map(|&idx| ChunkData {
                index: idx,
                data: &chunk_at_leaf[idx],
            })
            .collect();

        let ok = verify_subset(
            tree.root(),
            HashAlg::Blake3,
            &delivered,
            &proof,
            &grid,
            &sel,
            LeafOrder::Morton,
        )
        .unwrap();
        assert!(ok);
    }

    #[test]
    fn test_extract_subset_rank_too_high_errors_not_panics() {
        let (tree, grid, _chunks) = make_tree_and_grid(8); // grid is 1D
        let sel = Selection::slice(&[0..2, 0..2]); // 2D selection against a 1D grid
        let err = extract_subset(&tree, &grid, &sel, LeafOrder::RowMajor).unwrap_err();
        assert!(matches!(err, MerkleError::HyperslabOutOfBounds { idx: 2 }));
    }

    #[test]
    fn test_extract_subset_rank_too_low_errors_not_panics() {
        let dims = vec![4u64, 4u64];
        let chunk_shape = vec![1u64, 1u64];
        let grid = ChunkGridParams::new(dims, chunk_shape, HashAlg::Blake3); // grid is 2D
        let chunks: Vec<Vec<u8>> = (0..16).map(|i| format!("chunk-{i}").into_bytes()).collect();
        let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
        let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);

        let sel = Selection::slice(&[0..2]); // 1D selection against a 2D grid
        let err = extract_subset(&tree, &grid, &sel, LeafOrder::RowMajor).unwrap_err();
        assert!(matches!(err, MerkleError::HyperslabOutOfBounds { idx: 1 }));
    }

    #[test]
    fn test_verify_subset_leaf_hashes_length_mismatch_errors_not_panics() {
        let (tree, grid, chunks) = make_tree_and_grid(8);
        let sel = Selection::slice(&[2..5]);
        let mut proof = extract_subset(&tree, &grid, &sel, LeafOrder::RowMajor).unwrap();
        proof.leaf_hashes.pop(); // desync leaf_hashes from chunk_indices

        let delivered: Vec<ChunkData<'_>> = proof
            .chunk_indices
            .iter()
            .map(|&idx| ChunkData {
                index: idx,
                data: &chunks[idx],
            })
            .collect();

        let err = verify_subset(
            tree.root(),
            HashAlg::Blake3,
            &delivered,
            &proof,
            &grid,
            &sel,
            LeafOrder::RowMajor,
        )
        .unwrap_err();
        assert!(matches!(err, MerkleError::CompanionTampered));
    }

    #[test]
    fn test_verify_subset_grid_axis_mismatch_errors_not_panics() {
        let (tree, grid, chunks) = make_tree_and_grid(8);
        let sel = Selection::slice(&[2..5]);
        let proof = extract_subset(&tree, &grid, &sel, LeafOrder::RowMajor).unwrap();

        let delivered: Vec<ChunkData<'_>> = proof
            .chunk_indices
            .iter()
            .map(|&idx| ChunkData {
                index: idx,
                data: &chunks[idx],
            })
            .collect();

        // `expected_grid` (the verifier's own trusted grid, not the proof's)
        // is now what's structurally validated, so the malformed input must
        // be injected there.
        let mut bad_grid = grid.clone();
        bad_grid.chunk_shape.push(1); // now mismatched vs dims.len()

        let err = verify_subset(
            tree.root(),
            HashAlg::Blake3,
            &delivered,
            &proof,
            &bad_grid,
            &sel,
            LeafOrder::RowMajor,
        )
        .unwrap_err();
        assert!(matches!(err, MerkleError::CompanionTampered));
    }

    #[test]
    fn test_verify_subset_zero_chunk_shape_errors_not_panics() {
        let (tree, grid, chunks) = make_tree_and_grid(8);
        let sel = Selection::slice(&[2..5]);
        let proof = extract_subset(&tree, &grid, &sel, LeafOrder::RowMajor).unwrap();

        let delivered: Vec<ChunkData<'_>> = proof
            .chunk_indices
            .iter()
            .map(|&idx| ChunkData {
                index: idx,
                data: &chunks[idx],
            })
            .collect();

        let mut bad_grid = grid.clone();
        bad_grid.chunk_shape[0] = 0; // would divide-by-zero in div_ceil

        let err = verify_subset(
            tree.root(),
            HashAlg::Blake3,
            &delivered,
            &proof,
            &bad_grid,
            &sel,
            LeafOrder::RowMajor,
        )
        .unwrap_err();
        assert!(matches!(err, MerkleError::CompanionTampered));
    }

    #[test]
    fn test_verify_subset_grid_product_overflow_errors_not_panics() {
        let (tree, grid, chunks) = make_tree_and_grid(8);
        let sel = Selection::slice(&[2..5]);
        let proof = extract_subset(&tree, &grid, &sel, LeafOrder::RowMajor).unwrap();

        let delivered: Vec<ChunkData<'_>> = proof
            .chunk_indices
            .iter()
            .map(|&idx| ChunkData {
                index: idx,
                data: &chunks[idx],
            })
            .collect();

        // Two axes whose chunk counts overflow u64 when multiplied together.
        let bad_grid = ChunkGridParams::new(vec![u64::MAX, 2], vec![1, 1], HashAlg::Blake3);
        let bad_sel = Selection::slice(&[0..1, 0..1]); // rank must match bad_grid

        let err = verify_subset(
            tree.root(),
            HashAlg::Blake3,
            &delivered,
            &proof,
            &bad_grid,
            &bad_sel,
            LeafOrder::RowMajor,
        )
        .unwrap_err();
        assert!(matches!(err, MerkleError::TreeTooDeep { .. }));
    }

    #[test]
    fn test_verify_subset_grid_next_power_of_two_overflow_errors_not_panics() {
        let (tree, grid, chunks) = make_tree_and_grid(8);
        let sel = Selection::slice(&[2..5]);
        let proof = extract_subset(&tree, &grid, &sel, LeafOrder::RowMajor).unwrap();

        let delivered: Vec<ChunkData<'_>> = proof
            .chunk_indices
            .iter()
            .map(|&idx| ChunkData {
                index: idx,
                data: &chunks[idx],
            })
            .collect();

        // A single axis whose chunk count doesn't overflow the multiply but
        // whose next_power_of_two() would overflow usize.
        let bad_grid = ChunkGridParams::new(vec![u64::MAX], vec![1], HashAlg::Blake3);

        let err = verify_subset(
            tree.root(),
            HashAlg::Blake3,
            &delivered,
            &proof,
            &bad_grid,
            &sel,
            LeafOrder::RowMajor,
        )
        .unwrap_err();
        assert!(matches!(err, MerkleError::TreeTooDeep { .. }));
    }

    #[test]
    fn test_grid_hash_changes_with_params() {
        let g1 = ChunkGridParams::new(vec![10], vec![2], HashAlg::Blake3);
        let g2 = ChunkGridParams::new(vec![10], vec![5], HashAlg::Blake3);
        assert_ne!(g1.grid_hash, g2.grid_hash);
    }

    #[test]
    fn test_verify_subset_rejects_proof_for_different_selection() {
        // A proof extracted for one region must not verify against a
        // different region, even though the proof is internally
        // self-consistent (correct hashes, correct coverage cert) and the
        // delivered chunks genuinely belong to the real tree. This is the
        // gap closed by binding `verify_subset` to the verifier's own
        // `expected_grid`/`sel`/`order` instead of trusting `proof.grid_params`.
        let (tree, grid, chunks) = make_tree_and_grid(8);
        let extracted_sel = Selection::slice(&[0..3]);
        let proof = extract_subset(&tree, &grid, &extracted_sel, LeafOrder::RowMajor).unwrap();
        assert_eq!(proof.chunk_indices, vec![0, 1, 2]);

        let delivered: Vec<ChunkData<'_>> = proof
            .chunk_indices
            .iter()
            .map(|&idx| ChunkData {
                index: idx,
                data: &chunks[idx],
            })
            .collect();

        // Verifier actually requested a disjoint region.
        let requested_sel = Selection::slice(&[5..8]);
        let err = verify_subset(
            tree.root(),
            HashAlg::Blake3,
            &delivered,
            &proof,
            &grid,
            &requested_sel,
            LeafOrder::RowMajor,
        )
        .unwrap_err();
        assert!(matches!(err, MerkleError::SelectionMismatch));
    }
}
