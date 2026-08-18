//! Verification grid for HDF5's default (contiguous) layout (P2.8a).
//!
//! A contiguous dataset has no chunk grid to supply Merkle leaf granularity,
//! so one must be constructed. This module derives a *verification grid*
//! over the unmodified byte stream: a partition of the dataspace, chosen so
//! that every leaf is a single contiguous byte run, with no change to the
//! stored data or the `H5D_CONTIGUOUS` layout message. The grid-selection
//! rule is borrowed from the DAOS VOL connector (its automatic grid
//! selection, not its storage conversion — nothing here rewrites the file).
//!
//! See `S2-D2-Yr2/Merkle-tree-HDF5.tex` §"Leaf Granularity for Contiguous
//! Datasets" for the design rationale, the run-length cost model, and the
//! worked examples this module's test vectors are drawn from.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec;

use core::ops::Range;

/// Default grid-selection target leaf size (1 MiB): the DAOS VOL connector's
/// own default, and the value this module's test vectors
/// (`gen_verification_grid_vectors`) are generated against.
///
/// `write_merkle_attr` (P2.8b) uses this constant to independently re-derive
/// a contiguous dataset's leaf grid when binding `grid_hash`, so whoever
/// builds the Merkle tree over that dataset's raw bytes (e.g. via
/// [`crate::subset_proof::contiguous_tree_and_grid`]) must tile with this
/// same target -- otherwise the bound `grid_hash` describes a different
/// tiling than the tree it's supposed to authenticate.
pub const DEFAULT_TARGET_BYTES: u64 = 1024 * 1024;

/// Which grid semantics a [`crate::subset_proof::ChunkGridParams`] carries:
/// a real HDF5 chunk grid, or a verification grid constructed over a
/// contiguous byte stream. Bound into `grid_hash` so a proof produced under
/// one layout can never be replayed as the other with the same extents.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum LayoutClass {
    /// A real HDF5 chunk grid (P1.5 behavior).
    Chunked = 0,
    /// A verification grid over an unmodified contiguous byte stream.
    Contiguous = 1,
}

/// Derive the leaf-granularity grid for a contiguous dataset.
///
/// Ports the DAOS VOL connector's grid-selection rule: walk `dims` from
/// fastest- to slowest-varying (i.e. from the last axis to the first),
/// accumulating the byte size of a leaf that spans the full extent in every
/// dimension visited so far. While that running size does not exceed
/// `target_bytes / sqrt(2)`, the full extent is taken and the walk moves to
/// the next (slower) dimension. At the first dimension `k` where taking the
/// full extent *would* exceed that threshold, dimension `k` is split into
/// `ceil(size / target_bytes)` parts (capped at the extent, rounded up) and
/// every dimension slower than `k` is set to extent 1. The result always has
/// the form `T = (1, ..., 1, T_k, n_{k+1}, ..., n_{r-1})`: every dimension
/// faster than `k` is full (so those rows merge into one run) and every
/// dimension slower than `k` is a single element (so no gather occurs) —
/// every leaf this grid describes is therefore a single contiguous byte run.
///
/// Returns `None` when the dataset's total size does not exceed
/// `target_bytes * sqrt(2)` — too small to be worth splitting, matching the
/// DAOS connector's own threshold. Callers should treat `None` as "stay a
/// single leaf" (ClawHDF5's existing flat-hash tier), not as an error.
///
/// `target_bytes` is always an explicit parameter, never read from an
/// environment variable: a verifier that re-derived the grid from its own
/// configuration would accept whatever grid its own environment happened to
/// produce, defeating the point of binding the derived grid into
/// `grid_hash`. Only the *derivation* is automatic; the derived grid itself
/// must be stored and authenticated.
#[must_use]
pub fn verification_grid(dims: &[u64], elem_size: u32, target_bytes: u64) -> Option<Vec<u64>> {
    let r = dims.len();
    if r == 0 || target_bytes == 0 || elem_size == 0 {
        return None;
    }

    let target = u128::from(target_bytes);
    let elem = u128::from(elem_size);

    let total: u128 = dims.iter().fold(elem, |acc, &n| acc * u128::from(n));
    // total <= target * sqrt(2)  <=>  total^2 <= 2 * target^2
    if total * total <= 2 * target * target {
        return None;
    }

    let mut grid = dims.to_vec();
    let mut running = elem;
    // Walk fastest (last index) to slowest (first index).
    for i in (0..r).rev() {
        let extent = u128::from(dims[i]);
        let size_if_full = running * extent;
        // size_if_full > target / sqrt(2)  <=>  2 * size_if_full^2 > target^2
        if 2 * size_if_full * size_if_full > target * target {
            let num_parts = size_if_full.div_ceil(target).max(1);
            let tile_extent = extent.div_ceil(num_parts);
            grid[i] = tile_extent as u64;
            for slower in grid.iter_mut().take(i) {
                *slower = 1;
            }
            return Some(grid);
        }
        running = size_if_full;
    }

    // Walked every dimension at full extent without exceeding the lower
    // threshold; the earlier total-size check should already have caught
    // this, but stay consistent (single leaf) rather than return a
    // degenerate grid equal to `dims`.
    None
}

/// Byte range of leaf `idx`, relative to the dataset's raw-data start.
///
/// `grid` must be a shape returned by [`verification_grid`], whose leaves
/// are each a single contiguous byte run by construction — that is what
/// makes this arithmetic against the raw-data address rather than a lookup
/// in a stored index. It is *not* valid for a general HDF5 chunk grid,
/// whose tiles are a set of disjoint runs gathered in C-order.
///
/// Leaf indices are numbered row-major over the tile grid (axis
/// `dims.len() - 1` varies fastest), matching `LeafOrder::RowMajor` in
/// `subset_proof.rs`.
///
/// # Panics
///
/// In debug builds, panics if `grid.len() != dims.len()` or `idx` is out of
/// bounds for the tile grid. Both are caller bugs, not malicious input —
/// `idx` always comes from the verifier's own iteration over a grid it
/// derived or authenticated itself.
#[must_use]
pub fn leaf_byte_range(grid: &[u64], dims: &[u64], elem_size: u32, idx: u64) -> Range<u64> {
    debug_assert_eq!(grid.len(), dims.len());
    let r = dims.len();

    let n_tiles: Vec<u64> = (0..r).map(|i| dims[i].div_ceil(grid[i].max(1))).collect();

    let mut coords = vec![0u64; r];
    let mut rem = idx;
    for i in (0..r).rev() {
        debug_assert!(n_tiles[i] > 0);
        coords[i] = rem % n_tiles[i];
        rem /= n_tiles[i];
    }
    debug_assert_eq!(rem, 0, "leaf index out of bounds for this grid");

    let elem = u128::from(elem_size);
    let mut start_elem: u128 = 0;
    let mut run_elems: u128 = 1;
    let mut multiplier: u128 = 1; // prod_{m>i} dims[m], accumulated fastest->slowest
    for i in (0..r).rev() {
        let axis_start = coords[i] * grid[i];
        start_elem += u128::from(axis_start) * multiplier;
        let extent = grid[i].min(dims[i] - axis_start);
        run_elems *= u128::from(extent);
        multiplier *= u128::from(dims[i]);
    }

    let start = (start_elem * elem) as u64;
    let len = (run_elems * elem) as u64;
    start..start + len
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn daos_fixture_1000_cubed() {
        let grid = verification_grid(&[1000, 1000, 1000], 4, 1024 * 1024);
        assert_eq!(grid, Some(vec![1, 250, 1000]));
    }

    #[test]
    fn daos_fixture_time_series() {
        let grid = verification_grid(&[8760, 720, 1440], 4, 1024 * 1024);
        assert_eq!(grid, Some(vec![1, 180, 1440]));
    }

    #[test]
    fn daos_fixture_1d_2_30() {
        let grid = verification_grid(&[1u64 << 30], 8, 1024 * 1024);
        assert_eq!(grid, Some(vec![131_072]));
    }

    #[test]
    fn daos_fixture_1d_million() {
        let grid = verification_grid(&[1_000_000], 4, 1024 * 1024);
        assert_eq!(grid, Some(vec![250_000]));
    }

    #[test]
    fn below_threshold_2d_single_leaf() {
        assert_eq!(verification_grid(&[100, 100], 8, 1024 * 1024), None);
    }

    #[test]
    fn degenerate_small_dims_single_leaf() {
        assert_eq!(verification_grid(&[2, 3, 4], 4, 1024 * 1024), None);
    }

    /// Smallest `k` such that `grid[j] == dims[j]` for every `j > k` — the
    /// pivot dimension of the paper's run-structure model.
    fn pivot(grid: &[u64], dims: &[u64]) -> usize {
        let mut k = grid.len() - 1;
        while k > 0 && grid[k] == dims[k] {
            k -= 1;
        }
        k
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(10_000))]

        #[test]
        fn grid_invariants(
            dims in proptest::collection::vec(1u64..10_000, 1..=4),
            elem_size in proptest::sample::select(vec![1u32, 2, 4, 8, 16]),
            target in proptest::sample::select(vec![1u64 << 16, 1 << 20, 1 << 24]),
        ) {
            let Some(grid) = verification_grid(&dims, elem_size, target) else {
                return Ok(());
            };

            // Tiling: the grid covers the dataspace exactly, with a
            // possibly-short final leaf per axis.
            for i in 0..dims.len() {
                proptest::prop_assert!(
                    grid[i] >= 1 && grid[i] <= dims[i],
                    "tiling violated on axis {i}: grid={grid:?} dims={dims:?}"
                );
            }

            // Size band: the sqrt(2) thresholds plus the rounding rule cannot
            // escape [target/2, 2*target]. (The observed band is tighter; only
            // the loose band is asserted, per the design doc.)
            let leaf_bytes: u128 = grid
                .iter()
                .fold(u128::from(elem_size), |acc, &t| acc * u128::from(t));
            proptest::prop_assert!(
                leaf_bytes >= u128::from(target) / 2 && leaf_bytes <= 2 * u128::from(target),
                "leaf size {leaf_bytes} outside [target/2, 2*target] for \
                 target={target} grid={grid:?} dims={dims:?} elem_size={elem_size}"
            );

            // Byte-contiguity: every axis faster than the pivot is full extent
            // and every axis slower than it is 1, so each leaf is a single
            // contiguous byte run. THIS is the invariant the entire read-cost
            // argument rests on — if a future re-derivation breaks it, leaves
            // silently become scattered runs and the cost model no longer
            // describes the implementation.
            let k = pivot(&grid, &dims);
            for i in 0..k {
                proptest::prop_assert_eq!(
                    grid[i], 1,
                    "byte-contiguity violated: axis {} slower than pivot {} is not 1 \
                     (grid={:?} dims={:?})",
                    i, k, grid, dims
                );
            }
        }

        #[test]
        fn leaf_ranges_partition_the_stream(
            dims in proptest::collection::vec(1u64..2_000, 1..=3),
            elem_size in proptest::sample::select(vec![1u32, 4, 8]),
            target in proptest::sample::select(vec![1u64 << 16, 1 << 20]),
        ) {
            let Some(grid) = verification_grid(&dims, elem_size, target) else {
                return Ok(());
            };
            let n_tiles: Vec<u64> = (0..dims.len())
                .map(|i| dims[i].div_ceil(grid[i]))
                .collect();
            let total_leaves: u64 = n_tiles.iter().product();

            // Walk a bounded prefix for the no-gap/no-overlap property, then
            // check the final leaf's end separately — together these give
            // "the leaves partition the stream" without an unbounded
            // per-case cost (and without rejecting large cases, which would
            // silently skew the sampled distribution toward small grids).
            let mut prev_end = 0u64;
            for idx in 0..total_leaves.min(5_000) {
                let range = leaf_byte_range(&grid, &dims, elem_size, idx);
                proptest::prop_assert_eq!(
                    range.start, prev_end,
                    "gap or overlap at leaf {} (grid={:?} dims={:?})", idx, grid, dims
                );
                proptest::prop_assert!(range.end > range.start);
                prev_end = range.end;
            }

            let total_bytes = dims.iter().product::<u64>() * u64::from(elem_size);
            let last = leaf_byte_range(&grid, &dims, elem_size, total_leaves - 1);
            proptest::prop_assert_eq!(
                last.end, total_bytes,
                "leaves do not cover the stream to its end (grid={:?} dims={:?})",
                grid, dims
            );
        }
    }

    #[test]
    fn leaf_byte_range_covers_whole_dataset_no_gap_no_overlap() {
        let dims = [1000u64, 1000, 1000];
        let elem_size = 4u32;
        let grid = verification_grid(&dims, elem_size, 1024 * 1024).unwrap();
        let n_tiles: Vec<u64> = (0..3).map(|i| dims[i].div_ceil(grid[i])).collect();
        let total_leaves: u64 = n_tiles.iter().product();
        assert_eq!(total_leaves, 4000);

        let mut prev_end = 0u64;
        for idx in 0..total_leaves {
            let range = leaf_byte_range(&grid, &dims, elem_size, idx);
            assert_eq!(range.start, prev_end, "gap or overlap at leaf {idx}");
            assert!(range.end > range.start);
            prev_end = range.end;
        }
        let total_bytes = dims.iter().product::<u64>() * u64::from(elem_size);
        assert_eq!(prev_end, total_bytes);
    }
}
