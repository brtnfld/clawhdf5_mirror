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

use crate::merkle::{GRID_PREFIX, HASH_SIZE, HashAlg, MerkleError, MerkleTree, constant_time_eq};
use crate::selection::Selection;
use crate::verification_grid::{self, LayoutClass};

/// Trusted chunk-grid parameters anchored by the Merkle path to the
/// file-level signed root (the "coverage certificate" component (b)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkGridParams {
    /// Whether `chunk_shape` is a real HDF5 chunk grid or a verification
    /// grid constructed over a contiguous byte stream (P2.8a). Bound into
    /// `grid_hash`, which is what stops a contiguous dataset's proof from
    /// being replayed as a chunked one with the same extents.
    pub layout_class: LayoutClass,
    /// Dataset dimensions (element counts per axis).
    pub dims: Vec<u64>,
    /// Chunk dimensions (element counts per axis): real chunks under
    /// [`LayoutClass::Chunked`], the verification grid under
    /// [`LayoutClass::Contiguous`].
    pub chunk_shape: Vec<u64>,
    /// File datatype size in bytes. Bound into `grid_hash` so the same byte
    /// stream cannot be re-presented under a different element width (e.g.
    /// f32 claimed as f64, halving the apparent extent).
    pub elem_size: u32,
    /// `H(0x04 || layout_class || elem_size || dims || chunk_shape)`,
    /// binding the grid parameters.
    pub grid_hash: [u8; HASH_SIZE],
}

impl ChunkGridParams {
    /// Construct grid params, computing `grid_hash` from the layout class,
    /// element size, `dims`, and `chunk_shape`.
    #[must_use]
    pub fn new(
        dims: Vec<u64>,
        chunk_shape: Vec<u64>,
        elem_size: u32,
        layout_class: LayoutClass,
        alg: HashAlg,
    ) -> Self {
        let grid_hash = compute_grid_hash(&dims, &chunk_shape, elem_size, layout_class, alg);
        Self {
            layout_class,
            dims,
            chunk_shape,
            elem_size,
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

pub(crate) fn compute_grid_hash(
    dims: &[u64],
    chunk_shape: &[u64],
    elem_size: u32,
    layout_class: LayoutClass,
    alg: HashAlg,
) -> [u8; HASH_SIZE] {
    let mut buf = Vec::with_capacity(6 + (dims.len() + chunk_shape.len()) * 8);
    buf.push(GRID_PREFIX);
    buf.push(layout_class as u8);
    buf.extend_from_slice(&elem_size.to_le_bytes());
    for &d in dims {
        buf.extend_from_slice(&d.to_le_bytes());
    }
    for &c in chunk_shape {
        buf.extend_from_slice(&c.to_le_bytes());
    }
    alg.hash_grid(&buf)
}

/// Build a Merkle tree and matching [`ChunkGridParams`] for a *contiguous*
/// (unchunked) dataset's raw byte stream (P2.8b).
///
/// Derives the leaf-granularity grid via
/// [`verification_grid::verification_grid`], falling back to a single
/// whole-buffer leaf when the dataset is too small to be worth tiling (that
/// function's documented `None` case), then slices `data` into the
/// resulting leaf byte ranges and hashes each range as one Merkle leaf. The
/// returned [`ChunkGridParams`] carries [`LayoutClass::Contiguous`] and the
/// derived tiling, so a proof built from this tree can never be replayed
/// against a real chunk grid with the same extents (or vice versa).
///
/// `target_bytes` must match the value `write_merkle_attr` uses to
/// independently re-derive this same grid when binding `grid_hash` --
/// [`verification_grid::DEFAULT_TARGET_BYTES`] is that shared default.
///
/// # Panics
///
/// In debug builds, panics if `data.len()` does not equal the byte size
/// implied by `dims` and `elem_size` (`elem_size * product(dims)`) -- a
/// caller bug (inconsistent dataset dimensions), not malicious input, since
/// this runs on the trusted write path.
#[must_use]
pub fn contiguous_tree_and_grid(
    data: &[u8],
    dims: &[u64],
    elem_size: u32,
    target_bytes: u64,
    alg: HashAlg,
) -> (MerkleTree, ChunkGridParams) {
    debug_assert_eq!(
        dims.iter()
            .fold(u128::from(elem_size), |acc, &d| acc * u128::from(d)),
        data.len() as u128,
        "contiguous data length does not match dims * elem_size"
    );

    let chunk_shape = verification_grid::verification_grid(dims, elem_size, target_bytes)
        .unwrap_or_else(|| dims.to_vec());
    let grid = ChunkGridParams::new(
        dims.to_vec(),
        chunk_shape.clone(),
        elem_size,
        LayoutClass::Contiguous,
        alg,
    );

    let n_per_dim = grid.n_chunks_per_dim();
    let total_leaves: u64 = n_per_dim.iter().product();

    let leaves: Vec<&[u8]> = (0..total_leaves)
        .map(|idx| {
            let range = verification_grid::leaf_byte_range(&chunk_shape, dims, elem_size, idx);
            &data[range.start as usize..range.end as usize]
        })
        .collect();

    #[cfg(feature = "parallel")]
    let tree = MerkleTree::from_chunks_parallel(&leaves, alg);
    #[cfg(not(feature = "parallel"))]
    let tree = MerkleTree::from_chunks(&leaves, alg);
    (tree, grid)
}

/// Working-buffer budget for [`contiguous_tree_streaming`]. Chosen large
/// enough that a batch holds many leaves (so parallel hashing has work to
/// spread across cores) but small enough to stay far below the size of the
/// datasets this path exists for. A single leaf larger than this is still
/// handled -- the batch always takes at least one leaf.
#[cfg(feature = "std")]
const STREAM_BUDGET_BYTES: usize = 64 * 1024 * 1024;

/// Build the Merkle tree and grid for a contiguous dataset by streaming it
/// from `reader`, without ever holding the whole dataset in memory (P2.9).
///
/// [`contiguous_tree_and_grid`] needs the entire byte stream as one `&[u8]`,
/// which is impractical for the multi-gigabyte datasets contiguous support
/// exists to serve. This reads the stream once, front to back, in bounded
/// batches, and hashes each batch's leaves *in parallel* when the `parallel`
/// feature is enabled.
///
/// # Why this can be both streaming and parallel
///
/// Every leaf of a verification grid is a single contiguous byte run, and
/// row-major leaf indices run in increasing byte order with no gap or overlap
/// (the `leaf_ranges_partition_the_stream` property test pins exactly this).
/// So the leaves arrive in index order during one forward pass: a batch can be
/// read with a single sequential `read_exact` and then split into per-leaf
/// slices that are hashed independently. No interleaved per-leaf hasher state
/// is required, which is what makes the leaf hashing embarrassingly parallel
/// rather than a 1-pass scatter across thousands of live hashers.
///
/// The tree produced is identical to [`contiguous_tree_and_grid`]'s over the
/// same bytes -- same root, same leaf hashes. `streaming_matches_in_memory`
/// asserts that equivalence.
///
/// `reader` must be `Send` because it is driven from a scoped producer thread
/// so that reads overlap hashing; `File` and `Cursor` both satisfy this.
///
/// # Errors
///
/// Propagates any I/O error from `reader`, including a short stream: `reader`
/// must yield exactly `elem_size * product(dims)` bytes.
#[cfg(feature = "std")]
pub fn contiguous_tree_streaming<R: std::io::Read + Send>(
    reader: &mut R,
    dims: &[u64],
    elem_size: u32,
    target_bytes: u64,
    alg: HashAlg,
) -> std::io::Result<(MerkleTree, ChunkGridParams)> {
    let chunk_shape = verification_grid::verification_grid(dims, elem_size, target_bytes)
        .unwrap_or_else(|| dims.to_vec());
    let grid = ChunkGridParams::new(
        dims.to_vec(),
        chunk_shape.clone(),
        elem_size,
        LayoutClass::Contiguous,
        alg,
    );

    let total_leaves: u64 = grid.n_chunks_per_dim().iter().product();
    let ranges: Vec<core::ops::Range<u64>> = (0..total_leaves)
        .map(|i| verification_grid::leaf_byte_range(&chunk_shape, dims, elem_size, i))
        .collect();

    // Batch boundaries: greedily fill the budget, always taking at least one
    // leaf so an oversized leaf cannot stall the loop.
    let mut batches: Vec<(usize, usize)> = Vec::new();
    let mut i = 0usize;
    while i < ranges.len() {
        let base = ranges[i].start;
        let mut j = i;
        while j < ranges.len() && (ranges[j].end - base) as usize <= STREAM_BUDGET_BYTES {
            j += 1;
        }
        if j == i {
            j = i + 1;
        }
        batches.push((i, j));
        i = j;
    }

    let mut leaf_hashes: Vec<[u8; HASH_SIZE]> = Vec::with_capacity(ranges.len());

    // Reads are overlapped with hashing: a scoped producer thread fills one
    // batch while this thread hashes the previous one. Without that overlap the
    // two phases serialise, and since a large parallel hash is faster than the
    // read that feeds it, the read latency lands entirely on the critical path
    // (measured: ~62% of device bandwidth serialised, versus read-bound with
    // the overlap). `sync_channel(1)` bounds the producer to a single batch
    // ahead, and spent buffers are returned for reuse so steady state holds
    // exactly two allocations rather than one per batch.
    std::thread::scope(|scope| -> std::io::Result<()> {
        type Batch = (Vec<u8>, usize, usize);
        let (full_tx, full_rx) = std::sync::mpsc::sync_channel::<std::io::Result<Batch>>(1);
        let (empty_tx, empty_rx) = std::sync::mpsc::channel::<Vec<u8>>();
        // Two buffers ping-pong: one being filled, one being hashed.
        let _ = empty_tx.send(Vec::new());
        let _ = empty_tx.send(Vec::new());

        let ranges_ref = &ranges;
        let batches_ref = &batches;
        scope.spawn(move || {
            for &(i, j) in batches_ref {
                let base = ranges_ref[i].start;
                let span = (ranges_ref[j - 1].end - base) as usize;
                // A disconnected recycle channel just means the consumer is
                // gone; a fresh buffer keeps this correct either way.
                let mut buf = empty_rx.recv().unwrap_or_default();
                buf.resize(span, 0);
                if let Err(e) = reader.read_exact(&mut buf) {
                    let _ = full_tx.send(Err(e));
                    return;
                }
                if full_tx.send(Ok((buf, i, j))).is_err() {
                    return; // consumer bailed out
                }
            }
        });

        for msg in full_rx {
            let (buf, i, j) = msg?;
            let base = ranges[i].start;
            let slices: Vec<&[u8]> = ranges[i..j]
                .iter()
                .map(|r| &buf[(r.start - base) as usize..(r.end - base) as usize])
                .collect();

            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                let mut batch: Vec<[u8; HASH_SIZE]> = Vec::new();
                slices
                    .par_iter()
                    .map(|s| alg.hash_leaf(s))
                    .collect_into_vec(&mut batch);
                leaf_hashes.extend_from_slice(&batch);
            }
            #[cfg(not(feature = "parallel"))]
            {
                leaf_hashes.extend(slices.iter().map(|s| alg.hash_leaf(s)));
            }

            drop(slices);
            let _ = empty_tx.send(buf); // recycle
        }
        Ok(())
    })?;

    Ok((MerkleTree::from_leaf_hashes(&leaf_hashes, alg), grid))
}

/// A contiguous dataset's raw-data location and extents, read directly from
/// its (already-parsed) object header -- the Rust equivalent of
/// `H5Dget_offset` (P2.8b).
///
/// Deliberately does not touch or copy the raw bytes: callers slice
/// `file_bytes[data_addr as usize..(data_addr + nbytes) as usize]` themselves
/// (the same way [`contiguous_tree_and_grid`]'s caller slices its `data`
/// argument out of a file), keeping this function I/O-free.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContiguousLayout {
    /// File offset of the raw data.
    pub data_addr: u64,
    /// Size of the raw data in bytes (`product(dims) * elem_size`).
    pub nbytes: u64,
    /// Dataset dimensions (element counts per axis), from the Dataspace message.
    pub dims: Vec<u64>,
    /// File datatype size in bytes, from the Datatype message.
    pub elem_size: u32,
}

/// Why [`contiguous_layout`] rejected a dataset (P2.8b). Every variant is a
/// hard-error rejection, not a warning: silently hashing, e.g., a variable-
/// length dataset's global-heap identifiers as if they were the data would
/// produce a proof that certifies pointers rather than values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnsupportedLayoutReason {
    /// Layout is compact, chunked, or virtual -- not contiguous at all.
    NotContiguous,
    /// Contiguous layout with no allocated address (never written).
    Unallocated,
    /// External storage (`H5Pset_external`): the payload lives in another
    /// file this parser does not resolve. Detected via the presence of the
    /// raw External Data Files message (HDF5 message type `0x0007`), which
    /// this parser recognizes only well enough to reject, not decode.
    ExternalStorage,
    /// Variable-length or reference datatype: this byte range holds global-
    /// heap identifiers or object/region pointers, not the data itself.
    IndirectDatatype,
    /// A required message (Dataspace, Datatype, or DataLayout) is missing or
    /// fails to parse.
    MalformedHeader,
}

/// HDF5 message type `0x0007`, External Data Files -- not decoded by
/// [`crate::message_type::MessageType`] (it parses as `Unknown(0x0007)`),
/// but its mere presence is enough for [`contiguous_layout`] to reject the
/// dataset rather than treat an unrelated local placeholder range as real data.
const EXTERNAL_DATA_FILES_MSG_TYPE: u16 = 0x0007;

/// Read a contiguous dataset's raw-data address and extents straight out of
/// its object header messages, without modifying anything (P2.8b).
///
/// Rejects -- as [`MerkleError::UnsupportedLayout`], never silently -- every
/// case the contiguous-verification design flags as out of scope: external
/// storage, unallocated storage, and variable-length/reference datatypes.
///
/// # Errors
///
/// Returns [`MerkleError::UnsupportedLayout`] if the layout is not a real,
/// allocated contiguous layout; if external storage is present; if the
/// datatype is variable-length or a reference; or if the Dataspace,
/// Datatype, or DataLayout messages are missing or fail to parse.
pub fn contiguous_layout(
    header: &crate::object_header::ObjectHeader,
    offset_size: u8,
    length_size: u8,
) -> Result<ContiguousLayout, MerkleError> {
    use crate::data_layout::DataLayout;
    use crate::dataspace::Dataspace;
    use crate::datatype::Datatype;
    use crate::message_type::MessageType;

    let unsupported =
        |reason| MerkleError::UnsupportedLayout { reason };

    if header
        .messages
        .iter()
        .any(|m| m.msg_type == MessageType::Unknown(EXTERNAL_DATA_FILES_MSG_TYPE))
    {
        return Err(unsupported(UnsupportedLayoutReason::ExternalStorage));
    }

    let find = |t: MessageType| header.messages.iter().find(|m| m.msg_type == t);

    let ds_msg =
        find(MessageType::Dataspace).ok_or_else(|| unsupported(UnsupportedLayoutReason::MalformedHeader))?;
    let dataspace = Dataspace::parse(&ds_msg.data, length_size)
        .map_err(|_| unsupported(UnsupportedLayoutReason::MalformedHeader))?;

    let dt_msg =
        find(MessageType::Datatype).ok_or_else(|| unsupported(UnsupportedLayoutReason::MalformedHeader))?;
    let (datatype, _) = Datatype::parse(&dt_msg.data)
        .map_err(|_| unsupported(UnsupportedLayoutReason::MalformedHeader))?;
    if matches!(
        datatype,
        Datatype::VariableLength { .. } | Datatype::Reference { .. }
    ) {
        return Err(unsupported(UnsupportedLayoutReason::IndirectDatatype));
    }

    let dl_msg =
        find(MessageType::DataLayout).ok_or_else(|| unsupported(UnsupportedLayoutReason::MalformedHeader))?;
    let layout = DataLayout::parse(&dl_msg.data, offset_size, length_size)
        .map_err(|_| unsupported(UnsupportedLayoutReason::MalformedHeader))?;

    let (data_addr, nbytes) = match layout {
        DataLayout::Contiguous {
            address: Some(addr),
            size,
        } => (addr, size),
        DataLayout::Contiguous { address: None, .. } => {
            return Err(unsupported(UnsupportedLayoutReason::Unallocated));
        }
        _ => return Err(unsupported(UnsupportedLayoutReason::NotContiguous)),
    };

    Ok(ContiguousLayout {
        data_addr,
        nbytes,
        dims: dataspace.dimensions,
        elem_size: datatype.type_size(),
    })
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
/// # Trust model
///
/// `expected_grid.dims`/`expected_grid.chunk_shape` do **not** need to come
/// from the verifier's own out-of-band trusted knowledge — they can come
/// from anywhere convenient (e.g. a live read of the HDF5 object header),
/// because this function authenticates them itself: it recomputes
/// `H(dims || chunk_shape)` (the same way
/// [`ChunkGridParams::new`]/`compute_grid_hash` does) and rejects with
/// [`MerkleError::GridHashMismatch`] unless it matches `trusted_grid_hash`.
/// `trusted_grid_hash` must be the cryptographically-anchored 32-byte hash
/// obtained from an already-verified `MerkleAttr::grid_hash()` or
/// `MerkleAttrRef::grid_hash()` — i.e. bound into the file's signed Merkle
/// attribute. This is what closes the gap where a dataset's declared
/// shape/chunk grid could be tampered with while chunk data and version
/// counters remain untouched: the caller only needs to trust one small
/// hash, not `expected_grid`'s contents.
///
/// `sel`/`order` still must come from the verifier's own trusted knowledge
/// of the request — never from `proof` itself, or a prover holding the real
/// tree could satisfy every other check here while substituting a proof for
/// a different region than was requested. `proof.grid_params` is treated as
/// untrusted and is not used to derive the expected chunk set; it only
/// feeds the (redundant, defense-in-depth) coverage-certificate check below.
///
/// `chunks` must be in the same order as `proof.chunk_indices`, with
/// `chunks[i].index == proof.chunk_indices[i]` — this is what lets the
/// coverage certificate detect a silently omitted chunk: dropping an entry
/// changes the delivered index set's hash, which no longer matches
/// `proof.coverage_cert`.
///
/// # Errors
///
/// - [`MerkleError::GridHashMismatch`] if `expected_grid`'s `dims`/
///   `chunk_shape` do not hash to `trusted_grid_hash` — `expected_grid`
///   itself has been tampered with (the T1/T3 grid-tampering gap this
///   parameter closes).
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
#[allow(clippy::too_many_arguments)] // each argument is a distinct, independently-typed trust boundary (see doc comment above)
pub fn verify_subset(
    root: &[u8; HASH_SIZE],
    alg: HashAlg,
    chunks: &[ChunkData<'_>],
    proof: &SubsetProof,
    expected_grid: &ChunkGridParams,
    trusted_grid_hash: &[u8; HASH_SIZE],
    sel: &Selection,
    order: LeafOrder,
) -> Result<bool, MerkleError> {
    // Authenticate expected_grid's parameters against the caller's
    // cryptographically-anchored grid hash *before* trusting anything else
    // about expected_grid. Recomputing from expected_grid's own fields
    // (rather than trusting its `grid_hash` field, which is a plain pub
    // field an attacker could set to match tampered parameters) is what
    // prevents a coherent-looking but wrong `expected_grid` from slipping
    // through when its contents were sourced from an unauthenticated
    // location (e.g. a live object-header read). P2.8a extends the covered
    // set from dims/chunk_shape to elem_size and layout_class, so a
    // reinterpretation attack (f32 claimed as f64) or a cross-layout replay
    // (a contiguous proof presented as chunked) is caught here too.
    let recomputed_grid_hash = compute_grid_hash(
        &expected_grid.dims,
        &expected_grid.chunk_shape,
        expected_grid.elem_size,
        expected_grid.layout_class,
        alg,
    );
    if !constant_time_eq(&recomputed_grid_hash, trusted_grid_hash) {
        return Err(MerkleError::GridHashMismatch);
    }

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
        let grid = ChunkGridParams::new(vec![n_chunks as u64], vec![1], 4, LayoutClass::Chunked, HashAlg::Blake3);
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
        let grid = ChunkGridParams::new(vec![10, 10], vec![2, 0], 4, LayoutClass::Chunked, HashAlg::Blake3);
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
            &grid.grid_hash,
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
            &grid.grid_hash,
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
            &grid.grid_hash,
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
        let grid = ChunkGridParams::new(dims, chunk_shape, 4, LayoutClass::Chunked, HashAlg::Blake3);
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
            &grid.grid_hash,
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
        let grid = ChunkGridParams::new(dims, chunk_shape, 4, LayoutClass::Chunked, HashAlg::Blake3); // grid is 2D
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
            &grid.grid_hash,
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
        // be injected there. Its stale `.grid_hash` field (still the
        // original, pre-mutation hash) is irrelevant here — verify_subset
        // never reads `expected_grid.grid_hash`, it recomputes from
        // `dims`/`chunk_shape`. Pass a `trusted_grid_hash` that matches the
        // mutated grid so the (pre-existing) structural check below is what
        // gets exercised, not the new grid-hash-authentication check.
        let mut bad_grid = grid.clone();
        bad_grid.chunk_shape.push(1); // now mismatched vs dims.len()
        let bad_trusted_hash =
            compute_grid_hash(
                &bad_grid.dims,
                &bad_grid.chunk_shape,
                bad_grid.elem_size,
                bad_grid.layout_class,
                HashAlg::Blake3,
            );

        let err = verify_subset(
            tree.root(),
            HashAlg::Blake3,
            &delivered,
            &proof,
            &bad_grid,
            &bad_trusted_hash,
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
        let bad_trusted_hash =
            compute_grid_hash(
                &bad_grid.dims,
                &bad_grid.chunk_shape,
                bad_grid.elem_size,
                bad_grid.layout_class,
                HashAlg::Blake3,
            );

        let err = verify_subset(
            tree.root(),
            HashAlg::Blake3,
            &delivered,
            &proof,
            &bad_grid,
            &bad_trusted_hash,
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
        let bad_grid = ChunkGridParams::new(vec![u64::MAX, 2], vec![1, 1], 4, LayoutClass::Chunked, HashAlg::Blake3);
        let bad_sel = Selection::slice(&[0..1, 0..1]); // rank must match bad_grid

        let err = verify_subset(
            tree.root(),
            HashAlg::Blake3,
            &delivered,
            &proof,
            &bad_grid,
            &bad_grid.grid_hash,
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
        let bad_grid = ChunkGridParams::new(vec![u64::MAX], vec![1], 4, LayoutClass::Chunked, HashAlg::Blake3);

        let err = verify_subset(
            tree.root(),
            HashAlg::Blake3,
            &delivered,
            &proof,
            &bad_grid,
            &bad_grid.grid_hash,
            &sel,
            LeafOrder::RowMajor,
        )
        .unwrap_err();
        assert!(matches!(err, MerkleError::TreeTooDeep { .. }));
    }

    #[test]
    fn test_grid_hash_changes_with_params() {
        let g1 = ChunkGridParams::new(vec![10], vec![2], 4, LayoutClass::Chunked, HashAlg::Blake3);
        let g2 = ChunkGridParams::new(vec![10], vec![5], 4, LayoutClass::Chunked, HashAlg::Blake3);
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
            &grid.grid_hash,
            &requested_sel,
            LeafOrder::RowMajor,
        )
        .unwrap_err();
        assert!(matches!(err, MerkleError::SelectionMismatch));
    }

    #[test]
    fn test_verify_subset_rejects_tampered_grid_params_via_grid_hash_mismatch() {
        // The actual attack this closes: a dataset's declared chunk-grid
        // parameters (dims/chunk_shape) get tampered with while chunk data
        // and version counters remain untouched. Since `ChunkGridParams`'s
        // fields are all `pub`, an attacker can hand `verify_subset` a
        // forged `expected_grid` whose `grid_hash` field is *internally*
        // self-consistent with its own (forged) dims/chunk_shape — so a
        // check that only looked at `expected_grid.grid_hash` in isolation
        // would find nothing wrong. The fix is that `verify_subset` doesn't
        // trust `expected_grid.grid_hash` at all: it recomputes the hash
        // from `expected_grid.dims`/`chunk_shape` and compares it against
        // `trusted_grid_hash`, which the caller must obtain from an
        // independently-anchored, already-verified source (e.g.
        // `MerkleAttr::grid_hash()`/`MerkleAttrRef::grid_hash()`).
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

        // Forged grid: different dims/chunk_shape than the real dataset (8
        // chunks of shape [1]), but its own `grid_hash` field is coherently
        // recomputed for the forged values (i.e. it would pass a naive "is
        // grid_params self-consistent" check).
        let forged_grid = ChunkGridParams::new(vec![999], vec![1], 4, LayoutClass::Chunked, HashAlg::Blake3);
        assert_ne!(forged_grid.grid_hash, grid.grid_hash);

        // The verifier's trusted hash reflects the REAL dataset grid,
        // obtained independently (e.g. from an already-verified MerkleAttr).
        let trusted_grid_hash = grid.grid_hash;

        let err = verify_subset(
            tree.root(),
            HashAlg::Blake3,
            &delivered,
            &proof,
            &forged_grid,
            &trusted_grid_hash,
            &sel,
            LeafOrder::RowMajor,
        )
        .unwrap_err();
        assert!(matches!(err, MerkleError::GridHashMismatch));
    }

    #[test]
    fn test_contiguous_grid_falls_back_to_single_leaf_when_below_threshold() {
        // Below the verification_grid split threshold: chunk_shape == dims,
        // matching that function's documented "stay a single leaf" contract.
        let dims = vec![10u64];
        let elem_size = 4u32;
        let data: Vec<u8> = (0..10u32).flat_map(|v| v.to_le_bytes()).collect();

        let (tree, grid) =
            contiguous_tree_and_grid(&data, &dims, elem_size, 1024 * 1024, HashAlg::Blake3);

        assert_eq!(grid.layout_class, LayoutClass::Contiguous);
        assert_eq!(grid.chunk_shape, dims);
        assert_eq!(tree.leaf_count(), 1);
    }

    #[test]
    fn test_contiguous_layout_extract_and_verify_subset_round_trip() {
        // Full P2.8b round trip: derive a verification grid over a raw
        // contiguous byte stream, extract a subset proof, and verify it by
        // re-slicing the same raw bytes the way a real contiguous-dataset
        // read would -- not by peeking at the tree's internal leaves.
        let dims = vec![100u64];
        let elem_size = 4u32;
        let data: Vec<u8> = (0..100u32).flat_map(|v| v.to_le_bytes()).collect();
        // Small target so this modest buffer actually gets tiled instead of
        // falling back to the single-leaf case (covered separately above).
        let target_bytes = 64;

        let (tree, grid) =
            contiguous_tree_and_grid(&data, &dims, elem_size, target_bytes, HashAlg::Blake3);
        assert_eq!(grid.layout_class, LayoutClass::Contiguous);
        assert!(
            grid.chunk_shape[0] < dims[0],
            "buffer should actually be tiled, not fall back to a single leaf"
        );

        // Selection straddling two tiles' boundary.
        let sel = Selection::slice(&[10..20]);
        let proof = extract_subset(&tree, &grid, &sel, LeafOrder::RowMajor).unwrap();
        assert!(proof.chunk_indices.len() >= 2);

        let delivered: Vec<ChunkData<'_>> = proof
            .chunk_indices
            .iter()
            .map(|&idx| {
                let range =
                    verification_grid::leaf_byte_range(&grid.chunk_shape, &dims, elem_size, idx as u64);
                ChunkData {
                    index: idx,
                    data: &data[range.start as usize..range.end as usize],
                }
            })
            .collect();

        let ok = verify_subset(
            tree.root(),
            HashAlg::Blake3,
            &delivered,
            &proof,
            &grid,
            &grid.grid_hash,
            &sel,
            LeafOrder::RowMajor,
        )
        .unwrap();
        assert!(ok);
    }

    #[test]
    fn test_contiguous_layout_grid_hash_differs_from_chunked_same_extents() {
        // Domain separation (P2.8a): a Contiguous grid and a Chunked grid
        // with identical dims/chunk_shape/elem_size must not hash the same,
        // so a contiguous-dataset proof can never be replayed as a chunked
        // one (or vice versa).
        let dims = vec![10u64];
        let elem_size = 4u32;
        let data: Vec<u8> = (0..10u32).flat_map(|v| v.to_le_bytes()).collect();

        let (_, contiguous_grid) =
            contiguous_tree_and_grid(&data, &dims, elem_size, 1024 * 1024, HashAlg::Blake3);
        let chunked_grid = ChunkGridParams::new(
            dims,
            contiguous_grid.chunk_shape.clone(),
            elem_size,
            LayoutClass::Chunked,
            HashAlg::Blake3,
        );

        assert_ne!(contiguous_grid.grid_hash, chunked_grid.grid_hash);
    }

    /// P2.8b's load-bearing structural claim: the verifier must be
    /// layout-blind. If `verify_subset` ever has to branch on `LayoutClass`,
    /// the "contiguous reduces to chunked" design claim is false and the
    /// format must be corrected before it freezes. `LayoutClass` may appear
    /// only where the grid hash is computed (`compute_grid_hash` and the
    /// struct field it reads), never as control flow in the verify path.
    #[test]
    fn test_verify_subset_never_branches_on_layout_class() {
        let src = include_str!("subset_proof.rs");
        let start = src
            .find("pub fn verify_subset(")
            .expect("verify_subset should exist");
        // The verifier body ends at the start of the test module.
        let end = src[start..]
            .find("\n#[cfg(test)]")
            .map(|off| start + off)
            .expect("test module should follow verify_subset");
        let body = &src[start..end];

        for (i, line) in body.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            assert!(
                !code.contains("LayoutClass"),
                "verify_subset branches on LayoutClass at body line {i}: {line}\n\
                 The layout class must only ever reach the verifier through \
                 compute_grid_hash's preimage, never as control flow."
            );
        }
    }

    /// The sharpest statement of the reuse claim: a contiguous dataset and a
    /// chunked dataset holding the same bytes under the same grid produce
    /// proofs that are byte-identical in every field EXCEPT `layout_class`
    /// and the `grid_hash` it feeds. Same root, same leaf hashes, same
    /// proof nodes, same indices.
    #[test]
    fn test_contiguous_and_chunked_proofs_are_identical_except_grid_hash() {
        let dims = vec![100u64];
        let elem_size = 4u32;
        let data: Vec<u8> = (0..100u32).flat_map(|v| v.to_le_bytes()).collect();
        let target_bytes = 64;

        let (contiguous_tree, contiguous_grid) =
            contiguous_tree_and_grid(&data, &dims, elem_size, target_bytes, HashAlg::Blake3);

        // The "h5repack'ed chunked copy": same bytes, tiled the same way,
        // but presented as a real chunk grid.
        let chunk_shape = contiguous_grid.chunk_shape.clone();
        let n_leaves = dims[0].div_ceil(chunk_shape[0]);
        let leaves: Vec<&[u8]> = (0..n_leaves)
            .map(|idx| {
                let r = verification_grid::leaf_byte_range(&chunk_shape, &dims, elem_size, idx);
                &data[r.start as usize..r.end as usize]
            })
            .collect();
        let chunked_tree = MerkleTree::from_chunks(&leaves, HashAlg::Blake3);
        let chunked_grid = ChunkGridParams::new(
            dims.clone(),
            chunk_shape,
            elem_size,
            LayoutClass::Chunked,
            HashAlg::Blake3,
        );

        assert_eq!(contiguous_tree.root(), chunked_tree.root());

        let sel = Selection::slice(&[10..40]);
        let c_proof = extract_subset(&contiguous_tree, &contiguous_grid, &sel, LeafOrder::RowMajor)
            .unwrap();
        let k_proof =
            extract_subset(&chunked_tree, &chunked_grid, &sel, LeafOrder::RowMajor).unwrap();

        assert_eq!(c_proof.chunk_indices, k_proof.chunk_indices);
        assert_eq!(c_proof.leaf_hashes, k_proof.leaf_hashes);
        assert_eq!(c_proof.proof_nodes, k_proof.proof_nodes);
        assert_eq!(c_proof.grid_params.dims, k_proof.grid_params.dims);
        assert_eq!(c_proof.grid_params.chunk_shape, k_proof.grid_params.chunk_shape);
        assert_eq!(c_proof.grid_params.elem_size, k_proof.grid_params.elem_size);

        // ...differing ONLY in layout_class and the grid_hash it feeds (and
        // hence the coverage cert that binds the grid hash).
        assert_ne!(c_proof.grid_params.layout_class, k_proof.grid_params.layout_class);
        assert_ne!(c_proof.grid_params.grid_hash, k_proof.grid_params.grid_hash);

        // The SubsetProof struct itself is unchanged from P1.5: no
        // byte-range variant, no new field. Both proofs verify through the
        // identical, unmodified P1.5 entry point.
        for (tree, grid, proof) in [
            (&contiguous_tree, &contiguous_grid, &c_proof),
            (&chunked_tree, &chunked_grid, &k_proof),
        ] {
            let delivered: Vec<ChunkData<'_>> = proof
                .chunk_indices
                .iter()
                .map(|&idx| {
                    let r = verification_grid::leaf_byte_range(
                        &grid.chunk_shape,
                        &dims,
                        elem_size,
                        idx as u64,
                    );
                    ChunkData {
                        index: idx,
                        data: &data[r.start as usize..r.end as usize],
                    }
                })
                .collect();
            assert!(
                verify_subset(
                    tree.root(),
                    HashAlg::Blake3,
                    &delivered,
                    proof,
                    grid,
                    &grid.grid_hash,
                    &sel,
                    LeafOrder::RowMajor,
                )
                .unwrap()
            );
        }
    }

    /// P2.8b negative test 1: a verifier that re-derives the grid at a
    /// different target than the prover used must be rejected on the grid
    /// hash -- not succeed, and not fail with a confusing hash mismatch
    /// deeper in the proof path.
    #[test]
    fn test_grid_substitution_at_a_different_target_is_rejected() {
        let dims = vec![100u64];
        let elem_size = 4u32;
        let data: Vec<u8> = (0..100u32).flat_map(|v| v.to_le_bytes()).collect();

        let (tree, prover_grid) =
            contiguous_tree_and_grid(&data, &dims, elem_size, 64, HashAlg::Blake3);
        let (_, verifier_grid) =
            contiguous_tree_and_grid(&data, &dims, elem_size, 256, HashAlg::Blake3);
        assert_ne!(prover_grid.chunk_shape, verifier_grid.chunk_shape);

        let sel = Selection::slice(&[10..40]);
        let proof = extract_subset(&tree, &prover_grid, &sel, LeafOrder::RowMajor).unwrap();
        let delivered: Vec<ChunkData<'_>> = proof
            .chunk_indices
            .iter()
            .map(|&idx| {
                let r = verification_grid::leaf_byte_range(
                    &prover_grid.chunk_shape,
                    &dims,
                    elem_size,
                    idx as u64,
                );
                ChunkData {
                    index: idx,
                    data: &data[r.start as usize..r.end as usize],
                }
            })
            .collect();

        // The trusted grid hash is the prover's (it's what the file binds),
        // but the verifier re-derived a different grid from its own config.
        let err = verify_subset(
            tree.root(),
            HashAlg::Blake3,
            &delivered,
            &proof,
            &verifier_grid,
            &prover_grid.grid_hash,
            &sel,
            LeafOrder::RowMajor,
        )
        .unwrap_err();
        assert!(
            matches!(err, MerkleError::GridHashMismatch),
            "expected GridHashMismatch, got {err:?}"
        );
    }

    /// P2.8b negative test 2: a contiguous proof presented with
    /// `layout_class` flipped to `Chunked` (identical dims and shape) must
    /// fail on the grid hash. This is what stops a cross-layout replay.
    #[test]
    fn test_layout_class_flip_is_rejected_on_grid_hash() {
        let dims = vec![100u64];
        let elem_size = 4u32;
        let data: Vec<u8> = (0..100u32).flat_map(|v| v.to_le_bytes()).collect();

        let (tree, grid) = contiguous_tree_and_grid(&data, &dims, elem_size, 64, HashAlg::Blake3);
        let sel = Selection::slice(&[10..40]);
        let proof = extract_subset(&tree, &grid, &sel, LeafOrder::RowMajor).unwrap();
        let delivered: Vec<ChunkData<'_>> = proof
            .chunk_indices
            .iter()
            .map(|&idx| {
                let r = verification_grid::leaf_byte_range(
                    &grid.chunk_shape,
                    &dims,
                    elem_size,
                    idx as u64,
                );
                ChunkData {
                    index: idx,
                    data: &data[r.start as usize..r.end as usize],
                }
            })
            .collect();

        // Same dims, same chunk_shape, same elem_size -- only the layout
        // class differs.
        let mut flipped = grid.clone();
        flipped.layout_class = LayoutClass::Chunked;
        flipped.grid_hash = compute_grid_hash(
            &flipped.dims,
            &flipped.chunk_shape,
            flipped.elem_size,
            LayoutClass::Chunked,
            HashAlg::Blake3,
        );

        let err = verify_subset(
            tree.root(),
            HashAlg::Blake3,
            &delivered,
            &proof,
            &flipped,
            &grid.grid_hash, // trusted hash still says Contiguous
            &sel,
            LeafOrder::RowMajor,
        )
        .unwrap_err();
        assert!(
            matches!(err, MerkleError::GridHashMismatch),
            "expected GridHashMismatch, got {err:?}"
        );
    }

    /// P2.8b negative test 3: the reinterpretation attack -- the same bytes
    /// re-presented under a different element width (f32 claimed as f64),
    /// which halves the apparent extent -- must fail on the grid hash.
    #[test]
    fn test_elem_size_reinterpretation_is_rejected_on_grid_hash() {
        let dims = vec![100u64];
        let elem_size = 4u32;
        let data: Vec<u8> = (0..100u32).flat_map(|v| v.to_le_bytes()).collect();

        let (tree, grid) = contiguous_tree_and_grid(&data, &dims, elem_size, 64, HashAlg::Blake3);
        let sel = Selection::slice(&[10..40]);
        let proof = extract_subset(&tree, &grid, &sel, LeafOrder::RowMajor).unwrap();
        let delivered: Vec<ChunkData<'_>> = proof
            .chunk_indices
            .iter()
            .map(|&idx| {
                let r = verification_grid::leaf_byte_range(
                    &grid.chunk_shape,
                    &dims,
                    elem_size,
                    idx as u64,
                );
                ChunkData {
                    index: idx,
                    data: &data[r.start as usize..r.end as usize],
                }
            })
            .collect();

        // Same byte stream, claimed as f64 over half as many elements.
        let reinterpreted = ChunkGridParams::new(
            vec![50],
            grid.chunk_shape.clone(),
            8,
            LayoutClass::Contiguous,
            HashAlg::Blake3,
        );

        let err = verify_subset(
            tree.root(),
            HashAlg::Blake3,
            &delivered,
            &proof,
            &reinterpreted,
            &grid.grid_hash,
            &sel,
            LeafOrder::RowMajor,
        )
        .unwrap_err();
        assert!(
            matches!(err, MerkleError::GridHashMismatch),
            "expected GridHashMismatch, got {err:?}"
        );
    }

    /// P2.8b tamper localization: flipping one byte inside leaf `j` must be
    /// reported as a mismatch localized to leaf `j`, and `leaf_byte_range`'s
    /// reported range for that leaf must contain the flipped offset.
    /// Localization granularity is the practical benefit over flat hashing,
    /// so it's measured here rather than assumed.
    #[test]
    fn test_tamper_is_localized_to_the_correct_leaf() {
        let dims = vec![100u64];
        let elem_size = 4u32;
        let mut data: Vec<u8> = (0..100u32).flat_map(|v| v.to_le_bytes()).collect();
        let target_bytes = 64;

        let (tree, grid) =
            contiguous_tree_and_grid(&data, &dims, elem_size, target_bytes, HashAlg::Blake3);

        let tampered_leaf = 2u64;
        let leaf_range =
            verification_grid::leaf_byte_range(&grid.chunk_shape, &dims, elem_size, tampered_leaf);
        let flip_offset = leaf_range.start + 5;
        assert!(
            leaf_range.contains(&flip_offset),
            "the reported byte range for leaf {tampered_leaf} must contain the flipped offset"
        );
        data[flip_offset as usize] ^= 0xFF;

        let sel = Selection::All;
        let proof = extract_subset(&tree, &grid, &sel, LeafOrder::RowMajor).unwrap();
        let delivered: Vec<ChunkData<'_>> = proof
            .chunk_indices
            .iter()
            .map(|&idx| {
                let r = verification_grid::leaf_byte_range(
                    &grid.chunk_shape,
                    &dims,
                    elem_size,
                    idx as u64,
                );
                ChunkData {
                    index: idx,
                    data: &data[r.start as usize..r.end as usize],
                }
            })
            .collect();

        let err = verify_subset(
            tree.root(),
            HashAlg::Blake3,
            &delivered,
            &proof,
            &grid,
            &grid.grid_hash,
            &sel,
            LeafOrder::RowMajor,
        )
        .unwrap_err();

        match err {
            MerkleError::HashMismatch { chunk_idx } => {
                assert_eq!(
                    chunk_idx as u64, tampered_leaf,
                    "tamper should localize to leaf {tampered_leaf}, not {chunk_idx}"
                );
            }
            other => panic!("expected HashMismatch localized to a leaf, got {other:?}"),
        }
    }

    /// P2.8b byte-order hazard: leaf preimages must be computed over raw
    /// *file* bytes in file byte order, never over converted in-memory
    /// values, or a big-endian and a little-endian reader disagree on the
    /// root of the same file.
    ///
    /// Simulated with a byte-swapping reader: the same logical f64 values
    /// encoded big-endian are a *different byte stream* and must therefore
    /// produce a different root -- and, critically, a reader that byte-swaps
    /// the little-endian file into native values before hashing would
    /// produce that same wrong root. Getting the same root from both
    /// encodings would mean the implementation is hashing decoded values.
    #[test]
    fn test_root_is_over_raw_file_bytes_not_converted_values() {
        let dims = vec![100u64];
        let elem_size = 8u32;
        let values: Vec<f64> = (0..100).map(f64::from).collect();

        let le_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let be_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();

        let (le_tree, _) =
            contiguous_tree_and_grid(&le_bytes, &dims, elem_size, 64, HashAlg::Blake3);
        let (be_tree, _) =
            contiguous_tree_and_grid(&be_bytes, &dims, elem_size, 64, HashAlg::Blake3);

        assert_ne!(
            le_tree.root(),
            be_tree.root(),
            "the same logical values in different file byte orders are different \
             byte streams and must hash differently -- an equal root here would \
             mean leaves are computed over decoded values, not raw file bytes"
        );

        // The same raw bytes always give the same root regardless of what
        // machine reads them: hashing depends only on the byte stream.
        let (le_again, _) =
            contiguous_tree_and_grid(&le_bytes, &dims, elem_size, 64, HashAlg::Blake3);
        assert_eq!(le_tree.root(), le_again.root());
    }

    /// P2.9: the streaming builder is an optimisation, not a different tree.
    /// It must produce the identical root, leaf hashes, and grid as the
    /// in-memory path over the same bytes — including in the cases most likely
    /// to expose an off-by-one: a batch boundary falling mid-grid, a short
    /// final leaf, and the single-leaf fallback below the split threshold.
    #[test]
    #[cfg(feature = "std")]
    fn streaming_matches_in_memory() {
        // (dims, elem_size, target) chosen to cover: many small leaves, a
        // grid whose last leaf is truncated, a 3-D grid, and the None/
        // single-leaf fallback.
        let cases: &[(&[u64], u32, u64)] = &[
            (&[100], 4, 64),          // 400 B, tiled small
            (&[1000], 8, 1024),       // 8000 B, several leaves
            (&[37], 4, 64),           // truncated final leaf (37 not divisible)
            (&[16, 16, 16], 4, 1024), // 3-D
            (&[10], 4, 1024 * 1024),  // below threshold -> single leaf
        ];

        for &(dims, elem_size, target) in cases {
            let n: u64 = dims.iter().product::<u64>() * u64::from(elem_size);
            let data: Vec<u8> = (0..n).map(|v| (v % 251) as u8).collect();

            let (mem_tree, mem_grid) =
                contiguous_tree_and_grid(&data, dims, elem_size, target, HashAlg::Blake3);

            let mut cursor = std::io::Cursor::new(&data);
            let (stream_tree, stream_grid) =
                contiguous_tree_streaming(&mut cursor, dims, elem_size, target, HashAlg::Blake3)
                    .expect("streaming build should succeed");

            assert_eq!(
                stream_grid, mem_grid,
                "grid differs for dims={dims:?} target={target}"
            );
            assert_eq!(
                stream_tree.root(),
                mem_tree.root(),
                "root differs for dims={dims:?} target={target}"
            );
            assert_eq!(stream_tree.leaf_count(), mem_tree.leaf_count());
            for i in 0..mem_tree.leaf_count() {
                assert_eq!(
                    stream_tree.leaf_hash(i),
                    mem_tree.leaf_hash(i),
                    "leaf {i} differs for dims={dims:?} target={target}"
                );
            }
        }
    }

    /// A batch boundary must not corrupt the tree. Forcing many batches (by
    /// using a dataset far larger than one batch would be too slow for a unit
    /// test, so this instead checks the batching arithmetic directly against
    /// the in-memory path at a grid size that produces thousands of leaves).
    #[test]
    #[cfg(feature = "std")]
    fn streaming_handles_many_leaves() {
        let dims = [4096u64];
        let elem_size = 4u32;
        let target = 64; // 16 elements per leaf -> 256 leaves
        let n: u64 = dims[0] * u64::from(elem_size);
        let data: Vec<u8> = (0..n).map(|v| (v % 253) as u8).collect();

        let (mem_tree, _) =
            contiguous_tree_and_grid(&data, &dims, elem_size, target, HashAlg::Blake3);
        let mut cursor = std::io::Cursor::new(&data);
        let (stream_tree, _) =
            contiguous_tree_streaming(&mut cursor, &dims, elem_size, target, HashAlg::Blake3)
                .unwrap();

        assert!(mem_tree.leaf_count() > 100, "sanity: want many leaves");
        assert_eq!(stream_tree.root(), mem_tree.root());
    }

    /// A truncated stream is an error, not a silently short tree.
    #[test]
    #[cfg(feature = "std")]
    fn streaming_rejects_short_stream() {
        let dims = [100u64];
        let data = vec![0u8; 200]; // half of the 400 bytes dims implies
        let mut cursor = std::io::Cursor::new(&data);
        let err = contiguous_tree_streaming(&mut cursor, &dims, 4, 64, HashAlg::Blake3)
            .expect_err("a short stream must fail");
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
    }

    /// P2.8b step 1: `contiguous_layout` reads a real file's raw-data address
    /// and extents out of the object header, and the bytes at that address
    /// are the ones `contiguous_tree_and_grid` should be handed.
    #[test]
    #[cfg(feature = "blake3")]
    fn test_contiguous_layout_reads_real_file_address_and_extents() {
        use crate::file_writer::FileWriter;
        use crate::group_v2::resolve_path_any;
        use crate::object_header::ObjectHeader;
        use crate::signature::find_signature;
        use crate::superblock::Superblock;

        let values: Vec<f64> = (0..1000).map(f64::from).collect();
        let mut fw = FileWriter::new();
        let ds = fw.create_dataset("readings");
        ds.with_f64_data(&values);
        let expected_bytes = ds.data.clone().unwrap();
        let file_bytes = fw.finish().expect("file should build");

        let sig = find_signature(&file_bytes).unwrap();
        let sb = Superblock::parse(&file_bytes, sig).unwrap();
        let addr = resolve_path_any(&file_bytes, &sb, "readings").unwrap();
        let hdr =
            ObjectHeader::parse(&file_bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();

        let layout = contiguous_layout(&hdr, sb.offset_size, sb.length_size)
            .expect("a plain f64 dataset should be supported");

        assert_eq!(layout.dims, vec![1000]);
        assert_eq!(layout.elem_size, 8);
        assert_eq!(layout.nbytes, 8000);
        let on_disk = &file_bytes
            [layout.data_addr as usize..(layout.data_addr + layout.nbytes) as usize];
        assert_eq!(on_disk, expected_bytes.as_slice());
    }

    /// P2.8b step 1: a chunked dataset is not a contiguous byte stream and
    /// must be rejected rather than silently misread.
    #[test]
    #[cfg(feature = "blake3")]
    fn test_contiguous_layout_rejects_chunked() {
        use crate::file_writer::FileWriter;
        use crate::group_v2::resolve_path_any;
        use crate::object_header::ObjectHeader;
        use crate::signature::find_signature;
        use crate::superblock::Superblock;

        let values: Vec<f64> = (0..1000).map(f64::from).collect();
        let mut fw = FileWriter::new();
        let ds = fw.create_dataset("readings");
        ds.with_f64_data(&values);
        ds.with_chunks(&[100]);
        let file_bytes = fw.finish().expect("file should build");

        let sig = find_signature(&file_bytes).unwrap();
        let sb = Superblock::parse(&file_bytes, sig).unwrap();
        let addr = resolve_path_any(&file_bytes, &sb, "readings").unwrap();
        let hdr =
            ObjectHeader::parse(&file_bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();

        let err = contiguous_layout(&hdr, sb.offset_size, sb.length_size).unwrap_err();
        assert!(
            matches!(
                err,
                MerkleError::UnsupportedLayout {
                    reason: UnsupportedLayoutReason::NotContiguous
                }
            ),
            "expected NotContiguous, got {err:?}"
        );
    }

    /// P2.8b step 1: the three hard-error rejections that have no
    /// `FileWriter` path (this writer emits neither external storage,
    /// unallocated contiguous storage, nor variable-length datatypes), so
    /// the headers are assembled directly. Each is the "worst available
    /// outcome" case: hashing the byte range anyway would certify heap
    /// identifiers, another file's contents, or uninitialized bytes.
    #[test]
    fn test_contiguous_layout_rejects_external_unallocated_and_vlen() {
        use crate::message_type::MessageType;
        use crate::object_header::{HeaderMessage, ObjectHeader};

        fn msg(msg_type: MessageType, data: Vec<u8>) -> HeaderMessage {
            HeaderMessage {
                msg_type,
                size: data.len(),
                flags: 0,
                creation_order: None,
                data,
            }
        }
        fn header(messages: Vec<HeaderMessage>) -> ObjectHeader {
            ObjectHeader {
                version: 2,
                messages,
                reference_count: None,
                flags: 0,
                access_time: None,
                modification_time: None,
                change_time: None,
                birth_time: None,
            }
        }

        // Dataspace v1, rank 1, extent 100.
        let mut dataspace = vec![1u8, 1, 0, 0, 0, 0, 0, 0];
        dataspace.extend_from_slice(&100u64.to_le_bytes());
        // Datatype: class 1 (float), 8 bytes -- a plain f64.
        let f64_dt = {
            let mut d = vec![0x11u8, 0x20, 0x1f, 0x00];
            d.extend_from_slice(&8u32.to_le_bytes());
            d.extend_from_slice(&[0x00, 0x00, 0x34, 0x00, 0x00, 0x40, 0x0d, 0xff]);
            d.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
            d
        };
        // DataLayout v3, contiguous (class 1), address + size.
        let contiguous_layout_msg = |addr: [u8; 8]| {
            let mut d = vec![3u8, 1];
            d.extend_from_slice(&addr);
            d.extend_from_slice(&800u64.to_le_bytes());
            d
        };

        // (a) External storage: the External Data Files message is present.
        let external = header(vec![
            msg(MessageType::Dataspace, dataspace.clone()),
            msg(MessageType::Datatype, f64_dt.clone()),
            msg(MessageType::DataLayout, contiguous_layout_msg(1024u64.to_le_bytes())),
            msg(MessageType::Unknown(0x0007), vec![0u8; 8]),
        ]);
        assert!(
            matches!(
                contiguous_layout(&external, 8, 8).unwrap_err(),
                MerkleError::UnsupportedLayout {
                    reason: UnsupportedLayoutReason::ExternalStorage
                }
            ),
            "external storage must be a hard error"
        );

        // (b) Unallocated: the undefined-address sentinel (all 0xFF).
        let unallocated = header(vec![
            msg(MessageType::Dataspace, dataspace.clone()),
            msg(MessageType::Datatype, f64_dt.clone()),
            msg(MessageType::DataLayout, contiguous_layout_msg([0xFFu8; 8])),
        ]);
        assert!(
            matches!(
                contiguous_layout(&unallocated, 8, 8).unwrap_err(),
                MerkleError::UnsupportedLayout {
                    reason: UnsupportedLayoutReason::Unallocated
                }
            ),
            "unallocated storage must be a hard error"
        );

        // (c) Variable-length: the byte range holds global-heap identifiers,
        // so a proof over it would certify pointers, not data.
        let vlen_dt = {
            let mut d = vec![0x19u8, 0x00, 0x00, 0x00];
            d.extend_from_slice(&16u32.to_le_bytes());
            d.extend_from_slice(&f64_dt);
            d
        };
        let vlen = header(vec![
            msg(MessageType::Dataspace, dataspace.clone()),
            msg(MessageType::Datatype, vlen_dt),
            msg(MessageType::DataLayout, contiguous_layout_msg(1024u64.to_le_bytes())),
        ]);
        assert!(
            matches!(
                contiguous_layout(&vlen, 8, 8).unwrap_err(),
                MerkleError::UnsupportedLayout {
                    reason: UnsupportedLayoutReason::IndirectDatatype
                }
            ),
            "variable-length datatypes must be a hard error"
        );

        // (d) Missing DataLayout message entirely.
        let malformed = header(vec![
            msg(MessageType::Dataspace, dataspace),
            msg(MessageType::Datatype, f64_dt),
        ]);
        assert!(matches!(
            contiguous_layout(&malformed, 8, 8).unwrap_err(),
            MerkleError::UnsupportedLayout {
                reason: UnsupportedLayoutReason::MalformedHeader
            }
        ));
    }
}
