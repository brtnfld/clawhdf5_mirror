//! Hyperslab and point selection for partial dataset I/O.
//!
//! A [`Selection`] describes which elements of a dataset to read or write.
//! The most common form is a hyperslab — a regular, strided sub-region of
//! the dataspace.
//!
//! # Example
//!
//! ```ignore
//! use clawhdf5_format::selection::Selection;
//!
//! // Select rows 20..30, columns 40..60 from a 2D dataset
//! let sel = Selection::slice(&[20..30, 40..60]);
//! assert_eq!(sel.num_elements(&[100, 100]), 200); // 10 * 20
//! ```

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use core::ops::Range;

use crate::error::FormatError;

/// A selection describing which elements of a dataset to access.
#[derive(Debug, Clone, PartialEq)]
pub enum Selection {
    /// Select all elements (equivalent to the entire dataspace).
    All,

    /// Select no elements.
    None,

    /// A regular hyperslab selection defined by start, stride, count, and block.
    ///
    /// For each dimension:
    /// - `start[d]` — first element index
    /// - `stride[d]` — step between blocks (must be >= block[d])
    /// - `count[d]` — number of blocks
    /// - `block[d]` — number of consecutive elements per block
    ///
    /// When stride == block (or stride is 1 and block is 1), this reduces
    /// to a simple contiguous slice.
    Hyperslab {
        start: Vec<u64>,
        stride: Vec<u64>,
        count: Vec<u64>,
        block: Vec<u64>,
    },

    /// Select individual points by coordinate.
    Points(Vec<Vec<u64>>),
}

impl Selection {
    /// Create a simple contiguous hyperslab from ranges (one per dimension).
    ///
    /// This is equivalent to a hyperslab with stride=1 and block=1.
    pub fn slice(ranges: &[Range<u64>]) -> Self {
        let rank = ranges.len();
        let mut start = Vec::with_capacity(rank);
        let mut count = Vec::with_capacity(rank);
        for r in ranges {
            debug_assert!(
                r.end >= r.start,
                "Selection::slice: range end ({}) < start ({})",
                r.end,
                r.start,
            );
            start.push(r.start);
            count.push(r.end.saturating_sub(r.start));
        }
        Selection::Hyperslab {
            start,
            stride: vec![1; rank],
            count,
            block: vec![1; rank],
        }
    }

    /// Number of selected elements for a given dataspace shape.
    pub fn num_elements(&self, dims: &[u64]) -> u64 {
        match self {
            Selection::All => dims.iter().product(),
            Selection::None => 0,
            Selection::Hyperslab { count, block, .. } => count
                .iter()
                .zip(block.iter())
                .map(|(&c, &b)| c * b)
                .product(),
            Selection::Points(pts) => pts.len() as u64,
        }
    }

    /// The rank (number of dimensions) of this selection.
    pub fn rank(&self) -> Option<usize> {
        match self {
            Selection::All | Selection::None => Option::None,
            Selection::Hyperslab { start, .. } => Some(start.len()),
            Selection::Points(pts) => pts.first().map(|p| p.len()),
        }
    }

    /// The shape of the selected region (output dimensions).
    ///
    /// For hyperslabs, this is `count[d] * block[d]` per dimension.
    /// For `All`, returns the dataspace shape. For `None`, returns empty.
    pub fn output_shape(&self, dims: &[u64]) -> Vec<u64> {
        match self {
            Selection::All => dims.to_vec(),
            Selection::None => vec![],
            Selection::Hyperslab { count, block, .. } => count
                .iter()
                .zip(block.iter())
                .map(|(&c, &b)| c * b)
                .collect(),
            Selection::Points(pts) => vec![pts.len() as u64],
        }
    }

    /// Check whether a chunk at the given offset (with given chunk dimensions)
    /// intersects this selection.
    ///
    /// Returns `true` if any element in the chunk overlaps with the selection.
    pub fn intersects_chunk(&self, chunk_offset: &[u64], chunk_dims: &[u64]) -> bool {
        match self {
            Selection::All => true,
            Selection::None => false,
            Selection::Hyperslab {
                start,
                stride,
                count,
                block,
            } => {
                // For each dimension, check if the chunk range overlaps the hyperslab range
                for d in 0..start.len() {
                    let chunk_start = chunk_offset[d];
                    let chunk_end = chunk_start + chunk_dims[d];

                    // Compute the full extent of the hyperslab in this dimension
                    let sel_start = start[d];
                    let sel_end = if count[d] == 0 {
                        sel_start
                    } else {
                        start[d] + (count[d] - 1) * stride[d] + block[d]
                    };

                    // No overlap if chunk is entirely before or after selection
                    if chunk_end <= sel_start || chunk_start >= sel_end {
                        return false;
                    }
                }
                true
            }
            Selection::Points(pts) => pts.iter().any(|pt| {
                pt.iter()
                    .zip(chunk_offset.iter().zip(chunk_dims.iter()))
                    .all(|(&p, (&off, &dim))| p >= off && p < off + dim)
            }),
        }
    }

    /// For a given chunk, compute the local ranges within the chunk that
    /// overlap with this selection.
    ///
    /// Returns a list of (chunk_local_start, chunk_local_end, output_offset) per
    /// dimension, representing which elements from the chunk contribute to the
    /// output buffer. For simple contiguous slices, this returns exactly one range
    /// per dimension.
    pub fn chunk_local_ranges(&self, chunk_offset: &[u64], chunk_dims: &[u64]) -> Vec<Range<u64>> {
        match self {
            Selection::All => chunk_dims.iter().map(|&d| 0..d).collect(),
            Selection::None => vec![],
            Selection::Hyperslab {
                start,
                stride,
                count,
                block,
            } => {
                let mut ranges = Vec::with_capacity(start.len());
                for d in 0..start.len() {
                    let chunk_start = chunk_offset[d];
                    let chunk_end = chunk_start + chunk_dims[d];

                    // For simple contiguous selections (stride==1, block==1),
                    // just clamp the selection range to the chunk bounds
                    if stride[d] == 1 && block[d] == 1 {
                        let sel_start = start[d];
                        let sel_end = start[d] + count[d];
                        let local_start = sel_start.max(chunk_start) - chunk_start;
                        let local_end = sel_end.min(chunk_end) - chunk_start;
                        ranges.push(local_start..local_end);
                    } else {
                        // General strided case: find all blocks that overlap this chunk
                        let sel_start = start[d];
                        let mut min_local = chunk_dims[d];
                        let mut max_local = 0u64;

                        for bi in 0..count[d] {
                            let block_start = sel_start + bi * stride[d];
                            let block_end = block_start + block[d];
                            // Check overlap with chunk
                            if block_end > chunk_start && block_start < chunk_end {
                                let local_s = block_start.max(chunk_start) - chunk_start;
                                let local_e = block_end.min(chunk_end) - chunk_start;
                                min_local = min_local.min(local_s);
                                max_local = max_local.max(local_e);
                            }
                        }
                        if max_local > min_local {
                            ranges.push(min_local..max_local);
                        } else {
                            ranges.push(0..0);
                        }
                    }
                }
                ranges
            }
            Selection::Points(_) => {
                // For point selections, return the full chunk range
                // (filtering happens at the element level)
                chunk_dims.iter().map(|&d| 0..d).collect()
            }
        }
    }

    /// Decode a selection from its on-disk **`H5S_select_serialize`** form.
    ///
    /// Returns the selection and the number of bytes consumed (selections are
    /// self-describing in length, so the count lets a caller walk a packed list
    /// of selections — as the Virtual Dataset global-heap block does).
    ///
    /// Only the forms needed for VDS assembly are decoded: `ALL`, `NONE`, and
    /// **regular** hyperslabs serialized at **version 3** (the encoding HDF5
    /// 1.10+/2.0 emit). Point selections, irregular hyperslabs, and older
    /// hyperslab versions return an error rather than mis-decoding.
    pub fn decode_serialized(data: &[u8]) -> Result<(Selection, usize), FormatError> {
        if data.len() < 8 {
            return Err(FormatError::UnexpectedEof {
                expected: 8,
                available: data.len(),
            });
        }
        let sel_type = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);

        match sel_type {
            // ALL / NONE: type(4) + version(4) + reserved(4) + length(4) = 16 bytes.
            3 | 0 => {
                if data.len() < 16 {
                    return Err(FormatError::UnexpectedEof {
                        expected: 16,
                        available: data.len(),
                    });
                }
                let sel = if sel_type == 3 {
                    Selection::All
                } else {
                    Selection::None
                };
                Ok((sel, 16))
            }
            2 => decode_hyperslab_serialized(data, version),
            1 => Err(FormatError::ChunkedReadError(
                "VDS point selections are not supported".into(),
            )),
            _ => Err(FormatError::ChunkedReadError(
                "unknown dataspace selection type".into(),
            )),
        }
    }

    /// Enumerate the selected element indices of a **1-D** dataspace of the
    /// given `extent`, in row-major selection order.
    ///
    /// Convenience wrapper over [`Selection::iter_linear`] for rank-1 spaces.
    pub fn iter_linear_1d(&self, extent: u64) -> Result<Vec<u64>, FormatError> {
        self.iter_linear(&[extent])
    }

    /// Enumerate the **row-major linear indices** of the selected elements of a
    /// dataspace with shape `dims`, in row-major (C) iteration order.
    ///
    /// This is the order HDF5 uses to pair a virtual selection with a source
    /// selection in a Virtual Dataset, so the i-th index returned here for the
    /// virtual selection corresponds to the i-th index for the source
    /// selection. Hyperslab/point selections whose rank differs from
    /// `dims.len()` are rejected.
    pub fn iter_linear(&self, dims: &[u64]) -> Result<Vec<u64>, FormatError> {
        let overflow = || FormatError::Overflow("VDS selection index overflow".into());
        let total: u64 = dims
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .ok_or_else(overflow)?;
        // Row-major strides: row_stride[d] = product(dims[d+1..]).
        let rank = dims.len();
        let mut row_stride = vec![1u64; rank];
        for d in (0..rank.saturating_sub(1)).rev() {
            row_stride[d] = row_stride[d + 1]
                .checked_mul(dims[d + 1])
                .ok_or_else(overflow)?;
        }

        match self {
            Selection::All => Ok((0..total).collect()),
            Selection::None => Ok(Vec::new()),
            Selection::Hyperslab {
                start,
                stride,
                count,
                block,
            } => {
                if start.len() != rank {
                    return Err(FormatError::ChunkedReadError(
                        "VDS selection rank does not match dataspace rank".into(),
                    ));
                }
                // Selected coordinates along each dimension, in order.
                let mut per_dim: Vec<Vec<u64>> = Vec::with_capacity(rank);
                for d in 0..rank {
                    let mut coords = Vec::new();
                    for ci in 0..count[d] {
                        let base = ci
                            .checked_mul(stride[d])
                            .and_then(|o| start[d].checked_add(o))
                            .ok_or_else(overflow)?;
                        for bi in 0..block[d] {
                            let coord = base.checked_add(bi).ok_or_else(overflow)?;
                            // Anything past the extent is malformed; bail before the
                            // coordinate list can grow without bound.
                            if coord >= dims[d] {
                                return Err(FormatError::ChunkedReadError(
                                    "VDS hyperslab selection exceeds dataspace extent".into(),
                                ));
                            }
                            coords.push(coord);
                        }
                    }
                    per_dim.push(coords);
                }
                if per_dim.iter().any(|c| c.is_empty()) {
                    return Ok(Vec::new());
                }
                // Cartesian product in row-major order (dim 0 slowest-varying).
                let out_len: usize = per_dim
                    .iter()
                    .try_fold(1usize, |acc, c| acc.checked_mul(c.len()))
                    .ok_or_else(overflow)?;
                let mut out = Vec::with_capacity(out_len);
                let mut idx = vec![0usize; rank];
                loop {
                    let mut lin = 0u64;
                    for d in 0..rank {
                        lin = per_dim[d][idx[d]]
                            .checked_mul(row_stride[d])
                            .and_then(|o| lin.checked_add(o))
                            .ok_or_else(overflow)?;
                    }
                    out.push(lin);
                    // Increment the mixed-radix counter, last dimension fastest.
                    let mut carry = true;
                    for d in (0..rank).rev() {
                        idx[d] += 1;
                        if idx[d] < per_dim[d].len() {
                            carry = false;
                            break;
                        }
                        idx[d] = 0;
                    }
                    if carry {
                        break;
                    }
                }
                Ok(out)
            }
            Selection::Points(pts) => {
                let mut out = Vec::with_capacity(pts.len());
                for p in pts {
                    if p.len() != rank {
                        return Err(FormatError::ChunkedReadError(
                            "VDS point selection rank does not match dataspace rank".into(),
                        ));
                    }
                    let mut lin = 0u64;
                    for d in 0..rank {
                        if p[d] >= dims[d] {
                            return Err(FormatError::ChunkedReadError(
                                "VDS point selection exceeds dataspace extent".into(),
                            ));
                        }
                        lin = p[d]
                            .checked_mul(row_stride[d])
                            .and_then(|o| lin.checked_add(o))
                            .ok_or_else(overflow)?;
                    }
                    out.push(lin);
                }
                Ok(out)
            }
        }
    }
}

/// Decode an `H5S_SEL_HYPER` selection in its serialized form. Only version-3
/// **regular** hyperslabs are supported.
fn decode_hyperslab_serialized(
    data: &[u8],
    version: u32,
) -> Result<(Selection, usize), FormatError> {
    if version != 3 {
        return Err(FormatError::ChunkedReadError(
            "only version-3 hyperslab selections are supported".into(),
        ));
    }
    // type(4) ver(4) flags(1) enc_size(1) rank(4) [start,stride,count,block]*rank
    if data.len() < 14 {
        return Err(FormatError::UnexpectedEof {
            expected: 14,
            available: data.len(),
        });
    }
    let flags = data[8];
    let enc_size = data[9] as usize;
    // Bit 0 set => regular hyperslab. Irregular hyperslabs list explicit blocks.
    if flags & 0x01 == 0 {
        return Err(FormatError::ChunkedReadError(
            "irregular VDS hyperslab selections are not supported".into(),
        ));
    }
    if enc_size != 2 && enc_size != 4 && enc_size != 8 {
        return Err(FormatError::ChunkedReadError(
            "unsupported hyperslab coordinate encoding size".into(),
        ));
    }
    let rank = u32::from_le_bytes([data[10], data[11], data[12], data[13]]) as usize;
    // HDF5 caps dataspace rank at 32 (H5S_MAX_RANK). Reject anything larger so a
    // corrupt rank can't drive a huge allocation or read loop.
    if rank > 32 {
        return Err(FormatError::ChunkedReadError(
            "hyperslab selection rank exceeds maximum (32)".into(),
        ));
    }
    let mut pos = 14;
    let read_coord = |data: &[u8], pos: usize| -> Result<u64, FormatError> {
        if pos + enc_size > data.len() {
            return Err(FormatError::UnexpectedEof {
                expected: pos + enc_size,
                available: data.len(),
            });
        }
        let mut v = 0u64;
        for (i, &b) in data[pos..pos + enc_size].iter().enumerate() {
            v |= (b as u64) << (i * 8);
        }
        Ok(v)
    };
    let (mut start, mut stride, mut count, mut block) = (
        Vec::with_capacity(rank),
        Vec::with_capacity(rank),
        Vec::with_capacity(rank),
        Vec::with_capacity(rank),
    );
    for _ in 0..rank {
        start.push(read_coord(data, pos)?);
        pos += enc_size;
        stride.push(read_coord(data, pos)?);
        pos += enc_size;
        count.push(read_coord(data, pos)?);
        pos += enc_size;
        block.push(read_coord(data, pos)?);
        pos += enc_size;
    }
    Ok((
        Selection::Hyperslab {
            start,
            stride,
            count,
            block,
        },
        pos,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_all_num_elements() {
        let sel = Selection::All;
        assert_eq!(sel.num_elements(&[100, 200]), 20000);
    }

    #[test]
    fn selection_none_num_elements() {
        let sel = Selection::None;
        assert_eq!(sel.num_elements(&[100, 200]), 0);
    }

    #[test]
    fn selection_slice_basic() {
        let sel = Selection::slice(&[20..30, 40..60]);
        assert_eq!(sel.num_elements(&[100, 100]), 200); // 10 * 20
        assert_eq!(sel.output_shape(&[100, 100]), vec![10, 20]);
    }

    #[test]
    fn selection_slice_1d() {
        let sel = Selection::slice(&[5..15]);
        assert_eq!(sel.num_elements(&[100]), 10);
        assert_eq!(sel.output_shape(&[100]), vec![10]);
    }

    #[test]
    fn selection_intersects_chunk_basic() {
        let sel = Selection::slice(&[20..30, 40..60]);

        // Chunk [20..30, 40..50] — overlaps
        assert!(sel.intersects_chunk(&[20, 40], &[10, 10]));

        // Chunk [0..10, 0..10] — no overlap
        assert!(!sel.intersects_chunk(&[0, 0], &[10, 10]));

        // Chunk [20..30, 50..60] — overlaps
        assert!(sel.intersects_chunk(&[20, 50], &[10, 10]));

        // Chunk [30..40, 40..50] — no overlap (just past end in dim 0)
        assert!(!sel.intersects_chunk(&[30, 40], &[10, 10]));
    }

    #[test]
    fn selection_chunk_local_ranges_simple() {
        let sel = Selection::slice(&[25..35, 40..60]);

        // Chunk [20..30, 40..50]
        let ranges = sel.chunk_local_ranges(&[20, 40], &[10, 10]);
        assert_eq!(ranges[0], 5..10); // rows 25..30 within chunk starting at 20
        assert_eq!(ranges[1], 0..10); // cols 40..50 fully selected
    }

    #[test]
    fn selection_points() {
        let sel = Selection::Points(vec![vec![1, 2], vec![3, 4], vec![5, 6]]);
        assert_eq!(sel.num_elements(&[10, 10]), 3);
        assert_eq!(sel.rank(), Some(2));
    }

    #[test]
    fn selection_all_intersects_any_chunk() {
        let sel = Selection::All;
        assert!(sel.intersects_chunk(&[0, 0], &[10, 10]));
        assert!(sel.intersects_chunk(&[100, 100], &[1, 1]));
    }

    #[test]
    fn selection_hyperslab_strided() {
        // Select every other row: start=0, stride=2, count=5, block=1 in a 10-element dim
        let sel = Selection::Hyperslab {
            start: vec![0],
            stride: vec![2],
            count: vec![5],
            block: vec![1],
        };
        assert_eq!(sel.num_elements(&[10]), 5); // 5 blocks * 1 element each

        // Chunk [0..5] should intersect (contains rows 0, 2, 4)
        assert!(sel.intersects_chunk(&[0], &[5]));
        // Chunk [9..10] should not intersect (only row 9, but selection ends at row 8)
        assert!(!sel.intersects_chunk(&[9], &[1]));
    }

    #[test]
    fn decode_all_selection_16_bytes() {
        let bytes = [3u8, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let (sel, consumed) = Selection::decode_serialized(&bytes).unwrap();
        assert_eq!(sel, Selection::All);
        assert_eq!(consumed, 16);
        assert_eq!(sel.iter_linear_1d(4).unwrap(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn decode_regular_hyperslab_matches_vds_fixture() {
        // Exact virtual selection for src_a in the VDS fixture:
        // start=0 stride=1 count=1 block=4, version 3, enc_size 2, rank 1.
        let bytes = [
            0x02, 0, 0, 0, // type = HYPER
            0x03, 0, 0, 0, // version 3
            0x01, // flags = regular
            0x02, // enc_size = 2
            0x01, 0, 0, 0, // rank = 1
            0x00, 0x00, // start
            0x01, 0x00, // stride
            0x01, 0x00, // count
            0x04, 0x00, // block
        ];
        let (sel, consumed) = Selection::decode_serialized(&bytes).unwrap();
        assert_eq!(consumed, 22);
        assert_eq!(
            sel,
            Selection::Hyperslab {
                start: vec![0],
                stride: vec![1],
                count: vec![1],
                block: vec![4],
            }
        );
        assert_eq!(sel.iter_linear_1d(8).unwrap(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn decode_hyperslab_start4() {
        let bytes = [
            0x02, 0, 0, 0, 0x03, 0, 0, 0, 0x01, 0x02, 0x01, 0, 0, 0, //
            0x04, 0x00, 0x01, 0x00, 0x01, 0x00, 0x04, 0x00,
        ];
        let (sel, _) = Selection::decode_serialized(&bytes).unwrap();
        assert_eq!(sel.iter_linear_1d(8).unwrap(), vec![4, 5, 6, 7]);
    }

    #[test]
    fn decode_strided_hyperslab_iter() {
        // start=1 stride=3 count=2 block=2 => 1,2, 4,5
        let bytes = [
            0x02, 0, 0, 0, 0x03, 0, 0, 0, 0x01, 0x02, 0x01, 0, 0, 0, //
            0x01, 0x00, 0x03, 0x00, 0x02, 0x00, 0x02, 0x00,
        ];
        let (sel, _) = Selection::decode_serialized(&bytes).unwrap();
        assert_eq!(sel.iter_linear_1d(8).unwrap(), vec![1, 2, 4, 5]);
    }

    #[test]
    fn decode_nd_hyperslab_iter_rejected() {
        let bytes = [
            0x02, 0, 0, 0, 0x03, 0, 0, 0, 0x01, 0x02, 0x02, 0, 0, 0, // rank 2
            0, 0, 1, 0, 1, 0, 2, 0, 0, 0, 1, 0, 1, 0, 2, 0,
        ];
        let (sel, _) = Selection::decode_serialized(&bytes).unwrap();
        assert!(sel.iter_linear_1d(16).is_err());
    }

    #[test]
    fn decode_irregular_hyperslab_rejected() {
        let bytes = [0x02u8, 0, 0, 0, 0x03, 0, 0, 0, 0x00, 0x02, 0x01, 0, 0, 0];
        assert!(Selection::decode_serialized(&bytes).is_err());
    }

    #[test]
    fn iter_linear_2d_block_row_major() {
        // A 2x2 block at the top-left of a 4x4 space => linear 0,1,4,5.
        let sel = Selection::Hyperslab {
            start: vec![0, 0],
            stride: vec![1, 1],
            count: vec![1, 1],
            block: vec![2, 2],
        };
        assert_eq!(sel.iter_linear(&[4, 4]).unwrap(), vec![0, 1, 4, 5]);

        // The same block shifted to the bottom-right => 10,11,14,15.
        let sel2 = Selection::Hyperslab {
            start: vec![2, 2],
            stride: vec![1, 1],
            count: vec![1, 1],
            block: vec![2, 2],
        };
        assert_eq!(sel2.iter_linear(&[4, 4]).unwrap(), vec![10, 11, 14, 15]);
    }

    #[test]
    fn iter_linear_2d_strided() {
        // start=(0,0) stride=(2,2) count=(2,2) block=(1,1) over 4x4 =>
        // coords (0,0)(0,2)(2,0)(2,2) => linear 0,2,8,10.
        let sel = Selection::Hyperslab {
            start: vec![0, 0],
            stride: vec![2, 2],
            count: vec![2, 2],
            block: vec![1, 1],
        };
        assert_eq!(sel.iter_linear(&[4, 4]).unwrap(), vec![0, 2, 8, 10]);
    }

    #[test]
    fn iter_linear_all_2d() {
        assert_eq!(
            Selection::All.iter_linear(&[2, 3]).unwrap(),
            (0..6).collect::<Vec<_>>()
        );
    }

    #[test]
    fn iter_linear_rank_mismatch_rejected() {
        let sel = Selection::Hyperslab {
            start: vec![0],
            stride: vec![1],
            count: vec![1],
            block: vec![2],
        };
        assert!(sel.iter_linear(&[4, 4]).is_err());
    }

    // ----- Adversarial / hardening: malformed input must error, never panic -----

    #[test]
    fn decode_all_truncated_does_not_overrun() {
        // ALL claims to consume 16 bytes but only 8 are present.
        let bytes = [3u8, 0, 0, 0, 1, 0, 0, 0];
        assert!(Selection::decode_serialized(&bytes).is_err());
    }

    #[test]
    fn decode_hyperslab_huge_rank_rejected() {
        // rank = 0xFFFFFFFF must not drive a giant allocation.
        let bytes = [
            0x02u8, 0, 0, 0, 0x03, 0, 0, 0, 0x01, 0x02, 0xFF, 0xFF, 0xFF, 0xFF,
        ];
        assert!(Selection::decode_serialized(&bytes).is_err());
    }

    #[test]
    fn iter_linear_hyperslab_overflow_is_error() {
        // start/stride/count near u64::MAX must not panic on multiply/add.
        let sel = Selection::Hyperslab {
            start: vec![u64::MAX - 1],
            stride: vec![u64::MAX],
            count: vec![u64::MAX],
            block: vec![u64::MAX],
        };
        assert!(sel.iter_linear(&[100]).is_err());
    }

    #[test]
    fn iter_linear_dims_product_overflow_is_error() {
        assert!(Selection::All.iter_linear(&[u64::MAX, u64::MAX]).is_err());
    }

    #[test]
    fn decode_empty_or_short_is_error_not_panic() {
        assert!(Selection::decode_serialized(&[]).is_err());
        assert!(Selection::decode_serialized(&[2, 0, 0, 0, 3, 0]).is_err());
    }
}
