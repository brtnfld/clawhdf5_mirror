//! HDF5 Data Layout message parsing (message type 0x0008).

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

#[cfg(feature = "std")]
use std::string::String;

use crate::error::FormatError;

/// A single VDS (Virtual Dataset) source mapping.
///
/// Maps a region of the virtual dataset to a region of a source dataset
/// in a (possibly external) HDF5 file.
#[derive(Debug, Clone, PartialEq)]
pub struct VdsMapping {
    /// Source file name (may be "." for the same file).
    pub source_file: String,
    /// Source dataset path within the source file.
    pub source_dataset: String,
    /// Serialized source selection bytes (dataspace selection).
    pub source_selection: Vec<u8>,
    /// Serialized virtual selection bytes (dataspace selection).
    pub virtual_selection: Vec<u8>,
}

/// Parsed HDF5 data layout message.
#[derive(Debug, Clone, PartialEq)]
pub enum DataLayout {
    /// Compact: data stored inline in the message.
    Compact {
        /// The inline raw data bytes.
        data: Vec<u8>,
    },
    /// Contiguous: data stored at a single address in the file.
    Contiguous {
        /// File address of the data, or `None` if undefined (all 0xFF).
        address: Option<u64>,
        /// Size of the data in bytes.
        size: u64,
    },
    /// Chunked: data stored in chunks via a B-tree.
    Chunked {
        /// Chunk dimension sizes.
        chunk_dimensions: Vec<u32>,
        /// B-tree address, or `None` if undefined.
        btree_address: Option<u64>,
        /// Layout version (3 or 4).
        version: u8,
        /// Chunk index type (v4 only).
        chunk_index_type: Option<u8>,
        /// Filtered size for v4 single chunk with filters.
        single_chunk_filtered_size: Option<u64>,
        /// Filter mask for v4 single chunk with filters.
        single_chunk_filter_mask: Option<u32>,
    },
    /// Virtual dataset layout (v4 only).
    Virtual {
        /// Layout version.
        version: u8,
        /// Global heap address where VDS mappings are stored.
        global_heap_address: Option<u64>,
        /// Index of the object in the global heap collection.
        global_heap_index: u32,
        /// Parsed VDS source mappings (populated after global heap lookup).
        mappings: Vec<VdsMapping>,
    },
}

/// Parse VDS mappings from global-heap object data.
///
/// The global-heap block holding a VDS mapping list is laid out as
/// (reverse-engineered and validated against HDF5 2.0):
///
/// ```text
/// version(1) · nused(length_size, LE) · entry[nused] · checksum(4)
/// ```
///
/// Each entry is:
/// - source file name — a null-terminated string in **block version 0**; in
///   **block version 1** a same-file reference is encoded as a single `0x04`
///   marker byte (the source file is the virtual file itself) in place of the
///   name;
/// - source dataset name (null-terminated string);
/// - source selection (serialized `H5S` dataspace selection — self-describing
///   in length);
/// - virtual selection (serialized `H5S` dataspace selection).
///
/// The selections are decoded with [`crate::selection::Selection`] purely to
/// learn their byte length so the entry list can be walked; the raw selection
/// bytes are retained on each [`VdsMapping`] for the reader to interpret.
pub fn parse_vds_mappings(
    heap_data: &[u8],
    length_size: u8,
) -> Result<Vec<VdsMapping>, FormatError> {
    use crate::selection::Selection;

    let ls = length_size as usize;
    if heap_data.len() < 1 + ls {
        return Ok(Vec::new());
    }
    let version = heap_data[0];
    let mut pos = 1;
    let nused = read_length(heap_data, pos, length_size)?;
    pos += ls;

    // `nused` is untrusted; don't pre-allocate from it. Each entry consumes at
    // least a few bytes, so the loop is naturally bounded by the heap data and
    // a bogus `nused` simply errors out on the first short read.
    let mut mappings = Vec::new();
    // Reads one self-describing selection at `pos`, returning its raw bytes and
    // advancing past it — bounds-checked so a corrupt selection can't overrun.
    let read_selection = |heap_data: &[u8], pos: &mut usize| -> Result<Vec<u8>, FormatError> {
        let rest = heap_data.get(*pos..).ok_or(FormatError::UnexpectedEof {
            expected: *pos,
            available: heap_data.len(),
        })?;
        let (_, len) = Selection::decode_serialized(rest)?;
        let bytes = rest
            .get(..len)
            .ok_or(FormatError::UnexpectedEof {
                expected: pos.saturating_add(len),
                available: heap_data.len(),
            })?
            .to_vec();
        *pos += len;
        Ok(bytes)
    };

    for _ in 0..nused {
        // Source file name (with the version-1 same-file marker handled).
        let source_file = if version >= 1 && heap_data.get(pos) == Some(&0x04) {
            pos += 1;
            String::from(".")
        } else {
            read_null_terminated_string(heap_data, &mut pos)?
        };

        // Source dataset name.
        let source_dataset = read_null_terminated_string(heap_data, &mut pos)?;

        // Source selection, then virtual selection (both self-describing length).
        let source_selection = read_selection(heap_data, &mut pos)?;
        let virtual_selection = read_selection(heap_data, &mut pos)?;

        mappings.push(VdsMapping {
            source_file,
            source_dataset,
            source_selection,
            virtual_selection,
        });
    }

    Ok(mappings)
}

/// Read a null-terminated UTF-8 string from data starting at `pos`.
fn read_null_terminated_string(data: &[u8], pos: &mut usize) -> Result<String, FormatError> {
    let start = *pos;
    while *pos < data.len() && data[*pos] != 0 {
        *pos += 1;
    }
    if *pos >= data.len() {
        return Err(FormatError::UnexpectedEof {
            expected: start + 1,
            available: data.len(),
        });
    }
    let s = String::from_utf8_lossy(&data[start..*pos]).into_owned();
    *pos += 1; // skip null terminator
    Ok(s)
}

fn ensure_len(data: &[u8], offset: usize, needed: usize) -> Result<(), FormatError> {
    match offset.checked_add(needed) {
        Some(end) if end <= data.len() => Ok(()),
        _ => Err(FormatError::UnexpectedEof {
            expected: offset.saturating_add(needed),
            available: data.len(),
        }),
    }
}

fn read_offset(data: &[u8], pos: usize, size: u8) -> Result<u64, FormatError> {
    let s = size as usize;
    ensure_len(data, pos, s)?;
    let slice = &data[pos..pos + s];
    Ok(match size {
        2 => u16::from_le_bytes([slice[0], slice[1]]) as u64,
        4 => u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]) as u64,
        8 => u64::from_le_bytes([
            slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
        ]),
        _ => {
            return Err(FormatError::InvalidOffsetSize(size));
        }
    })
}

fn read_length(data: &[u8], pos: usize, size: u8) -> Result<u64, FormatError> {
    read_offset(data, pos, size)
}

/// Check if all bytes in a slice are 0xFF (undefined address).
fn is_undefined(data: &[u8], pos: usize, size: u8) -> bool {
    let s = size as usize;
    if pos + s > data.len() {
        return false;
    }
    data[pos..pos + s].iter().all(|&b| b == 0xFF)
}

impl DataLayout {
    /// For a Virtual layout, resolve VDS mappings from the global heap.
    ///
    /// Reads the global heap collection at the stored address and parses the
    /// VDS mapping entries from the referenced object. After calling this
    /// method, the `mappings` field will be populated.
    ///
    /// No-op for non-Virtual layouts.
    pub fn resolve_vds_mappings(
        &mut self,
        file_data: &[u8],
        length_size: u8,
    ) -> Result<(), FormatError> {
        if let DataLayout::Virtual {
            global_heap_address,
            global_heap_index,
            mappings,
            ..
        } = self
            && let Some(addr) = *global_heap_address
        {
            let coll = crate::global_heap::GlobalHeapCollection::parse(
                file_data,
                addr as usize,
                length_size,
            )?;
            let obj = coll.get_object(*global_heap_index as u16).ok_or(
                FormatError::GlobalHeapObjectNotFound {
                    collection_address: addr,
                    index: *global_heap_index as u16,
                },
            )?;
            *mappings = parse_vds_mappings(&obj.data, length_size)?;
        }
        Ok(())
    }

    /// Parse a data layout message from raw message bytes.
    ///
    /// `offset_size` and `length_size` come from the superblock.
    pub fn parse(data: &[u8], offset_size: u8, length_size: u8) -> Result<DataLayout, FormatError> {
        ensure_len(data, 0, 2)?;
        let version = data[0];
        let layout_class = data[1];

        match version {
            3 => Self::parse_v3(data, layout_class, offset_size, length_size),
            // v5 (emitted by HDF5 1.14+/2.0 with `libver=latest`) uses the same
            // message structure as v4 — only the version number was bumped.
            4 | 5 => Self::parse_v4(data, layout_class, offset_size, length_size),
            _ => Err(FormatError::InvalidLayoutVersion(version)),
        }
    }

    fn parse_v3(
        data: &[u8],
        layout_class: u8,
        offset_size: u8,
        length_size: u8,
    ) -> Result<DataLayout, FormatError> {
        let pos = 2;
        match layout_class {
            0 => {
                // Compact
                ensure_len(data, pos, 2)?;
                let data_size = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
                ensure_len(data, pos + 2, data_size)?;
                let raw = data[pos + 2..pos + 2 + data_size].to_vec();
                Ok(DataLayout::Compact { data: raw })
            }
            1 => {
                // Contiguous
                let os = offset_size as usize;
                let ls = length_size as usize;
                ensure_len(data, pos, os + ls)?;
                let address = if is_undefined(data, pos, offset_size) {
                    None
                } else {
                    Some(read_offset(data, pos, offset_size)?)
                };
                let size = read_length(data, pos + os, length_size)?;
                Ok(DataLayout::Contiguous { address, size })
            }
            2 => {
                // Chunked
                ensure_len(data, pos, 1)?;
                let dimensionality = data[pos] as usize;
                let mut p = pos + 1;
                // btree address first
                let os = offset_size as usize;
                ensure_len(data, p, os)?;
                let btree_address = if is_undefined(data, p, offset_size) {
                    None
                } else {
                    Some(read_offset(data, p, offset_size)?)
                };
                p += os;
                // chunk dim sizes: dimensionality × 4 bytes each
                ensure_len(data, p, dimensionality * 4)?;
                let mut chunk_dimensions = Vec::with_capacity(dimensionality);
                for _ in 0..dimensionality {
                    let dim = u32::from_le_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]]);
                    chunk_dimensions.push(dim);
                    p += 4;
                }
                Ok(DataLayout::Chunked {
                    chunk_dimensions,
                    btree_address,
                    version: 3,
                    chunk_index_type: None,
                    single_chunk_filtered_size: None,
                    single_chunk_filter_mask: None,
                })
            }
            _ => Err(FormatError::InvalidLayoutClass(layout_class)),
        }
    }

    fn parse_v4(
        data: &[u8],
        layout_class: u8,
        offset_size: u8,
        length_size: u8,
    ) -> Result<DataLayout, FormatError> {
        let pos = 2;
        match layout_class {
            0 => {
                // Compact — same as v3
                ensure_len(data, pos, 2)?;
                let data_size = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
                ensure_len(data, pos + 2, data_size)?;
                let raw = data[pos + 2..pos + 2 + data_size].to_vec();
                Ok(DataLayout::Compact { data: raw })
            }
            1 => {
                // Contiguous — same as v3
                let os = offset_size as usize;
                let ls = length_size as usize;
                ensure_len(data, pos, os + ls)?;
                let address = if is_undefined(data, pos, offset_size) {
                    None
                } else {
                    Some(read_offset(data, pos, offset_size)?)
                };
                let size = read_length(data, pos + os, length_size)?;
                Ok(DataLayout::Contiguous { address, size })
            }
            2 => {
                // Chunked v4
                ensure_len(data, pos, 3)?;
                let flags = data[pos];
                let dimensionality = data[pos + 1] as usize;
                let dim_size_encoded_length = data[pos + 2] as usize;
                let mut p = pos + 3;

                // dimension sizes
                ensure_len(data, p, dimensionality * dim_size_encoded_length)?;
                let mut chunk_dimensions = Vec::with_capacity(dimensionality);
                for _ in 0..dimensionality {
                    let val = match dim_size_encoded_length {
                        1 => data[p] as u32,
                        2 => u16::from_le_bytes([data[p], data[p + 1]]) as u32,
                        4 => u32::from_le_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]]),
                        8 => {
                            // V4 chunked encodes dimension sizes as 8 bytes, but
                            // our ChunkedStorageV4 stores them as u32. We read only
                            // the low 4 bytes (little-endian). This silently
                            // truncates dimensions > 4 GiB, which are not expected
                            // in practice (HDF5 chunk dimensions are always small).
                            // If the high bytes are non-zero, the file is malformed
                            // or uses dimensions we cannot represent.
                            let high = u32::from_le_bytes([
                                data[p + 4],
                                data[p + 5],
                                data[p + 6],
                                data[p + 7],
                            ]);
                            if high != 0 {
                                return Err(FormatError::UnexpectedEof {
                                    expected: p + 8,
                                    available: data.len(),
                                });
                            }
                            u32::from_le_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]])
                        }
                        _ => {
                            return Err(FormatError::UnexpectedEof {
                                expected: p + dim_size_encoded_length,
                                available: data.len(),
                            });
                        }
                    };
                    chunk_dimensions.push(val);
                    p += dim_size_encoded_length;
                }

                // chunk index type
                ensure_len(data, p, 1)?;
                let chunk_index_type = data[p];
                p += 1;

                // Parse index-specific fields
                let mut single_chunk_filtered_size = None;
                let mut single_chunk_filter_mask = None;
                let btree_address = match chunk_index_type {
                    1 => {
                        // Single chunk
                        // H5O_LAYOUT_CHUNK_SINGLE_INDEX_WITH_FILTER = 0x02
                        let filters_present = flags & 0x02 != 0;
                        if filters_present {
                            // filtered_size(length_size) + filter_mask(4) + address(offset_size)
                            let ls = length_size as usize;
                            let os = offset_size as usize;
                            ensure_len(data, p, ls + 4 + os)?;
                            single_chunk_filtered_size = Some(read_length(data, p, length_size)?);
                            p += ls;
                            single_chunk_filter_mask = Some(u32::from_le_bytes([
                                data[p],
                                data[p + 1],
                                data[p + 2],
                                data[p + 3],
                            ]));
                            p += 4;
                            if is_undefined(data, p, offset_size) {
                                None
                            } else {
                                Some(read_offset(data, p, offset_size)?)
                            }
                        } else {
                            // just address(offset_size)
                            ensure_len(data, p, offset_size as usize)?;
                            if is_undefined(data, p, offset_size) {
                                None
                            } else {
                                Some(read_offset(data, p, offset_size)?)
                            }
                        }
                    }
                    2 => {
                        // Implicit: just address
                        ensure_len(data, p, offset_size as usize)?;
                        if is_undefined(data, p, offset_size) {
                            None
                        } else {
                            Some(read_offset(data, p, offset_size)?)
                        }
                    }
                    3 => {
                        // Fixed Array: max_dblk_page_nelmts_bits(1) + address(offset_size)
                        ensure_len(data, p, 1 + offset_size as usize)?;
                        p += 1; // skip max_dblk_page_nelmts_bits
                        if is_undefined(data, p, offset_size) {
                            None
                        } else {
                            Some(read_offset(data, p, offset_size)?)
                        }
                    }
                    4 => {
                        // Extensible Array: 5 creation params + address(offset_size)
                        ensure_len(data, p, 5 + offset_size as usize)?;
                        p += 5; // skip EA creation parameters
                        if is_undefined(data, p, offset_size) {
                            None
                        } else {
                            Some(read_offset(data, p, offset_size)?)
                        }
                    }
                    5 => {
                        // B-tree v2: node_size(4) + split_percent(1) + merge_percent(1) + address
                        ensure_len(data, p, 6 + offset_size as usize)?;
                        p += 6;
                        if is_undefined(data, p, offset_size) {
                            None
                        } else {
                            Some(read_offset(data, p, offset_size)?)
                        }
                    }
                    _ => {
                        // Unknown index type: try just address
                        ensure_len(data, p, offset_size as usize)?;
                        if is_undefined(data, p, offset_size) {
                            None
                        } else {
                            Some(read_offset(data, p, offset_size)?)
                        }
                    }
                };

                Ok(DataLayout::Chunked {
                    chunk_dimensions,
                    btree_address,
                    version: 4,
                    chunk_index_type: Some(chunk_index_type),
                    single_chunk_filtered_size,
                    single_chunk_filter_mask,
                })
            }
            3 => {
                // Virtual: global_heap_address(offset_size) + global_heap_index(4)
                let os = offset_size as usize;
                ensure_len(data, pos, os + 4)?;
                let global_heap_address = if is_undefined(data, pos, offset_size) {
                    None
                } else {
                    Some(read_offset(data, pos, offset_size)?)
                };
                let idx_pos = pos + os;
                let global_heap_index = u32::from_le_bytes([
                    data[idx_pos],
                    data[idx_pos + 1],
                    data[idx_pos + 2],
                    data[idx_pos + 3],
                ]);
                Ok(DataLayout::Virtual {
                    version: 4,
                    global_heap_address,
                    global_heap_index,
                    mappings: Vec::new(),
                })
            }
            _ => Err(FormatError::InvalidLayoutClass(layout_class)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v3_compact() {
        let mut buf = vec![3u8, 0]; // version=3, class=0 (compact)
        buf.extend_from_slice(&5u16.to_le_bytes()); // data_size=5
        buf.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD, 0xEE]); // data
        let layout = DataLayout::parse(&buf, 8, 8).unwrap();
        assert_eq!(
            layout,
            DataLayout::Compact {
                data: vec![0xAA, 0xBB, 0xCC, 0xDD, 0xEE]
            }
        );
    }

    #[test]
    fn v3_contiguous() {
        let mut buf = vec![3u8, 1]; // version=3, class=1 (contiguous)
        buf.extend_from_slice(&0x1000u64.to_le_bytes()); // address
        buf.extend_from_slice(&256u64.to_le_bytes()); // size
        let layout = DataLayout::parse(&buf, 8, 8).unwrap();
        assert_eq!(
            layout,
            DataLayout::Contiguous {
                address: Some(0x1000),
                size: 256,
            }
        );
    }

    #[test]
    fn v3_contiguous_undefined_address() {
        let mut buf = vec![3u8, 1];
        buf.extend_from_slice(&[0xFF; 8]); // undefined address
        buf.extend_from_slice(&0u64.to_le_bytes()); // size
        let layout = DataLayout::parse(&buf, 8, 8).unwrap();
        assert_eq!(
            layout,
            DataLayout::Contiguous {
                address: None,
                size: 0,
            }
        );
    }

    #[test]
    fn v3_chunked() {
        let mut buf = vec![3u8, 2]; // version=3, class=2 (chunked)
        buf.push(3); // dimensionality=3 (rank+1)
        buf.extend_from_slice(&0x2000u64.to_le_bytes()); // btree address
        // 3 chunk dim sizes × 4 bytes
        buf.extend_from_slice(&100u32.to_le_bytes());
        buf.extend_from_slice(&200u32.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes()); // last = element size
        let layout = DataLayout::parse(&buf, 8, 8).unwrap();
        assert_eq!(
            layout,
            DataLayout::Chunked {
                chunk_dimensions: vec![100, 200, 8],
                btree_address: Some(0x2000),
                version: 3,
                chunk_index_type: None,
                single_chunk_filtered_size: None,
                single_chunk_filter_mask: None,
            }
        );
    }

    #[test]
    fn v4_compact() {
        let mut buf = vec![4u8, 0]; // version=4, class=0
        buf.extend_from_slice(&3u16.to_le_bytes());
        buf.extend_from_slice(&[1, 2, 3]);
        let layout = DataLayout::parse(&buf, 8, 8).unwrap();
        assert_eq!(
            layout,
            DataLayout::Compact {
                data: vec![1, 2, 3]
            }
        );
    }

    #[test]
    fn v4_contiguous() {
        let mut buf = vec![4u8, 1];
        buf.extend_from_slice(&0x5000u64.to_le_bytes());
        buf.extend_from_slice(&512u64.to_le_bytes());
        let layout = DataLayout::parse(&buf, 8, 8).unwrap();
        assert_eq!(
            layout,
            DataLayout::Contiguous {
                address: Some(0x5000),
                size: 512,
            }
        );
    }

    #[test]
    fn v5_chunked_from_hdf5_2_0() {
        // Real data layout message from h5py 3.16 / HDF5 2.0 (`libver=latest`)
        // for a gzip-compressed 1-D chunked dataset. Version 5 uses the same
        // structure as v4 (here: chunked, Fixed Array index). Regression guard
        // for reading modern-format chunked datasets.
        let bytes: [u8; 17] = [
            0x05, 0x02, 0x00, 0x02, 0x01, 0x0a, 0x08, 0x03, 0x0a, 0xef, 0x05, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00,
        ];
        let layout = DataLayout::parse(&bytes, 8, 8).unwrap();
        match layout {
            DataLayout::Chunked {
                chunk_dimensions,
                chunk_index_type,
                ..
            } => {
                assert_eq!(chunk_dimensions, vec![10, 8]);
                assert_eq!(chunk_index_type, Some(3)); // Fixed Array
            }
            other => panic!("expected Chunked, got {other:?}"),
        }
    }

    #[test]
    fn v4_chunked_single_chunk_no_filters() {
        let mut buf = vec![4u8, 2]; // version=4, class=2
        buf.push(0); // flags (no filters)
        buf.push(2); // dimensionality=2
        buf.push(4); // dim_size_encoded_length=4
        buf.extend_from_slice(&64u32.to_le_bytes()); // dim 0
        buf.extend_from_slice(&32u32.to_le_bytes()); // dim 1
        buf.push(1); // chunk_index_type=1 (single chunk)
        buf.extend_from_slice(&0x3000u64.to_le_bytes()); // chunk address
        let layout = DataLayout::parse(&buf, 8, 8).unwrap();
        assert_eq!(
            layout,
            DataLayout::Chunked {
                chunk_dimensions: vec![64, 32],
                btree_address: Some(0x3000),
                version: 4,
                chunk_index_type: Some(1),
                single_chunk_filtered_size: None,
                single_chunk_filter_mask: None,
            }
        );
    }

    #[test]
    fn v4_chunked_single_chunk_with_filters() {
        let mut buf = vec![4u8, 2]; // version=4, class=2
        buf.push(0x02); // flags bit 1 = single chunk with filter
        buf.push(1); // dimensionality=1
        buf.push(4); // dim_size_encoded_length=4
        buf.extend_from_slice(&128u32.to_le_bytes()); // dim 0
        buf.push(1); // chunk_index_type=1 (single chunk)
        // filters present: filtered_size(8) + filter_mask(4) + address(8)
        buf.extend_from_slice(&1024u64.to_le_bytes()); // filtered size
        buf.extend_from_slice(&0u32.to_le_bytes()); // filter mask
        buf.extend_from_slice(&0x4000u64.to_le_bytes()); // address
        let layout = DataLayout::parse(&buf, 8, 8).unwrap();
        assert_eq!(
            layout,
            DataLayout::Chunked {
                chunk_dimensions: vec![128],
                btree_address: Some(0x4000),
                version: 4,
                chunk_index_type: Some(1),
                single_chunk_filtered_size: Some(1024),
                single_chunk_filter_mask: Some(0),
            }
        );
    }

    #[test]
    fn invalid_version() {
        // v3-v5 are supported; v6 is not a real layout message version.
        let buf = vec![6u8, 0, 0, 0];
        let err = DataLayout::parse(&buf, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidLayoutVersion(6));
    }

    #[test]
    fn invalid_class_v3() {
        let buf = vec![3u8, 5];
        let err = DataLayout::parse(&buf, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidLayoutClass(5));
    }

    #[test]
    fn invalid_class_v4() {
        let buf = vec![4u8, 7];
        let err = DataLayout::parse(&buf, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidLayoutClass(7));
    }

    #[test]
    fn v3_contiguous_4byte_offsets() {
        let mut buf = vec![3u8, 1];
        buf.extend_from_slice(&0x800u32.to_le_bytes());
        buf.extend_from_slice(&24u32.to_le_bytes());
        let layout = DataLayout::parse(&buf, 4, 4).unwrap();
        assert_eq!(
            layout,
            DataLayout::Contiguous {
                address: Some(0x800),
                size: 24,
            }
        );
    }

    #[test]
    fn v4_virtual() {
        let mut buf = vec![4u8, 3]; // version=4, class=3 (virtual)
        buf.extend_from_slice(&0x5000u64.to_le_bytes()); // global heap address
        buf.extend_from_slice(&1u32.to_le_bytes()); // global heap index
        let layout = DataLayout::parse(&buf, 8, 8).unwrap();
        assert_eq!(
            layout,
            DataLayout::Virtual {
                version: 4,
                global_heap_address: Some(0x5000),
                global_heap_index: 1,
                mappings: Vec::new(),
            }
        );
    }

    #[test]
    fn v4_virtual_undefined_address() {
        let mut buf = vec![4u8, 3];
        buf.extend_from_slice(&[0xFF; 8]); // undefined address
        buf.extend_from_slice(&0u32.to_le_bytes());
        let layout = DataLayout::parse(&buf, 8, 8).unwrap();
        assert_eq!(
            layout,
            DataLayout::Virtual {
                version: 4,
                global_heap_address: None,
                global_heap_index: 0,
                mappings: Vec::new(),
            }
        );
    }

    #[test]
    fn parse_vds_mappings_same_file_v1() {
        // The exact global-heap block written by HDF5 2.0 for a same-file VDS
        // with two sources: src_a -> virtual[0:4], src_b -> virtual[4:8].
        let blob = [
            0x01u8, // block version 1
            0x02, 0, 0, 0, 0, 0, 0, 0, // nused = 2 (length_size = 8)
            // entry 0
            0x04, // same-file marker (replaces file name)
            0x73, 0x72, 0x63, 0x5f, 0x61, 0x00, // "src_a\0"
            0x03, 0, 0, 0, 0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // source sel = ALL
            0x02, 0, 0, 0, 0x03, 0, 0, 0, 0x01, 0x02, 0x01, 0, 0, 0, // virtual sel: HYPER v3
            0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x04, 0x00, // start0 stride1 count1 block4
            // entry 1
            0x04, 0x73, 0x72, 0x63, 0x5f, 0x62, 0x00, // "src_b\0"
            0x03, 0, 0, 0, 0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // source sel = ALL
            0x02, 0, 0, 0, 0x03, 0, 0, 0, 0x01, 0x02, 0x01, 0, 0, 0, // virtual sel: HYPER v3
            0x04, 0x00, 0x01, 0x00, 0x01, 0x00, 0x04, 0x00, // start4 stride1 count1 block4
            0x68, 0xf0, 0x3e, 0xe4, // checksum (ignored)
        ];
        let mappings = parse_vds_mappings(&blob, 8).unwrap();
        assert_eq!(mappings.len(), 2);
        assert_eq!(mappings[0].source_file, ".");
        assert_eq!(mappings[0].source_dataset, "src_a");
        assert_eq!(mappings[1].source_file, ".");
        assert_eq!(mappings[1].source_dataset, "src_b");

        // Virtual selections decode to [0:4] and [4:8].
        use crate::selection::Selection;
        let (v0, _) = Selection::decode_serialized(&mappings[0].virtual_selection).unwrap();
        let (v1, _) = Selection::decode_serialized(&mappings[1].virtual_selection).unwrap();
        assert_eq!(v0.iter_linear_1d(8).unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(v1.iter_linear_1d(8).unwrap(), vec![4, 5, 6, 7]);
    }

    #[test]
    fn parse_vds_mappings_external_v0() {
        // Block version 0 with an explicit (external) source file name.
        let blob = [
            0x00u8, // block version 0
            0x01, 0, 0, 0, 0, 0, 0, 0, // nused = 1
            0x73, 0x72, 0x63, 0x5f, 0x65, 0x78, 0x74, 0x2e, 0x68, 0x35, 0x00, // "src_ext.h5\0"
            0x64, 0x61, 0x74, 0x61, 0x00, // "data\0"
            0x03, 0, 0, 0, 0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // source sel = ALL
            0x03, 0, 0, 0, 0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // virtual sel = ALL
        ];
        let mappings = parse_vds_mappings(&blob, 8).unwrap();
        assert_eq!(mappings.len(), 1);
        assert_eq!(mappings[0].source_file, "src_ext.h5");
        assert_eq!(mappings[0].source_dataset, "data");
    }

    #[test]
    fn parse_vds_mappings_huge_nused_does_not_oom_or_panic() {
        // nused = u64::MAX with no entry data: must error, not pre-allocate or
        // overrun.
        let mut blob = vec![0x01u8];
        blob.extend_from_slice(&u64::MAX.to_le_bytes());
        assert!(parse_vds_mappings(&blob, 8).is_err());
    }

    #[test]
    fn parse_vds_mappings_truncated_selection_does_not_overrun() {
        // One entry whose source selection (ALL) is truncated to 8 of 16 bytes.
        let blob = [
            0x01u8, // version 1
            0x01, 0, 0, 0, 0, 0, 0, 0, // nused = 1
            0x04, // same-file marker
            0x78, 0x00, // "x\0"
            0x03, 0, 0, 0, 0x01, 0, 0, 0, // ALL header, truncated (8 of 16 bytes)
        ];
        assert!(parse_vds_mappings(&blob, 8).is_err());
    }

    #[test]
    fn parse_vds_mappings_empty_is_ok_empty() {
        assert!(parse_vds_mappings(&[], 8).unwrap().is_empty());
        // Header present, nused = 0.
        let blob = [0x01u8, 0, 0, 0, 0, 0, 0, 0, 0];
        assert!(parse_vds_mappings(&blob, 8).unwrap().is_empty());
    }
}
