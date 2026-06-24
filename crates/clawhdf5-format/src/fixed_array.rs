//! HDF5 Fixed Array index parsing for chunked datasets (v4 index type 3).

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};

use crate::chunked_read::ChunkInfo;
use crate::error::FormatError;

/// Parsed Fixed Array header (FAHD).
#[derive(Debug, Clone)]
pub struct FixedArrayHeader {
    /// Client ID: 0 = non-filtered chunks, 1 = filtered chunks.
    pub client_id: u8,
    /// Size of each array element in bytes.
    pub element_size: u8,
    /// Log2 of max number of elements in a data block page.
    pub max_nelmts_bits: u8,
    /// Total number of elements (chunks) in the array.
    pub num_elements: u64,
    /// Address of the data block.
    pub data_block_address: u64,
}

fn read_offset(data: &[u8], pos: usize, size: u8) -> Result<u64, FormatError> {
    let s = size as usize;
    if pos.checked_add(s).is_none_or(|end| end > data.len()) {
        return Err(FormatError::UnexpectedEof {
            expected: pos.saturating_add(s),
            available: data.len(),
        });
    }
    let slice = &data[pos..pos + s];
    Ok(match size {
        2 => u16::from_le_bytes([slice[0], slice[1]]) as u64,
        4 => u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]) as u64,
        8 => u64::from_le_bytes([
            slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
        ]),
        _ => return Err(FormatError::InvalidOffsetSize(size)),
    })
}

fn read_length(data: &[u8], pos: usize, size: u8) -> Result<u64, FormatError> {
    read_offset(data, pos, size)
}

fn is_undefined(data: &[u8], pos: usize, size: u8) -> bool {
    let s = size as usize;
    if pos + s > data.len() {
        return false;
    }
    data[pos..pos + s].iter().all(|&b| b == 0xFF)
}

impl FixedArrayHeader {
    /// Parse a Fixed Array header from file data at the given offset.
    pub fn parse(
        file_data: &[u8],
        offset: usize,
        offset_size: u8,
        length_size: u8,
    ) -> Result<Self, FormatError> {
        // FAHD signature(4) + version(1) + client_id(1) + element_size(1) +
        // max_nelmts_bits(1) + num_elements(length_size) + data_block_addr(offset_size) + checksum(4)
        let min_size = 4 + 1 + 1 + 1 + 1 + length_size as usize + offset_size as usize + 4;
        if offset + min_size > file_data.len() {
            return Err(FormatError::UnexpectedEof {
                expected: offset + min_size,
                available: file_data.len(),
            });
        }

        let d = &file_data[offset..];
        if &d[0..4] != b"FAHD" {
            return Err(FormatError::ChunkedReadError(
                "invalid Fixed Array header signature".into(),
            ));
        }

        let version = d[4];
        if version != 0 {
            return Err(FormatError::ChunkedReadError(format!(
                "unsupported Fixed Array header version: {version}"
            )));
        }

        let client_id = d[5];
        let element_size = d[6];
        let max_nelmts_bits = d[7];

        let mut pos = 8;
        let num_elements = read_length(d, pos, length_size)?;
        pos += length_size as usize;
        let data_block_address = read_offset(d, pos, offset_size)?;

        Ok(FixedArrayHeader {
            client_id,
            element_size,
            max_nelmts_bits,
            num_elements,
            data_block_address,
        })
    }
}

/// Read chunk records from a Fixed Array data block.
///
/// Returns a `Vec<ChunkInfo>` with one entry per allocated chunk.
/// `chunk_dimensions` should be the spatial chunk dims only (not including the element-size dim).
/// `element_size` is the datatype size in bytes.
#[allow(clippy::too_many_arguments)]
pub fn read_fixed_array_chunks(
    file_data: &[u8],
    header: &FixedArrayHeader,
    dataset_dims: &[u64],
    chunk_dimensions: &[u32],
    element_size: u32,
    offset_size: u8,
    _length_size: u8,
) -> Result<Vec<ChunkInfo>, FormatError> {
    let db_offset = header.data_block_address as usize;
    let rank = chunk_dimensions.len();

    // Parse data block header: FADB(4) + version(1) + client_id(1) + header_address(offset_size)
    let db_header_size = 4 + 1 + 1 + offset_size as usize;
    if db_offset + db_header_size > file_data.len() {
        return Err(FormatError::UnexpectedEof {
            expected: db_offset + db_header_size,
            available: file_data.len(),
        });
    }

    let d = &file_data[db_offset..];
    if &d[0..4] != b"FADB" {
        return Err(FormatError::ChunkedReadError(
            "invalid Fixed Array data block signature".into(),
        ));
    }

    // Elements start immediately after the data block prefix.
    let elements_start = db_offset + db_header_size;

    let num_elements = header.num_elements as usize;
    // A chunk index cannot describe more elements than the file has bytes (each
    // element occupies at least `offset_size` bytes). Reject a corrupt count
    // before it can drive a huge loop or overflow an offset computation.
    if num_elements > file_data.len() {
        return Err(FormatError::ChunkedReadError(
            "Fixed Array element count exceeds file size".into(),
        ));
    }
    let os = offset_size as usize;
    // On-disk stride of one element. For non-filtered arrays the element is just
    // the chunk address (== offset_size); for filtered arrays it is
    // address + chunk_size + filter_mask (== header.element_size).
    let elem_stride = (header.element_size as usize).max(os);

    // Absolute file offset of element `idx` within a run starting at `base`,
    // with overflow surfaced as a clean error rather than a panic/wrap.
    let elem_at = |base: usize, idx: usize| -> Result<usize, FormatError> {
        idx.checked_mul(elem_stride)
            .and_then(|o| base.checked_add(o))
            .ok_or(FormatError::ChunkedReadError(
                "Fixed Array element offset overflow".into(),
            ))
    };

    // Compute chunk offsets based on index.
    // Chunks are stored in row-major order within the dataset space.
    let mut num_chunks_per_dim = Vec::with_capacity(rank);
    for d_idx in 0..rank {
        let ch_dim = chunk_dimensions[d_idx] as u64;
        if ch_dim == 0 {
            return Err(FormatError::ChunkedReadError(
                "chunk dimension is zero".into(),
            ));
        }
        let ds_dim = dataset_dims[d_idx];
        num_chunks_per_dim.push(ds_dim.div_ceil(ch_dim));
    }

    let chunk_byte_size: u64 =
        chunk_dimensions.iter().map(|&d| d as u64).product::<u64>() * element_size as u64;

    let mut chunks = Vec::new();
    let push_element =
        |i: usize, abs: usize, chunks: &mut Vec<ChunkInfo>| -> Result<(), FormatError> {
            if let Some((address, chunk_size, filter_mask)) = parse_fa_element(
                file_data,
                abs,
                header.client_id,
                offset_size,
                header.element_size,
                chunk_byte_size,
            )? {
                let offsets = index_to_chunk_offsets(i, &num_chunks_per_dim, chunk_dimensions);
                chunks.push(ChunkInfo {
                    chunk_size,
                    filter_mask,
                    offsets,
                    address,
                });
            }
            Ok(())
        };

    // A data block is paged when it holds more elements than fit in one page.
    // `max_nelmts_bits` is an untrusted u8; a shift >= the pointer width would
    // panic, so reject it (real page-size bits are tiny — 10 by default).
    if header.max_nelmts_bits as u32 >= usize::BITS {
        return Err(FormatError::ChunkedReadError(
            "Fixed Array max_nelmts_bits too large".into(),
        ));
    }
    let page_nelmts = 1usize << header.max_nelmts_bits;
    let is_paged = num_elements > page_nelmts;

    if !is_paged {
        // Non-paged: prefix, then `num_elements` elements packed directly,
        // then a trailing checksum (which we don't validate).
        for i in 0..num_elements {
            push_element(i, elem_at(elements_start, i)?, &mut chunks)?;
        }
        return Ok(chunks);
    }

    // Paged layout: prefix, then a page-init bitmap (one bit per page, MSB-first
    // within each byte), then a 4-byte checksum, then the pages. Every page
    // occupies a full slot of `page_nelmts` elements plus a 4-byte checksum;
    // only the final page holds fewer elements. Uninitialized pages (bit clear)
    // still occupy their slot on disk but are zero-filled, so the bitmap — not a
    // 0xFF sentinel — is what marks a whole page as unallocated.
    let stride_overflow =
        || FormatError::ChunkedReadError("Fixed Array page offset overflow".into());
    let npages = num_elements.div_ceil(page_nelmts);
    let bitmap_size = npages.div_ceil(8);
    let bitmap_start = elements_start;
    // prefix(db_header_size) + bitmap + checksum(4)
    let pages_start = db_offset + db_header_size + bitmap_size + 4;
    let page_stride = page_nelmts
        .checked_mul(elem_stride)
        .and_then(|x| x.checked_add(4))
        .ok_or_else(stride_overflow)?;

    if bitmap_start + bitmap_size > file_data.len() {
        return Err(FormatError::UnexpectedEof {
            expected: bitmap_start + bitmap_size,
            available: file_data.len(),
        });
    }

    for p in 0..npages {
        let page_first = p * page_nelmts; // < num_elements, cannot overflow
        let page_count = core::cmp::min(page_nelmts, num_elements - page_first);

        // Check the page-init bit (MSB-first within each byte).
        let bit_byte = file_data[bitmap_start + p / 8];
        let bit_mask = 1u8 << (7 - (p % 8));
        if bit_byte & bit_mask == 0 {
            continue; // entire page unallocated
        }

        let page_off = p
            .checked_mul(page_stride)
            .and_then(|o| pages_start.checked_add(o))
            .ok_or_else(stride_overflow)?;
        for e in 0..page_count {
            push_element(page_first + e, elem_at(page_off, e)?, &mut chunks)?;
        }
    }

    Ok(chunks)
}

/// Parse a single Fixed Array element at absolute file offset `abs`.
///
/// Returns `Some((address, chunk_size, filter_mask))` for an allocated chunk, or
/// `None` if the element is undefined (an unallocated chunk, address all-`0xFF`).
fn parse_fa_element(
    file_data: &[u8],
    abs: usize,
    client_id: u8,
    offset_size: u8,
    element_size: u8,
    chunk_byte_size: u64,
) -> Result<Option<(u64, u32, u32)>, FormatError> {
    let os = offset_size as usize;
    if client_id == 0 {
        // Non-filtered: element is just the chunk address.
        if abs + os > file_data.len() {
            return Err(FormatError::UnexpectedEof {
                expected: abs + os,
                available: file_data.len(),
            });
        }
        if is_undefined(file_data, abs, offset_size) {
            return Ok(None);
        }
        let address = read_offset(file_data, abs, offset_size)?;
        Ok(Some((address, chunk_byte_size as u32, 0)))
    } else {
        // Filtered: address(offset_size) + chunk_size(variable) + filter_mask(4)
        let es = element_size as usize;
        if es < os + 4 {
            return Err(FormatError::ChunkedReadError(
                "element_size too small for filtered element".into(),
            ));
        }
        let chunk_size_bytes = es - os - 4;
        if abs + es > file_data.len() {
            return Err(FormatError::UnexpectedEof {
                expected: abs + es,
                available: file_data.len(),
            });
        }
        if is_undefined(file_data, abs, offset_size) {
            return Ok(None);
        }
        let address = read_offset(file_data, abs, offset_size)?;
        let chunk_size = read_variable_length(&file_data[abs + os..], chunk_size_bytes)?;
        let fm_off = abs + os + chunk_size_bytes;
        let filter_mask = u32::from_le_bytes([
            file_data[fm_off],
            file_data[fm_off + 1],
            file_data[fm_off + 2],
            file_data[fm_off + 3],
        ]);
        Ok(Some((address, chunk_size as u32, filter_mask)))
    }
}

/// Convert a linear chunk index to N-dimensional chunk offsets in dataset space.
fn index_to_chunk_offsets(
    index: usize,
    num_chunks_per_dim: &[u64],
    chunk_dimensions: &[u32],
) -> Vec<u64> {
    let rank = num_chunks_per_dim.len();
    let mut offsets = vec![0u64; rank];
    let mut remaining = index as u64;
    for d in (0..rank).rev() {
        let nchunks = num_chunks_per_dim[d];
        if nchunks == 0 {
            continue;
        }
        let chunk_idx = remaining % nchunks;
        remaining /= nchunks;
        offsets[d] = chunk_idx * chunk_dimensions[d] as u64;
    }
    offsets
}

/// Read a variable-length little-endian unsigned integer.
fn read_variable_length(data: &[u8], size: usize) -> Result<u64, FormatError> {
    if size > 8 || data.len() < size {
        return Err(FormatError::ChunkedReadError(
            "invalid variable-length size".into(),
        ));
    }
    let mut val = 0u64;
    for (i, &byte) in data.iter().enumerate().take(size) {
        val |= (byte as u64) << (i * 8);
    }
    Ok(val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_to_offsets_1d() {
        let num_chunks = vec![5u64];
        let chunk_dims = vec![20u32];
        assert_eq!(index_to_chunk_offsets(0, &num_chunks, &chunk_dims), vec![0]);
        assert_eq!(
            index_to_chunk_offsets(1, &num_chunks, &chunk_dims),
            vec![20]
        );
        assert_eq!(
            index_to_chunk_offsets(4, &num_chunks, &chunk_dims),
            vec![80]
        );
    }

    #[test]
    fn index_to_offsets_2d() {
        // 10x6 dataset with 4x3 chunks => ceil(10/4)=3, ceil(6/3)=2 => 6 chunks
        let num_chunks = vec![3u64, 2];
        let chunk_dims = vec![4u32, 3];
        assert_eq!(
            index_to_chunk_offsets(0, &num_chunks, &chunk_dims),
            vec![0, 0]
        );
        assert_eq!(
            index_to_chunk_offsets(1, &num_chunks, &chunk_dims),
            vec![0, 3]
        );
        assert_eq!(
            index_to_chunk_offsets(2, &num_chunks, &chunk_dims),
            vec![4, 0]
        );
        assert_eq!(
            index_to_chunk_offsets(3, &num_chunks, &chunk_dims),
            vec![4, 3]
        );
        assert_eq!(
            index_to_chunk_offsets(5, &num_chunks, &chunk_dims),
            vec![8, 3]
        );
    }

    #[test]
    fn read_variable_length_values() {
        assert_eq!(read_variable_length(&[0x78, 0x56], 2).unwrap(), 0x5678);
        assert_eq!(
            read_variable_length(&[0x01, 0x02, 0x03, 0x04], 4).unwrap(),
            0x04030201
        );
        assert_eq!(read_variable_length(&[0xFF], 1).unwrap(), 0xFF);
    }

    #[test]
    fn parse_fixed_array_header_valid() {
        let mut buf = vec![0u8; 256];
        // FAHD signature
        buf[0..4].copy_from_slice(b"FAHD");
        buf[4] = 0; // version
        buf[5] = 1; // client_id = filtered
        buf[6] = 16; // element_size
        buf[7] = 10; // max_nelmts_bits (page_size = 1024)
        // num_elements (length_size=8)
        buf[8..16].copy_from_slice(&5u64.to_le_bytes());
        // data_block_address (offset_size=8)
        buf[16..24].copy_from_slice(&0x1000u64.to_le_bytes());
        // checksum (4 bytes, we don't validate in parse)

        let header = FixedArrayHeader::parse(&buf, 0, 8, 8).unwrap();
        assert_eq!(header.client_id, 1);
        assert_eq!(header.element_size, 16);
        assert_eq!(header.max_nelmts_bits, 10);
        assert_eq!(header.num_elements, 5);
        assert_eq!(header.data_block_address, 0x1000);
    }

    #[test]
    fn parse_fixed_array_header_invalid_signature() {
        let mut buf = vec![0u8; 256];
        buf[0..4].copy_from_slice(b"XXXX");
        let result = FixedArrayHeader::parse(&buf, 0, 8, 8);
        assert!(result.is_err());
    }

    /// Malformed headers must error, never panic (shift overflow, huge counts).
    #[test]
    fn read_rejects_oversized_max_nelmts_bits() {
        let mut buf = vec![0u8; 512];
        let fahd = 0x40usize;
        buf[fahd..fahd + 4].copy_from_slice(b"FAHD");
        buf[fahd + 4] = 0; // version
        buf[fahd + 5] = 0; // client_id
        buf[fahd + 6] = 8; // element_size
        buf[fahd + 7] = 200; // max_nelmts_bits — absurd, would overflow a shift
        buf[fahd + 8..fahd + 16].copy_from_slice(&3u64.to_le_bytes()); // num_elements
        buf[fahd + 16..fahd + 24].copy_from_slice(&0x100u64.to_le_bytes());
        // FADB so parsing reaches the paged check
        let db = 0x100usize;
        buf[db..db + 4].copy_from_slice(b"FADB");
        let header = FixedArrayHeader::parse(&buf, fahd, 8, 8).unwrap();
        let r = read_fixed_array_chunks(&buf, &header, &[100], &[20], 8, 8, 8);
        assert!(r.is_err());
    }

    #[test]
    fn read_rejects_num_elements_larger_than_file() {
        let mut buf = vec![0u8; 256];
        let fahd = 0x40usize;
        buf[fahd..fahd + 4].copy_from_slice(b"FAHD");
        buf[fahd + 6] = 8;
        buf[fahd + 7] = 10;
        buf[fahd + 8..fahd + 16].copy_from_slice(&u64::MAX.to_le_bytes()); // absurd count
        buf[fahd + 16..fahd + 24].copy_from_slice(&0x80u64.to_le_bytes());
        buf[0x80..0x84].copy_from_slice(b"FADB");
        let header = FixedArrayHeader::parse(&buf, fahd, 8, 8).unwrap();
        let r = read_fixed_array_chunks(&buf, &header, &[100], &[20], 8, 8, 8);
        assert!(r.is_err());
    }

    #[test]
    fn parse_fixed_array_header_invalid_version() {
        let mut buf = vec![0u8; 256];
        buf[0..4].copy_from_slice(b"FAHD");
        buf[4] = 1; // unsupported version
        let result = FixedArrayHeader::parse(&buf, 0, 8, 8);
        assert!(result.is_err());
    }

    /// Build a synthetic Fixed Array (non-filtered) and verify reading.
    #[test]
    fn read_non_filtered_chunks() {
        let offset_size: u8 = 8;
        let length_size: u8 = 8;
        let os = offset_size as usize;
        let num_chunks = 5u64;

        let mut file_data = vec![0u8; 0x3000];

        // Build FAHD at offset 0x100
        let fahd_offset = 0x100usize;
        let db_offset = 0x200usize;
        file_data[fahd_offset..fahd_offset + 4].copy_from_slice(b"FAHD");
        file_data[fahd_offset + 4] = 0; // version
        file_data[fahd_offset + 5] = 0; // client_id = non-filtered
        file_data[fahd_offset + 6] = os as u8; // element_size = just address
        file_data[fahd_offset + 7] = 10; // max_nelmts_bits
        file_data[fahd_offset + 8..fahd_offset + 16].copy_from_slice(&num_chunks.to_le_bytes());
        file_data[fahd_offset + 16..fahd_offset + 24]
            .copy_from_slice(&(db_offset as u64).to_le_bytes());

        // Build FADB at db_offset
        file_data[db_offset..db_offset + 4].copy_from_slice(b"FADB");
        file_data[db_offset + 4] = 0; // version
        file_data[db_offset + 5] = 0; // client_id
        file_data[db_offset + 6..db_offset + 14]
            .copy_from_slice(&(fahd_offset as u64).to_le_bytes()); // header_address

        // Elements: 5 addresses
        let elem_start = db_offset + 6 + os;
        let base_addr = 0x1000u64;
        let chunk_byte_size = 20 * 8; // 20 elements × 8 bytes
        for i in 0..5 {
            let addr = base_addr + i as u64 * chunk_byte_size as u64;
            let pos = elem_start + i * os;
            file_data[pos..pos + os].copy_from_slice(&addr.to_le_bytes());
        }

        let header =
            FixedArrayHeader::parse(&file_data, fahd_offset, offset_size, length_size).unwrap();
        let ds_dims = vec![100u64];
        let chunk_dims = vec![20u32];
        let chunks = read_fixed_array_chunks(
            &file_data,
            &header,
            &ds_dims,
            &chunk_dims,
            8,
            offset_size,
            length_size,
        )
        .unwrap();

        assert_eq!(chunks.len(), 5);
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.address, base_addr + i as u64 * chunk_byte_size as u64);
            assert_eq!(c.offsets, vec![i as u64 * 20]);
            assert_eq!(c.filter_mask, 0);
            assert_eq!(c.chunk_size, chunk_byte_size as u32);
        }
    }

    /// Build a synthetic Fixed Array (filtered) and verify reading.
    #[test]
    fn read_filtered_chunks() {
        let offset_size: u8 = 8;
        let length_size: u8 = 8;
        let os = offset_size as usize;
        let num_chunks = 3u64;
        // element_size for filtered: offset_size + chunk_size_bytes + 4(filter_mask)
        // chunk_size_bytes: let's use 4 bytes
        let chunk_size_bytes = 4usize;
        let elem_size = os + chunk_size_bytes + 4;

        let mut file_data = vec![0u8; 0x3000];

        let fahd_offset = 0x100usize;
        let db_offset = 0x200usize;
        file_data[fahd_offset..fahd_offset + 4].copy_from_slice(b"FAHD");
        file_data[fahd_offset + 4] = 0;
        file_data[fahd_offset + 5] = 1; // client_id = filtered
        file_data[fahd_offset + 6] = elem_size as u8;
        file_data[fahd_offset + 7] = 10;
        file_data[fahd_offset + 8..fahd_offset + 16].copy_from_slice(&num_chunks.to_le_bytes());
        file_data[fahd_offset + 16..fahd_offset + 24]
            .copy_from_slice(&(db_offset as u64).to_le_bytes());

        file_data[db_offset..db_offset + 4].copy_from_slice(b"FADB");
        file_data[db_offset + 4] = 0;
        file_data[db_offset + 5] = 1;
        file_data[db_offset + 6..db_offset + 14]
            .copy_from_slice(&(fahd_offset as u64).to_le_bytes());

        let elem_start = db_offset + 6 + os;
        let test_chunks = [
            (0x1000u64, 120u32, 0u32),
            (0x2000u64, 115u32, 0u32),
            (0x3000u64, 100u32, 0u32),
        ];

        for (i, &(addr, csize, fmask)) in test_chunks.iter().enumerate() {
            let pos = elem_start + i * elem_size;
            file_data[pos..pos + os].copy_from_slice(&addr.to_le_bytes());
            // chunk_size as 4 bytes LE
            file_data[pos + os..pos + os + 4].copy_from_slice(&csize.to_le_bytes());
            file_data[pos + os + 4..pos + os + 8].copy_from_slice(&fmask.to_le_bytes());
        }

        let header =
            FixedArrayHeader::parse(&file_data, fahd_offset, offset_size, length_size).unwrap();
        let ds_dims = vec![60u64];
        let chunk_dims = vec![20u32];
        let chunks = read_fixed_array_chunks(
            &file_data,
            &header,
            &ds_dims,
            &chunk_dims,
            8,
            offset_size,
            length_size,
        )
        .unwrap();

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].address, 0x1000);
        assert_eq!(chunks[0].chunk_size, 120);
        assert_eq!(chunks[0].filter_mask, 0);
        assert_eq!(chunks[0].offsets, vec![0]);
        assert_eq!(chunks[1].address, 0x2000);
        assert_eq!(chunks[1].chunk_size, 115);
        assert_eq!(chunks[2].address, 0x3000);
        assert_eq!(chunks[2].chunk_size, 100);
    }

    /// Build a synthetic *paged* Fixed Array (non-filtered) and verify reading.
    ///
    /// Layout reverse-engineered and confirmed against an HDF5 2.0 file:
    /// after the FADB prefix comes a page-init bitmap (MSB-first within each
    /// byte), a 4-byte checksum, then full-size page slots (`page_nelmts`
    /// elements + a 4-byte checksum each), with only the last page shorter.
    /// Uninitialized pages occupy their slot but are skipped via the bitmap.
    #[test]
    fn read_paged_non_filtered_chunks() {
        let offset_size: u8 = 8;
        let length_size: u8 = 8;
        let os = offset_size as usize;

        // page_nelmts = 1 << 2 = 4. Use 11 elements => 3 pages
        // (page0: 4, page1: 4, page2: 3 short). Initialize pages 0 and 2; leave
        // page 1 uninitialized. 3 pages still fits one bitmap byte, but we place
        // the set bits at positions 7 and 5 to lock the MSB-first ordering.
        let max_nelmts_bits = 2u8;
        let page_nelmts = 1usize << max_nelmts_bits; // 4
        let num_elements = 11u64;
        let db_header_size = 4 + 1 + 1 + os; // FADB sig+ver+client+header_addr
        let bitmap_size = 1usize; // ceil(3/8)
        let page_total = page_nelmts * os + 4; // elements + checksum

        let fahd_offset = 0x100usize;
        let db_offset = 0x400usize;
        let mut file_data = vec![0u8; 0x4000];

        // FAHD
        file_data[fahd_offset..fahd_offset + 4].copy_from_slice(b"FAHD");
        file_data[fahd_offset + 4] = 0; // version
        file_data[fahd_offset + 5] = 0; // client_id = non-filtered
        file_data[fahd_offset + 6] = os as u8; // element_size = address only
        file_data[fahd_offset + 7] = max_nelmts_bits;
        file_data[fahd_offset + 8..fahd_offset + 16].copy_from_slice(&num_elements.to_le_bytes());
        file_data[fahd_offset + 16..fahd_offset + 24]
            .copy_from_slice(&(db_offset as u64).to_le_bytes());

        // FADB prefix
        file_data[db_offset..db_offset + 4].copy_from_slice(b"FADB");
        file_data[db_offset + 4] = 0; // version
        file_data[db_offset + 5] = 0; // client_id
        file_data[db_offset + 6..db_offset + 6 + os]
            .copy_from_slice(&(fahd_offset as u64).to_le_bytes());

        // Page-init bitmap: pages 0 and 2 initialized, page 1 not.
        // MSB-first => page0 -> bit7 (0x80), page2 -> bit5 (0x20) => 0xA0.
        let bitmap_off = db_offset + db_header_size;
        file_data[bitmap_off] = 0b1010_0000;

        // Pages start after bitmap + 4-byte checksum.
        let pages_start = db_offset + db_header_size + bitmap_size + 4;

        let base_addr = 0x1000u64;
        // Page 0 (elements 0..4) and page 2 (elements 8..11) carry addresses;
        // page 1's slot is left zero-filled and must be skipped.
        for &p in &[0usize, 2usize] {
            let page_off = pages_start + p * page_total;
            let count = core::cmp::min(page_nelmts, num_elements as usize - p * page_nelmts);
            for e in 0..count {
                let i = p * page_nelmts + e;
                let addr = base_addr + i as u64 * 0x100;
                let pos = page_off + e * os;
                file_data[pos..pos + os].copy_from_slice(&addr.to_le_bytes());
            }
        }

        let header =
            FixedArrayHeader::parse(&file_data, fahd_offset, offset_size, length_size).unwrap();
        assert_eq!(header.num_elements, 11);

        let ds_dims = vec![11u64 * 20];
        let chunk_dims = vec![20u32];
        let chunks = read_fixed_array_chunks(
            &file_data,
            &header,
            &ds_dims,
            &chunk_dims,
            8,
            offset_size,
            length_size,
        )
        .unwrap();

        // Page 1 (elements 4,5,6,7) is uninitialized => skipped. The remaining
        // 7 chunks (0..4 and 8..11) come back with their original linear index.
        assert_eq!(chunks.len(), 7);
        let mut got: Vec<(u64, u64)> = chunks.iter().map(|c| (c.offsets[0], c.address)).collect();
        got.sort();
        let expect: Vec<(u64, u64)> = [0usize, 1, 2, 3, 8, 9, 10]
            .iter()
            .map(|&i| (i as u64 * 20, base_addr + i as u64 * 0x100))
            .collect();
        assert_eq!(got, expect);
    }
}
