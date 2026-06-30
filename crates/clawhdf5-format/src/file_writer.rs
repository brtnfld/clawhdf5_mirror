//! HDF5 file creation (write pipeline).
//!
//! Produces valid HDF5 files with v3 superblock, v2 object headers,
//! link messages, contiguous datasets, inline and dense attributes.

#[cfg(not(feature = "std"))]
use alloc::{string::String, string::ToString, vec, vec::Vec};

use crate::attribute::AttributeMessage;
use crate::chunked_write::{
    ChunkOptions, PrecompressedChunks, build_chunked_data_from_precompressed, precompress_chunks,
};
use crate::data_layout::VdsMapping;
use crate::dataspace::{Dataspace, DataspaceType};
use crate::error::FormatError;
use crate::link_message::{LinkMessage, LinkTarget};
use crate::message_type::MessageType;
use crate::metadata_index::{DatasetMetadata, MetadataBlock, MetadataIndex};
use crate::object_header_writer::ObjectHeaderWriter;
use crate::superblock::Superblock;
use crate::type_builders::{
    DatasetBuilder, FillTime, FinishedGroup, GroupBuilder, build_attr_message,
};

// Re-export public types that moved to type_builders for API compatibility.
#[cfg(feature = "provenance")]
pub use crate::type_builders::ProvenanceConfig;
pub use crate::type_builders::{AttrValue, CompoundTypeBuilder, EnumTypeBuilder};

use crate::datatype::{CharacterSet, Datatype};

pub(crate) const OFFSET_SIZE: u8 = 8;
pub(crate) const LENGTH_SIZE: u8 = 8;
const SUPERBLOCK_SIZE: usize = 48;

/// Threshold for switching from compact (inline) to dense attribute storage.
const DENSE_ATTR_THRESHOLD: usize = 8;

/// Threshold for switching a group from compact (inline Link messages) to dense
/// link storage (fractal heap + v2 B-tree), matching libhdf5's default
/// `max_compact` of 8 links.
const DENSE_LINK_THRESHOLD: usize = 8;

// ---- OH builders ----

pub(crate) fn build_chunked_dataset_oh(
    dt: &Datatype,
    ds: &Dataspace,
    layout_message: &[u8],
    pipeline_message: Option<&[u8]>,
    attrs: &[AttributeMessage],
    dense_blob: Option<&DenseAttrBlob>,
    fill_time: FillTime,
) -> Vec<u8> {
    let mut w = ObjectHeaderWriter::new();
    w.add_message_with_flags(MessageType::Datatype, dt.serialize(), 0x01);
    w.add_message(MessageType::Dataspace, ds.serialize(LENGTH_SIZE));
    w.add_message_with_flags(MessageType::FillValue, vec![3, fill_time.to_byte()], 0x01);
    w.add_message(MessageType::DataLayout, layout_message.to_vec());
    if let Some(pm) = pipeline_message {
        w.add_message(MessageType::FilterPipeline, pm.to_vec());
    }
    if let Some(blob) = dense_blob {
        w.add_message(MessageType::AttributeInfo, blob.attr_info_message.clone());
    } else {
        for attr in attrs {
            w.add_message(MessageType::Attribute, attr.serialize(LENGTH_SIZE));
        }
    }
    w.serialize()
}

pub(crate) fn build_dataset_oh(
    dt: &Datatype,
    ds: &Dataspace,
    data_addr: u64,
    data_size: u64,
    attrs: &[AttributeMessage],
    dense_blob: Option<&DenseAttrBlob>,
    fill_time: FillTime,
) -> Vec<u8> {
    let mut w = ObjectHeaderWriter::new();
    w.add_message_with_flags(MessageType::Datatype, dt.serialize(), 0x01);
    w.add_message(MessageType::Dataspace, ds.serialize(LENGTH_SIZE));
    w.add_message_with_flags(MessageType::FillValue, vec![3, fill_time.to_byte()], 0x01);
    let mut dl = Vec::new();
    dl.push(4); // version
    dl.push(1); // class = contiguous
    dl.extend_from_slice(&data_addr.to_le_bytes());
    dl.extend_from_slice(&data_size.to_le_bytes());
    w.add_message(MessageType::DataLayout, dl);
    if let Some(blob) = dense_blob {
        w.add_message(MessageType::AttributeInfo, blob.attr_info_message.clone());
    } else {
        for attr in attrs {
            w.add_message(MessageType::Attribute, attr.serialize(LENGTH_SIZE));
        }
    }
    w.serialize()
}

/// Build a compact dataset object header where data is stored inline.
pub(crate) fn build_compact_dataset_oh(
    dt: &Datatype,
    ds: &Dataspace,
    data: &[u8],
    attrs: &[AttributeMessage],
    dense_blob: Option<&DenseAttrBlob>,
    fill_time: FillTime,
) -> Vec<u8> {
    let mut w = ObjectHeaderWriter::new();
    w.add_message_with_flags(MessageType::Datatype, dt.serialize(), 0x01);
    w.add_message(MessageType::Dataspace, ds.serialize(LENGTH_SIZE));
    w.add_message_with_flags(MessageType::FillValue, vec![3, fill_time.to_byte()], 0x01);
    // Compact layout message: version=4, class=0, u16 size, inline data
    let mut dl = Vec::new();
    dl.push(4); // version
    dl.push(0); // class = compact
    dl.extend_from_slice(&(data.len() as u16).to_le_bytes());
    dl.extend_from_slice(data);
    w.add_message(MessageType::DataLayout, dl);
    if let Some(blob) = dense_blob {
        w.add_message(MessageType::AttributeInfo, blob.attr_info_message.clone());
    } else {
        for attr in attrs {
            w.add_message(MessageType::Attribute, attr.serialize(LENGTH_SIZE));
        }
    }
    w.serialize()
}

pub(crate) fn build_group_oh(
    links: &[LinkMessage],
    dense_link_info: Option<&[u8]>,
    attrs: &[AttributeMessage],
    dense_blob: Option<&DenseAttrBlob>,
) -> Vec<u8> {
    let mut w = ObjectHeaderWriter::new();
    if let Some(li) = dense_link_info {
        // Dense link storage: a LinkInfo pointing at the fractal heap + name
        // B-tree, and no inline Link messages.
        w.add_message(MessageType::LinkInfo, li.to_vec());
    } else {
        let mut li = Vec::new();
        li.push(0); // version
        li.push(0); // flags
        li.extend_from_slice(&u64::MAX.to_le_bytes()); // fractal heap addr = UNDEF
        li.extend_from_slice(&u64::MAX.to_le_bytes()); // btree name index addr = UNDEF
        w.add_message(MessageType::LinkInfo, li);
        for link in links {
            w.add_message(MessageType::Link, link.serialize(OFFSET_SIZE));
        }
    }
    if let Some(blob) = dense_blob {
        w.add_message(MessageType::AttributeInfo, blob.attr_info_message.clone());
    } else {
        for attr in attrs {
            w.add_message(MessageType::Attribute, attr.serialize(LENGTH_SIZE));
        }
    }
    w.serialize()
}

pub(crate) fn make_link(name: &str, addr: u64) -> LinkMessage {
    LinkMessage {
        name: name.to_string(),
        link_target: LinkTarget::Hard {
            object_header_address: addr,
        },
        creation_order: None,
        charset: CharacterSet::Ascii,
    }
}

pub(crate) fn make_external_link(name: &str, filename: &str, object_path: &str) -> LinkMessage {
    LinkMessage {
        name: name.to_string(),
        link_target: LinkTarget::External {
            filename: filename.to_string(),
            object_path: object_path.to_string(),
        },
        creation_order: None,
        charset: CharacterSet::Ascii,
    }
}

// ---- Dense attribute blob ----

/// Pre-built dense attribute storage (fractal heap + B-tree v2 + attribute info message).
pub(crate) struct DenseAttrBlob {
    /// Serialized AttributeInfo message data (to embed in the object header).
    pub(crate) attr_info_message: Vec<u8>,
    /// The combined fractal heap header + direct block + B-tree v2 bytes.
    pub(crate) blob: Vec<u8>,
}

/// A fractal heap holding a set of serialized objects, plus the heap IDs that
/// address them. Shared by dense attribute and dense link storage, which differ
/// only in their v2 B-tree record layout.
pub(crate) struct FractalHeapBlock {
    /// The complete heap bytes: FRHP header, then either a single root direct
    /// block, or a root indirect block (FHIB) followed by its direct blocks.
    blob: Vec<u8>,
    /// Address of the fractal heap header.
    frhp_addr: u64,
    /// Address where the v2 B-tree should be placed (right after the heap).
    btree_addr: u64,
    /// Heap ID for each object, in input order.
    heap_ids: Vec<Vec<u8>>,
    /// Heap ID length (bytes).
    heap_id_length: u16,
}

/// Build a fractal heap for `serialized` objects, laid out at `base_address`.
///
/// Uses a single root direct block when the data fits in one (≤ the maximum
/// direct block size), otherwise a root indirect block over multiple direct
/// blocks following the doubling table. The caller builds the matching v2
/// B-tree (type 5 for links, type 8 for attributes) at the returned
/// `btree_addr`.
pub(crate) fn build_single_block_fractal_heap(
    serialized: &[Vec<u8>],
    base_address: u64,
    max_heap_size: u16,
    heap_id_length: u16,
) -> FractalHeapBlock {
    let os = OFFSET_SIZE as usize;
    let ls = LENGTH_SIZE as usize;
    let block_offset_bytes = (max_heap_size as usize).div_ceil(8);
    let max_direct_block_size: u64 = 65536;

    // Direct block layout: sig(4) + ver(1) + heap_addr(os) + block_offset(bo_bytes)
    //   + checksum(4) [when flags bit 1 set] + data...
    let dblock_header_size = 4 + 1 + os + block_offset_bytes + 4; // +4 for checksum
    let total_data_size: usize = serialized.iter().map(|s| s.len()).sum();
    let dblock_content_size = dblock_header_size + total_data_size;
    let starting_block_size = dblock_content_size.next_power_of_two().max(512) as u64;

    // When the objects don't fit in a single direct block, fall back to a
    // multi-block heap with a root indirect block.
    if starting_block_size > max_direct_block_size {
        return build_multiblock_fractal_heap(
            serialized,
            base_address,
            max_heap_size,
            heap_id_length,
        );
    }

    // Fractal heap header size
    let frhp_size = 4
        + 1
        + 2
        + 2
        + 1
        + 4
        + ls
        + os
        + ls
        + os
        + ls
        + ls
        + ls
        + ls
        + ls
        + ls
        + ls
        + ls
        + 2
        + ls
        + ls
        + 2
        + 2
        + os
        + 2
        + 4;

    let frhp_addr = base_address;
    let dblock_addr = frhp_addr + frhp_size as u64;
    let btree_addr = dblock_addr + starting_block_size;

    let data_space = starting_block_size as usize - dblock_header_size;
    let free_space = data_space - total_data_size;

    // Build fractal heap header
    let mut frhp = Vec::with_capacity(frhp_size);
    frhp.extend_from_slice(b"FRHP");
    frhp.push(0); // version
    frhp.extend_from_slice(&heap_id_length.to_le_bytes());
    frhp.extend_from_slice(&0u16.to_le_bytes()); // io_filter_encoded_length
    frhp.push(0x02); // flags: bit 1 = checksum direct blocks
    let max_managed = max_direct_block_size as u32 - dblock_header_size as u32;
    frhp.extend_from_slice(&max_managed.to_le_bytes());
    write_length(&mut frhp, 0, LENGTH_SIZE); // next_huge_object_id
    write_undef_offset(&mut frhp, OFFSET_SIZE); // btree_huge_objects_address
    write_length(&mut frhp, free_space as u64, LENGTH_SIZE); // free_space_managed_blocks
    write_undef_offset(&mut frhp, OFFSET_SIZE); // free_space_mgr_addr
    write_length(&mut frhp, starting_block_size, LENGTH_SIZE); // managed_space_in_heap
    write_length(&mut frhp, starting_block_size, LENGTH_SIZE); // allocated_managed_space
    write_length(&mut frhp, 0, LENGTH_SIZE); // dblock_alloc_iter
    write_length(&mut frhp, serialized.len() as u64, LENGTH_SIZE); // managed_objects_count
    write_length(&mut frhp, 0, LENGTH_SIZE); // huge_objects_size
    write_length(&mut frhp, 0, LENGTH_SIZE); // huge_objects_count
    write_length(&mut frhp, 0, LENGTH_SIZE); // tiny_objects_size
    write_length(&mut frhp, 0, LENGTH_SIZE); // tiny_objects_count
    frhp.extend_from_slice(&4u16.to_le_bytes()); // table_width
    write_length(&mut frhp, starting_block_size, LENGTH_SIZE);
    write_length(&mut frhp, max_direct_block_size, LENGTH_SIZE); // max_direct_block_size
    frhp.extend_from_slice(&max_heap_size.to_le_bytes());
    let sri: u16 = 1;
    frhp.extend_from_slice(&sri.to_le_bytes()); // starting_row_of_indirect_blocks
    write_offset(&mut frhp, dblock_addr, OFFSET_SIZE);
    frhp.extend_from_slice(&0u16.to_le_bytes()); // root is direct block
    let frhp_checksum = crate::checksum::jenkins_lookup3(&frhp);
    frhp.extend_from_slice(&frhp_checksum.to_le_bytes());
    debug_assert_eq!(frhp.len(), frhp_size);

    // Build direct block: header (with checksum) + data + padding
    let mut dblock = Vec::with_capacity(starting_block_size as usize);
    dblock.extend_from_slice(b"FHDB");
    dblock.push(0); // version
    write_offset(&mut dblock, frhp_addr, OFFSET_SIZE);
    dblock.extend_from_slice(&vec![0u8; block_offset_bytes]); // block_offset = 0 for root
    let cksum_pos = dblock.len();
    dblock.extend_from_slice(&[0u8; 4]); // checksum placeholder
    debug_assert_eq!(dblock.len(), dblock_header_size);

    // Data area starts after header
    let mut obj_offsets: Vec<(u64, u64)> = Vec::with_capacity(serialized.len());
    for s in serialized {
        let offset_in_heap = dblock.len() as u64;
        obj_offsets.push((offset_in_heap, s.len() as u64));
        dblock.extend_from_slice(s);
    }

    // Pad to full block size
    dblock.resize(starting_block_size as usize, 0);

    // Checksum: computed over entire block with checksum field zeroed
    let dblock_checksum = crate::checksum::jenkins_lookup3(&dblock);
    dblock[cksum_pos..cksum_pos + 4].copy_from_slice(&dblock_checksum.to_le_bytes());
    debug_assert_eq!(dblock.len(), starting_block_size as usize);

    // Build heap IDs
    let heap_ids: Vec<Vec<u8>> = obj_offsets
        .iter()
        .map(|(off, len)| encode_managed_id(*off, *len, max_heap_size, heap_id_length))
        .collect();

    let mut blob = Vec::with_capacity(frhp.len() + dblock.len());
    blob.extend_from_slice(&frhp);
    blob.extend_from_slice(&dblock);

    FractalHeapBlock {
        blob,
        frhp_addr,
        btree_addr,
        heap_ids,
        heap_id_length,
    }
}

/// Build a multi-block fractal heap: a root indirect block (FHIB) over multiple
/// direct blocks sized by the doubling table. Used when the objects don't fit
/// in a single direct block. Objects do not span blocks (no huge-object path).
fn build_multiblock_fractal_heap(
    serialized: &[Vec<u8>],
    base_address: u64,
    max_heap_size: u16,
    heap_id_length: u16,
) -> FractalHeapBlock {
    let os = OFFSET_SIZE as usize;
    let block_offset_bytes = (max_heap_size as usize).div_ceil(8);
    let max_direct_block_size: u64 = 65536;
    let table_width: u16 = 4;
    let starting_block_size: u64 = 512;
    let dblock_header_size = 4 + 1 + os + block_offset_bytes + 4;
    let block_capacity = |row: usize| block_size_for_row(starting_block_size, row) - dblock_header_size as u64;

    // ---- Pack objects into direct blocks (row-major over the doubling table) ----
    struct Blk {
        row: usize,
        size: u64,
        heap_offset: u64,
        data: Vec<u8>,
    }
    let mut blocks: Vec<Blk> = Vec::new();
    // Each object's (heap_offset, length) for the heap ID.
    let mut obj_loc: Vec<(u64, u64)> = vec![(0, 0); serialized.len()];

    let mut row = 0usize;
    let mut col = 0u16;
    let mut heap_off = 0u64;
    let mut cur: Option<Blk> = None;

    for (idx, s) in serialized.iter().enumerate() {
        loop {
            if cur.is_none() {
                let size = block_size_for_row(starting_block_size, row);
                cur = Some(Blk {
                    row,
                    size,
                    heap_offset: heap_off,
                    data: Vec::new(),
                });
            }
            let blk = cur.as_mut().unwrap();
            let cap = block_capacity(blk.row) as usize;
            if !blk.data.is_empty() && blk.data.len() + s.len() > cap {
                // Doesn't fit; finalize this block and advance to the next slot.
                let finished = cur.take().unwrap();
                heap_off += finished.size;
                blocks.push(finished);
                col += 1;
                if col >= table_width {
                    col = 0;
                    row += 1;
                }
                continue;
            }
            // Place the object (a fresh block always accepts at least one object
            // up to its capacity; objects larger than a max block are unsupported).
            let pos_in_block = dblock_header_size + blk.data.len();
            obj_loc[idx] = (blk.heap_offset + pos_in_block as u64, s.len() as u64);
            blk.data.extend_from_slice(s);
            break;
        }
    }
    if let Some(b) = cur.take() {
        blocks.push(b);
    }

    let cur_rows = (blocks.last().map(|b| b.row).unwrap_or(0) + 1) as u16;

    // ---- Addresses ----
    let frhp_size = frhp_header_size(os, LENGTH_SIZE as usize);
    let frhp_addr = base_address;
    let fhib_addr = frhp_addr + frhp_size as u64;
    let fhib_entries = cur_rows as usize * table_width as usize;
    let fhib_size = 5 + os + block_offset_bytes + fhib_entries * os + 4;
    let first_dblock_addr = fhib_addr + fhib_size as u64;

    // Assign each used block an address (laid out consecutively after the FHIB).
    let mut blk_addrs: Vec<u64> = Vec::with_capacity(blocks.len());
    let mut a = first_dblock_addr;
    for b in &blocks {
        blk_addrs.push(a);
        a += b.size;
    }
    let heap_end = a;
    let btree_addr = heap_end;

    // Bookkeeping totals.
    let managed_space: u64 = (0..cur_rows as usize)
        .map(|r| block_size_for_row(starting_block_size, r) * table_width as u64)
        .sum();
    let alloc_space: u64 = blocks.iter().map(|b| b.size).sum();
    let used: u64 = blocks
        .iter()
        .map(|b| dblock_header_size as u64 + b.data.len() as u64)
        .sum();
    let free_space = alloc_space.saturating_sub(used);

    // ---- FRHP header ----
    let max_managed = max_direct_block_size as u32 - dblock_header_size as u32;
    let frhp = write_frhp(WriteFrhp {
        heap_id_length,
        max_managed,
        free_space,
        managed_space,
        alloc_space,
        nobjects: serialized.len() as u64,
        table_width,
        starting_block_size,
        max_direct_block_size,
        max_heap_size,
        root_addr: fhib_addr,
        cur_rows,
    });
    debug_assert_eq!(frhp.len(), frhp_size);

    // ---- Root indirect block (FHIB) ----
    let mut fhib = Vec::with_capacity(fhib_size);
    fhib.extend_from_slice(b"FHIB");
    fhib.push(0); // version
    write_offset(&mut fhib, frhp_addr, OFFSET_SIZE);
    fhib.extend_from_slice(&vec![0u8; block_offset_bytes]); // block offset = 0 (root)
    for &addr in &blk_addrs {
        write_offset(&mut fhib, addr, OFFSET_SIZE);
    }
    // Remaining slots within the current rows are unallocated.
    for _ in blk_addrs.len()..fhib_entries {
        write_undef_offset(&mut fhib, OFFSET_SIZE);
    }
    let fhib_checksum = crate::checksum::jenkins_lookup3(&fhib);
    fhib.extend_from_slice(&fhib_checksum.to_le_bytes());
    debug_assert_eq!(fhib.len(), fhib_size);

    // ---- Direct blocks ----
    let mut blob = frhp;
    blob.extend_from_slice(&fhib);
    for b in &blocks {
        let mut dblock = Vec::with_capacity(b.size as usize);
        dblock.extend_from_slice(b"FHDB");
        dblock.push(0); // version
        write_offset(&mut dblock, frhp_addr, OFFSET_SIZE);
        let mut bo = b.heap_offset.to_le_bytes().to_vec();
        bo.truncate(block_offset_bytes);
        dblock.extend_from_slice(&bo);
        let cksum_pos = dblock.len();
        dblock.extend_from_slice(&[0u8; 4]); // checksum placeholder
        dblock.extend_from_slice(&b.data);
        dblock.resize(b.size as usize, 0);
        let cksum = crate::checksum::jenkins_lookup3(&dblock);
        dblock[cksum_pos..cksum_pos + 4].copy_from_slice(&cksum.to_le_bytes());
        blob.extend_from_slice(&dblock);
    }

    let heap_ids: Vec<Vec<u8>> = obj_loc
        .iter()
        .map(|(off, len)| encode_managed_id(*off, *len, max_heap_size, heap_id_length))
        .collect();

    FractalHeapBlock {
        blob,
        frhp_addr,
        btree_addr,
        heap_ids,
        heap_id_length,
    }
}

/// Doubling-table block size for `row`: rows 0 and 1 share the starting size;
/// row r (r ≥ 1) is `start * 2^(r-1)`.
fn block_size_for_row(starting_block_size: u64, row: usize) -> u64 {
    if row <= 1 {
        starting_block_size
    } else {
        starting_block_size << (row - 1)
    }
}

/// Size in bytes of the FRHP header for the given offset/length sizes.
fn frhp_header_size(os: usize, ls: usize) -> usize {
    4 + 1 + 2 + 2 + 1 + 4 + ls + os + ls + os + ls + ls + ls + ls + ls + ls + ls + ls + 2 + ls + ls
        + 2
        + 2
        + os
        + 2
        + 4
}

/// Parameters for [`write_frhp`].
struct WriteFrhp {
    heap_id_length: u16,
    max_managed: u32,
    free_space: u64,
    managed_space: u64,
    alloc_space: u64,
    nobjects: u64,
    table_width: u16,
    starting_block_size: u64,
    max_direct_block_size: u64,
    max_heap_size: u16,
    root_addr: u64,
    cur_rows: u16,
}

/// Serialize a fractal heap header (FRHP).
fn write_frhp(p: WriteFrhp) -> Vec<u8> {
    let mut frhp = Vec::with_capacity(frhp_header_size(OFFSET_SIZE as usize, LENGTH_SIZE as usize));
    frhp.extend_from_slice(b"FRHP");
    frhp.push(0); // version
    frhp.extend_from_slice(&p.heap_id_length.to_le_bytes());
    frhp.extend_from_slice(&0u16.to_le_bytes()); // io_filter_encoded_length
    frhp.push(0x02); // flags: bit 1 = checksum direct blocks
    frhp.extend_from_slice(&p.max_managed.to_le_bytes());
    write_length(&mut frhp, 0, LENGTH_SIZE); // next_huge_object_id
    write_undef_offset(&mut frhp, OFFSET_SIZE); // btree_huge_objects_address
    write_length(&mut frhp, p.free_space, LENGTH_SIZE); // free_space_managed_blocks
    write_undef_offset(&mut frhp, OFFSET_SIZE); // free_space_mgr_addr
    write_length(&mut frhp, p.managed_space, LENGTH_SIZE); // managed_space_in_heap
    write_length(&mut frhp, p.alloc_space, LENGTH_SIZE); // allocated_managed_space
    write_length(&mut frhp, 0, LENGTH_SIZE); // dblock_alloc_iter
    write_length(&mut frhp, p.nobjects, LENGTH_SIZE); // managed_objects_count
    write_length(&mut frhp, 0, LENGTH_SIZE); // huge_objects_size
    write_length(&mut frhp, 0, LENGTH_SIZE); // huge_objects_count
    write_length(&mut frhp, 0, LENGTH_SIZE); // tiny_objects_size
    write_length(&mut frhp, 0, LENGTH_SIZE); // tiny_objects_count
    frhp.extend_from_slice(&p.table_width.to_le_bytes());
    write_length(&mut frhp, p.starting_block_size, LENGTH_SIZE);
    write_length(&mut frhp, p.max_direct_block_size, LENGTH_SIZE);
    frhp.extend_from_slice(&p.max_heap_size.to_le_bytes());
    frhp.extend_from_slice(&1u16.to_le_bytes()); // starting # rows in root indirect block
    write_offset(&mut frhp, p.root_addr, OFFSET_SIZE);
    frhp.extend_from_slice(&p.cur_rows.to_le_bytes());
    let checksum = crate::checksum::jenkins_lookup3(&frhp);
    frhp.extend_from_slice(&checksum.to_le_bytes());
    frhp
}

/// Build dense attribute storage for a set of attributes.
pub(crate) fn build_dense_attrs(attrs: &[AttributeMessage], base_address: u64) -> DenseAttrBlob {
    // Dense attrs use v3 attribute messages (adds character set encoding byte).
    let serialized: Vec<Vec<u8>> = attrs.iter().map(|a| a.serialize_v3(LENGTH_SIZE)).collect();

    let name_hashes: Vec<u32> = attrs
        .iter()
        .map(|a| crate::checksum::jenkins_lookup3(a.name.as_bytes()))
        .collect();

    let os = OFFSET_SIZE as usize;
    let ls = LENGTH_SIZE as usize;

    // Attribute heaps use max_heap_size 40 / heap ID length 8 (matching libhdf5).
    let heap = build_single_block_fractal_heap(&serialized, base_address, 40, 8);
    let frhp_addr = heap.frhp_addr;
    let btree_addr = heap.btree_addr;
    let heap_id_length = heap.heap_id_length;
    let heap_ids = &heap.heap_ids;

    // Build B-tree v2 type 8 records (17 bytes each)
    let record_size: u16 = heap_id_length + 1 + 4 + 4;
    let mut records: Vec<(u32, u32, Vec<u8>)> = Vec::with_capacity(attrs.len());
    for (i, heap_id) in heap_ids.iter().enumerate() {
        let mut rec = Vec::with_capacity(record_size as usize);
        rec.extend_from_slice(heap_id);
        rec.push(0); // msg_flags
        rec.extend_from_slice(&(i as u32).to_le_bytes()); // creation_order
        rec.extend_from_slice(&name_hashes[i].to_le_bytes()); // hash
        records.push((name_hashes[i], i as u32, rec));
    }
    records.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let bthd_size = 4 + 1 + 1 + 4 + 2 + 2 + 1 + 1 + os + 2 + ls + 4;
    let num_records = attrs.len();
    let btlf_size = 4 + 1 + 1 + (num_records * record_size as usize) + 4;
    let node_size = btlf_size.next_power_of_two().max(512) as u32;

    let bthd_addr = btree_addr;
    let btlf_addr = bthd_addr + bthd_size as u64;

    let mut bthd = Vec::with_capacity(bthd_size);
    bthd.extend_from_slice(b"BTHD");
    bthd.push(0); // version
    bthd.push(8); // type = attribute name index
    bthd.extend_from_slice(&node_size.to_le_bytes());
    bthd.extend_from_slice(&record_size.to_le_bytes());
    bthd.extend_from_slice(&0u16.to_le_bytes()); // depth = 0
    bthd.push(100); // split_percent
    bthd.push(40); // merge_percent
    write_offset(&mut bthd, btlf_addr, OFFSET_SIZE);
    bthd.extend_from_slice(&(num_records as u16).to_le_bytes());
    write_length(&mut bthd, num_records as u64, LENGTH_SIZE);
    let bthd_checksum = crate::checksum::jenkins_lookup3(&bthd);
    bthd.extend_from_slice(&bthd_checksum.to_le_bytes());
    debug_assert_eq!(bthd.len(), bthd_size);

    let mut btlf = Vec::with_capacity(node_size as usize);
    btlf.extend_from_slice(b"BTLF");
    btlf.push(0); // version
    btlf.push(8); // type
    for (_, _, rec) in &records {
        btlf.extend_from_slice(rec);
    }
    // Checksum goes immediately after records (NOT at end of node).
    // HDF5 C library computes checksum over sig+ver+type+records only.
    let btlf_checksum = crate::checksum::jenkins_lookup3(&btlf);
    btlf.extend_from_slice(&btlf_checksum.to_le_bytes());
    // Pad to node_size
    btlf.resize(node_size as usize, 0);

    let mut blob =
        Vec::with_capacity(heap.blob.len() + bthd.len() + btlf.len());
    blob.extend_from_slice(&heap.blob);
    blob.extend_from_slice(&bthd);
    blob.extend_from_slice(&btlf);

    let attr_info = serialize_attribute_info(frhp_addr, bthd_addr);

    DenseAttrBlob {
        attr_info_message: attr_info,
        blob,
    }
}

// ---- Dense link blob ----

/// Pre-built dense link storage (fractal heap + B-tree v2 + link-info message).
pub(crate) struct DenseLinkBlob {
    /// Serialized LinkInfo message (to embed in the group's object header).
    pub(crate) link_info_message: Vec<u8>,
    /// The combined fractal heap header + direct block + B-tree v2 bytes.
    pub(crate) blob: Vec<u8>,
}

/// Build dense link storage for a group's links, laid out at `base_address`.
///
/// Mirrors [`build_dense_attrs`]: each link is stored as a serialized Link
/// message in a single-direct-block fractal heap, indexed by a v2 B-tree of
/// **type 5** (link-name index, record = name hash + heap ID). The returned
/// LinkInfo message points at the heap and the name B-tree.
pub(crate) fn build_dense_links(links: &[LinkMessage], base_address: u64) -> DenseLinkBlob {
    let serialized: Vec<Vec<u8>> = links.iter().map(|l| l.serialize(OFFSET_SIZE)).collect();
    let name_hashes: Vec<u32> = links
        .iter()
        .map(|l| crate::checksum::jenkins_lookup3(l.name.as_bytes()))
        .collect();

    let os = OFFSET_SIZE as usize;
    let ls = LENGTH_SIZE as usize;

    // libhdf5's link heap uses max_heap_size 32 / heap ID length 7 (vs 40/8 for
    // attributes), giving a 7-byte heap ID and an 11-byte type-5 record.
    let heap = build_single_block_fractal_heap(&serialized, base_address, 32, 7);
    let heap_id_length = heap.heap_id_length;

    // B-tree v2 type 5 records: hash(4) + heap_id(heap_id_length). The B-tree
    // search key is the name hash, so records are sorted by (hash, order).
    let record_size: u16 = 4 + heap_id_length;
    let mut records: Vec<(u32, u32, Vec<u8>)> = Vec::with_capacity(links.len());
    for (i, heap_id) in heap.heap_ids.iter().enumerate() {
        let mut rec = Vec::with_capacity(record_size as usize);
        rec.extend_from_slice(&name_hashes[i].to_le_bytes()); // hash
        rec.extend_from_slice(heap_id); // heap ID
        records.push((name_hashes[i], i as u32, rec));
    }
    records.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let bthd_size = 4 + 1 + 1 + 4 + 2 + 2 + 1 + 1 + os + 2 + ls + 4;
    let num_records = links.len();
    let btlf_size = 4 + 1 + 1 + (num_records * record_size as usize) + 4;
    let node_size = btlf_size.next_power_of_two().max(512) as u32;

    let bthd_addr = heap.btree_addr;
    let btlf_addr = bthd_addr + bthd_size as u64;

    let mut bthd = Vec::with_capacity(bthd_size);
    bthd.extend_from_slice(b"BTHD");
    bthd.push(0); // version
    bthd.push(5); // type = link name index
    bthd.extend_from_slice(&node_size.to_le_bytes());
    bthd.extend_from_slice(&record_size.to_le_bytes());
    bthd.extend_from_slice(&0u16.to_le_bytes()); // depth = 0 (single leaf)
    bthd.push(100); // split_percent
    bthd.push(40); // merge_percent
    write_offset(&mut bthd, btlf_addr, OFFSET_SIZE);
    bthd.extend_from_slice(&(num_records as u16).to_le_bytes());
    write_length(&mut bthd, num_records as u64, LENGTH_SIZE);
    let bthd_checksum = crate::checksum::jenkins_lookup3(&bthd);
    bthd.extend_from_slice(&bthd_checksum.to_le_bytes());
    debug_assert_eq!(bthd.len(), bthd_size);

    let mut btlf = Vec::with_capacity(node_size as usize);
    btlf.extend_from_slice(b"BTLF");
    btlf.push(0); // version
    btlf.push(5); // type
    for (_, _, rec) in &records {
        btlf.extend_from_slice(rec);
    }
    let btlf_checksum = crate::checksum::jenkins_lookup3(&btlf);
    btlf.extend_from_slice(&btlf_checksum.to_le_bytes());
    btlf.resize(node_size as usize, 0);

    let mut blob =
        Vec::with_capacity(heap.blob.len() + bthd.len() + btlf.len());
    blob.extend_from_slice(&heap.blob);
    blob.extend_from_slice(&bthd);
    blob.extend_from_slice(&btlf);

    DenseLinkBlob {
        link_info_message: serialize_link_info(heap.frhp_addr, bthd_addr),
        blob,
    }
}

/// Serialize a LinkInfo message (version 0, no creation-order index) pointing
/// at a fractal heap and a v2 B-tree name index.
fn serialize_link_info(fh_addr: u64, btree_name_addr: u64) -> Vec<u8> {
    let mut data = Vec::new();
    data.push(0); // version
    data.push(0x00); // flags: no creation-order tracking
    write_offset(&mut data, fh_addr, OFFSET_SIZE);
    write_offset(&mut data, btree_name_addr, OFFSET_SIZE);
    data
}

fn encode_managed_id(offset: u64, length: u64, max_heap_size: u16, id_length: u16) -> Vec<u8> {
    let mut id = vec![0u8; id_length as usize];
    id[0] = 0x00; // type = 0 (managed)
    let combined = offset | (length << max_heap_size);
    let payload_len = (id_length as usize) - 1;
    for i in 0..payload_len.min(8) {
        id[1 + i] = ((combined >> (i * 8)) & 0xFF) as u8;
    }
    id
}

fn serialize_attribute_info(fh_addr: u64, btree_name_addr: u64) -> Vec<u8> {
    let mut data = Vec::new();
    data.push(0); // version
    data.push(0x00); // flags
    data.extend_from_slice(&fh_addr.to_le_bytes());
    data.extend_from_slice(&btree_name_addr.to_le_bytes());
    data
}

// ---- VDS helpers ----

/// Serialize VDS mappings for storage in a global heap object.
///
/// Delegates to `data_layout_write::serialize_vds_mappings` (the canonical
/// implementation with full version/external-file handling), then appends a
/// trailing 4-byte Jenkins lookup3 checksum that parsers skip after consuming
/// all `nused` entries.
pub(crate) fn serialize_vds_mappings(mappings: &[VdsMapping]) -> Vec<u8> {
    let mut buf = crate::data_layout_write::serialize_vds_mappings(mappings, 8);
    let cksum = crate::checksum::jenkins_lookup3(&buf);
    buf.extend_from_slice(&cksum.to_le_bytes());
    buf
}

/// Build a minimal global heap collection containing a single object.
///
/// Returns the serialized collection bytes. The object index is always 1.
///
/// Global heap collection layout:
/// ```text
/// "GCOL"(4) · version(1) · reserved(3) · collection_size(8)
/// · [index(2) · ref_count(2) · reserved(4) · object_size(8) · data · padding]
/// · free-space-marker(2)
/// ```
pub(crate) fn build_global_heap_collection(object_data: &[u8]) -> Vec<u8> {
    let ls = LENGTH_SIZE as usize;
    let header_size = 8 + ls; // sig(4)+ver(1)+rsv(3)+coll_size(ls)
    let obj_header_size = 8 + ls; // idx(2)+rc(2)+rsv(4)+obj_size(ls)
    let padded_data_len = pad8(object_data.len());
    let free_marker_size = 2;
    let collection_size = header_size + obj_header_size + padded_data_len + free_marker_size;

    let mut buf = Vec::with_capacity(collection_size);
    buf.extend_from_slice(b"GCOL");
    buf.push(1); // version
    buf.extend_from_slice(&[0u8; 3]); // reserved
    buf.extend_from_slice(&(collection_size as u64).to_le_bytes()); // collection_size

    // Object 1
    buf.extend_from_slice(&1u16.to_le_bytes()); // index
    buf.extend_from_slice(&1u16.to_le_bytes()); // reference count
    buf.extend_from_slice(&[0u8; 4]); // reserved
    buf.extend_from_slice(&(object_data.len() as u64).to_le_bytes()); // object size
    buf.extend_from_slice(object_data);
    // Pad object data to 8-byte boundary
    let pad = padded_data_len - object_data.len();
    buf.extend_from_slice(&vec![0u8; pad]);

    // Free space marker
    buf.extend_from_slice(&0u16.to_le_bytes());

    debug_assert_eq!(buf.len(), collection_size);
    buf
}

/// Round up to the next multiple of 8.
fn pad8(x: usize) -> usize {
    (x + 7) & !7
}

/// Build a Virtual Dataset object header.
///
/// The layout message for a VDS dataset is:
/// ```text
/// version(1=4) · class(1=3) · global_heap_address(8) · global_heap_index(4)
/// ```
pub(crate) fn build_vds_dataset_oh(
    dt: &Datatype,
    ds: &Dataspace,
    global_heap_addr: u64,
    attrs: &[AttributeMessage],
    dense_blob: Option<&DenseAttrBlob>,
    fill_time: FillTime,
) -> Vec<u8> {
    let mut w = ObjectHeaderWriter::new();
    w.add_message_with_flags(MessageType::Datatype, dt.serialize(), 0x01);
    w.add_message(MessageType::Dataspace, ds.serialize(LENGTH_SIZE));
    w.add_message_with_flags(MessageType::FillValue, vec![3, fill_time.to_byte()], 0x01);
    // VDS layout message: version=4, class=3, global_heap_address(8), global_heap_index=1(4)
    let mut dl = Vec::new();
    dl.push(4u8); // version
    dl.push(3u8); // class = virtual
    dl.extend_from_slice(&global_heap_addr.to_le_bytes());
    dl.extend_from_slice(&1u32.to_le_bytes()); // object index 1 in the collection
    w.add_message(MessageType::DataLayout, dl);
    if let Some(blob) = dense_blob {
        w.add_message(MessageType::AttributeInfo, blob.attr_info_message.clone());
    } else {
        for attr in attrs {
            w.add_message(MessageType::Attribute, attr.serialize(LENGTH_SIZE));
        }
    }
    w.serialize()
}

fn write_offset(buf: &mut Vec<u8>, val: u64, offset_size: u8) {
    match offset_size {
        2 => buf.extend_from_slice(&(val as u16).to_le_bytes()),
        4 => buf.extend_from_slice(&(val as u32).to_le_bytes()),
        8 => buf.extend_from_slice(&val.to_le_bytes()),
        _ => {}
    }
}

fn write_length(buf: &mut Vec<u8>, val: u64, length_size: u8) {
    write_offset(buf, val, length_size);
}

fn write_undef_offset(buf: &mut Vec<u8>, offset_size: u8) {
    for _ in 0..offset_size {
        buf.push(0xFF);
    }
}

// ---- FileWriter ----

/// The main file creation API.
pub struct FileWriter {
    root_datasets: Vec<DatasetBuilder>,
    root_attrs: Vec<(String, AttrValue)>,
    groups: Vec<FinishedGroup>,
    /// Global alignment threshold: datasets with raw data >= this many bytes
    /// will have their data aligned to `alignment_bytes`.
    alignment_threshold: usize,
    /// Global alignment boundary in bytes (0 = disabled).
    alignment_bytes: usize,
}

impl Default for FileWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl FileWriter {
    pub fn new() -> Self {
        Self {
            root_datasets: Vec::new(),
            root_attrs: Vec::new(),
            groups: Vec::new(),
            alignment_threshold: 0,
            alignment_bytes: 0,
        }
    }

    /// Set global file alignment: datasets with raw data >= `threshold` bytes
    /// will have their data aligned to `bytes` boundary.
    ///
    /// For example, `.alignment(1, 4096)` aligns all datasets to 4KB pages.
    pub fn alignment(&mut self, threshold: usize, bytes: usize) -> &mut Self {
        self.alignment_threshold = threshold;
        self.alignment_bytes = bytes;
        self
    }

    pub fn create_group(&mut self, name: &str) -> GroupBuilder {
        GroupBuilder::new(name)
    }

    pub fn add_group(&mut self, group: FinishedGroup) {
        self.groups.push(group);
    }

    pub fn create_dataset(&mut self, name: &str) -> &mut DatasetBuilder {
        self.root_datasets.push(DatasetBuilder::new(name));
        self.root_datasets.last_mut().unwrap()
    }

    pub fn set_root_attr(&mut self, name: &str, value: AttrValue) {
        self.root_attrs.push((name.to_string(), value));
    }

    pub fn finish(self) -> Result<Vec<u8>, FormatError> {
        struct DsFlat {
            name: String,
            dt: Datatype,
            ds: Dataspace,
            raw: Vec<u8>,
            attrs: Vec<AttributeMessage>,
            chunk_options: ChunkOptions,
            maxshape: Option<Vec<u64>>,
            fill_time: FillTime,
            compact: bool,
            alignment: usize,
            /// VDS source mappings (set for Virtual datasets).
            virtual_sources: Option<Vec<VdsMapping>>,
        }
        struct GrpFlat {
            name: String,
            attrs: Vec<AttributeMessage>,
            ds_indices: Vec<usize>,
            /// (link_name, target_file, target_path)
            external_links: Vec<(String, String, String)>,
        }

        // Helper: convert a DatasetBuilder into DsFlat, handling VDS (which
        // does not require a `data` field).
        let flatten_ds = |db: DatasetBuilder| -> Result<DsFlat, FormatError> {
            let dt = db.datatype.ok_or(FormatError::DatasetMissingData)?;
            let shape = db.shape.ok_or(FormatError::DatasetMissingShape)?;
            let is_vds = db.virtual_sources.is_some();
            let raw = if is_vds {
                // VDS datasets have no raw data stored in this file.
                db.data.unwrap_or_default()
            } else {
                db.data.ok_or(FormatError::DatasetMissingData)?
            };
            let max_dimensions = db.maxshape.clone();
            let dspace = Dataspace {
                space_type: if shape.is_empty() {
                    DataspaceType::Scalar
                } else {
                    DataspaceType::Simple
                },
                rank: shape.len() as u8,
                dimensions: shape,
                max_dimensions,
            };
            let mut attrs = Vec::new();
            for (n, v) in &db.attrs {
                attrs.push(build_attr_message(n, v));
            }
            #[cfg(feature = "provenance")]
            if let Some(ref prov) = db.provenance {
                let p = crate::provenance::Provenance {
                    creator: prov.creator.clone(),
                    timestamp: prov.timestamp.clone(),
                    source: prov.source.clone(),
                };
                attrs.extend(p.build_attrs(&raw));
            }
            Ok(DsFlat {
                name: db.name,
                dt,
                ds: dspace,
                raw,
                attrs,
                chunk_options: db.chunk_options,
                maxshape: db.maxshape,
                fill_time: db.fill_time,
                compact: db.compact,
                alignment: db.alignment,
                virtual_sources: db.virtual_sources,
            })
        };

        let mut all_ds: Vec<DsFlat> = Vec::new();
        let mut groups: Vec<GrpFlat> = Vec::new();
        let mut root_ds_indices: Vec<usize> = Vec::new();

        for db in self.root_datasets {
            root_ds_indices.push(all_ds.len());
            all_ds.push(flatten_ds(db)?);
        }

        for g in self.groups.into_iter() {
            let mut gattrs = Vec::new();
            for (n, v) in &g.attrs {
                gattrs.push(build_attr_message(n, v));
            }
            let mut ds_idx = Vec::new();
            for db in g.datasets {
                ds_idx.push(all_ds.len());
                all_ds.push(flatten_ds(db)?);
            }
            groups.push(GrpFlat {
                name: g.name,
                attrs: gattrs,
                ds_indices: ds_idx,
                external_links: g.external_links,
            });
        }

        let mut root_attrs: Vec<AttributeMessage> = Vec::new();
        for (n, v) in &self.root_attrs {
            root_attrs.push(build_attr_message(n, v));
        }

        let is_vds: Vec<bool> = all_ds
            .iter()
            .map(|d| d.virtual_sources.is_some())
            .collect();
        let is_chunked: Vec<bool> = all_ds
            .iter()
            .enumerate()
            .map(|(i, d)| !is_vds[i] && (d.chunk_options.is_chunked() || d.maxshape.is_some()))
            .collect();
        // Determine which datasets use compact storage
        let is_compact: Vec<bool> = all_ds
            .iter()
            .enumerate()
            .map(|(i, d)| !is_vds[i] && !is_chunked[i] && d.compact && d.raw.len() <= 65535)
            .collect();
        let root_dense = root_attrs.len() > DENSE_ATTR_THRESHOLD;
        let group_dense: Vec<bool> = groups
            .iter()
            .map(|g| g.attrs.len() > DENSE_ATTR_THRESHOLD)
            .collect();
        let ds_dense: Vec<bool> = all_ds
            .iter()
            .map(|d| d.attrs.len() > DENSE_ATTR_THRESHOLD)
            .collect();

        // Dense link decision: a group with more than the compact threshold of
        // links stores them in a fractal heap + v2 B-tree instead of inline.
        let root_link_count = root_ds_indices.len() + groups.len();
        let root_links_dense = root_link_count > DENSE_LINK_THRESHOLD;
        let group_links_dense: Vec<bool> = groups
            .iter()
            .map(|g| g.ds_indices.len() + g.external_links.len() > DENSE_LINK_THRESHOLD)
            .collect();
        // The dense LinkInfo message is a fixed size regardless of address, so a
        // dummy is sufficient for OH size computation.
        let dummy_link_info = serialize_link_info(0, 0);

        // Pass 1: compute OH sizes with dummy addresses
        let group_oh_sizes: Vec<usize> = groups
            .iter()
            .enumerate()
            .map(|(gi, g)| {
                let mut dummy_links: Vec<LinkMessage> = g
                    .ds_indices
                    .iter()
                    .map(|&i| make_link(&all_ds[i].name, 0))
                    .collect();
                for (lname, fname, opath) in &g.external_links {
                    dummy_links.push(make_external_link(lname, fname, opath));
                }
                let attr_blob = group_dense[gi].then(|| build_dense_attrs(&g.attrs, 0));
                let dl = group_links_dense[gi].then_some(dummy_link_info.as_slice());
                build_group_oh(&dummy_links, dl, &g.attrs, attr_blob.as_ref()).len()
            })
            .collect();

        let root_dummy_links: Vec<LinkMessage> = {
            let mut links = Vec::new();
            for &i in &root_ds_indices {
                links.push(make_link(&all_ds[i].name, 0));
            }
            for g in &groups {
                links.push(make_link(&g.name, 0));
            }
            links
        };
        let root_oh_size = {
            let attr_blob = root_dense.then(|| build_dense_attrs(&root_attrs, 0));
            let dl = root_links_dense.then_some(dummy_link_info.as_slice());
            build_group_oh(&root_dummy_links, dl, &root_attrs, attr_blob.as_ref()).len()
        };

        struct DataBlob {
            data: Vec<u8>,
            oh_bytes: Vec<u8>,
            /// Cached compressed chunks for chunked datasets; reused in Pass 2
            /// to avoid re-compressing the same data.
            precompressed: Option<PrecompressedChunks>,
        }

        let mut dummy_blobs: Vec<DataBlob> = Vec::new();
        let mut dummy_cursor = 0u64;
        for (i, d) in all_ds.iter().enumerate() {
            if is_vds[i] {
                // VDS: dummy OH with address 0 to get the OH size. The global
                // heap blob will be placed after the OHs in pass 2.
                let dense_blob = if ds_dense[i] {
                    Some(build_dense_attrs(&d.attrs, 0))
                } else {
                    None
                };
                let oh = build_vds_dataset_oh(
                    &d.dt,
                    &d.ds,
                    0, // dummy address
                    &d.attrs,
                    dense_blob.as_ref(),
                    d.fill_time,
                );
                // Global heap blob size is address-independent; compute it now
                // so pass 2 can place it correctly.
                let vds_mappings = d.virtual_sources.as_deref().unwrap_or(&[]);
                let gcol_bytes = build_global_heap_collection(&serialize_vds_mappings(vds_mappings));
                dummy_blobs.push(DataBlob {
                    data: gcol_bytes, // store heap blob here temporarily
                    oh_bytes: oh,
                    precompressed: None,
                });
            } else if is_chunked[i] {
                let chunk_dims = d.chunk_options.resolve_chunk_dims(&d.ds.dimensions);
                let elem_size = d.dt.type_size() as usize;
                // Compress once in Pass 1; cache the result so Pass 2 can skip
                // re-compression and just rebuild the index with real addresses.
                let pre = precompress_chunks(
                    &d.raw,
                    &d.ds.dimensions,
                    &chunk_dims,
                    elem_size,
                    &d.chunk_options,
                )?;
                let result =
                    build_chunked_data_from_precompressed(&pre, dummy_cursor, d.maxshape.as_deref());
                dummy_cursor += result.data_bytes.len() as u64;
                let dense_blob = if ds_dense[i] {
                    Some(build_dense_attrs(&d.attrs, 0))
                } else {
                    None
                };
                let oh = build_chunked_dataset_oh(
                    &d.dt,
                    &d.ds,
                    &result.layout_message,
                    result.pipeline_message.as_deref(),
                    &d.attrs,
                    dense_blob.as_ref(),
                    d.fill_time,
                );
                dummy_blobs.push(DataBlob {
                    data: result.data_bytes,
                    oh_bytes: oh,
                    precompressed: Some(pre),
                });
            } else if is_compact[i] {
                let dense_blob = if ds_dense[i] {
                    Some(build_dense_attrs(&d.attrs, 0))
                } else {
                    None
                };
                let oh = build_compact_dataset_oh(
                    &d.dt,
                    &d.ds,
                    &d.raw,
                    &d.attrs,
                    dense_blob.as_ref(),
                    d.fill_time,
                );
                dummy_blobs.push(DataBlob {
                    data: vec![],
                    oh_bytes: oh,
                    precompressed: None,
                });
            } else {
                let dense_blob = if ds_dense[i] {
                    Some(build_dense_attrs(&d.attrs, 0))
                } else {
                    None
                };
                let oh = build_dataset_oh(
                    &d.dt,
                    &d.ds,
                    0,
                    d.raw.len() as u64,
                    &d.attrs,
                    dense_blob.as_ref(),
                    d.fill_time,
                );
                dummy_blobs.push(DataBlob {
                    data: d.raw.clone(),
                    oh_bytes: oh,
                    precompressed: None,
                });
            }
        }

        let actual_ds_oh_sizes: Vec<usize> = dummy_blobs.iter().map(|b| b.oh_bytes.len()).collect();

        // Pass 2: compute real addresses
        let root_group_addr = SUPERBLOCK_SIZE as u64;
        let mut cursor2 = SUPERBLOCK_SIZE + root_oh_size;

        // Each group is laid out as: object header, then (if dense) its link
        // blob, then (if dense) its attribute blob. Link blobs are sized with
        // dummy target addresses here — link message size is address-independent
        // — and rebuilt with real addresses in the final pass.
        let root_link_blob_addr = if root_links_dense {
            let addr = cursor2 as u64;
            cursor2 += build_dense_links(&root_dummy_links, addr).blob.len();
            Some(addr)
        } else {
            None
        };
        let root_dense_blob = if root_dense {
            let blob = build_dense_attrs(&root_attrs, cursor2 as u64);
            cursor2 += blob.blob.len();
            Some(blob)
        } else {
            None
        };

        let mut group_link_blob_addrs: Vec<Option<u64>> = Vec::new();
        let mut group_dense_blobs: Vec<Option<DenseAttrBlob>> = Vec::new();
        let group_addrs2: Vec<u64> = group_oh_sizes
            .iter()
            .enumerate()
            .map(|(gi, &sz)| {
                let addr = cursor2 as u64;
                cursor2 += sz;
                if group_links_dense[gi] {
                    let mut dummy_links: Vec<LinkMessage> = groups[gi]
                        .ds_indices
                        .iter()
                        .map(|&i| make_link(&all_ds[i].name, 0))
                        .collect();
                    for (lname, fname, opath) in &groups[gi].external_links {
                        dummy_links.push(make_external_link(lname, fname, opath));
                    }
                    let blob_addr = cursor2 as u64;
                    cursor2 += build_dense_links(&dummy_links, blob_addr).blob.len();
                    group_link_blob_addrs.push(Some(blob_addr));
                } else {
                    group_link_blob_addrs.push(None);
                }
                if group_dense[gi] {
                    let blob = build_dense_attrs(&groups[gi].attrs, cursor2 as u64);
                    cursor2 += blob.blob.len();
                    group_dense_blobs.push(Some(blob));
                } else {
                    group_dense_blobs.push(None);
                }
                addr
            })
            .collect();

        let mut ds_dense_blobs: Vec<Option<DenseAttrBlob>> = Vec::new();
        let ds_oh_addrs2: Vec<u64> = actual_ds_oh_sizes
            .iter()
            .enumerate()
            .map(|(i, &sz)| {
                let addr = cursor2 as u64;
                cursor2 += sz;
                if ds_dense[i] {
                    let blob = build_dense_attrs(&all_ds[i].attrs, cursor2 as u64);
                    cursor2 += blob.blob.len();
                    ds_dense_blobs.push(Some(blob));
                } else {
                    ds_dense_blobs.push(None);
                }
                addr
            })
            .collect();

        let mut ds_blobs2: Vec<DataBlob> = Vec::new();
        let global_align_threshold = self.alignment_threshold;
        let global_align_bytes = self.alignment_bytes;
        for (i, d) in all_ds.iter().enumerate() {
            if is_vds[i] {
                // VDS: place the global heap collection right after the OHs,
                // then rebuild the OH with the real heap address.
                let gcol_bytes = &dummy_blobs[i].data; // pre-computed in pass 1
                let heap_addr = cursor2 as u64;
                cursor2 += gcol_bytes.len();
                let oh = build_vds_dataset_oh(
                    &d.dt,
                    &d.ds,
                    heap_addr,
                    &d.attrs,
                    ds_dense_blobs[i].as_ref(),
                    d.fill_time,
                );
                ds_blobs2.push(DataBlob {
                    data: gcol_bytes.clone(),
                    oh_bytes: oh,
                    precompressed: None,
                });
            } else if is_chunked[i] {
                let base_address = cursor2 as u64;
                // Reuse precompressed chunks from Pass 1 — avoids re-compressing
                // the same data a second time.
                let result = build_chunked_data_from_precompressed(
                    dummy_blobs[i].precompressed.as_ref().expect("chunked dataset missing precompressed cache"),
                    base_address,
                    d.maxshape.as_deref(),
                );
                cursor2 += result.data_bytes.len();
                let oh = build_chunked_dataset_oh(
                    &d.dt,
                    &d.ds,
                    &result.layout_message,
                    result.pipeline_message.as_deref(),
                    &d.attrs,
                    ds_dense_blobs[i].as_ref(),
                    d.fill_time,
                );
                ds_blobs2.push(DataBlob {
                    data: result.data_bytes,
                    oh_bytes: oh,
                    precompressed: None,
                });
            } else if is_compact[i] {
                // Compact: data is inline in the object header, no external blob
                let oh = build_compact_dataset_oh(
                    &d.dt,
                    &d.ds,
                    &d.raw,
                    &d.attrs,
                    ds_dense_blobs[i].as_ref(),
                    d.fill_time,
                );
                ds_blobs2.push(DataBlob {
                    data: vec![],
                    oh_bytes: oh,
                    precompressed: None,
                });
            } else {
                // Determine alignment: per-dataset overrides global
                let align = if d.alignment > 0 {
                    d.alignment
                } else if global_align_bytes > 0 && d.raw.len() >= global_align_threshold {
                    global_align_bytes
                } else {
                    8 // default: 8-byte alignment for zero-copy read support
                };
                let padding = (align - (cursor2 % align)) % align;
                cursor2 += padding;
                let oh = build_dataset_oh(
                    &d.dt,
                    &d.ds,
                    cursor2 as u64,
                    d.raw.len() as u64,
                    &d.attrs,
                    ds_dense_blobs[i].as_ref(),
                    d.fill_time,
                );
                let mut data = vec![0u8; padding];
                data.extend_from_slice(&d.raw);
                cursor2 += d.raw.len();
                ds_blobs2.push(DataBlob {
                    data,
                    oh_bytes: oh,
                    precompressed: None,
                });
            }
        }

        let actual_ds_oh_sizes2: Vec<usize> = ds_blobs2.iter().map(|b| b.oh_bytes.len()).collect();
        debug_assert_eq!(actual_ds_oh_sizes, actual_ds_oh_sizes2);

        let eof_addr2 = cursor2 as u64;
        let mut buf = Vec::with_capacity(cursor2);

        let sb = Superblock {
            version: 3,
            offset_size: OFFSET_SIZE,
            length_size: LENGTH_SIZE,
            base_address: 0,
            eof_address: eof_addr2,
            root_group_address: root_group_addr,
            group_leaf_node_k: None,
            group_internal_node_k: None,
            indexed_storage_internal_node_k: None,
            free_space_address: None,
            driver_info_address: None,
            consistency_flags: 0,
            superblock_extension_address: Some(u64::MAX),
            checksum: None,
        };
        buf.extend_from_slice(&sb.serialize());

        // Root group OH
        let mut root_links: Vec<LinkMessage> = Vec::new();
        for &i in &root_ds_indices {
            root_links.push(make_link(&all_ds[i].name, ds_oh_addrs2[i]));
        }
        for (gi, g) in groups.iter().enumerate() {
            root_links.push(make_link(&g.name, group_addrs2[gi]));
        }
        // Rebuild the root link blob with real target addresses (same size as
        // the dummy used for layout); its LinkInfo goes in the OH.
        let root_link_blob = root_link_blob_addr.map(|addr| build_dense_links(&root_links, addr));
        let root_dl = root_link_blob.as_ref().map(|b| b.link_info_message.as_slice());
        buf.extend_from_slice(&build_group_oh(
            &root_links,
            root_dl,
            &root_attrs,
            root_dense_blob.as_ref(),
        ));
        if let Some(ref b) = root_link_blob {
            buf.extend_from_slice(&b.blob);
        }
        if let Some(ref blob) = root_dense_blob {
            buf.extend_from_slice(&blob.blob);
        }

        // Group OHs + dense blobs (link blob, then attr blob, matching pass 2)
        for (gi, g) in groups.iter().enumerate() {
            let mut links: Vec<LinkMessage> = g
                .ds_indices
                .iter()
                .map(|&i| make_link(&all_ds[i].name, ds_oh_addrs2[i]))
                .collect();
            for (lname, fname, opath) in &g.external_links {
                links.push(make_external_link(lname, fname, opath));
            }
            let link_blob = group_link_blob_addrs[gi].map(|addr| build_dense_links(&links, addr));
            let dl = link_blob.as_ref().map(|b| b.link_info_message.as_slice());
            buf.extend_from_slice(&build_group_oh(
                &links,
                dl,
                &g.attrs,
                group_dense_blobs[gi].as_ref(),
            ));
            if let Some(ref b) = link_blob {
                buf.extend_from_slice(&b.blob);
            }
            if let Some(ref blob) = group_dense_blobs[gi] {
                buf.extend_from_slice(&blob.blob);
            }
        }

        // Dataset OHs + dense blobs
        for (i, blob) in ds_blobs2.iter().enumerate() {
            buf.extend_from_slice(&blob.oh_bytes);
            if let Some(ref dense) = ds_dense_blobs[i] {
                buf.extend_from_slice(&dense.blob);
            }
        }

        // Data
        for blob in &ds_blobs2 {
            buf.extend_from_slice(&blob.data);
        }

        debug_assert_eq!(buf.len(), cursor2);
        Ok(buf)
    }
}

// ---- Independent parallel dataset creation ----

/// Builder that creates datasets without locking the file header.
///
/// Each `IndependentDatasetBuilder` accumulates its own [`MetadataBlock`]
/// independently. On [`IndependentDatasetBuilder::finish`], the block is
/// returned for later merging.
///
/// Thread-safety: each thread should own its own builder instance.
pub struct IndependentDatasetBuilder {
    block: MetadataBlock,
}

impl IndependentDatasetBuilder {
    /// Create a new independent builder with the given creator id.
    pub fn new(creator_id: u32) -> Self {
        Self {
            block: MetadataBlock::new(creator_id),
        }
    }

    /// Add a dataset specification to this builder.
    pub fn add_dataset(&mut self, meta: DatasetMetadata) {
        self.block.add_dataset(meta);
    }

    /// Consume the builder and return the metadata block.
    pub fn finish(self) -> MetadataBlock {
        self.block
    }
}

/// Finalize multiple independently-created metadata blocks into a complete HDF5 file.
///
/// This implements the write-ahead approach: each block's data is laid out
/// sequentially, then the index table (root group with links) is written last
/// to point at all the dataset object headers.
pub fn finalize_parallel(blocks: Vec<MetadataBlock>) -> Result<Vec<u8>, FormatError> {
    let index = MetadataIndex::merge_blocks(&blocks)?;
    finalize_from_index(index)
}

/// Build a complete HDF5 file from a merged MetadataIndex.
fn finalize_from_index(index: MetadataIndex) -> Result<Vec<u8>, FormatError> {
    // Convert DatasetMetadata into the internal DsFlat representation and
    // delegate to the same two-pass algorithm used by FileWriter.
    let mut fw = FileWriter::new();
    for ds_meta in &index.datasets {
        let db = fw.create_dataset(&ds_meta.name);
        // Set the datatype and raw data directly via internal fields
        db.datatype = Some(ds_meta.datatype.clone());
        db.shape = Some(ds_meta.dataspace.dimensions.clone());
        db.maxshape = ds_meta.maxshape.clone();
        db.data = Some(ds_meta.raw_data.clone());
        db.chunk_options = ds_meta.chunk_options.clone();
        for (name, val) in &ds_meta.attrs {
            db.set_attr(name, val.clone());
        }
    }
    fw.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group_v2::resolve_path_any;
    use crate::object_header::ObjectHeader;
    use crate::signature;

    fn parse_file(bytes: &[u8]) -> (Superblock, ObjectHeader) {
        let sig = signature::find_signature(bytes).unwrap();
        let sb = Superblock::parse(bytes, sig).unwrap();
        let oh = ObjectHeader::parse(
            bytes,
            sb.root_group_address as usize,
            sb.offset_size,
            sb.length_size,
        )
        .unwrap();
        (sb, oh)
    }

    fn read_dataset_f64(bytes: &[u8], path: &str) -> Vec<f64> {
        let sig = signature::find_signature(bytes).unwrap();
        let sb = Superblock::parse(bytes, sig).unwrap();
        let addr = resolve_path_any(bytes, &sb, path).unwrap();
        let hdr =
            ObjectHeader::parse(bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
        let dt_data = &hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::Datatype)
            .unwrap()
            .data;
        let ds_data = &hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::Dataspace)
            .unwrap()
            .data;
        let dl_data = &hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::DataLayout)
            .unwrap()
            .data;
        let (dt, _) = Datatype::parse(dt_data).unwrap();
        let ds = Dataspace::parse(ds_data, sb.length_size).unwrap();
        let dl =
            crate::data_layout::DataLayout::parse(dl_data, sb.offset_size, sb.length_size).unwrap();
        let raw = crate::data_read::read_raw_data(bytes, &dl, &ds, &dt).unwrap();
        crate::data_read::read_as_f64(&raw, &dt).unwrap()
    }

    #[test]
    fn empty_file_root_group_only() {
        let fw = FileWriter::new();
        let bytes = fw.finish().unwrap();
        let (sb, oh) = parse_file(&bytes);
        assert_eq!(sb.version, 3);
        assert_eq!(oh.version, 2);
    }

    #[test]
    fn file_with_f64_dataset() {
        let mut fw = FileWriter::new();
        fw.create_dataset("data").with_f64_data(&[1.0, 2.0, 3.0]);
        let bytes = fw.finish().unwrap();
        assert_eq!(read_dataset_f64(&bytes, "data"), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn file_with_dataset_attrs() {
        let mut fw = FileWriter::new();
        fw.create_dataset("data")
            .with_f64_data(&[1.0, 2.0])
            .set_attr("scale", AttrValue::F64(0.5));
        let bytes = fw.finish().unwrap();
        assert_eq!(read_dataset_f64(&bytes, "data"), vec![1.0, 2.0]);
        let sig = signature::find_signature(&bytes).unwrap();
        let sb = Superblock::parse(&bytes, sig).unwrap();
        let addr = resolve_path_any(&bytes, &sb, "data").unwrap();
        let hdr =
            ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
        let attrs = crate::attribute::extract_attributes(&hdr, sb.length_size).unwrap();
        assert_eq!(attrs.len(), 1);
        assert_eq!(attrs[0].name, "scale");
    }

    #[test]
    fn file_with_group_and_dataset() {
        let mut fw = FileWriter::new();
        let mut gb = fw.create_group("grp");
        gb.create_dataset("vals").with_f64_data(&[10.0, 20.0]);
        fw.add_group(gb.finish());
        let bytes = fw.finish().unwrap();
        assert_eq!(read_dataset_f64(&bytes, "grp/vals"), vec![10.0, 20.0]);
    }

    #[test]
    fn file_with_root_attr() {
        let mut fw = FileWriter::new();
        fw.set_root_attr("version", AttrValue::I64(42));
        let bytes = fw.finish().unwrap();
        let (sb, oh) = parse_file(&bytes);
        let attrs = crate::attribute::extract_attributes(&oh, sb.length_size).unwrap();
        assert_eq!(attrs[0].name, "version");
    }

    #[test]
    fn dense_attrs_self_roundtrip() {
        let mut fw = FileWriter::new();
        let ds = fw.create_dataset("data");
        ds.with_f64_data(&[1.0, 2.0, 3.0]);
        for i in 0..20 {
            ds.set_attr(&format!("attr_{i:03}"), AttrValue::F64(i as f64 * 1.5));
        }
        let bytes = fw.finish().unwrap();
        let sig = signature::find_signature(&bytes).unwrap();
        let sb = Superblock::parse(&bytes, sig).unwrap();
        let addr = resolve_path_any(&bytes, &sb, "data").unwrap();
        let hdr =
            ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
        let attrs =
            crate::attribute::extract_attributes_full(&bytes, &hdr, sb.offset_size, sb.length_size)
                .unwrap();
        assert_eq!(attrs.len(), 20);
        for i in 0..20 {
            let attr = attrs
                .iter()
                .find(|a| a.name == format!("attr_{i:03}"))
                .unwrap();
            let v = attr.read_as_f64().unwrap();
            assert!((v[0] - i as f64 * 1.5).abs() < 1e-10);
        }
        assert_eq!(read_dataset_f64(&bytes, "data"), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn dense_attrs_root_group_self_roundtrip() {
        let mut fw = FileWriter::new();
        fw.create_dataset("dummy").with_f64_data(&[0.0]);
        for i in 0..15 {
            fw.set_root_attr(&format!("root_{i:02}"), AttrValue::F64(i as f64 * 2.0));
        }
        let bytes = fw.finish().unwrap();
        let sig = signature::find_signature(&bytes).unwrap();
        let sb = Superblock::parse(&bytes, sig).unwrap();
        let oh = ObjectHeader::parse(
            &bytes,
            sb.root_group_address as usize,
            sb.offset_size,
            sb.length_size,
        )
        .unwrap();
        let attrs =
            crate::attribute::extract_attributes_full(&bytes, &oh, sb.offset_size, sb.length_size)
                .unwrap();
        assert_eq!(attrs.len(), 15);
    }

    #[test]
    fn inline_attrs_below_threshold() {
        let mut fw = FileWriter::new();
        let ds = fw.create_dataset("data");
        ds.with_f64_data(&[1.0]);
        for i in 0..5 {
            ds.set_attr(&format!("a{i}"), AttrValue::F64(i as f64));
        }
        let bytes = fw.finish().unwrap();
        let sig = signature::find_signature(&bytes).unwrap();
        let sb = Superblock::parse(&bytes, sig).unwrap();
        let addr = resolve_path_any(&bytes, &sb, "data").unwrap();
        let hdr =
            ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
        assert!(
            !hdr.messages
                .iter()
                .any(|m| m.msg_type == MessageType::AttributeInfo)
        );
        let attrs = crate::attribute::extract_attributes(&hdr, sb.length_size).unwrap();
        assert_eq!(attrs.len(), 5);
    }

    #[test]
    fn encode_decode_managed_id_roundtrip() {
        let id = encode_managed_id(100, 42, 40, 8);
        let fh = crate::fractal_heap::FractalHeapHeader {
            heap_id_length: 8,
            io_filter_encoded_length: 0,
            max_managed_object_size: 1024,
            table_width: 4,
            starting_block_size: 4096,
            max_direct_block_size: 65536,
            max_heap_size: 40,
            starting_row_of_indirect_blocks: 1,
            root_block_address: 0,
            current_rows_in_root_indirect_block: 0,
            managed_objects_count: 0,
        };
        let (off, len) = fh.decode_managed_id(&id).unwrap();
        assert_eq!(off, 100);
        assert_eq!(len, 42);
    }

    #[test]
    fn finalize_parallel_basic() {
        use crate::chunked_write::ChunkOptions;
        use crate::metadata_index::{MetadataBlock, build_dataset_metadata};
        use crate::type_builders::make_f64_type;

        let mut b0 = MetadataBlock::new(0);
        let data_a: Vec<u8> = [1.0f64, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        b0.add_dataset(build_dataset_metadata(
            "alpha",
            make_f64_type(),
            vec![3],
            data_a,
            ChunkOptions::default(),
            None,
            vec![],
        ));

        let mut b1 = MetadataBlock::new(1);
        let data_b: Vec<u8> = [10.0f64, 20.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        b1.add_dataset(build_dataset_metadata(
            "beta",
            make_f64_type(),
            vec![2],
            data_b,
            ChunkOptions::default(),
            None,
            vec![],
        ));

        let bytes = finalize_parallel(vec![b0, b1]).unwrap();
        assert_eq!(read_dataset_f64(&bytes, "alpha"), vec![1.0, 2.0, 3.0]);
        assert_eq!(read_dataset_f64(&bytes, "beta"), vec![10.0, 20.0]);
    }

    #[test]
    fn finalize_parallel_duplicate_error() {
        use crate::chunked_write::ChunkOptions;
        use crate::metadata_index::{MetadataBlock, build_dataset_metadata};
        use crate::type_builders::make_f64_type;

        let mut b0 = MetadataBlock::new(0);
        b0.add_dataset(build_dataset_metadata(
            "dup",
            make_f64_type(),
            vec![1],
            vec![0u8; 8],
            ChunkOptions::default(),
            None,
            vec![],
        ));
        let mut b1 = MetadataBlock::new(1);
        b1.add_dataset(build_dataset_metadata(
            "dup",
            make_f64_type(),
            vec![1],
            vec![0u8; 8],
            ChunkOptions::default(),
            None,
            vec![],
        ));
        let err = finalize_parallel(vec![b0, b1]).unwrap_err();
        assert!(matches!(err, FormatError::DuplicateDatasetName(_)));
    }

    // ---- Virtual Dataset (VDS) round-trip tests ----

    /// Serialize an H5S ALL selection (type=3, version=1, 16 bytes).
    fn sel_all() -> Vec<u8> {
        vec![
            3, 0, 0, 0, // type = ALL
            1, 0, 0, 0, // version
            0, 0, 0, 0, // reserved
            0, 0, 0, 0, // length (unused for ALL)
        ]
    }

    /// Serialize an H5S HYPER selection (version 3, rank 1, enc_size 2).
    /// Encodes start=`start`, stride=1, count=1, block=`block`.
    fn sel_hyper_1d(start: u16, block: u16) -> Vec<u8> {
        let mut v = vec![
            2, 0, 0, 0, // type = HYPER
            3, 0, 0, 0, // version 3
            0x01,       // flags = regular
            0x02,       // enc_size = 2 (u16 per coordinate)
            1, 0, 0, 0, // rank = 1
        ];
        v.extend_from_slice(&start.to_le_bytes()); // start
        v.extend_from_slice(&1u16.to_le_bytes());  // stride
        v.extend_from_slice(&1u16.to_le_bytes());  // count
        v.extend_from_slice(&block.to_le_bytes()); // block
        v
    }

    #[test]
    fn vds_write_read_virtual_layout() {
        use crate::data_layout::DataLayout;

        // A virtual dataset /vds of shape [8] backed by two same-file sources:
        // /src_a maps to virtual[0:4] and /src_b maps to virtual[4:8].
        let mapping_a = VdsMapping {
            source_file: ".".into(),
            source_dataset: "src_a".into(),
            source_selection: sel_all(),
            virtual_selection: sel_hyper_1d(0, 4),
        };
        let mapping_b = VdsMapping {
            source_file: ".".into(),
            source_dataset: "src_b".into(),
            source_selection: sel_all(),
            virtual_selection: sel_hyper_1d(4, 4),
        };

        let mut fw = FileWriter::new();
        // Source datasets (real data in this file)
        fw.create_dataset("src_a").with_f64_data(&[1.0, 2.0, 3.0, 4.0]);
        fw.create_dataset("src_b").with_f64_data(&[5.0, 6.0, 7.0, 8.0]);
        // Virtual dataset
        fw.create_dataset("vds")
            .with_shape(&[8])
            .with_f64_data(&[]) // shape hint; raw data is ignored for VDS
            .with_virtual_sources(vec![mapping_a, mapping_b]);

        let bytes = fw.finish().unwrap();

        // Verify the virtual dataset resolves to DataLayout::Virtual
        let sig = signature::find_signature(&bytes).unwrap();
        let sb = Superblock::parse(&bytes, sig).unwrap();
        let vds_addr = resolve_path_any(&bytes, &sb, "vds").unwrap();
        let hdr = ObjectHeader::parse(
            &bytes,
            vds_addr as usize,
            sb.offset_size,
            sb.length_size,
        )
        .unwrap();

        let dl_data = &hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::DataLayout)
            .unwrap()
            .data;

        let mut layout =
            DataLayout::parse(dl_data, sb.offset_size, sb.length_size).unwrap();

        // Before resolution, mappings field is empty.
        assert!(
            matches!(layout, DataLayout::Virtual { .. }),
            "expected Virtual layout, got {layout:?}"
        );

        // Resolve VDS mappings from the global heap.
        layout.resolve_vds_mappings(&bytes, sb.length_size).unwrap();

        match &layout {
            DataLayout::Virtual { mappings, .. } => {
                assert_eq!(mappings.len(), 2, "expected 2 VDS mappings");
                assert_eq!(mappings[0].source_file, ".");
                assert_eq!(mappings[0].source_dataset, "src_a");
                assert_eq!(mappings[1].source_file, ".");
                assert_eq!(mappings[1].source_dataset, "src_b");

                // Verify the virtual selections cover [0:4] and [4:8].
                use crate::selection::Selection;
                let (vsel_a, _) =
                    Selection::decode_serialized(&mappings[0].virtual_selection).unwrap();
                let (vsel_b, _) =
                    Selection::decode_serialized(&mappings[1].virtual_selection).unwrap();
                assert_eq!(vsel_a.iter_linear_1d(8).unwrap(), vec![0, 1, 2, 3]);
                assert_eq!(vsel_b.iter_linear_1d(8).unwrap(), vec![4, 5, 6, 7]);
            }
            other => panic!("expected Virtual layout after resolution, got {other:?}"),
        }

        // Source datasets still readable normally.
        assert_eq!(read_dataset_f64(&bytes, "src_a"), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(read_dataset_f64(&bytes, "src_b"), vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn vds_external_source_file() {
        use crate::data_layout::DataLayout;

        // A VDS mapping referencing an external file ("other.h5").
        let mapping_ext = VdsMapping {
            source_file: "other.h5".into(),
            source_dataset: "data".into(),
            source_selection: sel_all(),
            virtual_selection: sel_all(),
        };

        let mut fw = FileWriter::new();
        fw.create_dataset("ext_vds")
            .with_shape(&[10])
            .with_f64_data(&[]) // shape hint only
            .with_virtual_sources(vec![mapping_ext]);

        let bytes = fw.finish().unwrap();

        let sig = signature::find_signature(&bytes).unwrap();
        let sb = Superblock::parse(&bytes, sig).unwrap();
        let addr = resolve_path_any(&bytes, &sb, "ext_vds").unwrap();
        let hdr =
            ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
        let dl_data = &hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::DataLayout)
            .unwrap()
            .data;
        let mut layout =
            DataLayout::parse(dl_data, sb.offset_size, sb.length_size).unwrap();
        layout.resolve_vds_mappings(&bytes, sb.length_size).unwrap();

        match &layout {
            DataLayout::Virtual { mappings, .. } => {
                assert_eq!(mappings.len(), 1);
                assert_eq!(mappings[0].source_file, "other.h5");
                assert_eq!(mappings[0].source_dataset, "data");
            }
            other => panic!("expected Virtual, got {other:?}"),
        }
    }

    #[test]
    fn vds_empty_mapping_list() {
        // Calling with_virtual_sources([]) is silently ignored — the dataset
        // falls back to a normal contiguous layout rather than writing an empty VDS.
        use crate::data_layout::DataLayout;

        let mut fw = FileWriter::new();
        fw.create_dataset("empty_vds")
            .with_shape(&[0])
            .with_f64_data(&[])
            .with_virtual_sources(vec![]);

        let bytes = fw.finish().unwrap();

        let sig = signature::find_signature(&bytes).unwrap();
        let sb = Superblock::parse(&bytes, sig).unwrap();
        let addr = resolve_path_any(&bytes, &sb, "empty_vds").unwrap();
        let hdr =
            ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
        let dl_data = &hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::DataLayout)
            .unwrap()
            .data;
        let layout = DataLayout::parse(dl_data, sb.offset_size, sb.length_size).unwrap();

        // Empty mapping list → no VDS layout; should be Contiguous or Compact.
        assert!(
            !matches!(layout, DataLayout::Virtual { .. }),
            "empty with_virtual_sources should NOT produce a VDS layout, got {layout:?}"
        );
    }

    #[test]
    fn external_link_write_roundtrip() {
        let mut fw = FileWriter::new();
        let mut grp = fw.create_group("sensors");
        grp.create_dataset("local_ds").with_f64_data(&[1.0, 2.0]);
        grp.add_external_link("remote_temp", "other_file.h5", "/temperature");
        fw.add_group(grp.finish());

        let bytes = fw.finish().unwrap();

        let sig = signature::find_signature(&bytes).unwrap();
        let sb = Superblock::parse(&bytes, sig).unwrap();
        let sensors_addr = resolve_path_any(&bytes, &sb, "sensors").unwrap();
        let hdr = ObjectHeader::parse(
            &bytes,
            sensors_addr as usize,
            sb.offset_size,
            sb.length_size,
        )
        .unwrap();

        // Find the external LinkMessage directly in the object header.
        let ext_link = hdr
            .messages
            .iter()
            .filter(|m| m.msg_type == MessageType::Link)
            .filter_map(|m| crate::link_message::LinkMessage::parse(&m.data, sb.offset_size).ok())
            .find(|l| l.name == "remote_temp")
            .expect("external link 'remote_temp' not found in group OH");

        match &ext_link.link_target {
            crate::link_message::LinkTarget::External { filename, object_path } => {
                assert_eq!(filename, "other_file.h5");
                assert_eq!(object_path, "/temperature");
            }
            other => panic!("expected External link, got {other:?}"),
        }
    }
}
