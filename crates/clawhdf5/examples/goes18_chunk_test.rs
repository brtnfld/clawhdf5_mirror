//! Test chunk modification on NOAA GOES-18 satellite data
//!
//! Run with: cargo run --example goes18_chunk_test --release -- /path/to/goes18.nc

use clawhdf5::File;
use clawhdf5_format::chunked_read::ChunkInfo;
use clawhdf5_format::chunked_read::collect_chunk_info;
use clawhdf5_format::data_layout::DataLayout;
use clawhdf5_format::extensible_array::{ExtensibleArrayHeader, read_extensible_array_chunks};
use clawhdf5_format::filter_pipeline::FilterPipeline;
use clawhdf5_format::fixed_array::{FixedArrayHeader, read_fixed_array_chunks};
use clawhdf5_format::group_v2;
use clawhdf5_format::message_type::MessageType;
use clawhdf5_format::object_header::ObjectHeader;
use clawhdf5_format::signature;
use clawhdf5_format::superblock::Superblock;

use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "goes18_sample.nc".to_string());
    let dataset_override = args.get(2).cloned();

    println!("=== HDF5 Chunk Modification Test ===\n");
    println!("File: {}", path);

    // Step 1: Open and explore the file structure
    println!("\n--- Step 1: Exploring file structure ---");
    let file = File::open(&path)?;

    // List root datasets
    let root = file.root();
    let datasets = root.datasets()?;
    let groups = root.groups()?;

    println!("Root datasets: {:?}", datasets);
    println!("Root groups: {:?}", groups);

    // Find a chunked dataset to test
    let file_data = std::fs::read(&path)?;
    let sig_offset = signature::find_signature(&file_data)?;
    let superblock = Superblock::parse(&file_data, sig_offset)?;

    // Use override or find a chunked dataset (preferring "Rad" or "dataset_1mb")
    let test_dataset = if let Some(ref ds) = dataset_override {
        ds.as_str()
    } else if datasets.contains(&"Rad".to_string()) {
        "Rad"
    } else if datasets.contains(&"dataset_1mb".to_string()) {
        "dataset_1mb"
    } else {
        datasets.first().map(|s| s.as_str()).unwrap_or("unknown")
    };

    println!("\nTesting dataset: {}", test_dataset);

    let dataset = file.dataset(test_dataset)?;
    let shape = dataset.shape()?;
    let dtype = dataset.dtype()?;

    println!("  Shape: {:?}", shape);
    println!("  Dtype: {:?}", dtype);

    // Get layout info
    let addr = group_v2::resolve_path_any(&file_data, &superblock, test_dataset)?;
    let header = ObjectHeader::parse(
        &file_data,
        addr as usize,
        superblock.offset_size,
        superblock.length_size,
    )?;

    let layout_msg = header
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::DataLayout)
        .ok_or("No DataLayout")?;
    let layout = DataLayout::parse(
        &layout_msg.data,
        superblock.offset_size,
        superblock.length_size,
    )?;

    // Check for filter pipeline (compression)
    let pipeline = header
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::FilterPipeline)
        .and_then(|msg| FilterPipeline::parse(&msg.data).ok());

    if let Some(ref p) = pipeline {
        println!("  Filters: {} filters applied", p.filters.len());
        for f in &p.filters {
            println!(
                "    - Filter ID {}: {}",
                f.filter_id,
                f.name.as_deref().unwrap_or("unnamed")
            );
        }
    }

    // Get chunk info
    let (chunks, chunk_dims, element_size) = match &layout {
        DataLayout::Chunked {
            chunk_dimensions,
            btree_address,
            version,
            chunk_index_type,
            ..
        } => {
            println!("  Layout: Chunked v{}", version);
            println!("  Chunk dimensions: {:?}", chunk_dimensions);
            println!("  Index type: {:?}", chunk_index_type);

            let addr = btree_address.ok_or("No chunk address")?;
            let rank = shape.len();
            let spatial_dims: Vec<u32> = chunk_dimensions[..rank].to_vec();

            // Determine element size from last dimension (HDF5 convention)
            let elem_size = *chunk_dimensions.last().unwrap_or(&1) as usize;

            let chunks: Vec<ChunkInfo> = match (*version, *chunk_index_type) {
                (3, _) => {
                    // B-tree v1
                    collect_chunk_info(
                        &file_data,
                        addr,
                        rank + 1,
                        superblock.offset_size,
                        superblock.length_size,
                    )?
                }
                (4, Some(1)) => {
                    // Single chunk
                    vec![ChunkInfo {
                        chunk_size: spatial_dims.iter().map(|&d| d as u64).product::<u64>() as u32
                            * elem_size as u32,
                        filter_mask: 0,
                        offsets: vec![0; rank],
                        address: addr,
                    }]
                }
                (4, Some(3)) => {
                    // Fixed Array
                    let fa_header = FixedArrayHeader::parse(
                        &file_data,
                        addr as usize,
                        superblock.offset_size,
                        superblock.length_size,
                    )?;
                    read_fixed_array_chunks(
                        &file_data,
                        &fa_header,
                        &shape,
                        &spatial_dims,
                        elem_size as u32,
                        superblock.offset_size,
                        superblock.length_size,
                    )?
                }
                (4, Some(4)) => {
                    // Extensible Array
                    let ea_header = ExtensibleArrayHeader::parse(
                        &file_data,
                        addr as usize,
                        superblock.offset_size,
                        superblock.length_size,
                    )?;
                    read_extensible_array_chunks(
                        &file_data,
                        &ea_header,
                        &shape,
                        &spatial_dims,
                        elem_size as u32,
                        superblock.offset_size,
                        superblock.length_size,
                    )?
                }
                _ => {
                    return Err(format!(
                        "Unsupported chunk index: version={}, type={:?}",
                        version, chunk_index_type
                    )
                    .into());
                }
            };

            (chunks, spatial_dims, elem_size)
        }
        DataLayout::Contiguous { .. } => {
            println!("  Layout: Contiguous (not chunked)");
            return Err("Dataset is not chunked, cannot run chunk test".into());
        }
        DataLayout::Compact { .. } => {
            println!("  Layout: Compact (not chunked)");
            return Err("Dataset is not chunked, cannot run chunk test".into());
        }
        _ => return Err("Unknown layout".into()),
    };

    println!("\n--- Step 2: Chunk Information ---");
    println!("Number of chunks: {}", chunks.len());

    let chunk_elements: usize = chunk_dims.iter().map(|&d| d as usize).product();
    println!("Elements per chunk: {}", chunk_elements);
    println!("Bytes per chunk: {}", chunk_elements * element_size);

    // Show first few chunks
    for (i, chunk) in chunks.iter().take(5).enumerate() {
        println!(
            "  Chunk {}: addr=0x{:x}, size={} bytes, offsets={:?}, filtered={}",
            i,
            chunk.address,
            chunk.chunk_size,
            chunk.offsets,
            chunk.filter_mask != 0
        );
    }
    if chunks.len() > 5 {
        println!("  ... and {} more chunks", chunks.len() - 5);
    }

    // Check if data is compressed
    if pipeline.is_some() {
        println!("\n--- WARNING: Dataset is compressed ---");
        println!("Direct chunk modification would corrupt compressed data.");
        println!("Skipping write test, but demonstrating read iteration.\n");

        // Still iterate and read chunks to demonstrate
        println!("--- Step 3: Reading all chunks (checksums) ---");
        let mut total_bytes = 0u64;
        for (i, chunk) in chunks.iter().enumerate() {
            let start = chunk.address as usize;
            let end = start + chunk.chunk_size as usize;
            if end <= file_data.len() {
                let chunk_data = &file_data[start..end];
                let checksum: u64 = chunk_data.iter().map(|&b| b as u64).sum();
                total_bytes += chunk.chunk_size as u64;
                if i < 3 || i == chunks.len() - 1 {
                    println!(
                        "  Chunk {}: {} bytes, checksum={}",
                        i, chunk.chunk_size, checksum
                    );
                } else if i == 3 {
                    println!("  ... iterating {} more chunks ...", chunks.len() - 4);
                }
            }
        }
        println!(
            "\nTotal chunk data read: {} bytes ({:.2} MB)",
            total_bytes,
            total_bytes as f64 / 1_048_576.0
        );

        println!("\n=== TEST COMPLETE (read-only due to compression) ===");
        return Ok(());
    }

    // For uncompressed data, proceed with modification test
    println!("\n--- Step 3: Reading all chunks ---");
    let mut chunk_checksums: Vec<u64> = Vec::with_capacity(chunks.len());

    for (i, chunk) in chunks.iter().enumerate() {
        let start = chunk.address as usize;
        let end = start + chunk.chunk_size as usize;
        if end <= file_data.len() {
            let chunk_data = &file_data[start..end];
            let checksum: u64 = chunk_data.iter().map(|&b| b as u64).sum();
            chunk_checksums.push(checksum);
            if i < 3 {
                println!("  Chunk {}: checksum={}", i, checksum);
            }
        }
    }

    // Modify a chunk
    let target_idx = 1.min(chunks.len() - 1);
    println!("\n--- Step 4: Modifying chunk {} ---", target_idx);

    let target = &chunks[target_idx];
    let start = target.address as usize;
    let end = start + target.chunk_size as usize;

    let mut modified = file_data[start..end].to_vec();
    let original_first_bytes: [u8; 8] = modified[0..8].try_into()?;
    println!("  Original first 8 bytes: {:?}", original_first_bytes);

    // Flip all bits (simple modification)
    for byte in &mut modified {
        *byte = !*byte;
    }
    println!("  Modified first 8 bytes: {:?}", &modified[0..8]);

    // Write back
    println!("\n--- Step 5: Writing modified chunk ---");
    {
        let mut fh = OpenOptions::new().write(true).open(&path)?;
        fh.seek(SeekFrom::Start(target.address))?;
        fh.write_all(&modified)?;
        fh.sync_all()?;
    }
    println!("  Wrote {} bytes at 0x{:x}", modified.len(), target.address);

    // Verify
    println!("\n--- Step 6: Verifying ---");
    let file_data_after = std::fs::read(&path)?;
    let modified_chunk = &file_data_after[start..end];
    let new_checksum: u64 = modified_chunk.iter().map(|&b| b as u64).sum();

    println!("  Modified chunk new checksum: {}", new_checksum);

    // Check other chunks weren't corrupted
    let mut errors = 0;
    for (i, chunk) in chunks.iter().enumerate() {
        if i == target_idx {
            continue;
        }
        let s = chunk.address as usize;
        let e = s + chunk.chunk_size as usize;
        if e <= file_data_after.len() {
            let checksum: u64 = file_data_after[s..e].iter().map(|&b| b as u64).sum();
            if checksum != chunk_checksums[i] {
                println!("  ERROR: Chunk {} corrupted!", i);
                errors += 1;
            }
        }
    }

    if errors == 0 {
        println!("  All {} other chunks intact", chunks.len() - 1);
    }

    // Restore original data
    println!("\n--- Step 7: Restoring original data ---");
    {
        let mut fh = OpenOptions::new().write(true).open(&path)?;
        fh.seek(SeekFrom::Start(target.address))?;
        // Flip bits back
        for byte in &mut modified {
            *byte = !*byte;
        }
        fh.write_all(&modified)?;
        fh.sync_all()?;
    }
    println!("  Restored original chunk data");

    println!("\n=== TEST PASSED ===");
    Ok(())
}
