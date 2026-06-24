//! Uncompress GOES-18 data for chunk modification testing
//!
//! Run with: cargo run --example uncompress_goes --release -- goes18_test.nc goes18_uncompressed.h5

use clawhdf5::{AttrValue, File, FileBuilder};
use clawhdf5_format::datatype::Datatype;
use clawhdf5_format::group_v2;
use clawhdf5_format::message_type::MessageType;
use clawhdf5_format::object_header::ObjectHeader;
use clawhdf5_format::signature;
use clawhdf5_format::superblock::Superblock;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let input_path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "goes18_test.nc".to_string());
    let output_path = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "goes18_uncompressed.h5".to_string());

    println!("=== GOES Data Uncompressor ===\n");
    println!("Input:  {}", input_path);
    println!("Output: {}", output_path);

    // Open the compressed file
    let file = File::open(&input_path)?;
    let file_data = std::fs::read(&input_path)?;
    let sig_offset = signature::find_signature(&file_data)?;
    let superblock = Superblock::parse(&file_data, sig_offset)?;

    // Find chunked datasets to uncompress
    let root = file.root();
    let datasets = root.datasets()?;

    println!("\nDatasets found: {:?}", datasets);

    let mut builder = FileBuilder::new();
    builder.set_attr("source", AttrValue::String(input_path.clone()));
    builder.set_attr("uncompressed", AttrValue::I64(1));

    // Process Rad dataset (main radiance data)
    if datasets.contains(&"Rad".to_string()) {
        println!("\n--- Processing 'Rad' dataset ---");

        let dataset = file.dataset("Rad")?;
        let shape = dataset.shape()?;
        let dtype = dataset.dtype()?;

        println!("  Shape: {:?}", shape);
        println!("  DType: {:?}", dtype);

        // Get detailed datatype info
        let addr = group_v2::resolve_path_any(&file_data, &superblock, "Rad")?;
        let header = ObjectHeader::parse(
            &file_data,
            addr as usize,
            superblock.offset_size,
            superblock.length_size,
        )?;

        let dt_msg = header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::Datatype)
            .ok_or("No Datatype")?;
        let (datatype, _) = Datatype::parse(&dt_msg.data)?;

        println!("  Raw datatype: {:?}", datatype);

        // Read the raw data (clawhdf5 handles decompression automatically)
        match &datatype {
            Datatype::FixedPoint { size, signed, .. } => {
                println!("  FixedPoint: size={}, signed={}", size, signed);

                if *size == 2 && *signed {
                    // i16 data - read as raw bytes and convert
                    let raw_data = read_raw_dataset(&file, "Rad")?;
                    let num_elements = raw_data.len() / 2;

                    // Convert to i32 for writing (clawhdf5 supports i32)
                    let mut i32_data = Vec::with_capacity(num_elements);
                    for chunk in raw_data.chunks_exact(2) {
                        let val = i16::from_le_bytes([chunk[0], chunk[1]]);
                        i32_data.push(val as i32);
                    }

                    println!(
                        "  Read {} elements, first 5: {:?}",
                        num_elements,
                        &i32_data[..5.min(num_elements)]
                    );

                    // Create chunked dataset (250x250 to match original)
                    builder
                        .create_dataset("Rad")
                        .with_i32_data(&i32_data)
                        .with_shape(&shape)
                        .with_chunks(&[250, 250])
                        .set_attr("original_dtype", AttrValue::String("int16".into()));

                    println!("  Added to output (as i32, chunked 250x250)");
                }
            }
            _ => {
                println!("  Skipping - unsupported datatype");
            }
        }
    }

    // Process DQF dataset (data quality flags)
    if datasets.contains(&"DQF".to_string()) {
        println!("\n--- Processing 'DQF' dataset ---");

        let dataset = file.dataset("DQF")?;
        let shape = dataset.shape()?;
        let dtype = dataset.dtype()?;

        println!("  Shape: {:?}", shape);
        println!("  DType: {:?}", dtype);

        // Get detailed datatype info
        let addr = group_v2::resolve_path_any(&file_data, &superblock, "DQF")?;
        let header = ObjectHeader::parse(
            &file_data,
            addr as usize,
            superblock.offset_size,
            superblock.length_size,
        )?;

        let dt_msg = header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::Datatype)
            .ok_or("No Datatype")?;
        let (datatype, _) = Datatype::parse(&dt_msg.data)?;

        match &datatype {
            Datatype::FixedPoint { size, signed, .. } => {
                println!("  FixedPoint: size={}, signed={}", size, signed);

                // Read raw bytes
                let raw_data = read_raw_dataset(&file, "DQF")?;

                if *size == 1 {
                    // u8/i8 data - convert to i32
                    let i32_data: Vec<i32> = raw_data.iter().map(|&b| b as i32).collect();
                    println!("  Read {} elements", i32_data.len());

                    builder
                        .create_dataset("DQF")
                        .with_i32_data(&i32_data)
                        .with_shape(&shape)
                        .with_chunks(&[250, 250])
                        .set_attr("original_dtype", AttrValue::String("uint8".into()));

                    println!("  Added to output (as i32, chunked 250x250)");
                }
            }
            _ => {
                println!("  Skipping - unsupported datatype");
            }
        }
    }

    // Write the uncompressed file
    println!("\n--- Writing uncompressed file ---");
    builder.write(&output_path)?;

    let output_size = std::fs::metadata(&output_path)?.len();
    println!(
        "  Written: {} bytes ({:.2} KB)",
        output_size,
        output_size as f64 / 1024.0
    );

    // Verify the output
    println!("\n--- Verifying output ---");
    let verify_file = File::open(&output_path)?;
    let verify_root = verify_file.root();
    let verify_datasets = verify_root.datasets()?;
    println!("  Datasets in output: {:?}", verify_datasets);

    for name in &verify_datasets {
        let ds = verify_file.dataset(name)?;
        println!(
            "  {}: shape={:?}, dtype={:?}",
            name,
            ds.shape()?,
            ds.dtype()?
        );
    }

    println!("\n=== Done! ===");
    println!(
        "You can now run chunk modification tests on: {}",
        output_path
    );

    Ok(())
}

/// Read raw bytes from a dataset (decompressed by clawhdf5)
fn read_raw_dataset(file: &File, name: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    use clawhdf5_format::data_read;
    use clawhdf5_format::dataspace::Dataspace;

    let file_data = file.as_bytes();
    let sig_offset = signature::find_signature(file_data)?;
    let superblock = Superblock::parse(file_data, sig_offset)?;

    let addr = group_v2::resolve_path_any(file_data, &superblock, name)?;
    let header = ObjectHeader::parse(
        file_data,
        addr as usize,
        superblock.offset_size,
        superblock.length_size,
    )?;

    // Get datatype
    let dt_msg = header
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::Datatype)
        .ok_or("No Datatype")?;
    let (datatype, _) = Datatype::parse(&dt_msg.data)?;

    // Get dataspace
    let ds_msg = header
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::Dataspace)
        .ok_or("No Dataspace")?;
    let dataspace = Dataspace::parse(&ds_msg.data, superblock.length_size)?;

    // Get layout
    let dl_msg = header
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::DataLayout)
        .ok_or("No DataLayout")?;
    let layout = clawhdf5_format::data_layout::DataLayout::parse(
        &dl_msg.data,
        superblock.offset_size,
        superblock.length_size,
    )?;

    // Get filter pipeline if present
    let pipeline = header
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::FilterPipeline)
        .and_then(|msg| clawhdf5_format::filter_pipeline::FilterPipeline::parse(&msg.data).ok());

    // Read raw data with decompression via read_raw_data_full
    let raw = data_read::read_raw_data_full(
        file_data,
        &layout,
        &dataspace,
        &datatype,
        pipeline.as_ref(),
        superblock.offset_size,
        superblock.length_size,
    )?;

    Ok(raw)
}
