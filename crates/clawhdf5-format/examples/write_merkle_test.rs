//! Write a test HDF5 file with Merkle companion dataset.
//!
//! Run with: cargo run --features merkle --example write_merkle_test

use clawhdf5_format::file_writer::FileWriter;
use clawhdf5_format::merkle::{
    hash_chunk, write_merkle_companion, HashAlg, MerkleAttr, MerkleCompanionResult,
    MERKLE_ATTR_NAME,
};
use clawhdf5_format::type_builders::AttrValue;
use std::fs;

fn main() {
    // Create 1024 synthetic chunks (above the 256 inline threshold)
    let chunks: Vec<Vec<u8>> = (0..1024)
        .map(|i| {
            let mut chunk = vec![0u8; 64];
            for (j, byte) in chunk.iter_mut().enumerate() {
                *byte = ((i + j) % 256) as u8;
            }
            chunk
        })
        .collect();

    // Build Merkle tree
    let leaf_hashes: Vec<[u8; 32]> = chunks
        .iter()
        .map(|c| hash_chunk(c, HashAlg::Blake3))
        .collect();
    let tree = clawhdf5_format::merkle::MerkleTree::build(&leaf_hashes, HashAlg::Blake3)
        .expect("build tree");

    println!("Merkle tree built:");
    println!("  Leaf count: {}", tree.leaf_count());
    println!("  Padded leaf count: {}", tree.padded_leaf_count());
    println!("  Total nodes: {}", tree.nodes().len());
    println!("  Root: {}", hex::encode(tree.root()));

    // Write the file
    let mut fw = FileWriter::new();

    // Write merkle companion first (creates /merkle/sensor_data)
    let result = write_merkle_companion(&mut fw, "sensor_data", &tree)
        .expect("write_merkle_companion should succeed");

    let companion_hash = match &result {
        MerkleCompanionResult::Dataset { companion_hash } => {
            println!("  Companion stored as dataset at /merkle/sensor_data");
            *companion_hash
        }
        MerkleCompanionResult::Inline { companion_hash, .. } => {
            println!("  Companion stored inline in attribute");
            *companion_hash
        }
    };
    println!("  Companion hash: {}", hex::encode(&companion_hash));

    // Create the main dataset
    let ds = fw.create_dataset("sensor_data");
    let all_data: Vec<u8> = chunks.iter().flatten().copied().collect();
    ds.with_u8_data(&all_data);

    // Write merkle_root attribute
    let attr = MerkleAttr::from_tree_with_companion(&tree, companion_hash);
    ds.set_attr(MERKLE_ATTR_NAME, AttrValue::Bytes(attr.pack().to_vec()));

    // Finish and write to disk
    let file_bytes = fw.finish().expect("file should build");

    let output_path = "merkle_test.h5";
    fs::write(output_path, &file_bytes).expect("write file");

    println!("\nWrote {} bytes to {}", file_bytes.len(), output_path);
    println!("Verify with: python3 -c \"import h5py; f=h5py.File('{}', 'r'); print(list(f.keys())); print(list(f['merkle'].keys()))\"", output_path);
}

mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }
}
