//! Generate merkle test vectors JSON.
//!
//! Run with: cargo run --features merkle --example gen_merkle_vectors

use clawhdf5_format::merkle::{HashAlg, MerkleTree, hash_chunk};

const INTERNAL_PREFIX: u8 = 0x01;

fn hash_pair_blake3(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut combined = [0u8; 65];
    combined[0] = INTERNAL_PREFIX;
    combined[1..33].copy_from_slice(left);
    combined[33..65].copy_from_slice(right);
    blake3::hash(&combined).into()
}

fn hash_pair_sha256(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    use sha2::{Sha256, Digest};
    let mut combined = [0u8; 65];
    combined[0] = INTERNAL_PREFIX;
    combined[1..33].copy_from_slice(left);
    combined[33..65].copy_from_slice(right);
    Sha256::digest(&combined).into()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn main() {
    println!("{{");
    println!("  \"description\": \"Merkle tree test vectors for clawhdf5\",");
    println!("  \"specification\": \"§5.5 Domain-separated Merkle trees\",");
    println!("  \"domain_separation\": {{");
    println!("    \"leaf_prefix\": \"0x00\",");
    println!("    \"internal_prefix\": \"0x01\",");
    println!("    \"null_prefix\": \"0x02\"");
    println!("  }},");

    // Single leaf test (BLAKE3)
    let leaf_data = b"single leaf data";
    let leaf_hash = hash_chunk(leaf_data, HashAlg::Blake3);
    let tree1 = MerkleTree::build(&[leaf_hash], HashAlg::Blake3).unwrap();

    println!("  \"single_leaf_blake3\": {{");
    println!("    \"input\": \"single leaf data\",");
    println!("    \"leaf_hash\": \"{}\",", hex(&leaf_hash));
    println!("    \"root\": \"{}\",", hex(tree1.root()));
    println!("    \"note\": \"For single-leaf tree, root equals leaf hash\"");
    println!("  }},");

    // Two leaf test (BLAKE3)
    let leaf0 = hash_chunk(b"leaf zero", HashAlg::Blake3);
    let leaf1 = hash_chunk(b"leaf one", HashAlg::Blake3);
    let root2 = hash_pair_blake3(&leaf0, &leaf1);
    let tree2 = MerkleTree::build(&[leaf0, leaf1], HashAlg::Blake3).unwrap();

    println!("  \"two_leaf_blake3\": {{");
    println!("    \"inputs\": [\"leaf zero\", \"leaf one\"],");
    println!("    \"leaf_hashes\": [");
    println!("      \"{}\",", hex(&leaf0));
    println!("      \"{}\"", hex(&leaf1));
    println!("    ],");
    println!("    \"root\": \"{}\",", hex(tree2.root()));
    println!("    \"computation\": \"H(0x01 || leaf0 || leaf1)\"");
    println!("  }},");

    // Verify our manual computation matches
    assert_eq!(&root2, tree2.root(), "Two-leaf root mismatch");

    // Eight leaf test (BLAKE3)
    let leaves: Vec<[u8; 32]> = (0u8..8)
        .map(|i| hash_chunk(&[b'L', i], HashAlg::Blake3))
        .collect();

    let tree8 = MerkleTree::build(&leaves, HashAlg::Blake3).unwrap();

    // Level 2
    let n3 = hash_pair_blake3(&leaves[0], &leaves[1]);
    let n4 = hash_pair_blake3(&leaves[2], &leaves[3]);
    let n5 = hash_pair_blake3(&leaves[4], &leaves[5]);
    let n6 = hash_pair_blake3(&leaves[6], &leaves[7]);

    // Level 1
    let n1 = hash_pair_blake3(&n3, &n4);
    let n2 = hash_pair_blake3(&n5, &n6);

    // Root
    let root8 = hash_pair_blake3(&n1, &n2);

    // Verify
    assert_eq!(&root8, tree8.root(), "Eight-leaf root mismatch");

    println!("  \"eight_leaf_blake3\": {{");
    println!("    \"inputs\": [\"L\\\\x00\", \"L\\\\x01\", \"L\\\\x02\", \"L\\\\x03\", \"L\\\\x04\", \"L\\\\x05\", \"L\\\\x06\", \"L\\\\x07\"],");
    println!("    \"leaf_hashes\": [");
    for (i, leaf) in leaves.iter().enumerate() {
        let comma = if i < 7 { "," } else { "" };
        println!("      \"{}\"{}", hex(leaf), comma);
    }
    println!("    ],");
    println!("    \"internal_nodes\": {{");
    println!("      \"n3_H(L0,L1)\": \"{}\",", hex(&n3));
    println!("      \"n4_H(L2,L3)\": \"{}\",", hex(&n4));
    println!("      \"n5_H(L4,L5)\": \"{}\",", hex(&n5));
    println!("      \"n6_H(L6,L7)\": \"{}\",", hex(&n6));
    println!("      \"n1_H(n3,n4)\": \"{}\",", hex(&n1));
    println!("      \"n2_H(n5,n6)\": \"{}\"", hex(&n2));
    println!("    }},");
    println!("    \"root\": \"{}\"", hex(tree8.root()));
    println!("  }},");

    // SHA-256 vectors for interoperability
    let leaf0_sha = hash_chunk(b"leaf zero", HashAlg::Sha256);
    let leaf1_sha = hash_chunk(b"leaf one", HashAlg::Sha256);
    let root2_sha = hash_pair_sha256(&leaf0_sha, &leaf1_sha);
    let tree2_sha = MerkleTree::build(&[leaf0_sha, leaf1_sha], HashAlg::Sha256).unwrap();

    assert_eq!(&root2_sha, tree2_sha.root(), "SHA-256 two-leaf root mismatch");

    println!("  \"two_leaf_sha256\": {{");
    println!("    \"inputs\": [\"leaf zero\", \"leaf one\"],");
    println!("    \"leaf_hashes\": [");
    println!("      \"{}\",", hex(&leaf0_sha));
    println!("      \"{}\"", hex(&leaf1_sha));
    println!("    ],");
    println!("    \"root\": \"{}\"", hex(tree2_sha.root()));
    println!("  }}");
    println!("}}");

    eprintln!("All assertions passed!");
}
