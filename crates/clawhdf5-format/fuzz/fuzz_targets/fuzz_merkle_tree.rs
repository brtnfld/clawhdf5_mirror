#![no_main]
use libfuzzer_sys::fuzz_target;

use clawhdf5_format::merkle::{Dataset, HashAlg, MerkleAttr};

fuzz_target!(|data: &[u8]| {
    // Create a minimal valid MerkleAttr for testing
    // We're primarily fuzzing the tree_nodes parsing, not the attribute itself
    let merkle_attr = MerkleAttr {
        root: [0u8; 32],
        algorithm: HashAlg::Blake3,
        integrity: [0u8; 32],
        companion_hash: [1u8; 32], // Non-zero to indicate companion exists
    };

    // Fuzz Dataset::reconstruct_tree with arbitrary tree_nodes data
    // This should handle any input without panicking:
    // - Empty data
    // - Non-multiple of 32 bytes
    // - Odd node count check
    // - Non-power-of-two padded_count check
    // - Size validation
    let dataset = Dataset::from_owned(merkle_attr, data.to_vec(), vec![]);
    let _ = dataset.reconstruct_tree();

    // If reconstruction succeeds, try verification functions
    // These should not panic even with mismatched chunks
    if let Ok(tree) = dataset.reconstruct_tree() {
        // Try leaf_hash access at various indices
        let _ = tree.leaf_hash(0);
        let _ = tree.leaf_hash(tree.leaf_count().saturating_sub(1));
        let _ = tree.leaf_hash(tree.leaf_count()); // Out of bounds - should return None
        let _ = tree.leaf_hash(usize::MAX); // Way out of bounds

        // Access tree properties
        let _ = tree.root();
        let _ = tree.depth();
        let _ = tree.leaf_count();
        let _ = tree.padded_leaf_count();
    }
});
