//! Merkle tree implementation for chunk integrity verification.
//!
//! Supports SHA-256, BLAKE3, and KangarooTwelve (K12) hash algorithms.
//! Useful for verifying individual chunks without rehashing the entire dataset.
//!
//! # Security
//!
//! This implementation uses:
//! - Domain separation (leaf prefix `0x00`, internal prefix `0x01`) to prevent
//!   second-preimage attacks
//! - Constant-time hash comparison to prevent timing attacks
//!
//! # Example
//!
//! ```ignore
//! use clawhdf5_format::merkle::{MerkleTree, HashAlg};
//!
//! // Build tree from chunk data
//! let chunks: Vec<&[u8]> = vec![&[1, 2, 3], &[4, 5, 6], &[7, 8, 9]];
//! let tree = MerkleTree::from_chunks(&chunks, HashAlg::Blake3);
//!
//! // Generate proof for chunk 1
//! let proof = tree.proof(1).unwrap();
//!
//! // Verify the proof
//! assert!(tree.verify_proof(1, &chunks[1], &proof));
//! ```

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Size of hash output in bytes (256 bits).
const HASH_SIZE: usize = 32;

/// Domain separator for leaf node hashes.
const LEAF_PREFIX: u8 = 0x00;

/// Domain separator for internal node hashes.
const INTERNAL_PREFIX: u8 = 0x01;

/// Domain separator for null/unallocated sparse-chunk slots.
const NULL_PREFIX: u8 = 0x02;

/// Maximum tree depth (supports up to 2^40 ≈ 1 trillion chunks).
/// Prevents out-of-memory attacks from maliciously large inputs.
const MAX_DEPTH: usize = 40;

/// The null sentinel string used for padding sparse-chunk slots.
const NULL_SENTINEL_DATA: &[u8] = b"null";

/// Errors that can occur during Merkle tree construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MerkleError {
    /// Tree depth exceeds maximum allowed (40 levels, supporting up to 2^40 chunks).
    /// This prevents out-of-memory attacks from maliciously large inputs.
    TreeTooDeep {
        /// Requested depth.
        requested: usize,
        /// Maximum allowed depth.
        max: usize,
    },
}

impl core::fmt::Display for MerkleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MerkleError::TreeTooDeep { requested, max } => {
                write!(
                    f,
                    "Merkle tree depth {} exceeds maximum allowed depth {}",
                    requested, max
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MerkleError {}

/// Hash algorithm selection for Merkle tree construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum HashAlg {
    /// SHA-256 (FIPS 180-4) - widely supported, good for interoperability
    Sha256,
    /// BLAKE3 - fast, parallelizable, modern design (default)
    #[default]
    Blake3,
    /// KangarooTwelve - very fast, based on Keccak/SHA-3
    K12,
}

impl HashAlg {
    /// Hash a single block of data, returning a 32-byte digest.
    #[inline]
    fn hash_raw(&self, data: &[u8]) -> [u8; HASH_SIZE] {
        match self {
            HashAlg::Sha256 => hash_sha256(data),
            HashAlg::Blake3 => hash_blake3(data),
            HashAlg::K12 => hash_k12(data),
        }
    }

    /// Hash leaf data with domain separation prefix.
    ///
    /// Prefixes data with `0x00` to distinguish leaf hashes from internal nodes.
    /// Uses incremental hashing APIs to avoid memory allocation.
    ///
    /// This is equivalent to calling [`hash_chunk`] with the same algorithm.
    #[inline]
    pub fn hash_leaf(&self, data: &[u8]) -> [u8; HASH_SIZE] {
        hash_chunk(data, *self)
    }

    /// Hash two 32-byte child hashes together with domain separation.
    ///
    /// Prefixes with `0x01` to distinguish internal node hashes from leaves.
    #[inline]
    pub fn hash_pair(&self, left: &[u8; HASH_SIZE], right: &[u8; HASH_SIZE]) -> [u8; HASH_SIZE] {
        let mut combined = [0u8; 1 + HASH_SIZE * 2];
        combined[0] = INTERNAL_PREFIX;
        combined[1..HASH_SIZE + 1].copy_from_slice(left);
        combined[HASH_SIZE + 1..].copy_from_slice(right);
        self.hash_raw(&combined)
    }

    /// Compute the null sentinel hash for padding sparse-chunk slots.
    ///
    /// Returns `H(0x02 || "null")` as specified in §5.5 for unallocated chunks.
    /// This domain-separated value prevents crafted payloads from colliding
    /// with the null constant.
    #[inline]
    #[must_use]
    pub fn null_sentinel(&self) -> [u8; HASH_SIZE] {
        let mut prefixed = [0u8; 1 + NULL_SENTINEL_DATA.len()];
        prefixed[0] = NULL_PREFIX;
        prefixed[1..].copy_from_slice(NULL_SENTINEL_DATA);
        self.hash_raw(&prefixed)
    }
}

/// Compute the leaf hash for a raw chunk of data.
///
/// This is the primary entry point for hashing chunk data with the correct
/// domain separation prefix (`0x00`). Uses incremental hashing APIs to avoid
/// memory allocation for large chunks.
///
/// # BLAKE3 API Note
///
/// Uses `Hasher::new().update(&[0x00]).update(data).finalize()` (the flat hash API).
/// Does **not** use BLAKE3's internal tree mode (`new_derive_key` or implicit 1KB
/// chunking), which does not expose intermediate nodes and cannot produce per-chunk
/// proofs. BLAKE3 tree mode is benchmarked for throughput only (RQ4).
///
/// # Example
///
/// ```ignore
/// use clawhdf5_format::merkle::{hash_chunk, HashAlg};
///
/// let chunk_data = b"some chunk data";
/// let leaf_hash = hash_chunk(chunk_data, HashAlg::Blake3);
/// ```
#[inline]
#[must_use]
pub fn hash_chunk(data: &[u8], alg: HashAlg) -> [u8; HASH_SIZE] {
    match alg {
        HashAlg::Sha256 => hash_chunk_sha256(data),
        HashAlg::Blake3 => hash_chunk_blake3(data),
        HashAlg::K12 => hash_chunk_k12(data),
    }
}

/// Constant-time comparison of two hash values.
///
/// Prevents timing attacks by always comparing all bytes regardless of
/// where the first difference occurs.
#[inline]
fn constant_time_eq(a: &[u8; HASH_SIZE], b: &[u8; HASH_SIZE]) -> bool {
    let mut result = 0u8;
    for i in 0..HASH_SIZE {
        result |= a[i] ^ b[i];
    }
    result == 0
}

/// A Merkle tree for verifying chunk integrity.
///
/// The tree is stored in a flat array in level-order (breadth-first),
/// with the root at index 0. This layout is cache-friendly and allows
/// O(1) parent/child index calculations.
///
/// For `n` leaves (padded to power of 2), the tree contains `2n - 1` nodes.
#[derive(Debug, Clone)]
pub struct MerkleTree {
    /// All nodes in level-order (root at index 0).
    nodes: Vec<[u8; HASH_SIZE]>,
    /// Number of actual data chunks (before padding).
    leaf_count: usize,
    /// Number of leaves after padding to power of 2.
    padded_count: usize,
    /// Hash algorithm used.
    alg: HashAlg,
}

/// Compute the depth required for a given padded leaf count.
///
/// Returns `ceil(log2(padded_count))` for padded_count > 1, or 1 for single leaf.
#[inline]
fn compute_depth(padded_count: usize) -> usize {
    if padded_count <= 1 {
        1
    } else {
        // Since padded_count is always a power of 2, ilog2 is exact
        padded_count.ilog2() as usize + 1
    }
}

impl MerkleTree {
    /// Build a Merkle tree from pre-hashed leaves with depth validation.
    ///
    /// This is the recommended constructor for building trees from pre-computed
    /// leaf hashes. It enforces the maximum depth constraint (40 levels, supporting
    /// up to 2^40 ≈ 1 trillion chunks) and uses the cryptographically correct
    /// null sentinel `H(0x02 || "null")` for padding sparse-chunk slots.
    ///
    /// # Arguments
    ///
    /// * `leaves` - Pre-hashed leaf values (32-byte digests with leaf domain
    ///   separation already applied via `HashAlg::hash_leaf`)
    /// * `alg` - Hash algorithm to use for internal node computation
    ///
    /// # Returns
    ///
    /// * `Ok(MerkleTree)` - Successfully constructed tree
    /// * `Err(MerkleError::TreeTooDeep)` - Leaf count exceeds maximum (2^40)
    ///
    /// # Tree Structure
    ///
    /// Scientific datasets rarely have power-of-2 chunk counts. This method
    /// pads the leaf array with the null sentinel up to the next power of 2,
    /// maintaining correct level-order index arithmetic (left = 2i + 1,
    /// right = 2i + 2).
    pub fn build(leaves: &[[u8; HASH_SIZE]], alg: HashAlg) -> Result<Self, MerkleError> {
        let leaf_count = leaves.len();
        if leaf_count == 0 {
            return Ok(Self {
                nodes: vec![alg.null_sentinel()],
                leaf_count: 0,
                padded_count: 1,
                alg,
            });
        }

        // Pad to next power of 2
        let padded_count = leaf_count.next_power_of_two();
        let depth = compute_depth(padded_count);

        // Enforce maximum depth to prevent out-of-memory attacks (threat T7)
        if depth > MAX_DEPTH {
            return Err(MerkleError::TreeTooDeep {
                requested: depth,
                max: MAX_DEPTH,
            });
        }

        let total_nodes = 2 * padded_count - 1;
        let internal_nodes = padded_count - 1;

        // Compute null sentinel once for padding
        let null_sentinel = alg.null_sentinel();

        let mut nodes = vec![[0u8; HASH_SIZE]; total_nodes];

        // Copy actual leaf hashes to leaf positions (after internal nodes)
        for (i, hash) in leaves.iter().enumerate() {
            nodes[internal_nodes + i] = *hash;
        }

        // Fill padding positions with null sentinel
        for i in leaf_count..padded_count {
            nodes[internal_nodes + i] = null_sentinel;
        }

        // Build internal nodes from bottom up
        // Parent at index i has children at 2i+1 and 2i+2
        for i in (0..internal_nodes).rev() {
            let left_idx = 2 * i + 1;
            let right_idx = 2 * i + 2;
            nodes[i] = alg.hash_pair(&nodes[left_idx], &nodes[right_idx]);
        }

        Ok(Self {
            nodes,
            leaf_count,
            padded_count,
            alg,
        })
    }

    /// Build a Merkle tree from pre-computed leaf hashes.
    ///
    /// Each hash should be a 32-byte digest of a chunk (with leaf domain separation
    /// already applied via `HashAlg::hash_leaf`).
    ///
    /// **Note:** This method does not enforce the maximum depth constraint. For
    /// untrusted input sizes, use [`build`](Self::build) instead, which returns
    /// an error for trees exceeding 2^40 leaves.
    #[must_use]
    pub fn from_leaf_hashes(leaf_hashes: &[[u8; HASH_SIZE]], alg: HashAlg) -> Self {
        let leaf_count = leaf_hashes.len();
        if leaf_count == 0 {
            return Self {
                nodes: vec![alg.null_sentinel()],
                leaf_count: 0,
                padded_count: 1,
                alg,
            };
        }

        // Pad to next power of 2
        let padded_count = leaf_count.next_power_of_two();
        let total_nodes = 2 * padded_count - 1;
        let internal_nodes = padded_count - 1;

        // Compute null sentinel once for padding
        let null_sentinel = alg.null_sentinel();

        let mut nodes = vec![[0u8; HASH_SIZE]; total_nodes];

        // Copy leaf hashes to the leaf positions (after internal nodes)
        for (i, hash) in leaf_hashes.iter().enumerate() {
            nodes[internal_nodes + i] = *hash;
        }

        // Fill padding positions with null sentinel (not zero hashes)
        for i in leaf_count..padded_count {
            nodes[internal_nodes + i] = null_sentinel;
        }

        // Build internal nodes from bottom up
        // Parent at index i has children at 2i+1 and 2i+2
        for i in (0..internal_nodes).rev() {
            let left_idx = 2 * i + 1;
            let right_idx = 2 * i + 2;
            nodes[i] = alg.hash_pair(&nodes[left_idx], &nodes[right_idx]);
        }

        Self {
            nodes,
            leaf_count,
            padded_count,
            alg,
        }
    }

    /// Build a Merkle tree by hashing raw chunk data.
    ///
    /// Applies leaf domain separation automatically.
    #[must_use]
    pub fn from_chunks(chunks: &[&[u8]], alg: HashAlg) -> Self {
        let leaf_hashes: Vec<[u8; HASH_SIZE]> =
            chunks.iter().map(|chunk| alg.hash_leaf(chunk)).collect();
        Self::from_leaf_hashes(&leaf_hashes, alg)
    }

    /// Build a Merkle tree by hashing owned chunk data.
    ///
    /// Applies leaf domain separation automatically.
    #[must_use]
    pub fn from_chunks_owned(chunks: &[Vec<u8>], alg: HashAlg) -> Self {
        let leaf_hashes: Vec<[u8; HASH_SIZE]> =
            chunks.iter().map(|chunk| alg.hash_leaf(chunk)).collect();
        Self::from_leaf_hashes(&leaf_hashes, alg)
    }

    /// Build a Merkle tree with parallel hashing (requires `parallel` feature).
    #[cfg(feature = "parallel")]
    #[must_use]
    pub fn from_chunks_parallel(chunks: &[&[u8]], alg: HashAlg) -> Self {
        use rayon::prelude::*;
        let leaf_hashes: Vec<[u8; HASH_SIZE]> = chunks
            .par_iter()
            .map(|chunk| alg.hash_leaf(chunk))
            .collect();
        Self::from_leaf_hashes(&leaf_hashes, alg)
    }

    /// Get the root hash of the tree.
    #[inline]
    #[must_use]
    pub fn root(&self) -> &[u8; HASH_SIZE] {
        &self.nodes[0]
    }

    /// Get the hash algorithm used.
    #[inline]
    #[must_use]
    pub fn algorithm(&self) -> HashAlg {
        self.alg
    }

    /// Get the number of actual (non-padded) leaves.
    #[inline]
    #[must_use]
    pub fn leaf_count(&self) -> usize {
        self.leaf_count
    }

    /// Get the number of leaves after padding to power of 2.
    #[inline]
    #[must_use]
    pub fn padded_leaf_count(&self) -> usize {
        self.padded_count
    }

    /// Get the leaf hash at the given index.
    #[must_use]
    pub fn leaf_hash(&self, index: usize) -> Option<&[u8; HASH_SIZE]> {
        if index >= self.leaf_count {
            return None;
        }
        let internal_nodes = self.padded_count - 1;
        Some(&self.nodes[internal_nodes + index])
    }

    /// Get the depth of the tree (number of levels, including root).
    ///
    /// A single-leaf tree has depth 1, a 2-leaf tree has depth 2, etc.
    #[inline]
    #[must_use]
    pub fn depth(&self) -> usize {
        compute_depth(self.padded_count)
    }

    /// Generate an inclusion proof for a leaf at the given index.
    ///
    /// The proof consists of sibling hashes from leaf to root.
    /// Returns `None` if the index is out of bounds.
    #[must_use]
    pub fn proof(&self, leaf_index: usize) -> Option<MerkleProof> {
        if leaf_index >= self.leaf_count {
            return None;
        }

        let internal_nodes = self.padded_count - 1;
        let mut node_idx = internal_nodes + leaf_index;
        let mut siblings = Vec::with_capacity(self.depth().saturating_sub(1));

        while node_idx > 0 {
            // Sibling index: if we're at odd index (left child), sibling is +1
            // if we're at even index (right child), sibling is -1
            let sibling_idx = if node_idx % 2 == 1 {
                node_idx + 1
            } else {
                node_idx - 1
            };
            siblings.push(self.nodes[sibling_idx]);

            // Move to parent: (idx - 1) / 2
            node_idx = (node_idx - 1) / 2;
        }

        Some(MerkleProof {
            leaf_index,
            siblings,
            alg: self.alg,
        })
    }

    /// Verify that a chunk belongs to this tree at the given index.
    ///
    /// Uses constant-time comparison to prevent timing attacks.
    #[must_use]
    pub fn verify_chunk(&self, leaf_index: usize, chunk_data: &[u8]) -> bool {
        if leaf_index >= self.leaf_count {
            return false;
        }

        let chunk_hash = self.alg.hash_leaf(chunk_data);
        let expected_hash = self.leaf_hash(leaf_index).unwrap();
        constant_time_eq(&chunk_hash, expected_hash)
    }

    /// Verify a proof against this tree's root.
    ///
    /// Uses constant-time comparison to prevent timing attacks.
    #[must_use]
    pub fn verify_proof(&self, leaf_index: usize, chunk_data: &[u8], proof: &MerkleProof) -> bool {
        if leaf_index != proof.leaf_index || self.alg != proof.alg {
            return false;
        }

        // Validate proof length matches tree depth
        let expected_siblings = self.depth().saturating_sub(1);
        if proof.siblings.len() != expected_siblings {
            return false;
        }

        let computed_root = proof.compute_root(chunk_data);
        constant_time_eq(&computed_root, self.root())
    }

    /// Verify a proof given only the root hash (no full tree needed).
    ///
    /// Note: Cannot validate proof length without knowing the tree structure.
    /// Use `verify_proof` when the full tree is available.
    #[must_use]
    pub fn verify_proof_standalone(
        root: &[u8; HASH_SIZE],
        leaf_index: usize,
        chunk_data: &[u8],
        proof: &MerkleProof,
    ) -> bool {
        if leaf_index != proof.leaf_index {
            return false;
        }

        let computed_root = proof.compute_root(chunk_data);
        constant_time_eq(&computed_root, root)
    }

    /// Get all node hashes (for serialization).
    #[must_use]
    pub fn nodes(&self) -> &[[u8; HASH_SIZE]] {
        &self.nodes
    }
}

/// A Merkle inclusion proof for a single leaf.
///
/// Contains the sibling hashes needed to recompute the root from a leaf.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MerkleProof {
    /// Index of the leaf being proved.
    leaf_index: usize,
    /// Sibling hashes from leaf to root.
    siblings: Vec<[u8; HASH_SIZE]>,
    /// Hash algorithm (needed for verification).
    alg: HashAlg,
}

impl MerkleProof {
    /// Get the leaf index this proof is for.
    #[inline]
    #[must_use]
    pub fn leaf_index(&self) -> usize {
        self.leaf_index
    }

    /// Get the sibling hashes.
    #[inline]
    #[must_use]
    pub fn siblings(&self) -> &[[u8; HASH_SIZE]] {
        &self.siblings
    }

    /// Get the hash algorithm.
    #[inline]
    #[must_use]
    pub fn algorithm(&self) -> HashAlg {
        self.alg
    }

    /// Compute the root hash from this proof and the given chunk data.
    #[must_use]
    pub fn compute_root(&self, chunk_data: &[u8]) -> [u8; HASH_SIZE] {
        let leaf_hash = self.alg.hash_leaf(chunk_data);
        self.compute_root_from_hash(&leaf_hash)
    }

    /// Compute the root hash from a pre-computed leaf hash.
    #[must_use]
    pub fn compute_root_from_hash(&self, leaf_hash: &[u8; HASH_SIZE]) -> [u8; HASH_SIZE] {
        let mut current = *leaf_hash;
        let mut idx = self.leaf_index;

        for sibling in &self.siblings {
            current = if idx.is_multiple_of(2) {
                // Current is left child
                self.alg.hash_pair(&current, sibling)
            } else {
                // Current is right child
                self.alg.hash_pair(sibling, &current)
            };
            idx /= 2;
        }

        current
    }

    /// Size of the proof in bytes (for serialization planning).
    ///
    /// Layout: 1 byte alg + 8 bytes leaf_index + N * 32 bytes siblings
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        1 + 8 + self.siblings.len() * HASH_SIZE
    }
}

// ============================================================================
// Hash function implementations
// ============================================================================

#[cfg(feature = "sha2")]
fn hash_sha256(data: &[u8]) -> [u8; HASH_SIZE] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

#[cfg(not(feature = "sha2"))]
fn hash_sha256(_data: &[u8]) -> [u8; HASH_SIZE] {
    panic!("SHA-256 support requires the 'sha2' or 'merkle' feature")
}

#[cfg(feature = "blake3")]
fn hash_blake3(data: &[u8]) -> [u8; HASH_SIZE] {
    blake3::hash(data).into()
}

#[cfg(not(feature = "blake3"))]
fn hash_blake3(_data: &[u8]) -> [u8; HASH_SIZE] {
    panic!("BLAKE3 support requires the 'blake3' or 'merkle' feature")
}

#[cfg(feature = "k12")]
fn hash_k12(data: &[u8]) -> [u8; HASH_SIZE] {
    use k12::digest::{ExtendableOutput, Update};
    use k12::KangarooTwelve;

    let mut hasher = KangarooTwelve::default();
    hasher.update(data);
    let mut output = [0u8; HASH_SIZE];
    hasher.finalize_xof_into(&mut output);
    output
}

#[cfg(not(feature = "k12"))]
fn hash_k12(_data: &[u8]) -> [u8; HASH_SIZE] {
    panic!("KangarooTwelve support requires the 'k12' or 'merkle' feature")
}

// ============================================================================
// Optimized chunk hashing with incremental APIs (avoids allocation)
// ============================================================================

/// Hash chunk data with SHA-256 using incremental API.
/// Computes H(0x00 || data) without allocating a combined buffer.
#[cfg(feature = "sha2")]
#[inline]
fn hash_chunk_sha256(data: &[u8]) -> [u8; HASH_SIZE] {
    use sha2::{Digest, Sha256};
    Sha256::new()
        .chain_update([LEAF_PREFIX])
        .chain_update(data)
        .finalize()
        .into()
}

#[cfg(not(feature = "sha2"))]
fn hash_chunk_sha256(_data: &[u8]) -> [u8; HASH_SIZE] {
    panic!("SHA-256 support requires the 'sha2' or 'merkle' feature")
}

/// Hash chunk data with BLAKE3 using flat hash API.
/// Computes H(0x00 || data) using Hasher::new().update().update().finalize().
///
/// IMPORTANT: Does NOT use BLAKE3's internal tree mode (new_derive_key or
/// implicit 1KB chunking), which cannot produce per-chunk proofs.
#[cfg(feature = "blake3")]
#[inline]
fn hash_chunk_blake3(data: &[u8]) -> [u8; HASH_SIZE] {
    blake3::Hasher::new()
        .update(&[LEAF_PREFIX])
        .update(data)
        .finalize()
        .into()
}

#[cfg(not(feature = "blake3"))]
fn hash_chunk_blake3(_data: &[u8]) -> [u8; HASH_SIZE] {
    panic!("BLAKE3 support requires the 'blake3' or 'merkle' feature")
}

/// Hash chunk data with KangarooTwelve using incremental API.
/// Computes H(0x00 || data) without allocating a combined buffer.
#[cfg(feature = "k12")]
#[inline]
fn hash_chunk_k12(data: &[u8]) -> [u8; HASH_SIZE] {
    use k12::digest::{ExtendableOutput, Update};
    use k12::KangarooTwelve;

    let mut hasher = KangarooTwelve::default();
    hasher.update(&[LEAF_PREFIX]);
    hasher.update(data);
    let mut output = [0u8; HASH_SIZE];
    hasher.finalize_xof_into(&mut output);
    output
}

#[cfg(not(feature = "k12"))]
fn hash_chunk_k12(_data: &[u8]) -> [u8; HASH_SIZE] {
    panic!("KangarooTwelve support requires the 'k12' or 'merkle' feature")
}

// ============================================================================
// HDF5 Merkle Attribute Support
// ============================================================================

/// Algorithm identifier bytes for the merkle_root attribute.
const ALG_ID_SHA256: u8 = 0x00;
const ALG_ID_BLAKE3: u8 = 0x01;
const ALG_ID_K12: u8 = 0x02;

/// Domain separator for companion integrity hash.
const INTEGRITY_PREFIX: u8 = 0x03;

/// Size of the packed merkle_root attribute (root + alg_id + integrity).
pub const MERKLE_ATTR_SIZE: usize = HASH_SIZE + 1 + HASH_SIZE; // 65 bytes

/// Name of the HDF5 attribute storing merkle root information.
pub const MERKLE_ATTR_NAME: &str = "merkle_root";

impl HashAlg {
    /// Get the algorithm identifier byte for serialization.
    #[inline]
    #[must_use]
    pub const fn to_id(self) -> u8 {
        match self {
            HashAlg::Sha256 => ALG_ID_SHA256,
            HashAlg::Blake3 => ALG_ID_BLAKE3,
            HashAlg::K12 => ALG_ID_K12,
        }
    }

    /// Parse an algorithm identifier byte.
    ///
    /// Returns `None` for unknown algorithm IDs.
    #[inline]
    #[must_use]
    pub const fn from_id(id: u8) -> Option<Self> {
        match id {
            ALG_ID_SHA256 => Some(HashAlg::Sha256),
            ALG_ID_BLAKE3 => Some(HashAlg::Blake3),
            ALG_ID_K12 => Some(HashAlg::K12),
            _ => None,
        }
    }
}

/// Packed merkle root attribute data.
///
/// Layout (65 bytes total):
/// - Bytes 0-31: Root hash (32 bytes)
/// - Byte 32: Algorithm identifier (1 byte)
/// - Bytes 33-64: Companion integrity hash (32 bytes)
///
/// The companion integrity hash is `H(0x03 || root || alg_id)` and prevents
/// an attacker from modifying the algorithm ID without detection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MerkleAttr {
    /// The Merkle tree root hash.
    pub root: [u8; HASH_SIZE],
    /// The hash algorithm used.
    pub algorithm: HashAlg,
    /// Companion integrity hash binding root and algorithm.
    pub integrity: [u8; HASH_SIZE],
}

impl MerkleAttr {
    /// Create a new merkle attribute from a tree.
    ///
    /// Computes the companion integrity hash as `H(0x03 || root || alg_id)`.
    #[must_use]
    pub fn from_tree(tree: &MerkleTree) -> Self {
        let root = *tree.root();
        let algorithm = tree.algorithm();
        let integrity = Self::compute_integrity(&root, algorithm);

        Self {
            root,
            algorithm,
            integrity,
        }
    }

    /// Compute the companion integrity hash.
    ///
    /// This binds the root hash and algorithm ID together, preventing
    /// an attacker from changing the algorithm without detection.
    fn compute_integrity(root: &[u8; HASH_SIZE], alg: HashAlg) -> [u8; HASH_SIZE] {
        let mut data = [0u8; 1 + HASH_SIZE + 1];
        data[0] = INTEGRITY_PREFIX;
        data[1..HASH_SIZE + 1].copy_from_slice(root);
        data[HASH_SIZE + 1] = alg.to_id();
        alg.hash_raw(&data)
    }

    /// Pack the attribute into a 65-byte binary blob.
    #[must_use]
    pub fn pack(&self) -> [u8; MERKLE_ATTR_SIZE] {
        let mut buf = [0u8; MERKLE_ATTR_SIZE];
        buf[0..HASH_SIZE].copy_from_slice(&self.root);
        buf[HASH_SIZE] = self.algorithm.to_id();
        buf[HASH_SIZE + 1..].copy_from_slice(&self.integrity);
        buf
    }

    /// Unpack from a 65-byte binary blob.
    ///
    /// Returns `None` if:
    /// - The data is not exactly 65 bytes
    /// - The algorithm ID is unknown
    /// - The integrity hash does not match
    #[must_use]
    pub fn unpack(data: &[u8]) -> Option<Self> {
        if data.len() != MERKLE_ATTR_SIZE {
            return None;
        }

        let mut root = [0u8; HASH_SIZE];
        root.copy_from_slice(&data[0..HASH_SIZE]);

        let alg_id = data[HASH_SIZE];
        let algorithm = HashAlg::from_id(alg_id)?;

        let mut integrity = [0u8; HASH_SIZE];
        integrity.copy_from_slice(&data[HASH_SIZE + 1..]);

        // Verify integrity hash
        let expected_integrity = Self::compute_integrity(&root, algorithm);
        if !constant_time_eq(&integrity, &expected_integrity) {
            return None;
        }

        Some(Self {
            root,
            algorithm,
            integrity,
        })
    }

    /// Verify that a Merkle tree matches this attribute.
    #[must_use]
    pub fn verify_tree(&self, tree: &MerkleTree) -> bool {
        tree.algorithm() == self.algorithm && constant_time_eq(tree.root(), &self.root)
    }
}

/// Write the merkle_root attribute to a dataset.
///
/// Packs the root hash (32 bytes), algorithm identifier (1 byte), and
/// companion integrity hash (32 bytes) into a fixed-width 65-byte binary
/// blob and writes it as the HDF5 attribute `merkle_root`.
///
/// # Arguments
///
/// * `dataset` - The dataset builder to add the attribute to
/// * `tree` - The Merkle tree to store
///
/// # Example
///
/// ```ignore
/// use clawhdf5_format::merkle::{MerkleTree, HashAlg, write_merkle_attr};
/// use clawhdf5_format::file_writer::FileWriter;
///
/// let chunks: Vec<&[u8]> = vec![&[1, 2, 3], &[4, 5, 6]];
/// let tree = MerkleTree::from_chunks(&chunks, HashAlg::Blake3);
///
/// let mut fw = FileWriter::new();
/// let ds = fw.create_dataset("data");
/// ds.with_u8_data(&[1, 2, 3, 4, 5, 6]);
/// write_merkle_attr(ds, &tree);
/// ```
pub fn write_merkle_attr(
    dataset: &mut crate::type_builders::DatasetBuilder,
    tree: &MerkleTree,
) {
    let attr = MerkleAttr::from_tree(tree);
    let packed = attr.pack();
    dataset.set_attr(MERKLE_ATTR_NAME, crate::type_builders::AttrValue::Bytes(packed.to_vec()));
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_chunks() -> Vec<Vec<u8>> {
        vec![
            b"chunk0".to_vec(),
            b"chunk1".to_vec(),
            b"chunk2".to_vec(),
            b"chunk3".to_vec(),
        ]
    }

    #[test]
    #[cfg(all(feature = "sha2", feature = "blake3", feature = "k12"))]
    fn test_different_algorithms_produce_different_roots() {
        let chunks = make_test_chunks();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();

        let tree_sha = MerkleTree::from_chunks(&refs, HashAlg::Sha256);
        let tree_blake = MerkleTree::from_chunks(&refs, HashAlg::Blake3);
        let tree_k12 = MerkleTree::from_chunks(&refs, HashAlg::K12);

        assert_ne!(tree_sha.root(), tree_blake.root());
        assert_ne!(tree_blake.root(), tree_k12.root());
        assert_ne!(tree_sha.root(), tree_k12.root());
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_proof_generation_and_verification() {
        let chunks = make_test_chunks();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();

        let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);

        for i in 0..chunks.len() {
            let proof = tree.proof(i).expect("proof should exist");
            assert!(tree.verify_proof(i, &chunks[i], &proof));

            // Verify standalone (only root needed)
            assert!(MerkleTree::verify_proof_standalone(
                tree.root(),
                i,
                &chunks[i],
                &proof
            ));
        }
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_invalid_proof_fails() {
        let chunks = make_test_chunks();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();

        let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);
        let proof = tree.proof(0).unwrap();

        // Wrong chunk data should fail
        assert!(!tree.verify_proof(0, b"wrong_data", &proof));

        // Wrong index should fail
        assert!(!tree.verify_proof(1, &chunks[0], &proof));
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_non_power_of_two_leaves() {
        // 5 leaves requires padding to 8
        let chunks: Vec<Vec<u8>> = (0..5).map(|i| format!("chunk{}", i).into_bytes()).collect();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();

        let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);

        assert_eq!(tree.leaf_count(), 5);
        assert_eq!(tree.padded_leaf_count(), 8);
        assert_eq!(tree.depth(), 4); // log2(8) + 1

        // All real leaves should have valid proofs
        for i in 0..5 {
            let proof = tree.proof(i).expect("proof should exist");
            assert!(tree.verify_proof(i, &chunks[i], &proof));
        }

        // Padding leaves should return None
        assert!(tree.proof(5).is_none());
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_single_chunk() {
        let chunks = vec![b"single".to_vec()];
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();

        let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);

        assert_eq!(tree.leaf_count(), 1);
        assert_eq!(tree.padded_leaf_count(), 1);
        assert_eq!(tree.depth(), 1);

        let proof = tree.proof(0).unwrap();
        assert!(tree.verify_proof(0, &chunks[0], &proof));
        assert!(proof.siblings().is_empty()); // No siblings for single leaf
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_empty_tree() {
        let alg = HashAlg::Blake3;
        let tree = MerkleTree::from_chunks(&[], alg);

        assert_eq!(tree.leaf_count(), 0);
        // Empty tree root should be the null sentinel, not zero
        assert_eq!(tree.root(), &alg.null_sentinel());
        assert_eq!(tree.depth(), 1);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_deterministic() {
        let chunks = make_test_chunks();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();

        let tree1 = MerkleTree::from_chunks(&refs, HashAlg::Blake3);
        let tree2 = MerkleTree::from_chunks(&refs, HashAlg::Blake3);

        assert_eq!(tree1.root(), tree2.root());
        assert_eq!(tree1.nodes(), tree2.nodes());
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_chunk_verification() {
        let chunks = make_test_chunks();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();

        let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);

        // Correct chunks verify
        for i in 0..chunks.len() {
            assert!(tree.verify_chunk(i, &chunks[i]));
        }

        // Wrong chunk fails
        assert!(!tree.verify_chunk(0, b"wrong"));
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_domain_separation() {
        // Verify that leaf and internal hashes use different prefixes
        let alg = HashAlg::Blake3;

        let data = b"test data";
        let leaf_hash = alg.hash_leaf(data);

        // Compute an internal hash (of two zero hashes) for comparison
        // This should differ from the leaf hash due to domain separation
        let internal_hash = alg.hash_pair(&[0u8; HASH_SIZE], &[0u8; HASH_SIZE]);

        // The leaf hash should not equal any internal node construction
        // (This is a basic sanity check, not exhaustive)
        assert_ne!(leaf_hash, internal_hash);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_constant_time_eq() {
        let a = [1u8; HASH_SIZE];
        let b = [1u8; HASH_SIZE];
        let c = [2u8; HASH_SIZE];

        assert!(constant_time_eq(&a, &b));
        assert!(!constant_time_eq(&a, &c));

        // Differs only in last byte
        let mut d = a;
        d[HASH_SIZE - 1] = 2;
        assert!(!constant_time_eq(&a, &d));
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_proof_length_validation() {
        let chunks = make_test_chunks();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();

        let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);
        let mut proof = tree.proof(0).unwrap();

        // Tamper with proof length
        proof.siblings.push([0u8; HASH_SIZE]);

        // Should fail due to length mismatch
        assert!(!tree.verify_proof(0, &chunks[0], &proof));
    }

    #[test]
    fn test_default_algorithm() {
        assert_eq!(HashAlg::default(), HashAlg::Blake3);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_null_sentinel_domain_separation() {
        // Verify that null sentinel uses 0x02 prefix, distinct from leaf (0x00)
        // and internal (0x01) prefixes
        let alg = HashAlg::Blake3;

        let leaf_hash = alg.hash_leaf(b"null");
        let null_sentinel = alg.null_sentinel();

        // Leaf hash of "null" should differ from null sentinel H(0x02 || "null")
        assert_ne!(leaf_hash, null_sentinel);

        // Internal hash should also differ
        let internal_hash = alg.hash_pair(&[0u8; HASH_SIZE], &[0u8; HASH_SIZE]);
        assert_ne!(null_sentinel, internal_hash);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_build_method_success() {
        let alg = HashAlg::Blake3;
        let chunks = make_test_chunks();

        // Pre-hash the leaves
        let leaf_hashes: Vec<[u8; HASH_SIZE]> =
            chunks.iter().map(|c| alg.hash_leaf(c)).collect();

        // Build tree using the new build method
        let tree = MerkleTree::build(&leaf_hashes, alg).expect("build should succeed");

        assert_eq!(tree.leaf_count(), 4);
        assert_eq!(tree.padded_leaf_count(), 4);
        assert_eq!(tree.depth(), 3);

        // Verify proofs work
        for i in 0..chunks.len() {
            let proof = tree.proof(i).expect("proof should exist");
            assert!(tree.verify_proof(i, &chunks[i], &proof));
        }
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_build_method_non_power_of_two() {
        let alg = HashAlg::Blake3;
        // 5 leaves requires padding to 8
        let chunks: Vec<Vec<u8>> = (0..5).map(|i| format!("chunk{}", i).into_bytes()).collect();

        let leaf_hashes: Vec<[u8; HASH_SIZE]> =
            chunks.iter().map(|c| alg.hash_leaf(c)).collect();

        let tree = MerkleTree::build(&leaf_hashes, alg).expect("build should succeed");

        assert_eq!(tree.leaf_count(), 5);
        assert_eq!(tree.padded_leaf_count(), 8);
        assert_eq!(tree.depth(), 4);

        // Padding positions should contain null sentinel
        let null_sentinel = alg.null_sentinel();
        let internal_nodes = tree.padded_leaf_count() - 1;

        // Access padding slots directly from nodes
        for i in 5..8 {
            assert_eq!(tree.nodes()[internal_nodes + i], null_sentinel);
        }
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_build_empty() {
        let alg = HashAlg::Blake3;
        let tree = MerkleTree::build(&[], alg).expect("build should succeed");

        assert_eq!(tree.leaf_count(), 0);
        assert_eq!(tree.root(), &alg.null_sentinel());
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_build_and_from_leaf_hashes_equivalent() {
        let alg = HashAlg::Blake3;
        let chunks = make_test_chunks();

        let leaf_hashes: Vec<[u8; HASH_SIZE]> =
            chunks.iter().map(|c| alg.hash_leaf(c)).collect();

        let tree_build = MerkleTree::build(&leaf_hashes, alg).expect("build should succeed");
        let tree_from_leaf = MerkleTree::from_leaf_hashes(&leaf_hashes, alg);

        // Both methods should produce identical trees
        assert_eq!(tree_build.root(), tree_from_leaf.root());
        assert_eq!(tree_build.nodes(), tree_from_leaf.nodes());
    }

    #[test]
    fn test_merkle_error_display() {
        let err = MerkleError::TreeTooDeep {
            requested: 50,
            max: 40,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("50"));
        assert!(msg.contains("40"));
    }

    #[test]
    fn test_compute_depth() {
        assert_eq!(compute_depth(1), 1);
        assert_eq!(compute_depth(2), 2);
        assert_eq!(compute_depth(4), 3);
        assert_eq!(compute_depth(8), 4);
        assert_eq!(compute_depth(16), 5);
        assert_eq!(compute_depth(1024), 11);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_hash_chunk_equivalent_to_hash_leaf() {
        let data = b"test chunk data for hashing";

        // hash_chunk should produce identical results to HashAlg::hash_leaf
        let from_helper = hash_chunk(data, HashAlg::Blake3);
        let from_method = HashAlg::Blake3.hash_leaf(data);

        assert_eq!(from_helper, from_method);
    }

    #[test]
    #[cfg(all(feature = "sha2", feature = "blake3", feature = "k12"))]
    fn test_hash_chunk_all_algorithms() {
        let data = b"chunk data for all algorithms";

        // Each algorithm should produce different (but consistent) results
        let sha256_hash = hash_chunk(data, HashAlg::Sha256);
        let blake3_hash = hash_chunk(data, HashAlg::Blake3);
        let k12_hash = hash_chunk(data, HashAlg::K12);

        // All should be different from each other
        assert_ne!(sha256_hash, blake3_hash);
        assert_ne!(blake3_hash, k12_hash);
        assert_ne!(sha256_hash, k12_hash);

        // Should be deterministic
        assert_eq!(hash_chunk(data, HashAlg::Sha256), sha256_hash);
        assert_eq!(hash_chunk(data, HashAlg::Blake3), blake3_hash);
        assert_eq!(hash_chunk(data, HashAlg::K12), k12_hash);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_hash_chunk_domain_separation() {
        let data = b"some data";

        // hash_chunk (leaf hash) should differ from raw hash of same data
        let leaf_hash = hash_chunk(data, HashAlg::Blake3);
        let raw_hash: [u8; HASH_SIZE] = blake3::hash(data).into();

        // The leaf hash includes the 0x00 prefix, so it should differ
        assert_ne!(leaf_hash, raw_hash);
    }

    // =========================================================================
    // §5.5 Specification Tests: Manual verification of tree structure
    // =========================================================================

    #[test]
    #[cfg(feature = "blake3")]
    fn test_single_leaf_root_equals_leaf_hash() {
        // For a single-leaf tree, the root must equal the leaf hash directly.
        // No internal hashing should occur.
        let alg = HashAlg::Blake3;

        // Create a leaf hash
        let leaf_data = b"single leaf data";
        let leaf_hash = alg.hash_leaf(leaf_data);

        // Build tree from the single pre-hashed leaf
        let tree = MerkleTree::build(&[leaf_hash], alg).expect("build should succeed");

        // Root must equal the leaf hash exactly
        assert_eq!(
            tree.root(),
            &leaf_hash,
            "Single-leaf tree root must equal the leaf hash"
        );

        // Verify tree structure
        assert_eq!(tree.leaf_count(), 1);
        assert_eq!(tree.padded_leaf_count(), 1);
        assert_eq!(tree.depth(), 1);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_two_leaf_tree_manual_verification() {
        // For a two-leaf tree, manually compute the root as:
        // root = H(0x01 || leaf0 || leaf1)
        let alg = HashAlg::Blake3;

        // Create two distinct leaf hashes
        let leaf0 = alg.hash_leaf(b"leaf zero");
        let leaf1 = alg.hash_leaf(b"leaf one");

        // Build tree from the two leaves
        let tree = MerkleTree::build(&[leaf0, leaf1], alg).expect("build should succeed");

        // Manually compute expected root: H(0x01 || leaf0 || leaf1)
        let mut combined = [0u8; 1 + HASH_SIZE * 2];
        combined[0] = INTERNAL_PREFIX; // 0x01
        combined[1..HASH_SIZE + 1].copy_from_slice(&leaf0);
        combined[HASH_SIZE + 1..].copy_from_slice(&leaf1);
        let expected_root: [u8; HASH_SIZE] = blake3::hash(&combined).into();

        // Verify root matches hand computation
        assert_eq!(
            tree.root(),
            &expected_root,
            "Two-leaf tree root must equal H(0x01 || leaf0 || leaf1)"
        );

        // Verify tree structure
        assert_eq!(tree.leaf_count(), 2);
        assert_eq!(tree.padded_leaf_count(), 2);
        assert_eq!(tree.depth(), 2);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_eight_leaf_tree_reference_computation() {
        // Build an 8-leaf tree and verify root against hand-computed reference.
        //
        // Tree structure (level-order storage):
        //                     root (0)
        //                    /        \
        //               n1 (1)         n2 (2)
        //              /     \        /      \
        //           n3 (3)  n4 (4)  n5 (5)  n6 (6)
        //           /  \    /  \    /  \    /   \
        //         L0  L1  L2  L3  L4  L5  L6   L7
        //
        // Computation steps:
        //   n3 = H(0x01 || L0 || L1)
        //   n4 = H(0x01 || L2 || L3)
        //   n5 = H(0x01 || L4 || L5)
        //   n6 = H(0x01 || L6 || L7)
        //   n1 = H(0x01 || n3 || n4)
        //   n2 = H(0x01 || n5 || n6)
        //   root = H(0x01 || n1 || n2)

        let alg = HashAlg::Blake3;

        // Create 8 distinct leaf hashes
        let leaves: Vec<[u8; HASH_SIZE]> = (0u8..8)
            .map(|i| alg.hash_leaf(&[b'L', i]))
            .collect();

        // Build tree
        let tree = MerkleTree::build(&leaves, alg).expect("build should succeed");

        // Reference implementation: compute expected root step by step
        let hash_pair_ref = |left: &[u8; HASH_SIZE], right: &[u8; HASH_SIZE]| -> [u8; HASH_SIZE] {
            let mut combined = [0u8; 1 + HASH_SIZE * 2];
            combined[0] = INTERNAL_PREFIX;
            combined[1..HASH_SIZE + 1].copy_from_slice(left);
            combined[HASH_SIZE + 1..].copy_from_slice(right);
            blake3::hash(&combined).into()
        };

        // Level 2: pair up leaves
        let n3 = hash_pair_ref(&leaves[0], &leaves[1]);
        let n4 = hash_pair_ref(&leaves[2], &leaves[3]);
        let n5 = hash_pair_ref(&leaves[4], &leaves[5]);
        let n6 = hash_pair_ref(&leaves[6], &leaves[7]);

        // Level 1: pair up level 2 nodes
        let n1 = hash_pair_ref(&n3, &n4);
        let n2 = hash_pair_ref(&n5, &n6);

        // Level 0: compute root
        let expected_root = hash_pair_ref(&n1, &n2);

        // Verify root matches reference computation
        assert_eq!(
            tree.root(),
            &expected_root,
            "Eight-leaf tree root must match reference computation"
        );

        // Verify tree structure
        assert_eq!(tree.leaf_count(), 8);
        assert_eq!(tree.padded_leaf_count(), 8);
        assert_eq!(tree.depth(), 4); // log2(8) + 1 = 4

        // Additionally verify all proofs work
        for i in 0..8 {
            let proof = tree.proof(i).expect("proof should exist");
            // Verify using raw chunk data
            let chunk_data = [b'L', i as u8];
            assert!(
                tree.verify_proof(i, &chunk_data, &proof),
                "Proof for leaf {} should verify",
                i
            );
        }
    }

    // =========================================================================
    // MerkleAttr Tests
    // =========================================================================

    #[test]
    fn test_algorithm_id_roundtrip() {
        assert_eq!(HashAlg::from_id(HashAlg::Sha256.to_id()), Some(HashAlg::Sha256));
        assert_eq!(HashAlg::from_id(HashAlg::Blake3.to_id()), Some(HashAlg::Blake3));
        assert_eq!(HashAlg::from_id(HashAlg::K12.to_id()), Some(HashAlg::K12));
        assert_eq!(HashAlg::from_id(0xFF), None);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_merkle_attr_pack_unpack() {
        let chunks = make_test_chunks();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);

        let attr = MerkleAttr::from_tree(&tree);
        let packed = attr.pack();

        // Verify size
        assert_eq!(packed.len(), MERKLE_ATTR_SIZE);
        assert_eq!(packed.len(), 65);

        // Unpack and verify round-trip
        let unpacked = MerkleAttr::unpack(&packed).expect("unpack should succeed");
        assert_eq!(unpacked.root, attr.root);
        assert_eq!(unpacked.algorithm, attr.algorithm);
        assert_eq!(unpacked.integrity, attr.integrity);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_merkle_attr_integrity_verification() {
        let chunks = make_test_chunks();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);

        let attr = MerkleAttr::from_tree(&tree);
        let mut packed = attr.pack();

        // Tamper with the algorithm ID
        packed[HASH_SIZE] = 0x00; // Change from BLAKE3 to SHA256

        // Unpack should fail due to integrity mismatch
        assert!(
            MerkleAttr::unpack(&packed).is_none(),
            "Tampered algorithm ID should fail integrity check"
        );
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_merkle_attr_root_tampering_detected() {
        let chunks = make_test_chunks();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);

        let attr = MerkleAttr::from_tree(&tree);
        let mut packed = attr.pack();

        // Tamper with the root hash
        packed[0] ^= 0xFF;

        // Unpack should fail due to integrity mismatch
        assert!(
            MerkleAttr::unpack(&packed).is_none(),
            "Tampered root hash should fail integrity check"
        );
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_merkle_attr_verify_tree() {
        let chunks = make_test_chunks();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);

        let attr = MerkleAttr::from_tree(&tree);

        // Same tree should verify
        assert!(attr.verify_tree(&tree));

        // Different tree should not verify
        let other_chunks = vec![b"different".to_vec()];
        let other_refs: Vec<&[u8]> = other_chunks.iter().map(|c| c.as_slice()).collect();
        let other_tree = MerkleTree::from_chunks(&other_refs, HashAlg::Blake3);
        assert!(!attr.verify_tree(&other_tree));
    }

    #[test]
    fn test_merkle_attr_invalid_size() {
        // Too short
        assert!(MerkleAttr::unpack(&[0u8; 64]).is_none());
        // Too long
        assert!(MerkleAttr::unpack(&[0u8; 66]).is_none());
        // Empty
        assert!(MerkleAttr::unpack(&[]).is_none());
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_write_merkle_attr() {
        use crate::file_writer::FileWriter;

        let chunks = make_test_chunks();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);

        let mut fw = FileWriter::new();
        let ds = fw.create_dataset("data");
        ds.with_u8_data(&[1, 2, 3, 4]);

        // Write merkle attribute
        write_merkle_attr(ds, &tree);

        // Finish and verify the file is valid
        let bytes = fw.finish().expect("file should build");
        assert!(!bytes.is_empty());

        // The attribute should be readable (basic check)
        // Full parsing would require reading the attribute back
    }
}
