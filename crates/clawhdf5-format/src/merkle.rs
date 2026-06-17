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

#[cfg(not(feature = "std"))]
use alloc::borrow::Cow;

#[cfg(feature = "std")]
use std::borrow::Cow;

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

/// Errors that can occur during Merkle tree operations.
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
    /// Invalid attribute data during unpacking.
    InvalidAttribute {
        /// Reason for the failure.
        reason: InvalidAttrReason,
    },
}

/// Reasons why a merkle attribute is invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidAttrReason {
    /// Attribute size is not 97 bytes.
    WrongSize,
    /// Unknown algorithm identifier.
    UnknownAlgorithm,
    /// Integrity hash does not match.
    IntegrityMismatch,
}

/// Result of companion data verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompanionVerifyResult {
    /// Verification passed: companion hash matches.
    Valid,
    /// No companion data present (hash is all zeros).
    NoCompanion,
    /// Verification failed: hash mismatch (possible tampering).
    HashMismatch,
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
            MerkleError::InvalidAttribute { reason } => {
                let msg = match reason {
                    InvalidAttrReason::WrongSize => "attribute size is not valid (expected 97 bytes for v0)",
                    InvalidAttrReason::UnknownAlgorithm => "unknown algorithm identifier",
                    InvalidAttrReason::IntegrityMismatch => "integrity hash mismatch",
                };
                write!(f, "Invalid merkle attribute: {}", msg)
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

/// Compute SHA-256 hash of arbitrary data.
///
/// Used for companion integrity verification. Always uses SHA-256 regardless
/// of the tree's hash algorithm to provide a consistent integrity check.
///
/// Note: No feature gate needed here since the merkle module is only compiled
/// when `feature = "merkle"` is enabled, which implies `sha2`.
fn compute_sha256(data: &[u8]) -> [u8; HASH_SIZE] {
    use sha2::{Digest, Sha256};
    Sha256::digest(data).into()
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

// ---- Attribute format versioning ----
//
// Version 0 (implicit): 97 bytes, current format
// Future versions may add a version byte prefix

/// Attribute format version 0 (current, 97 bytes with companion hash).
pub const MERKLE_ATTR_VERSION_0: u8 = 0;

// ---- 97-byte attribute layout offsets ----
//
// ┌─────────────────────────────────┬───────┬─────────────────────────────────┬─────────────────────────────────┐
// │         Root Hash (32B)         │Alg(1B)│     Integrity Hash (32B)        │   Companion Hash (32B)          │
// └─────────────────────────────────┴───────┴─────────────────────────────────┴─────────────────────────────────┘
// 0                                32      33                                65                                97
//
/// Offset of root hash in packed attribute.
const ATTR_ROOT_OFFSET: usize = 0;
/// Size of root hash field.
const ATTR_ROOT_SIZE: usize = HASH_SIZE;
/// End offset of root hash (exclusive).
const ATTR_ROOT_END: usize = ATTR_ROOT_OFFSET + ATTR_ROOT_SIZE; // 32

/// Offset of algorithm identifier in packed attribute.
const ATTR_ALG_OFFSET: usize = ATTR_ROOT_END; // 32
/// Size of algorithm identifier field.
const ATTR_ALG_SIZE: usize = 1;

/// Offset of integrity hash in packed attribute.
const ATTR_INTEGRITY_OFFSET: usize = ATTR_ALG_OFFSET + ATTR_ALG_SIZE; // 33
/// Size of integrity hash field.
const ATTR_INTEGRITY_SIZE: usize = HASH_SIZE;
/// End offset of integrity hash (exclusive).
const ATTR_INTEGRITY_END: usize = ATTR_INTEGRITY_OFFSET + ATTR_INTEGRITY_SIZE; // 65

/// Offset of companion hash in packed attribute.
const ATTR_COMPANION_OFFSET: usize = ATTR_INTEGRITY_END; // 65
/// Size of companion hash field.
const ATTR_COMPANION_SIZE: usize = HASH_SIZE;
/// End offset of companion hash (exclusive).
const ATTR_COMPANION_END: usize = ATTR_COMPANION_OFFSET + ATTR_COMPANION_SIZE; // 97

/// Size of the packed merkle_root attribute (root + alg_id + integrity + companion_hash).
pub const MERKLE_ATTR_SIZE: usize = ATTR_COMPANION_END; // 97 bytes

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
/// Layout (97 bytes total):
/// - Bytes 0-31: Root hash (32 bytes)
/// - Byte 32: Algorithm identifier (1 byte)
/// - Bytes 33-64: Integrity hash (32 bytes) - binds root and algorithm
/// - Bytes 65-96: Companion hash (32 bytes) - SHA-256 of nodes data
///
/// The integrity hash is `H(0x03 || root || alg_id)` and prevents
/// an attacker from modifying the algorithm ID without detection.
///
/// The companion hash is SHA-256 of the full nodes array (either inline
/// in `merkle_nodes` attribute or in `/merkle/{name}` companion dataset).
/// This allows quick tamper detection before walking the tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MerkleAttr {
    /// The Merkle tree root hash.
    pub root: [u8; HASH_SIZE],
    /// The hash algorithm used.
    pub algorithm: HashAlg,
    /// Integrity hash binding root and algorithm.
    pub integrity: [u8; HASH_SIZE],
    /// SHA-256 hash of the companion/inline nodes data.
    /// All zeros if no companion data (root-only attribute).
    pub companion_hash: [u8; HASH_SIZE],
}

impl MerkleAttr {
    /// Create a new merkle attribute from a tree without companion data.
    ///
    /// Computes the integrity hash as `H(0x03 || root || alg_id)`.
    /// Sets companion_hash to all zeros.
    #[must_use]
    pub fn from_tree(tree: &MerkleTree) -> Self {
        Self::from_tree_with_companion(tree, [0u8; HASH_SIZE])
    }

    /// Create a new merkle attribute from a tree with companion data hash.
    ///
    /// The companion_hash should be SHA-256 of the nodes data (either inline
    /// in `merkle_nodes` attribute or in `/merkle/{name}` companion dataset).
    #[must_use]
    pub fn from_tree_with_companion(tree: &MerkleTree, companion_hash: [u8; HASH_SIZE]) -> Self {
        let root = *tree.root();
        let algorithm = tree.algorithm();
        let integrity = Self::compute_integrity(&root, algorithm);

        Self {
            root,
            algorithm,
            integrity,
            companion_hash,
        }
    }

    /// Compute the integrity hash.
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

    /// Pack the attribute into a 97-byte binary blob.
    ///
    /// Layout: `[root:32][alg:1][integrity:32][companion_hash:32]`
    #[must_use]
    pub fn pack(&self) -> [u8; MERKLE_ATTR_SIZE] {
        let mut buf = [0u8; MERKLE_ATTR_SIZE];
        buf[ATTR_ROOT_OFFSET..ATTR_ROOT_END].copy_from_slice(&self.root);
        buf[ATTR_ALG_OFFSET] = self.algorithm.to_id();
        buf[ATTR_INTEGRITY_OFFSET..ATTR_INTEGRITY_END].copy_from_slice(&self.integrity);
        buf[ATTR_COMPANION_OFFSET..ATTR_COMPANION_END].copy_from_slice(&self.companion_hash);
        buf
    }

    /// Unpack from a 97-byte binary blob.
    ///
    /// Layout: `[root:32][alg:1][integrity:32][companion_hash:32]`
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// - The data is not 97 bytes (`WrongSize`)
    /// - The algorithm ID is unknown (`UnknownAlgorithm`)
    /// - The integrity hash does not match (`IntegrityMismatch`)
    pub fn unpack(data: &[u8]) -> Result<Self, MerkleError> {
        if data.len() != MERKLE_ATTR_SIZE {
            return Err(MerkleError::InvalidAttribute {
                reason: InvalidAttrReason::WrongSize,
            });
        }

        let mut root = [0u8; ATTR_ROOT_SIZE];
        root.copy_from_slice(&data[ATTR_ROOT_OFFSET..ATTR_ROOT_END]);

        let alg_id = data[ATTR_ALG_OFFSET];
        let algorithm = HashAlg::from_id(alg_id).ok_or(MerkleError::InvalidAttribute {
            reason: InvalidAttrReason::UnknownAlgorithm,
        })?;

        let mut integrity = [0u8; ATTR_INTEGRITY_SIZE];
        integrity.copy_from_slice(&data[ATTR_INTEGRITY_OFFSET..ATTR_INTEGRITY_END]);

        let mut companion_hash = [0u8; ATTR_COMPANION_SIZE];
        companion_hash.copy_from_slice(&data[ATTR_COMPANION_OFFSET..ATTR_COMPANION_END]);

        // Verify integrity hash
        let expected_integrity = Self::compute_integrity(&root, algorithm);
        if !constant_time_eq(&integrity, &expected_integrity) {
            return Err(MerkleError::InvalidAttribute {
                reason: InvalidAttrReason::IntegrityMismatch,
            });
        }

        Ok(Self {
            root,
            algorithm,
            integrity,
            companion_hash,
        })
    }

    /// Verify that a Merkle tree matches this attribute.
    #[must_use]
    pub fn verify_tree(&self, tree: &MerkleTree) -> bool {
        tree.algorithm() == self.algorithm && constant_time_eq(tree.root(), &self.root)
    }

    /// Verify the companion data integrity.
    ///
    /// Computes SHA-256 of the provided nodes data and compares with
    /// the stored companion_hash.
    ///
    /// # Returns
    ///
    /// - `Valid`: Companion hash matches the provided data
    /// - `NoCompanion`: No companion data present (hash is all zeros)
    /// - `HashMismatch`: Verification failed (possible tampering)
    #[must_use]
    pub fn verify_companion(&self, nodes_data: &[u8]) -> CompanionVerifyResult {
        // All zeros means no companion data
        if self.companion_hash == [0u8; HASH_SIZE] {
            return CompanionVerifyResult::NoCompanion;
        }
        let computed = compute_sha256(nodes_data);
        if constant_time_eq(&computed, &self.companion_hash) {
            CompanionVerifyResult::Valid
        } else {
            CompanionVerifyResult::HashMismatch
        }
    }

    /// Check if this attribute has companion data.
    #[must_use]
    pub fn has_companion(&self) -> bool {
        self.companion_hash != [0u8; HASH_SIZE]
    }

    /// Get the format version of this attribute.
    #[must_use]
    pub const fn version(&self) -> u8 {
        MERKLE_ATTR_VERSION_0
    }
}

/// Zero-copy reference to packed merkle attribute data.
///
/// This struct holds a reference to the raw attribute bytes using [`Cow`],
/// allowing zero-copy reads when the data is borrowed directly from HDF5
/// file memory, while still supporting owned data when needed.
///
/// # Format Versioning
///
/// The format version is determined implicitly by the data size:
/// - 97 bytes: Version 0 (current format)
///
/// Future versions may add an explicit version byte prefix.
///
/// # Example
///
/// ```ignore
/// use clawhdf5_format::merkle::MerkleAttrRef;
///
/// // Zero-copy read from HDF5 attribute data
/// let attr_data: &[u8] = /* read from HDF5 */;
/// let attr_ref = MerkleAttrRef::from_slice(attr_data)?;
///
/// // Access fields without copying
/// let root = attr_ref.root();
/// let alg = attr_ref.algorithm()?;
///
/// // Convert to owned if needed
/// let owned: MerkleAttr = attr_ref.to_owned_attr()?;
/// ```
#[derive(Debug, Clone)]
pub struct MerkleAttrRef<'a> {
    /// Raw attribute data (borrowed or owned).
    data: Cow<'a, [u8]>,
}

impl<'a> MerkleAttrRef<'a> {
    /// Create a reference from a borrowed slice (zero-copy).
    ///
    /// # Errors
    ///
    /// Returns `Err` if the data size is not 97 bytes.
    pub fn from_slice(data: &'a [u8]) -> Result<Self, MerkleError> {
        Self::validate_size(data.len())?;
        Ok(Self {
            data: Cow::Borrowed(data),
        })
    }

    /// Create a reference from owned data.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the data size is not 97 bytes.
    pub fn from_vec(data: Vec<u8>) -> Result<MerkleAttrRef<'static>, MerkleError> {
        Self::validate_size(data.len())?;
        Ok(MerkleAttrRef {
            data: Cow::Owned(data),
        })
    }

    /// Create from a packed MerkleAttr.
    #[must_use]
    pub fn from_attr(attr: &MerkleAttr) -> MerkleAttrRef<'static> {
        MerkleAttrRef {
            data: Cow::Owned(attr.pack().to_vec()),
        }
    }

    /// Validate data size matches expected attribute size.
    fn validate_size(size: usize) -> Result<(), MerkleError> {
        if size != MERKLE_ATTR_SIZE {
            return Err(MerkleError::InvalidAttribute {
                reason: InvalidAttrReason::WrongSize,
            });
        }
        Ok(())
    }

    /// Get the format version.
    #[must_use]
    pub const fn version(&self) -> u8 {
        MERKLE_ATTR_VERSION_0
    }

    /// Get a reference to the raw data.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Get the root hash (zero-copy slice).
    #[must_use]
    pub fn root(&self) -> &[u8] {
        &self.data[ATTR_ROOT_OFFSET..ATTR_ROOT_END]
    }

    /// Get the root hash as a fixed-size array.
    #[must_use]
    pub fn root_array(&self) -> [u8; HASH_SIZE] {
        let mut arr = [0u8; HASH_SIZE];
        arr.copy_from_slice(self.root());
        arr
    }

    /// Get the algorithm identifier byte.
    #[must_use]
    pub fn algorithm_id(&self) -> u8 {
        self.data[ATTR_ALG_OFFSET]
    }

    /// Get the hash algorithm.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the algorithm ID is unknown.
    pub fn algorithm(&self) -> Result<HashAlg, MerkleError> {
        HashAlg::from_id(self.algorithm_id()).ok_or(MerkleError::InvalidAttribute {
            reason: InvalidAttrReason::UnknownAlgorithm,
        })
    }

    /// Get the integrity hash (zero-copy slice).
    #[must_use]
    pub fn integrity(&self) -> &[u8] {
        &self.data[ATTR_INTEGRITY_OFFSET..ATTR_INTEGRITY_END]
    }

    /// Get the companion hash (zero-copy slice).
    #[must_use]
    pub fn companion_hash(&self) -> &[u8] {
        &self.data[ATTR_COMPANION_OFFSET..ATTR_COMPANION_END]
    }

    /// Check if this attribute has companion data.
    ///
    /// Returns `false` when companion hash is all zeros.
    #[must_use]
    pub fn has_companion(&self) -> bool {
        self.companion_hash().iter().any(|&b| b != 0)
    }

    /// Verify the integrity hash without fully unpacking.
    ///
    /// This validates that the root and algorithm haven't been tampered with.
    pub fn verify_integrity(&self) -> Result<(), MerkleError> {
        let algorithm = self.algorithm()?;
        let expected = MerkleAttr::compute_integrity(&self.root_array(), algorithm);
        // Safe: integrity() always returns exactly HASH_SIZE bytes for v0
        let integrity_arr: &[u8; HASH_SIZE] = self.integrity().try_into().map_err(|_| {
            MerkleError::InvalidAttribute {
                reason: InvalidAttrReason::WrongSize,
            }
        })?;
        if !constant_time_eq(integrity_arr, &expected) {
            return Err(MerkleError::InvalidAttribute {
                reason: InvalidAttrReason::IntegrityMismatch,
            });
        }
        Ok(())
    }

    /// Verify companion data against the stored hash.
    ///
    /// # Returns
    ///
    /// - `Valid`: Companion hash matches the provided data
    /// - `NoCompanion`: No companion data (hash is all zeros)
    /// - `HashMismatch`: Verification failed (possible tampering)
    #[must_use]
    pub fn verify_companion(&self, nodes_data: &[u8]) -> CompanionVerifyResult {
        if !self.has_companion() {
            return CompanionVerifyResult::NoCompanion;
        }
        let computed = compute_sha256(nodes_data);
        // Safe: companion_hash() always returns exactly HASH_SIZE bytes for v0
        let companion_arr: &[u8; HASH_SIZE] = match self.companion_hash().try_into() {
            Ok(arr) => arr,
            Err(_) => return CompanionVerifyResult::NoCompanion,
        };
        if constant_time_eq(&computed, companion_arr) {
            CompanionVerifyResult::Valid
        } else {
            CompanionVerifyResult::HashMismatch
        }
    }

    /// Convert to an owned `MerkleAttr`, verifying integrity.
    ///
    /// # Errors
    ///
    /// Returns `Err` if validation fails (unknown algorithm or integrity mismatch).
    pub fn to_owned_attr(&self) -> Result<MerkleAttr, MerkleError> {
        MerkleAttr::unpack(&self.data)
    }

    /// Convert to owned data, consuming the reference.
    #[must_use]
    pub fn into_owned(self) -> Vec<u8> {
        self.data.into_owned()
    }

    /// Check if the data is borrowed (zero-copy).
    #[must_use]
    pub fn is_borrowed(&self) -> bool {
        matches!(self.data, Cow::Borrowed(_))
    }
}

impl<'a> AsRef<[u8]> for MerkleAttrRef<'a> {
    fn as_ref(&self) -> &[u8] {
        &self.data
    }
}

impl<'a> From<&'a MerkleAttr> for MerkleAttrRef<'static> {
    fn from(attr: &'a MerkleAttr) -> Self {
        Self::from_attr(attr)
    }
}

/// Write the merkle_root attribute to a dataset.
///
/// Packs the root hash (32 bytes), algorithm identifier (1 byte),
/// integrity hash (32 bytes), and companion hash (32 bytes) into a
/// fixed-width 97-byte binary blob and writes it as the HDF5 attribute
/// `merkle_root`.
///
/// # Arguments
///
/// * `dataset` - The dataset builder to add the attribute to
/// * `tree` - The Merkle tree to store
///
/// # Errors
///
/// Currently infallible, but returns `Result` to allow future extension
/// for HDF5 write error handling without breaking API changes.
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
/// write_merkle_attr(ds, &tree)?;
/// ```
pub fn write_merkle_attr(
    dataset: &mut crate::type_builders::DatasetBuilder,
    tree: &MerkleTree,
) -> Result<(), MerkleError> {
    let attr = MerkleAttr::from_tree(tree);
    let packed = attr.pack();
    dataset.set_attr(MERKLE_ATTR_NAME, crate::type_builders::AttrValue::Bytes(packed.to_vec()));
    Ok(())
}

/// Threshold for inline node storage vs companion dataset.
/// Trees with up to this many leaf chunks will have their nodes stored
/// directly in an attribute. Larger trees use a companion dataset.
pub const INLINE_CHUNK_THRESHOLD: usize = 256;

/// Name of the attribute used for inline merkle nodes.
pub const MERKLE_NODES_ATTR_NAME: &str = "merkle_nodes";

/// Name of the group containing companion merkle datasets.
pub const MERKLE_GROUP_NAME: &str = "merkle";

/// Pending companion dataset to be written.
#[derive(Debug, Clone)]
struct PendingCompanion {
    name: String,
    data: Vec<u8>,
}

/// Batched writer for merkle companion datasets.
///
/// Collects companion datasets and writes them to a single `/merkle` group
/// when finalized. This avoids creating duplicate groups when multiple
/// datasets require companion storage.
///
/// # Example
///
/// ```ignore
/// use clawhdf5_format::merkle::{MerkleTree, HashAlg, MerkleCompanionWriter};
/// use clawhdf5_format::file_writer::FileWriter;
///
/// let mut fw = FileWriter::new();
/// let mut companion_writer = MerkleCompanionWriter::new();
///
/// // Add multiple datasets with merkle trees
/// let tree1 = MerkleTree::from_chunks(&chunks1, HashAlg::Blake3);
/// let result1 = companion_writer.add("dataset1", &tree1);
///
/// let tree2 = MerkleTree::from_chunks(&chunks2, HashAlg::Blake3);
/// let result2 = companion_writer.add("dataset2", &tree2);
///
/// // Write all companion datasets to a single /merkle group
/// companion_writer.finish(&mut fw);
/// ```
#[derive(Debug, Default)]
pub struct MerkleCompanionWriter {
    pending: Vec<PendingCompanion>,
}

impl MerkleCompanionWriter {
    /// Create a new companion writer.
    #[must_use]
    pub fn new() -> Self {
        Self { pending: Vec::new() }
    }

    /// Add a merkle tree's nodes, returning the storage result.
    ///
    /// For trees with ≤256 chunks, returns `Inline` with the packed nodes.
    /// For larger trees, queues the companion dataset and returns `Dataset`.
    pub fn add(&mut self, name: &str, tree: &MerkleTree) -> MerkleCompanionResult {
        let nodes = tree.nodes();
        let mut flat_nodes = Vec::with_capacity(nodes.len() * HASH_SIZE);
        for node in nodes {
            flat_nodes.extend_from_slice(node);
        }

        let companion_hash = compute_sha256(&flat_nodes);

        if tree.leaf_count() <= INLINE_CHUNK_THRESHOLD {
            MerkleCompanionResult::Inline {
                nodes: flat_nodes,
                companion_hash,
            }
        } else {
            self.pending.push(PendingCompanion {
                name: name.to_string(),
                data: flat_nodes,
            });
            MerkleCompanionResult::Dataset { companion_hash }
        }
    }

    /// Write all pending companion datasets to a single `/merkle` group.
    ///
    /// Does nothing if no datasets require companion storage.
    pub fn finish(self, file: &mut crate::file_writer::FileWriter) {
        if self.pending.is_empty() {
            return;
        }

        let mut group = file.create_group(MERKLE_GROUP_NAME);
        for companion in self.pending {
            let ds = group.create_dataset(&companion.name);
            ds.with_u8_data(&companion.data);
        }
        file.add_group(group.finish());
    }

    /// Check if any companion datasets are pending.
    #[must_use]
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Get the number of pending companion datasets.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

/// Result of `write_merkle_companion` indicating storage method and companion hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MerkleCompanionResult {
    /// Nodes were small enough to be inlined.
    ///
    /// Contains:
    /// - `nodes`: packed bytes to write as `merkle_nodes` attribute
    /// - `companion_hash`: SHA-256 of nodes for integrity verification
    Inline {
        /// Packed node hashes to write as attribute.
        nodes: Vec<u8>,
        /// SHA-256 hash of the nodes data for integrity verification.
        companion_hash: [u8; HASH_SIZE],
    },
    /// Nodes were written as a companion dataset at `/merkle/{name}`.
    ///
    /// Contains the SHA-256 hash of the companion dataset content.
    Dataset {
        /// SHA-256 hash of the companion dataset for integrity verification.
        companion_hash: [u8; HASH_SIZE],
    },
}

/// Write merkle tree node data, using companion dataset for large trees.
///
/// **Warning**: This function creates a new `/merkle` group each time it's called
/// for a large dataset. For files with multiple datasets requiring companion storage,
/// use [`MerkleCompanionWriter`] instead to batch writes to a single group.
///
/// For datasets with 256 or fewer chunks, returns packed node hashes that
/// should be written as a `merkle_nodes` attribute on the dataset. For larger
/// datasets, writes the nodes as a flat u8 dataset at `/merkle/{name}`.
///
/// The nodes array contains all internal and leaf hashes in level-order
/// (breadth-first) layout. Each hash is 32 bytes, so the total size is
/// `node_count * 32` bytes.
///
/// # Arguments
///
/// * `file` - The FileWriter to write companion datasets to (used only for large trees)
/// * `name` - The name of the dataset this tree belongs to
/// * `tree` - The Merkle tree whose nodes to write
///
/// # Returns
///
/// - `Ok(Inline { nodes, companion_hash })` - For trees with ≤256 chunks
/// - `Ok(Dataset { companion_hash })` - For larger trees
///
/// # Layout
///
/// For a tree with N leaves (padded to next power of 2):
/// - Total nodes: `2 * padded_leaf_count - 1`
/// - Node 0: root
/// - Nodes 1..padded_leaf_count-1: internal nodes (level-order)
/// - Nodes padded_leaf_count-1..2*padded_leaf_count-1: leaf hashes
///
/// # Example
///
/// ```ignore
/// use clawhdf5_format::merkle::{MerkleTree, HashAlg, write_merkle_companion, MerkleCompanionResult};
/// use clawhdf5_format::file_writer::FileWriter;
/// use clawhdf5_format::type_builders::AttrValue;
///
/// let chunks: Vec<&[u8]> = vec![&[1, 2, 3], &[4, 5, 6]];
/// let tree = MerkleTree::from_chunks(&chunks, HashAlg::Blake3);
///
/// let mut fw = FileWriter::new();
/// let ds = fw.create_dataset("data");
/// ds.with_u8_data(&[1, 2, 3, 4, 5, 6]);
///
/// match write_merkle_companion(&mut fw, "data", &tree)? {
///     MerkleCompanionResult::Inline { nodes, companion_hash } => {
///         ds.set_attr("merkle_nodes", AttrValue::Bytes(nodes));
///     }
///     MerkleCompanionResult::Dataset { companion_hash } => {
///         // Companion dataset already written at /merkle/data
///     }
/// }
/// ```
pub fn write_merkle_companion(
    file: &mut crate::file_writer::FileWriter,
    name: &str,
    tree: &MerkleTree,
) -> Result<MerkleCompanionResult, MerkleError> {
    // Flatten nodes to bytes
    let nodes = tree.nodes();
    let mut flat_nodes = Vec::with_capacity(nodes.len() * HASH_SIZE);
    for node in nodes {
        flat_nodes.extend_from_slice(node);
    }

    // Compute SHA-256 companion integrity hash
    let companion_hash = compute_sha256(&flat_nodes);

    if tree.leaf_count() <= INLINE_CHUNK_THRESHOLD {
        // Return packed nodes for caller to add as attribute
        Ok(MerkleCompanionResult::Inline {
            nodes: flat_nodes,
            companion_hash,
        })
    } else {
        // Create companion dataset at /merkle/{name}
        let mut group = file.create_group(MERKLE_GROUP_NAME);
        let companion = group.create_dataset(name);
        companion.with_u8_data(&flat_nodes);
        file.add_group(group.finish());
        Ok(MerkleCompanionResult::Dataset { companion_hash })
    }
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
    fn test_three_leaf_tree_manual_verification() {
        // 3-leaf tree tests padding with null sentinel
        // Tree structure (padded to 4 leaves):
        //
        //              root
        //             /    \
        //           n1      n2
        //          / \     /  \
        //        L0  L1   L2  NULL
        //
        let alg = HashAlg::Blake3;

        // Create leaves with known inputs matching gen_merkle_vectors
        let leaf0 = hash_chunk(b"leaf A", alg);
        let leaf1 = hash_chunk(b"leaf B", alg);
        let leaf2 = hash_chunk(b"leaf C", alg);
        let null_sentinel = alg.null_sentinel();

        // Build tree from leaf hashes
        let tree =
            MerkleTree::build(&[leaf0, leaf1, leaf2], alg).expect("build should succeed");

        // Verify tree structure
        assert_eq!(tree.leaf_count(), 3);
        assert_eq!(tree.padded_leaf_count(), 4);
        assert_eq!(tree.depth(), 3); // root + 1 internal level + leaves

        // Manually compute expected internal nodes and root
        // n1 = H(0x01 || L0 || L1)
        let mut combined = [0u8; 65];
        combined[0] = INTERNAL_PREFIX;
        combined[1..33].copy_from_slice(&leaf0);
        combined[33..65].copy_from_slice(&leaf1);
        let n1: [u8; 32] = blake3::hash(&combined).into();

        // n2 = H(0x01 || L2 || NULL)
        combined[1..33].copy_from_slice(&leaf2);
        combined[33..65].copy_from_slice(&null_sentinel);
        let n2: [u8; 32] = blake3::hash(&combined).into();

        // root = H(0x01 || n1 || n2)
        combined[1..33].copy_from_slice(&n1);
        combined[33..65].copy_from_slice(&n2);
        let expected_root: [u8; 32] = blake3::hash(&combined).into();

        assert_eq!(tree.root(), &expected_root, "3-leaf tree root mismatch");

        // Verify the padding slot contains null sentinel
        let internal_nodes = tree.padded_leaf_count() - 1; // 3
        assert_eq!(
            tree.nodes()[internal_nodes + 3],
            null_sentinel,
            "Padding slot should contain null sentinel"
        );

        // Verify proofs work for all 3 leaves
        for i in 0..3 {
            let proof = tree.proof(i).expect("proof should exist");
            let chunk = match i {
                0 => b"leaf A".as_slice(),
                1 => b"leaf B".as_slice(),
                2 => b"leaf C".as_slice(),
                _ => unreachable!(),
            };
            assert!(
                tree.verify_proof(i, chunk, &proof),
                "Proof verification failed for leaf {}",
                i
            );
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

        // Verify size (now 97 bytes with companion hash)
        assert_eq!(packed.len(), MERKLE_ATTR_SIZE);
        assert_eq!(packed.len(), 97);

        // Unpack and verify round-trip
        let unpacked = MerkleAttr::unpack(&packed).expect("unpack should succeed");
        assert_eq!(unpacked.root, attr.root);
        assert_eq!(unpacked.algorithm, attr.algorithm);
        assert_eq!(unpacked.integrity, attr.integrity);
        assert_eq!(unpacked.companion_hash, [0u8; HASH_SIZE]); // No companion for basic from_tree
        assert!(!unpacked.has_companion());
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
            MerkleAttr::unpack(&packed).is_err(),
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
            MerkleAttr::unpack(&packed).is_err(),
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
        // Too short (97 bytes expected)
        assert!(MerkleAttr::unpack(&[0u8; 96]).is_err());
        // Too long
        assert!(MerkleAttr::unpack(&[0u8; 98]).is_err());
        // Empty
        assert!(MerkleAttr::unpack(&[]).is_err());
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
        write_merkle_attr(ds, &tree).expect("write_merkle_attr should succeed");

        // Finish and verify the file is valid
        let bytes = fw.finish().expect("file should build");
        assert!(!bytes.is_empty());

        // The attribute should be readable (basic check)
        // Full parsing would require reading the attribute back
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_write_merkle_companion_inline() {
        use crate::file_writer::FileWriter;
        use crate::type_builders::AttrValue;

        // Create a small tree (< 256 chunks) - should return inline data
        let chunks: Vec<Vec<u8>> = (0..10).map(|i| vec![i as u8; 64]).collect();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);

        let mut fw = FileWriter::new();

        // Write merkle companion - should return Inline result
        let result = write_merkle_companion(&mut fw, "small_data", &tree)
            .expect("write_merkle_companion should succeed");

        match result {
            MerkleCompanionResult::Inline { nodes, companion_hash } => {
                // Verify expected size: 10 leaves padded to 16, so 31 nodes * 32 bytes
                assert_eq!(nodes.len(), 31 * HASH_SIZE);

                // Verify companion hash is SHA-256 of nodes
                let expected_hash = compute_sha256(&nodes);
                assert_eq!(companion_hash, expected_hash);

                // Add as attribute to verify it works
                let ds = fw.create_dataset("small_data");
                ds.with_u8_data(&[1, 2, 3, 4]);
                ds.set_attr(MERKLE_NODES_ATTR_NAME, AttrValue::Bytes(nodes));
            }
            MerkleCompanionResult::Dataset { .. } => {
                panic!("Expected Inline result for small tree");
            }
        }

        // Finish and verify the file is valid
        let bytes = fw.finish().expect("file should build");
        assert!(!bytes.is_empty());

        assert_eq!(tree.padded_leaf_count(), 16);
        assert_eq!(tree.nodes().len(), 31);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_write_merkle_companion_dataset() {
        use crate::file_writer::FileWriter;

        // Create a large tree (> 256 chunks) - should create companion dataset
        let chunks: Vec<Vec<u8>> = (0..300).map(|i| vec![(i % 256) as u8; 64]).collect();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);

        let mut fw = FileWriter::new();

        // Also create the main dataset
        let ds = fw.create_dataset("large_data");
        ds.with_u8_data(&[1, 2, 3, 4]);

        // Write merkle companion - should create /merkle/large_data dataset
        let result = write_merkle_companion(&mut fw, "large_data", &tree)
            .expect("write_merkle_companion should succeed");

        match result {
            MerkleCompanionResult::Dataset { companion_hash } => {
                // Verify companion hash is not all zeros
                assert_ne!(companion_hash, [0u8; HASH_SIZE]);
            }
            MerkleCompanionResult::Inline { .. } => {
                panic!("Expected Dataset result for large tree");
            }
        }

        // Finish and verify the file is valid
        let bytes = fw.finish().expect("file should build");
        assert!(!bytes.is_empty());

        // Verify expected node count: 300 leaves padded to 512, so 1023 nodes
        assert_eq!(tree.padded_leaf_count(), 512);
        assert_eq!(tree.nodes().len(), 1023);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_write_merkle_companion_threshold_boundary() {
        use crate::file_writer::FileWriter;

        // Test exactly at the threshold (256 chunks) - should use inline
        let chunks: Vec<Vec<u8>> = (0..256).map(|i| vec![(i % 256) as u8; 32]).collect();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);

        assert_eq!(tree.leaf_count(), 256);

        let mut fw = FileWriter::new();

        let result = write_merkle_companion(&mut fw, "boundary_data", &tree)
            .expect("write_merkle_companion should succeed");

        match result {
            MerkleCompanionResult::Inline { nodes, companion_hash } => {
                // 256 leaves = 256 padded (power of 2), so 511 nodes * 32 bytes
                assert_eq!(nodes.len(), 511 * HASH_SIZE);
                assert_ne!(companion_hash, [0u8; HASH_SIZE]);
            }
            MerkleCompanionResult::Dataset { .. } => {
                panic!("Expected Inline result at threshold");
            }
        }

        // Create dataset and finish file
        let ds = fw.create_dataset("boundary_data");
        ds.with_u8_data(&[1, 2, 3, 4]);

        let bytes = fw.finish().expect("file should build");
        assert!(!bytes.is_empty());

        assert_eq!(tree.padded_leaf_count(), 256);
        assert_eq!(tree.nodes().len(), 511);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_write_merkle_companion_just_over_threshold() {
        use crate::file_writer::FileWriter;

        // Test just over threshold (257 chunks) - should create dataset
        let chunks: Vec<Vec<u8>> = (0..257).map(|i| vec![(i % 256) as u8; 32]).collect();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);

        assert_eq!(tree.leaf_count(), 257);

        let mut fw = FileWriter::new();

        let result = write_merkle_companion(&mut fw, "over_threshold", &tree)
            .expect("write_merkle_companion should succeed");

        match result {
            MerkleCompanionResult::Dataset { companion_hash } => {
                assert_ne!(companion_hash, [0u8; HASH_SIZE]);
            }
            MerkleCompanionResult::Inline { .. } => {
                panic!("Expected Dataset result for large tree");
            }
        }

        // Create main dataset and finish file
        let ds = fw.create_dataset("over_threshold");
        ds.with_u8_data(&[1, 2, 3, 4]);

        let bytes = fw.finish().expect("file should build");
        assert!(!bytes.is_empty());

        // 257 leaves padded to 512, so 1023 nodes
        assert_eq!(tree.padded_leaf_count(), 512);
        assert_eq!(tree.nodes().len(), 1023);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_merkle_companion_writer_multi_dataset() {
        use crate::file_writer::FileWriter;

        // Create two large trees (>256 chunks each) that require companion datasets
        let chunks1: Vec<Vec<u8>> = (0..300).map(|i| vec![(i % 256) as u8; 64]).collect();
        let refs1: Vec<&[u8]> = chunks1.iter().map(|c| c.as_slice()).collect();
        let tree1 = MerkleTree::from_chunks(&refs1, HashAlg::Blake3);

        let chunks2: Vec<Vec<u8>> = (0..400).map(|i| vec![((i + 50) % 256) as u8; 64]).collect();
        let refs2: Vec<&[u8]> = chunks2.iter().map(|c| c.as_slice()).collect();
        let tree2 = MerkleTree::from_chunks(&refs2, HashAlg::Blake3);

        // Both trees should require companion datasets (>256 chunks)
        assert!(tree1.leaf_count() > INLINE_CHUNK_THRESHOLD);
        assert!(tree2.leaf_count() > INLINE_CHUNK_THRESHOLD);

        let mut fw = FileWriter::new();
        let mut companion_writer = MerkleCompanionWriter::new();

        // Add both trees to the batched writer
        let result1 = companion_writer.add("dataset1", &tree1);
        let result2 = companion_writer.add("dataset2", &tree2);

        // Both should return Dataset results (queued for writing)
        match result1 {
            MerkleCompanionResult::Dataset { companion_hash } => {
                assert_ne!(companion_hash, [0u8; HASH_SIZE]);
            }
            MerkleCompanionResult::Inline { .. } => {
                panic!("Expected Dataset result for large tree1");
            }
        }

        match result2 {
            MerkleCompanionResult::Dataset { companion_hash } => {
                assert_ne!(companion_hash, [0u8; HASH_SIZE]);
            }
            MerkleCompanionResult::Inline { .. } => {
                panic!("Expected Dataset result for large tree2");
            }
        }

        // Verify pending state
        assert!(companion_writer.has_pending());
        assert_eq!(companion_writer.pending_count(), 2);

        // Create main datasets
        let ds1 = fw.create_dataset("dataset1");
        ds1.with_u8_data(&[1, 2, 3, 4]);
        let ds2 = fw.create_dataset("dataset2");
        ds2.with_u8_data(&[5, 6, 7, 8]);

        // Write all companion datasets to single /merkle group
        companion_writer.finish(&mut fw);

        // Finish file and verify it's valid
        let bytes = fw.finish().expect("file should build");
        assert!(!bytes.is_empty());
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_merkle_attr_with_companion_hash() {
        let chunks = make_test_chunks();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);

        // Create a fake companion hash
        let companion_hash = compute_sha256(b"test companion data");

        let attr = MerkleAttr::from_tree_with_companion(&tree, companion_hash);

        // Pack and unpack
        let packed = attr.pack();
        assert_eq!(packed.len(), MERKLE_ATTR_SIZE); // 97 bytes

        let unpacked = MerkleAttr::unpack(&packed).expect("unpack should succeed");
        assert_eq!(unpacked.root, attr.root);
        assert_eq!(unpacked.algorithm, attr.algorithm);
        assert_eq!(unpacked.companion_hash, companion_hash);
        assert!(unpacked.has_companion());
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_merkle_attr_verify_companion() {
        let chunks = make_test_chunks();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);

        // Flatten nodes
        let nodes = tree.nodes();
        let mut flat_nodes = Vec::new();
        for node in nodes {
            flat_nodes.extend_from_slice(node);
        }

        let companion_hash = compute_sha256(&flat_nodes);
        let attr = MerkleAttr::from_tree_with_companion(&tree, companion_hash);

        // Verify with correct data
        assert_eq!(
            attr.verify_companion(&flat_nodes),
            CompanionVerifyResult::Valid
        );

        // Verify with tampered data fails
        let mut tampered = flat_nodes.clone();
        tampered[0] ^= 0xFF;
        assert_eq!(
            attr.verify_companion(&tampered),
            CompanionVerifyResult::HashMismatch
        );
    }

    /// Round-trip test: write 1024-chunk dataset with merkle companion,
    /// read it back, verify companion-integrity hash matches.
    #[test]
    #[cfg(feature = "blake3")]
    fn test_merkle_roundtrip_1024_chunks() {
        use crate::attribute::{extract_attributes, find_attribute};
        use crate::data_layout::DataLayout;
        use crate::file_writer::FileWriter;
        use crate::group_v2::resolve_path_any;
        use crate::object_header::ObjectHeader;
        use crate::signature::find_signature;
        use crate::superblock::Superblock;
        use crate::type_builders::AttrValue;

        // 1. Create 1024 synthetic chunks (each 64 bytes)
        let chunks: Vec<Vec<u8>> = (0..1024)
            .map(|i| {
                let mut chunk = vec![0u8; 64];
                // Fill with predictable pattern
                for (j, byte) in chunk.iter_mut().enumerate() {
                    *byte = ((i + j) % 256) as u8;
                }
                chunk
            })
            .collect();

        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);

        // Verify we're above the inline threshold (256)
        assert_eq!(tree.leaf_count(), 1024);
        assert!(tree.leaf_count() > INLINE_CHUNK_THRESHOLD);

        // 2. Write the file with merkle companion and attribute
        let mut fw = FileWriter::new();

        // Write merkle companion first - should create /merkle/sensor_data dataset
        let result = write_merkle_companion(&mut fw, "sensor_data", &tree)
            .expect("write_merkle_companion should succeed");

        let companion_hash = match &result {
            MerkleCompanionResult::Dataset { companion_hash } => *companion_hash,
            MerkleCompanionResult::Inline { .. } => {
                panic!("Expected Dataset result for 1024 chunks");
            }
        };

        // Now create the main dataset with the merkle attribute
        let ds = fw.create_dataset("sensor_data");
        // Flatten all chunk data for the dataset
        let all_data: Vec<u8> = chunks.iter().flatten().copied().collect();
        ds.with_u8_data(&all_data);

        // Write merkle_root attribute with companion hash
        let attr = MerkleAttr::from_tree_with_companion(&tree, companion_hash);
        ds.set_attr(MERKLE_ATTR_NAME, AttrValue::Bytes(attr.pack().to_vec()));

        // 3. Finish and get file bytes
        let file_bytes = fw.finish().expect("file should build");
        assert!(!file_bytes.is_empty());

        // 4. Re-open and parse the file
        let sig_offset = find_signature(&file_bytes).expect("signature not found");
        let sb = Superblock::parse(&file_bytes, sig_offset).expect("superblock parse failed");

        // 5. Read back the merkle_root attribute from sensor_data dataset
        let data_addr =
            resolve_path_any(&file_bytes, &sb, "sensor_data").expect("sensor_data not found");
        let data_hdr = ObjectHeader::parse(
            &file_bytes,
            data_addr as usize,
            sb.offset_size,
            sb.length_size,
        )
        .expect("dataset header parse failed");

        let attrs = extract_attributes(&data_hdr, sb.length_size).expect("extract attrs failed");
        let merkle_attr = find_attribute(&attrs, MERKLE_ATTR_NAME).expect("merkle_root attr not found");

        // Verify attribute size
        assert_eq!(merkle_attr.raw_data.len(), MERKLE_ATTR_SIZE);

        // Unpack and verify
        let unpacked =
            MerkleAttr::unpack(&merkle_attr.raw_data).expect("merkle attr unpack failed");
        assert_eq!(unpacked.root, *tree.root());
        assert_eq!(unpacked.algorithm, HashAlg::Blake3);
        assert!(unpacked.has_companion());

        // 6. Read back the companion dataset from /merkle/sensor_data
        let companion_addr = resolve_path_any(&file_bytes, &sb, "merkle/sensor_data")
            .expect("companion dataset not found");
        let companion_hdr = ObjectHeader::parse(
            &file_bytes,
            companion_addr as usize,
            sb.offset_size,
            sb.length_size,
        )
        .expect("companion header parse failed");

        // Find the data layout message to get the companion data
        let mut companion_data: Option<Vec<u8>> = None;
        for msg in &companion_hdr.messages {
            if msg.msg_type == crate::message_type::MessageType::DataLayout {
                let layout = DataLayout::parse(&msg.data, sb.offset_size, sb.length_size)
                    .expect("data layout parse failed");
                if let DataLayout::Contiguous { address, size } = layout {
                    if let Some(addr) = address {
                        let start = addr as usize;
                        let end = start + size as usize;
                        companion_data = Some(file_bytes[start..end].to_vec());
                    }
                }
            }
        }

        let companion_bytes = companion_data.expect("companion data not found in layout");

        // 7. Verify companion-integrity hash matches recomputed value
        let recomputed_hash = compute_sha256(&companion_bytes);
        assert_eq!(
            unpacked.companion_hash, recomputed_hash,
            "Companion hash mismatch: stored vs recomputed"
        );

        // Also verify using the verify_companion method
        assert_eq!(
            unpacked.verify_companion(&companion_bytes),
            CompanionVerifyResult::Valid,
            "verify_companion should return Valid"
        );

        // Verify node count: 1024 leaves padded to 1024 (power of 2), so 2047 nodes
        // Each node is 32 bytes, so 2047 * 32 = 65504 bytes
        assert_eq!(tree.padded_leaf_count(), 1024);
        assert_eq!(tree.nodes().len(), 2047);
        assert_eq!(companion_bytes.len(), 2047 * HASH_SIZE);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_merkle_attr_ref_zero_copy() {
        let chunks = make_test_chunks();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);

        let attr = MerkleAttr::from_tree(&tree);
        let packed = attr.pack();

        // Create zero-copy reference from slice
        let attr_ref = MerkleAttrRef::from_slice(&packed).expect("should parse");

        // Verify it's borrowed (zero-copy)
        assert!(attr_ref.is_borrowed());

        // Access fields without copying
        assert_eq!(attr_ref.root(), &attr.root);
        assert_eq!(attr_ref.algorithm_id(), attr.algorithm.to_id());
        assert_eq!(attr_ref.algorithm().unwrap(), attr.algorithm);
        assert_eq!(attr_ref.integrity(), &attr.integrity);
        assert_eq!(attr_ref.companion_hash(), &attr.companion_hash);

        // Version should be 0
        assert_eq!(attr_ref.version(), MERKLE_ATTR_VERSION_0);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_merkle_attr_ref_from_vec() {
        let chunks = make_test_chunks();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);

        let attr = MerkleAttr::from_tree(&tree);
        let packed = attr.pack().to_vec();

        // Create from owned vec
        let attr_ref = MerkleAttrRef::from_vec(packed).expect("should parse");

        // Verify it's owned (not borrowed)
        assert!(!attr_ref.is_borrowed());

        // Should still work correctly
        assert_eq!(attr_ref.root_array(), attr.root);
        assert_eq!(attr_ref.algorithm().unwrap(), attr.algorithm);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_merkle_attr_ref_verify_integrity() {
        let chunks = make_test_chunks();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);

        let attr = MerkleAttr::from_tree(&tree);
        let packed = attr.pack();

        // Valid data should pass integrity check
        let attr_ref = MerkleAttrRef::from_slice(&packed).expect("should parse");
        assert!(attr_ref.verify_integrity().is_ok());

        // Tampered data should fail
        let mut tampered = packed;
        tampered[0] ^= 0xFF;
        let tampered_ref = MerkleAttrRef::from_slice(&tampered).expect("should parse");
        assert!(tampered_ref.verify_integrity().is_err());
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_merkle_attr_ref_to_owned() {
        let chunks = make_test_chunks();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);

        let attr = MerkleAttr::from_tree(&tree);
        let packed = attr.pack();

        // Create reference and convert to owned
        let attr_ref = MerkleAttrRef::from_slice(&packed).expect("should parse");
        let owned = attr_ref.to_owned_attr().expect("should convert");

        // Should match original
        assert_eq!(owned.root, attr.root);
        assert_eq!(owned.algorithm, attr.algorithm);
        assert_eq!(owned.integrity, attr.integrity);
        assert_eq!(owned.companion_hash, attr.companion_hash);
    }

    #[test]
    fn test_merkle_attr_ref_invalid_size() {
        // Too short
        assert!(MerkleAttrRef::from_slice(&[0u8; 96]).is_err());
        // Too long
        assert!(MerkleAttrRef::from_slice(&[0u8; 98]).is_err());
        // Empty
        assert!(MerkleAttrRef::from_slice(&[]).is_err());
        // Current size (97 bytes) should work
        assert!(MerkleAttrRef::from_slice(&[0u8; 97]).is_ok());
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_merkle_attr_ref_verify_companion() {
        let chunks = make_test_chunks();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);

        // Create attribute with companion hash
        let nodes = tree.nodes();
        let mut flat_nodes = Vec::with_capacity(nodes.len() * HASH_SIZE);
        for node in nodes {
            flat_nodes.extend_from_slice(node);
        }
        let companion_hash = compute_sha256(&flat_nodes);
        let attr = MerkleAttr::from_tree_with_companion(&tree, companion_hash);
        let packed = attr.pack();

        // Zero-copy reference should verify companion
        let attr_ref = MerkleAttrRef::from_slice(&packed).expect("should parse");
        assert!(attr_ref.has_companion());
        assert_eq!(
            attr_ref.verify_companion(&flat_nodes),
            CompanionVerifyResult::Valid
        );

        // Wrong data should fail
        let wrong_data = vec![0u8; flat_nodes.len()];
        assert_eq!(
            attr_ref.verify_companion(&wrong_data),
            CompanionVerifyResult::HashMismatch
        );
    }

    #[test]
    fn test_merkle_attr_version() {
        // MerkleAttr should report version 0
        let attr = MerkleAttr {
            root: [0u8; HASH_SIZE],
            algorithm: HashAlg::Blake3,
            integrity: [0u8; HASH_SIZE],
            companion_hash: [0u8; HASH_SIZE],
        };
        assert_eq!(attr.version(), MERKLE_ATTR_VERSION_0);

        // Size constants
        assert_eq!(MERKLE_ATTR_SIZE, 97);
    }
}
