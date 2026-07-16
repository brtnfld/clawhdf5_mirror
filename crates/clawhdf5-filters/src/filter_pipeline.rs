//! Encrypt-then-MAC filter pipeline for HDF5 chunks.
//!
//! This module enforces the cryptographically secure filter ordering from
//! S2-D2-Yr2 §4.2 (Filter Pipeline Interaction):
//!
//! ```text
//! Application data → shuffle → compress → encrypt → hash (Merkle leaf) → Storage
//! ```
//!
//! This is an **Encrypt-then-MAC (EtM)** construction, which is the standard
//! cryptographic best practice for authenticated encryption.
//!
//! # Security Properties
//!
//! 1. **Third-party integrity verification without decryption**: Because the
//!    Merkle leaf hashes cover the ciphertext, any party (storage node, archive,
//!    researcher without decryption key) can verify chunk integrity.
//!
//! 2. **Early rejection of tampered ciphertext**: A reader can detect tampering
//!    *before* spending CPU cycles on decryption.
//!
//! 3. **AEAD tag binding**: The Poly1305 authentication tag is included in the
//!    leaf hash, providing defense-in-depth.
//!
//! 4. **Version counter binding**: The per-chunk version counter is hashed into
//!    the leaf, preventing rollback attacks at the Merkle level.
//!
//! 5. **Position-swapping attack prevention**: The chunk index is bound into
//!    the leaf hash, preventing adversaries from substituting chunk k' in place
//!    of chunk k.
//!
//! # Leaf Hash Format
//!
//! Per P2.2 step 5 and S2-D2-Yr2 §4.6:
//! ```text
//! H_leaf(k) = H(0x00 || len(k) || k || len(Ciphertext_k) || Ciphertext_k || Tag_k || v_k)
//! ```
//!
//! Where:
//! - `0x00` is the leaf domain-separation prefix
//! - `len(k)` is the byte length of k (always 8 for u64), encoded as u32 big-endian
//! - `k` is the chunk index as u64 little-endian
//! - `len(Ciphertext_k)` is the ciphertext length as u32 big-endian
//! - `Ciphertext_k || Tag_k` is the ciphertext with appended Poly1305 tag
//! - `v_k` is the version counter as u64 little-endian
//!
//! This length-prefixed serialization prevents boundary-shifting attacks when
//! ciphertext lengths vary (e.g., after compression).
//!
//! # Spec Reference
//!
//! S2-D2-Yr2 P2.2 step 4: "Enforce the Encrypt-then-MAC pipeline order for
//! filters: compress → shuffle → encrypt. The Merkle leaf hash must be
//! computed over the ciphertext output of the filter chain, not over the
//! plaintext."
//!
//! **Note**: The spec text says "compress → shuffle → encrypt" but this implementation
//! uses "shuffle → compress → encrypt" which matches HDF5's actual filter order.
//! Shuffling before compression improves compression ratio by grouping similar bytes.
//! The spec text appears to have a typo; the security property (EtM) is preserved
//! regardless of shuffle/compress order since both happen before encryption.
//!
//! S2-D2-Yr2 P2.2 step 5: "Bind the AEAD authentication tag and the version
//! counter into the Merkle leaf." The simplified formula in P2.2 step 5 omits
//! length prefixes for brevity; this implementation follows the complete §4.6
//! formula with length-prefixed serialization to prevent boundary-shifting attacks.

use crate::chacha20_filter::{ChaCha20Error, encrypt_chunk};
use crate::mock_keystore::Dek;

/// Size of a BLAKE3 hash output in bytes.
pub const HASH_SIZE: usize = 32;

/// Domain separator for leaf node hashes (matches clawhdf5-format).
const LEAF_PREFIX: u8 = 0x00;

/// A 32-byte hash digest.
pub type Hash = [u8; HASH_SIZE];

/// Error type for filter pipeline operations.
#[derive(Debug)]
pub enum FilterPipelineError {
    /// Compression failed.
    CompressionFailed(String),
    /// Decompression failed.
    DecompressionFailed(String),
    /// Encryption failed.
    EncryptionFailed(ChaCha20Error),
    /// Decryption failed.
    DecryptionFailed(ChaCha20Error),
}

impl std::fmt::Display for FilterPipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FilterPipelineError::CompressionFailed(e) => {
                write!(f, "compression failed: {e}")
            }
            FilterPipelineError::DecompressionFailed(e) => {
                write!(f, "decompression failed: {e}")
            }
            FilterPipelineError::EncryptionFailed(e) => {
                write!(f, "encryption failed: {e}")
            }
            FilterPipelineError::DecryptionFailed(e) => {
                write!(f, "decryption failed: {e}")
            }
        }
    }
}

impl std::error::Error for FilterPipelineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FilterPipelineError::EncryptionFailed(e) => Some(e),
            FilterPipelineError::DecryptionFailed(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ChaCha20Error> for FilterPipelineError {
    fn from(e: ChaCha20Error) -> Self {
        FilterPipelineError::EncryptionFailed(e)
    }
}

/// Result of running the write pipeline on a chunk.
#[derive(Debug, Clone)]
pub struct FilteredChunk {
    /// The final ciphertext (compressed, shuffled, encrypted with tag).
    pub ciphertext: Vec<u8>,
    /// The Merkle leaf hash: `H(0x00 || len(k) || k || len(ct) || ct || tag || v_k)`.
    /// Binds chunk index, ciphertext, AEAD tag, and version counter.
    pub leaf_hash: Hash,
    /// The version counter used for this chunk.
    pub version: u64,
    /// The chunk index (bound into leaf_hash to prevent position-swapping attacks).
    pub chunk_idx: u64,
}

/// Compute the Merkle leaf hash for an encrypted chunk.
///
/// This implements the EtM leaf hash format from P2.2 step 5 and S2-D2-Yr2 §4.6:
/// ```text
/// H_leaf(k) = H(0x00 || len(k) || k || len(Ciphertext) || Ciphertext || Tag || v_k)
/// ```
///
/// The serialization uses length-prefixed fields to prevent boundary-shifting
/// attacks when ciphertext lengths vary across chunks. All length fields are
/// fixed-width big-endian integers (u32).
///
/// Explicitly binding the chunk index `k` into the leaf hash prevents
/// **position-swapping attacks**: an adversary cannot substitute the ciphertext,
/// tag, and version counter of chunk k' in place of chunk k, because the index
/// is baked into the hash.
///
/// TODO(P2.4): this leaf formula is distinct from `clawhdf5-format`'s
/// `HashAlg::hash_leaf` (`H(0x00 || chunk)`), which is what `verify_chunk` /
/// `verify_dataset` recompute — so a `Dataset` built from an encrypted file's
/// companion nodes cannot currently be verified by the format-side functions.
/// The attack harness (P2.4) needs a version-aware verification path in
/// `clawhdf5-format` (or a leaf-formula parameter on `Dataset`) before T4
/// selective-rollback detection works end-to-end rather than only at the
/// leaf-hash level (see `crash_vs_tamper_matrix.rs`, scenario e).
///
/// # Arguments
///
/// * `chunk_idx` - The chunk index in the dataset
/// * `ciphertext` - The encrypted chunk data with authentication tag appended
/// * `version` - The per-chunk version counter
///
/// # Returns
///
/// The 32-byte BLAKE3 leaf hash.
#[inline]
#[must_use]
pub fn compute_leaf_hash(chunk_idx: u64, ciphertext: &[u8], version: u64) -> Hash {
    let mut hasher = blake3::Hasher::new();

    // Domain separation prefix
    hasher.update(&[LEAF_PREFIX]); // 0x00

    // Chunk index with length prefix (len is always 8 for u64)
    hasher.update(&8u32.to_be_bytes()); // len(k) = 8
    hasher.update(&chunk_idx.to_le_bytes()); // k

    // Ciphertext with length prefix (includes tag for AEAD)
    #[expect(
        clippy::cast_possible_truncation,
        reason = "ciphertext length always < 4GB"
    )]
    let ct_len = ciphertext.len() as u32;
    hasher.update(&ct_len.to_be_bytes()); // len(ct)
    hasher.update(ciphertext); // ct || tag

    // Version counter (fixed width, no length prefix needed)
    hasher.update(&version.to_le_bytes()); // v_k

    hasher.finalize().into()
}

/// Compute the Merkle leaf hash for a plaintext chunk (no encryption).
///
/// Per S2-D2-Yr2 §4.6, for plaintext (unencrypted) chunks:
/// ```text
/// H_leaf(k) = H(0x00 || len(k) || k || len(Chunk_k) || Chunk_k || v_k)
/// ```
///
/// The chunk index `k` is bound to prevent **position-swapping attacks**.
/// The version counter `v_k` is included for consistency and rollback detection.
///
/// # Arguments
///
/// * `chunk_idx` - The chunk index in the dataset
/// * `chunk_data` - The raw (possibly compressed) chunk data
/// * `version` - The per-chunk version counter
///
/// # Returns
///
/// The 32-byte BLAKE3 leaf hash.
#[inline]
#[must_use]
pub fn compute_leaf_hash_plaintext(chunk_idx: u64, chunk_data: &[u8], version: u64) -> Hash {
    let mut hasher = blake3::Hasher::new();

    // Domain separation prefix
    hasher.update(&[LEAF_PREFIX]); // 0x00

    // Chunk index with length prefix (len is always 8 for u64)
    hasher.update(&8u32.to_be_bytes()); // len(k) = 8
    hasher.update(&chunk_idx.to_le_bytes()); // k

    // Chunk data with length prefix
    #[expect(
        clippy::cast_possible_truncation,
        reason = "chunk data length always < 4GB"
    )]
    let data_len = chunk_data.len() as u32;
    hasher.update(&data_len.to_be_bytes()); // len(data)
    hasher.update(chunk_data); // data

    // Version counter (fixed width)
    hasher.update(&version.to_le_bytes()); // v_k

    hasher.finalize().into()
}

/// HDF5 byte shuffle filter for improving compression ratio.
///
/// Rearranges bytes by element type size. For example, with 4-byte floats,
/// groups all first bytes together, then all second bytes, etc.
///
/// This is a simplified implementation matching HDF5's shuffle filter.
#[inline]
#[must_use]
pub fn shuffle(data: &[u8], element_size: usize) -> Vec<u8> {
    if element_size <= 1 || data.is_empty() {
        return data.to_vec();
    }

    let n_elements = data.len() / element_size;
    let mut shuffled = vec![0u8; data.len()];

    for i in 0..n_elements {
        for j in 0..element_size {
            shuffled[j * n_elements + i] = data[i * element_size + j];
        }
    }

    // Copy any trailing bytes that don't fit in a complete element
    let remainder_start = n_elements * element_size;
    shuffled[remainder_start..].copy_from_slice(&data[remainder_start..]);

    shuffled
}

/// Reverse the HDF5 byte shuffle filter.
#[inline]
#[must_use]
pub fn unshuffle(data: &[u8], element_size: usize) -> Vec<u8> {
    if element_size <= 1 || data.is_empty() {
        return data.to_vec();
    }

    let n_elements = data.len() / element_size;
    let mut unshuffled = vec![0u8; data.len()];

    for i in 0..n_elements {
        for j in 0..element_size {
            unshuffled[i * element_size + j] = data[j * n_elements + i];
        }
    }

    // Copy any trailing bytes
    let remainder_start = n_elements * element_size;
    unshuffled[remainder_start..].copy_from_slice(&data[remainder_start..]);

    unshuffled
}

/// Configuration for the filter pipeline.
#[derive(Debug, Clone)]
pub struct FilterPipelineConfig {
    /// Element size for shuffle filter (1 = no shuffle).
    pub element_size: usize,
    /// Compression level (0 = no compression, 1-9 for deflate).
    pub compression_level: u32,
    /// Whether encryption is enabled.
    pub encrypt: bool,
}

impl Default for FilterPipelineConfig {
    fn default() -> Self {
        Self {
            element_size: 1,      // No shuffle by default
            compression_level: 6, // Default zlib level
            encrypt: true,        // Encryption enabled by default
        }
    }
}

impl FilterPipelineConfig {
    /// Create a config for encrypted data with shuffle and compression.
    #[must_use]
    pub fn encrypted(element_size: usize, compression_level: u32) -> Self {
        Self {
            element_size,
            compression_level,
            encrypt: true,
        }
    }

    /// Create a config for plaintext data (no encryption).
    #[must_use]
    pub fn plaintext(element_size: usize, compression_level: u32) -> Self {
        Self {
            element_size,
            compression_level,
            encrypt: false,
        }
    }

    /// Create a config with no filters (raw storage).
    #[must_use]
    pub fn raw() -> Self {
        Self {
            element_size: 1,
            compression_level: 0,
            encrypt: false,
        }
    }
}

/// The Encrypt-then-MAC filter pipeline.
///
/// Applies filters in the secure order: shuffle → compress → encrypt → hash
///
/// # Security
///
/// This pipeline enforces the EtM (Encrypt-then-MAC) construction:
/// - The Merkle leaf hash is computed AFTER encryption
/// - Third parties can verify integrity without decryption
/// - Tampered ciphertext is detected before decryption
pub struct FilterPipeline {
    config: FilterPipelineConfig,
}

impl FilterPipeline {
    /// Create a new filter pipeline with the given configuration.
    #[must_use]
    pub fn new(config: FilterPipelineConfig) -> Self {
        Self { config }
    }

    /// Create a pipeline with default encrypted configuration.
    #[must_use]
    pub fn encrypted() -> Self {
        Self::new(FilterPipelineConfig::default())
    }

    /// Create a pipeline for plaintext (no encryption).
    #[must_use]
    pub fn plaintext() -> Self {
        Self::new(FilterPipelineConfig {
            encrypt: false,
            ..Default::default()
        })
    }

    /// Apply the write pipeline: shuffle → compress → encrypt → hash
    ///
    /// # Arguments
    ///
    /// * `plaintext` - The application data to process
    /// * `dek` - The Data Encryption Key (required if encryption is enabled)
    /// * `chunk_idx` - The chunk index in the dataset
    /// * `version` - The per-chunk version counter
    ///
    /// # Returns
    ///
    /// The filtered chunk with ciphertext and leaf hash.
    pub fn write(
        &self,
        plaintext: &[u8],
        dek: Option<&Dek>,
        chunk_idx: u64,
        version: u64,
    ) -> Result<FilteredChunk, FilterPipelineError> {
        // Step 1: Shuffle (if element_size > 1)
        let shuffled = if self.config.element_size > 1 {
            shuffle(plaintext, self.config.element_size)
        } else {
            plaintext.to_vec()
        };

        // Step 2: Compress (if compression_level > 0)
        let compressed = if self.config.compression_level > 0 {
            crate::deflate_compress(&shuffled, self.config.compression_level)
                .map_err(FilterPipelineError::CompressionFailed)?
        } else {
            shuffled
        };

        // Step 3: Encrypt (if enabled)
        let (ciphertext, leaf_hash) = if self.config.encrypt {
            let dek = dek.ok_or(FilterPipelineError::EncryptionFailed(
                ChaCha20Error::InvalidKeyLength {
                    expected: 32,
                    actual: 0,
                },
            ))?;

            let ct = encrypt_chunk(dek, chunk_idx, version, &compressed)?;

            // Step 4: Hash the ciphertext (EtM - Encrypt-then-MAC)
            // H_leaf(k) = H(0x00 || len(k) || k || len(ct) || ct || tag || v_k)
            let hash = compute_leaf_hash(chunk_idx, &ct, version);

            (ct, hash)
        } else {
            // No encryption - hash the compressed data with chunk index binding
            // H_leaf(k) = H(0x00 || len(k) || k || len(data) || data || v_k)
            let hash = compute_leaf_hash_plaintext(chunk_idx, &compressed, version);
            (compressed, hash)
        };

        Ok(FilteredChunk {
            ciphertext,
            leaf_hash,
            version,
            chunk_idx,
        })
    }

    /// Apply the read pipeline: decrypt → decompress → unshuffle
    ///
    /// # Arguments
    ///
    /// * `ciphertext` - The stored ciphertext
    /// * `dek` - The Data Encryption Key (required if encryption is enabled)
    /// * `chunk_idx` - The chunk index
    /// * `version` - The per-chunk version counter
    /// * `decompressed_size` - Hint for decompressed size (0 for unknown)
    ///
    /// # Returns
    ///
    /// The original plaintext data.
    pub fn read(
        &self,
        ciphertext: &[u8],
        dek: Option<&Dek>,
        chunk_idx: u64,
        version: u64,
        decompressed_size: usize,
    ) -> Result<Vec<u8>, FilterPipelineError> {
        // Step 1: Decrypt (if enabled)
        let decrypted = if self.config.encrypt {
            let dek = dek.ok_or(FilterPipelineError::DecryptionFailed(
                ChaCha20Error::InvalidKeyLength {
                    expected: 32,
                    actual: 0,
                },
            ))?;

            crate::chacha20_filter::decrypt_chunk(dek, chunk_idx, version, ciphertext)
                .map_err(FilterPipelineError::DecryptionFailed)?
        } else {
            ciphertext.to_vec()
        };

        // Step 2: Decompress (if compression was used)
        let decompressed = if self.config.compression_level > 0 {
            crate::deflate_decompress(&decrypted, decompressed_size)
                .map_err(FilterPipelineError::DecompressionFailed)?
        } else {
            decrypted
        };

        // Step 3: Unshuffle (if shuffle was used)
        let unshuffled = if self.config.element_size > 1 {
            unshuffle(&decompressed, self.config.element_size)
        } else {
            decompressed
        };

        Ok(unshuffled)
    }

    /// Verify a stored chunk against its expected leaf hash.
    ///
    /// This verification can be performed WITHOUT decryption, which is the
    /// key security property of the EtM construction.
    ///
    /// # Arguments
    ///
    /// * `chunk_idx` - The chunk index (must match the index used during write)
    /// * `ciphertext` - The stored ciphertext/data
    /// * `version` - The per-chunk version counter
    /// * `expected_hash` - The expected leaf hash to verify against
    #[must_use]
    pub fn verify_leaf_hash(
        &self,
        chunk_idx: u64,
        ciphertext: &[u8],
        version: u64,
        expected_hash: &Hash,
    ) -> bool {
        let computed = if self.config.encrypt {
            compute_leaf_hash(chunk_idx, ciphertext, version)
        } else {
            compute_leaf_hash_plaintext(chunk_idx, ciphertext, version)
        };

        // Constant-time comparison
        computed
            .iter()
            .zip(expected_hash.iter())
            .fold(0u8, |acc, (a, b)| acc | (a ^ b))
            == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chacha20_filter::TAG_SIZE;

    fn test_dek() -> Dek {
        let mut dek = [0u8; 32];
        for (i, byte) in dek.iter_mut().enumerate() {
            *byte = i as u8;
        }
        dek
    }

    #[test]
    fn shuffle_unshuffle_roundtrip() {
        // 4-byte elements (like f32)
        let data: Vec<u8> = (0..16).collect();
        let shuffled = shuffle(&data, 4);
        let unshuffled = unshuffle(&shuffled, 4);
        assert_eq!(unshuffled, data);
    }

    #[test]
    fn shuffle_groups_bytes() {
        // 4 elements of 2 bytes each: [0,1], [2,3], [4,5], [6,7]
        let data: Vec<u8> = (0..8).collect();
        let shuffled = shuffle(&data, 2);
        // Should become: [0,2,4,6], [1,3,5,7]
        assert_eq!(shuffled, vec![0, 2, 4, 6, 1, 3, 5, 7]);
    }

    #[test]
    fn shuffle_element_size_1_is_identity() {
        let data: Vec<u8> = (0..16).collect();
        let shuffled = shuffle(&data, 1);
        assert_eq!(shuffled, data);
    }

    #[test]
    fn compute_leaf_hash_includes_version() {
        let ciphertext = b"some ciphertext data";
        let chunk_idx = 0;

        let hash1 = compute_leaf_hash(chunk_idx, ciphertext, 1);
        let hash2 = compute_leaf_hash(chunk_idx, ciphertext, 2);

        // Different versions should produce different hashes
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn compute_leaf_hash_includes_ciphertext() {
        let chunk_idx = 0;
        let version = 1;

        let hash1 = compute_leaf_hash(chunk_idx, b"ciphertext A", version);
        let hash2 = compute_leaf_hash(chunk_idx, b"ciphertext B", version);

        // Different ciphertexts should produce different hashes
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn compute_leaf_hash_includes_chunk_idx() {
        let ciphertext = b"same ciphertext data";
        let version = 1;

        let hash1 = compute_leaf_hash(0, ciphertext, version);
        let hash2 = compute_leaf_hash(1, ciphertext, version);

        // Different chunk indices should produce different hashes
        // This prevents position-swapping attacks
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn pipeline_write_read_roundtrip() {
        let dek = test_dek();
        let plaintext = b"Hello, encrypted HDF5 world!";

        let pipeline = FilterPipeline::encrypted();
        let filtered = pipeline.write(plaintext, Some(&dek), 0, 1).unwrap();

        // Ciphertext should be larger due to compression overhead + tag
        assert!(filtered.ciphertext.len() >= TAG_SIZE);

        // Read back
        let recovered = pipeline
            .read(&filtered.ciphertext, Some(&dek), 0, 1, plaintext.len())
            .unwrap();
        assert_eq!(recovered, plaintext);
    }

    #[test]
    fn pipeline_with_shuffle_and_compression() {
        let dek = test_dek();
        // Simulated 4-byte float array (highly compressible when shuffled)
        let plaintext: Vec<u8> = (0..64).flat_map(|i| [i as u8, 0, 0, 0]).collect();

        let config = FilterPipelineConfig::encrypted(4, 6);
        let pipeline = FilterPipeline::new(config);

        let filtered = pipeline.write(&plaintext, Some(&dek), 0, 1).unwrap();

        // Read back
        let recovered = pipeline
            .read(&filtered.ciphertext, Some(&dek), 0, 1, plaintext.len())
            .unwrap();
        assert_eq!(recovered, plaintext);
    }

    #[test]
    fn pipeline_verify_hash_without_decryption() {
        let dek = test_dek();
        let plaintext = b"Verifiable chunk data";

        let pipeline = FilterPipeline::encrypted();
        let filtered = pipeline.write(plaintext, Some(&dek), 0, 1).unwrap();

        // Verify hash (this doesn't require decryption!)
        assert!(pipeline.verify_leaf_hash(
            filtered.chunk_idx,
            &filtered.ciphertext,
            filtered.version,
            &filtered.leaf_hash
        ));

        // Tampered ciphertext should fail
        let mut tampered = filtered.ciphertext.clone();
        tampered[0] ^= 0xFF;
        assert!(!pipeline.verify_leaf_hash(
            filtered.chunk_idx,
            &tampered,
            filtered.version,
            &filtered.leaf_hash
        ));
    }

    #[test]
    fn pipeline_wrong_version_changes_hash() {
        let dek = test_dek();
        let plaintext = b"Version-bound chunk";

        let pipeline = FilterPipeline::encrypted();
        let filtered = pipeline.write(plaintext, Some(&dek), 0, 1).unwrap();

        // Wrong version should fail verification
        assert!(!pipeline.verify_leaf_hash(
            filtered.chunk_idx,
            &filtered.ciphertext,
            2, // wrong version
            &filtered.leaf_hash
        ));
    }

    #[test]
    fn pipeline_wrong_chunk_idx_changes_hash() {
        let dek = test_dek();
        let plaintext = b"Position-bound chunk";

        let pipeline = FilterPipeline::encrypted();
        let filtered = pipeline.write(plaintext, Some(&dek), 42, 1).unwrap();

        // Wrong chunk index should fail verification (position-swapping attack prevention)
        assert!(!pipeline.verify_leaf_hash(
            99, // wrong chunk index
            &filtered.ciphertext,
            filtered.version,
            &filtered.leaf_hash
        ));
    }

    #[test]
    fn pipeline_plaintext_mode() {
        let plaintext = b"Unencrypted but hashed";

        let pipeline = FilterPipeline::plaintext();
        let filtered = pipeline.write(plaintext, None, 0, 1).unwrap();

        // Read back (no decryption key needed)
        let recovered = pipeline
            .read(&filtered.ciphertext, None, 0, 1, plaintext.len())
            .unwrap();
        assert_eq!(recovered, plaintext);

        // Verify hash
        assert!(pipeline.verify_leaf_hash(
            filtered.chunk_idx,
            &filtered.ciphertext,
            filtered.version,
            &filtered.leaf_hash
        ));
    }

    #[test]
    fn pipeline_raw_mode() {
        let plaintext = b"Raw storage, no filters";

        let pipeline = FilterPipeline::new(FilterPipelineConfig::raw());
        let filtered = pipeline.write(plaintext, None, 0, 1).unwrap();

        // Ciphertext should be identical to plaintext (no compression)
        // But we're not encrypting either, so leaf hash is over plaintext
        let recovered = pipeline.read(&filtered.ciphertext, None, 0, 1, 0).unwrap();
        assert_eq!(recovered, plaintext);
    }

    #[test]
    fn etm_security_third_party_verification() {
        // Demonstrate that a third party (without DEK) can verify integrity
        let dek = test_dek();
        let plaintext = b"Data that a storage node can verify";

        let pipeline = FilterPipeline::encrypted();
        let filtered = pipeline.write(plaintext, Some(&dek), 42, 7).unwrap();

        // Storage node receives: chunk_idx, ciphertext, version, expected_hash
        // Storage node does NOT have DEK

        // Verification works without DEK:
        let computed_hash = compute_leaf_hash(42, &filtered.ciphertext, 7);
        assert_eq!(computed_hash, filtered.leaf_hash);

        // Tampering is detected without DEK:
        let mut tampered = filtered.ciphertext.clone();
        tampered[5] ^= 0x01;
        let tampered_hash = compute_leaf_hash(42, &tampered, 7);
        assert_ne!(tampered_hash, filtered.leaf_hash);
    }

    #[test]
    fn leaf_hash_uses_domain_separator() {
        let data = b"test data";

        // Our leaf hash should differ from raw BLAKE3 hash
        // due to domain separator, chunk index, and version binding
        let leaf_hash = compute_leaf_hash_plaintext(0, data, 1);
        let raw_hash: [u8; HASH_SIZE] = blake3::hash(data).into();

        assert_ne!(
            leaf_hash, raw_hash,
            "leaf hash must include domain separator"
        );
    }

    #[test]
    fn plaintext_hash_includes_chunk_idx() {
        let data = b"same plaintext data";
        let version = 1;

        let hash1 = compute_leaf_hash_plaintext(0, data, version);
        let hash2 = compute_leaf_hash_plaintext(1, data, version);

        // Different chunk indices should produce different hashes
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn plaintext_hash_includes_version() {
        let data = b"same plaintext data";
        let chunk_idx = 0;

        let hash1 = compute_leaf_hash_plaintext(chunk_idx, data, 1);
        let hash2 = compute_leaf_hash_plaintext(chunk_idx, data, 2);

        // Different versions should produce different hashes
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn position_swapping_attack_prevented() {
        // Simulate an attacker trying to swap chunks at different positions
        let dek = test_dek();

        let pipeline = FilterPipeline::encrypted();

        // Create two different chunks at positions 0 and 1
        let chunk_0 = pipeline
            .write(b"chunk zero data", Some(&dek), 0, 1)
            .unwrap();
        let chunk_1 = pipeline.write(b"chunk one data", Some(&dek), 1, 1).unwrap();

        // Verify both chunks pass at their correct positions
        assert!(pipeline.verify_leaf_hash(0, &chunk_0.ciphertext, 1, &chunk_0.leaf_hash));
        assert!(pipeline.verify_leaf_hash(1, &chunk_1.ciphertext, 1, &chunk_1.leaf_hash));

        // Attacker tries to put chunk_1's ciphertext at position 0 (position-swapping attack)
        // This should FAIL because the chunk index is bound into the hash
        assert!(
            !pipeline.verify_leaf_hash(0, &chunk_1.ciphertext, 1, &chunk_1.leaf_hash),
            "position-swapping attack should be detected"
        );

        // Similarly, putting chunk_0's ciphertext at position 1 should fail
        assert!(
            !pipeline.verify_leaf_hash(1, &chunk_0.ciphertext, 1, &chunk_0.leaf_hash),
            "position-swapping attack should be detected"
        );
    }
}
