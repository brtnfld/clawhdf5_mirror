//! ChaCha20-Poly1305 AEAD filter for HDF5 chunk encryption.
//!
//! This module provides authenticated encryption for HDF5 chunks using
//! ChaCha20-Poly1305 (RFC 8439). It is part of the P2.2 implementation
//! for the ClawHDF5 Merkle-tree integrity system.
//!
//! # Security Properties
//!
//! - **Confidentiality**: ChaCha20 stream cipher
//! - **Integrity**: Poly1305 MAC (16-byte authentication tag)
//! - **AEAD**: Authenticated Encryption with Associated Data
//!
//! # Nonce Derivation
//!
//! To safely support in-place chunk updates, nonces are derived using BLAKE3:
//!
//! ```text
//! nonce = BLAKE3-derive(DEK, chunk_idx || version_counter)
//! ```
//!
//! This guarantees a unique nonce per (key, chunk, version) triple, preventing
//! keystream reuse even when chunks are updated in place.
//!
//! # Spec Reference
//!
//! S2-D2-Yr2 P2.2 step 2: "Add a `chacha20_filter` module to `clawhdf5-filters`
//! with Cargo dependency `chacha20poly1305`."
//!
//! See also §7.3 (Nonce Derivation for Stream Ciphers) in the spec.

use chacha20poly1305::{
    ChaCha20Poly1305, Nonce,
    aead::{Aead, KeyInit},
};

use crate::mock_keystore::Dek;

/// Size of the ChaCha20-Poly1305 nonce in bytes (96 bits).
pub const NONCE_SIZE: usize = 12;

/// Size of the Poly1305 authentication tag in bytes (128 bits).
pub const TAG_SIZE: usize = 16;

/// Size of the ChaCha20-Poly1305 key in bytes (256 bits).
pub const KEY_SIZE: usize = 32;

/// A 96-bit nonce for ChaCha20-Poly1305.
pub type ChaCha20Nonce = [u8; NONCE_SIZE];

/// Error type for ChaCha20 filter operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChaCha20Error {
    /// Encryption failed.
    EncryptionFailed,
    /// Decryption failed (authentication tag mismatch or corrupted ciphertext).
    DecryptionFailed,
    /// Invalid key length.
    InvalidKeyLength { expected: usize, actual: usize },
    /// Invalid nonce length.
    InvalidNonceLength { expected: usize, actual: usize },
}

impl core::fmt::Display for ChaCha20Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ChaCha20Error::EncryptionFailed => {
                write!(f, "ChaCha20-Poly1305 encryption failed")
            }
            ChaCha20Error::DecryptionFailed => {
                write!(
                    f,
                    "ChaCha20-Poly1305 decryption failed: authentication tag mismatch"
                )
            }
            ChaCha20Error::InvalidKeyLength { expected, actual } => {
                write!(
                    f,
                    "invalid key length: expected {expected} bytes, got {actual}"
                )
            }
            ChaCha20Error::InvalidNonceLength { expected, actual } => {
                write!(
                    f,
                    "invalid nonce length: expected {expected} bytes, got {actual}"
                )
            }
        }
    }
}

impl std::error::Error for ChaCha20Error {}

/// Derive a nonce from the DEK, chunk index, and version counter using BLAKE3.
///
/// This implements the version-counter nonce derivation from S2-D2-Yr2 §7.3:
///
/// ```text
/// nonce = BLAKE3-derive(DEK, chunk_idx || version_counter)
/// ```
///
/// The derived nonce is 12 bytes (96 bits) as required by ChaCha20-Poly1305.
///
/// # Arguments
///
/// * `dek` - The 32-byte Data Encryption Key
/// * `chunk_idx` - The chunk index (u64)
/// * `version` - The per-chunk version counter (u64)
///
/// # Returns
///
/// A 12-byte nonce suitable for ChaCha20-Poly1305.
#[inline]
#[must_use]
pub fn derive_nonce(dek: &Dek, chunk_idx: u64, version: u64) -> ChaCha20Nonce {
    // Create context: chunk_idx || version (16 bytes total)
    let mut context = [0u8; 16];
    context[..8].copy_from_slice(&chunk_idx.to_le_bytes());
    context[8..].copy_from_slice(&version.to_le_bytes());

    // Use BLAKE3's key derivation mode
    // Context string identifies this as nonce derivation for ClawHDF5
    let mut hasher = blake3::Hasher::new_derive_key("clawhdf5 chacha20 nonce v1");
    hasher.update(dek);
    hasher.update(&context);

    // Take first 12 bytes of the output as the nonce
    let hash = hasher.finalize();
    let mut nonce = [0u8; NONCE_SIZE];
    nonce.copy_from_slice(&hash.as_bytes()[..NONCE_SIZE]);
    nonce
}

/// Encrypt a plaintext chunk using ChaCha20-Poly1305.
///
/// # Arguments
///
/// * `dek` - The 32-byte Data Encryption Key
/// * `nonce` - The 12-byte nonce (use `derive_nonce` for safe nonce generation)
/// * `plaintext` - The plaintext data to encrypt
/// * `aad` - Additional Authenticated Data (optional, can be empty)
///
/// # Returns
///
/// The ciphertext with the 16-byte Poly1305 authentication tag appended.
/// Output length = plaintext.len() + TAG_SIZE (16 bytes).
///
/// # Errors
///
/// Returns `ChaCha20Error::EncryptionFailed` if encryption fails.
pub fn encrypt(
    dek: &Dek,
    nonce: &ChaCha20Nonce,
    plaintext: &[u8],
    aad: &[u8],
) -> Result<Vec<u8>, ChaCha20Error> {
    let cipher =
        ChaCha20Poly1305::new_from_slice(dek).map_err(|_| ChaCha20Error::InvalidKeyLength {
            expected: KEY_SIZE,
            actual: dek.len(),
        })?;

    let nonce = Nonce::from_slice(nonce);

    if aad.is_empty() {
        cipher
            .encrypt(nonce, plaintext)
            .map_err(|_| ChaCha20Error::EncryptionFailed)
    } else {
        use chacha20poly1305::aead::Payload;
        cipher
            .encrypt(
                nonce,
                Payload {
                    msg: plaintext,
                    aad,
                },
            )
            .map_err(|_| ChaCha20Error::EncryptionFailed)
    }
}

/// Decrypt a ciphertext chunk using ChaCha20-Poly1305.
///
/// # Arguments
///
/// * `dek` - The 32-byte Data Encryption Key
/// * `nonce` - The 12-byte nonce (must match the nonce used for encryption)
/// * `ciphertext` - The ciphertext with appended 16-byte authentication tag
/// * `aad` - Additional Authenticated Data (must match what was used for encryption)
///
/// # Returns
///
/// The decrypted plaintext.
///
/// # Errors
///
/// Returns `ChaCha20Error::DecryptionFailed` if:
/// - The authentication tag does not match (tampered ciphertext)
/// - The ciphertext is corrupted
/// - The wrong key or nonce was used
pub fn decrypt(
    dek: &Dek,
    nonce: &ChaCha20Nonce,
    ciphertext: &[u8],
    aad: &[u8],
) -> Result<Vec<u8>, ChaCha20Error> {
    let cipher =
        ChaCha20Poly1305::new_from_slice(dek).map_err(|_| ChaCha20Error::InvalidKeyLength {
            expected: KEY_SIZE,
            actual: dek.len(),
        })?;

    let nonce = Nonce::from_slice(nonce);

    if aad.is_empty() {
        cipher
            .decrypt(nonce, ciphertext)
            .map_err(|_| ChaCha20Error::DecryptionFailed)
    } else {
        use chacha20poly1305::aead::Payload;
        cipher
            .decrypt(
                nonce,
                Payload {
                    msg: ciphertext,
                    aad,
                },
            )
            .map_err(|_| ChaCha20Error::DecryptionFailed)
    }
}

/// Encrypt a chunk with automatic nonce derivation.
///
/// This is the primary API for chunk encryption in the HDF5 filter pipeline.
/// It automatically derives a unique nonce from the chunk index and version counter.
///
/// # Arguments
///
/// * `dek` - The 32-byte Data Encryption Key
/// * `chunk_idx` - The chunk index in the dataset
/// * `version` - The per-chunk version counter (must be incremented on each write)
/// * `plaintext` - The plaintext chunk data
///
/// # Returns
///
/// The ciphertext with appended authentication tag.
pub fn encrypt_chunk(
    dek: &Dek,
    chunk_idx: u64,
    version: u64,
    plaintext: &[u8],
) -> Result<Vec<u8>, ChaCha20Error> {
    let nonce = derive_nonce(dek, chunk_idx, version);
    encrypt(dek, &nonce, plaintext, &[])
}

/// Decrypt a chunk with automatic nonce derivation.
///
/// This is the primary API for chunk decryption in the HDF5 filter pipeline.
///
/// # Arguments
///
/// * `dek` - The 32-byte Data Encryption Key
/// * `chunk_idx` - The chunk index in the dataset
/// * `version` - The per-chunk version counter (must match what was used for encryption)
/// * `ciphertext` - The ciphertext with appended authentication tag
///
/// # Returns
///
/// The decrypted plaintext chunk.
pub fn decrypt_chunk(
    dek: &Dek,
    chunk_idx: u64,
    version: u64,
    ciphertext: &[u8],
) -> Result<Vec<u8>, ChaCha20Error> {
    let nonce = derive_nonce(dek, chunk_idx, version);
    decrypt(dek, &nonce, ciphertext, &[])
}

// ---- EncryptedChunkWriter: High-level API integrating WAL + version counters ----

use crate::version_wal::{VersionCounterStore, VersionWal, WalError};
use std::io::{Read, Seek, Write};

/// Error type for encrypted chunk write operations.
#[derive(Debug)]
pub enum EncryptedWriteError {
    /// WAL operation failed.
    Wal(WalError),
    /// Encryption failed.
    Encryption(ChaCha20Error),
}

impl std::fmt::Display for EncryptedWriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncryptedWriteError::Wal(e) => write!(f, "WAL error: {e}"),
            EncryptedWriteError::Encryption(e) => write!(f, "encryption error: {e}"),
        }
    }
}

impl std::error::Error for EncryptedWriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            EncryptedWriteError::Wal(e) => Some(e),
            EncryptedWriteError::Encryption(e) => Some(e),
        }
    }
}

impl From<WalError> for EncryptedWriteError {
    fn from(e: WalError) -> Self {
        EncryptedWriteError::Wal(e)
    }
}

impl From<ChaCha20Error> for EncryptedWriteError {
    fn from(e: ChaCha20Error) -> Self {
        EncryptedWriteError::Encryption(e)
    }
}

/// Result of an encrypted chunk write operation.
#[derive(Debug, Clone)]
pub struct EncryptedChunkResult {
    /// The encrypted ciphertext (with authentication tag appended).
    pub ciphertext: Vec<u8>,
    /// The version counter used for this write.
    pub version: u64,
    /// The chunk index.
    pub chunk_idx: u64,
}

/// High-level API for encrypted chunk writes with crash-safe version management.
///
/// This struct implements the WAL protocol from S2-D2-Yr2 §7.3:
///
/// 1. Journal `(chunk_idx, v_new)` to the WAL before deriving the nonce
/// 2. Derive the nonce using `v_new`, encrypt the chunk
/// 3. (Caller writes chunk and updates companion dataset)
/// 4. Call `commit()` to mark the journal record as committed
///
/// # Nonce Reuse Prevention
///
/// Nonce reuse requires `v_chunk` rollback, which is prevented by:
/// - The WAL protocol ensuring `v_new` is durably recorded before encryption
/// - Monotonic version counters that only increment
/// - Recovery replaying uncommitted journal records with their journaled versions
///
/// # Example
///
/// ```ignore
/// let mut writer = EncryptedChunkWriter::new(dek, wal, version_store);
///
/// // Encrypt chunk 7
/// let result = writer.encrypt_chunk(7, plaintext)?;
///
/// // ... write ciphertext to storage ...
/// // ... update companion dataset with new version ...
///
/// // Mark as committed (pass the version from the result)
/// writer.commit(7, result.version)?;
/// ```
pub struct EncryptedChunkWriter<W> {
    /// The Data Encryption Key.
    dek: Dek,
    /// Write-Ahead Log for crash-safe version management.
    wal: VersionWal<W>,
    /// In-memory version counter store.
    versions: VersionCounterStore,
}

impl<W: Read + Write + Seek> EncryptedChunkWriter<W> {
    /// Create a new encrypted chunk writer.
    pub fn new(dek: Dek, wal: VersionWal<W>, versions: VersionCounterStore) -> Self {
        Self { dek, wal, versions }
    }

    /// Encrypt a chunk with automatic version management.
    ///
    /// This method:
    /// 1. Computes the next version counter for the chunk
    /// 2. Journals the new version to the WAL (crash-safe)
    /// 3. **Reserves** that version in the in-memory store so it can never be
    ///    handed out again — even if `commit` is not called before the next
    ///    `encrypt_chunk` on the same chunk
    /// 4. Derives the nonce and encrypts the plaintext
    ///
    /// After calling this, you should:
    /// 1. Write the ciphertext to storage
    /// 2. Update the companion dataset with the new version
    /// 3. Call `commit()` to finalize
    ///
    /// # Nonce Safety
    ///
    /// The in-memory reservation in step 3 is what makes nonce reuse impossible
    /// *within a session*: without it, two `encrypt_chunk` calls on the same chunk
    /// before a `commit` would both read `next_version() == current + 1`, re-derive
    /// the identical `(DEK, chunk_idx, version)` nonce, and reuse the keystream on
    /// two different plaintexts. Reserving eagerly means an uncommitted (e.g. failed
    /// or retried) write only *burns* a version number rather than risking reuse.
    pub fn encrypt_chunk(
        &mut self,
        chunk_idx: u64,
        plaintext: &[u8],
    ) -> Result<EncryptedChunkResult, EncryptedWriteError> {
        // Step 1: Compute next version
        let version = self.versions.next_version(chunk_idx);

        // Step 2: Journal to WAL BEFORE deriving nonce (crash safety)
        self.wal.journal_version(chunk_idx, version)?;

        // Step 3: Reserve the version in-memory so the nonce can't be reused by a
        // subsequent encrypt of the same chunk before commit. `seed` is monotonic,
        // so this only ever advances the counter.
        self.versions.seed(chunk_idx, version);

        // Step 4: Derive nonce and encrypt
        let ciphertext = encrypt_chunk(&self.dek, chunk_idx, version, plaintext)?;

        Ok(EncryptedChunkResult {
            ciphertext,
            version,
            chunk_idx,
        })
    }

    /// Mark a chunk write as committed.
    ///
    /// This should be called after the ciphertext is written to storage
    /// and the companion dataset is updated with the new version.
    ///
    /// Uses the monotonic [`seed`](VersionCounterStore::seed) primitive rather than a
    /// strict update, so it is idempotent when `version` equals the value already seeded
    /// by [`recover`](Self::recover) (i.e. committing a replayed crash-recovery write).
    ///
    /// # Arguments
    ///
    /// * `chunk_idx` - The chunk index
    /// * `version` - The version from the `EncryptedChunkResult` returned by `encrypt_chunk`
    pub fn commit(&mut self, chunk_idx: u64, version: u64) -> Result<(), EncryptedWriteError> {
        // Update in-memory version store with the version used for encryption.
        // Monotonic: never regresses below a version already consumed/seeded.
        self.versions.seed(chunk_idx, version);

        // Mark WAL record as committed
        self.wal.commit_version(chunk_idx)?;

        Ok(())
    }

    /// Get the current version counter for a chunk.
    #[must_use]
    pub fn get_version(&self, chunk_idx: u64) -> u64 {
        self.versions.get(chunk_idx)
    }

    /// Get the maximum version across all chunks (dataset-level version).
    #[must_use]
    pub fn max_version(&self) -> u64 {
        self.versions.max_version()
    }

    /// Recover from a crash by replaying uncommitted writes.
    ///
    /// **Must be called after reopening an existing WAL, before any new write.**
    /// It seeds the in-memory version store with every journaled-but-uncommitted
    /// version, so that:
    ///
    /// - a subsequent [`encrypt_chunk`](Self::encrypt_chunk) of *new* data to a recovered
    ///   chunk uses a strictly higher version (a fresh nonce), and
    /// - an idempotent replay of the recovered write (re-encrypting the *same* plaintext
    ///   at the journaled version) reproduces the identical ciphertext.
    ///
    /// Either way the `(DEK, chunk_idx, version)` nonce can never be reused — closing the
    /// crash-window nonce-reuse hole that would otherwise exist if the store started at 0
    /// while the WAL still held a pending version.
    ///
    /// Returns the list of `(chunk_idx, version)` pairs that were pending. For each pair
    /// the caller should promote the companion dataset to `version` and call
    /// [`commit`](Self::commit) to finalize the record.
    pub fn recover(&mut self) -> Result<Vec<(u64, u64)>, EncryptedWriteError> {
        let uncommitted = self.wal.recover()?;

        // Seed the version store so journaled versions are treated as already consumed.
        // Without this, next_version() could hand back a version that was already used
        // to derive a nonce before the crash -> catastrophic keystream reuse.
        for &(chunk_idx, version) in &uncommitted {
            self.versions.seed(chunk_idx, version);
        }

        Ok(uncommitted)
    }

    /// Get a reference to the version counter store.
    #[must_use]
    pub fn versions(&self) -> &VersionCounterStore {
        &self.versions
    }

    /// Get a mutable reference to the version counter store.
    pub fn versions_mut(&mut self) -> &mut VersionCounterStore {
        &mut self.versions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dek() -> Dek {
        // Use a fixed test key
        let mut dek = [0u8; 32];
        for (i, byte) in dek.iter_mut().enumerate() {
            *byte = i as u8;
        }
        dek
    }

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let dek = test_dek();
        let nonce = [0u8; NONCE_SIZE];
        let plaintext = b"Hello, ChaCha20-Poly1305!";

        let ciphertext = encrypt(&dek, &nonce, plaintext, &[]).unwrap();
        assert_eq!(ciphertext.len(), plaintext.len() + TAG_SIZE);

        let decrypted = decrypt(&dek, &nonce, &ciphertext, &[]).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn encrypt_decrypt_with_aad() {
        let dek = test_dek();
        let nonce = [1u8; NONCE_SIZE];
        let plaintext = b"Secret data";
        let aad = b"chunk metadata";

        let ciphertext = encrypt(&dek, &nonce, plaintext, aad).unwrap();
        let decrypted = decrypt(&dek, &nonce, &ciphertext, aad).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn wrong_aad_fails() {
        let dek = test_dek();
        let nonce = [2u8; NONCE_SIZE];
        let plaintext = b"Secret data";

        let ciphertext = encrypt(&dek, &nonce, plaintext, b"correct aad").unwrap();
        let result = decrypt(&dek, &nonce, &ciphertext, b"wrong aad");
        assert!(matches!(result, Err(ChaCha20Error::DecryptionFailed)));
    }

    #[test]
    fn tampered_ciphertext_fails() {
        let dek = test_dek();
        let nonce = [3u8; NONCE_SIZE];
        let plaintext = b"Secret data";

        let mut ciphertext = encrypt(&dek, &nonce, plaintext, &[]).unwrap();
        // Tamper with the ciphertext
        ciphertext[0] ^= 0xFF;

        let result = decrypt(&dek, &nonce, &ciphertext, &[]);
        assert!(matches!(result, Err(ChaCha20Error::DecryptionFailed)));
    }

    #[test]
    fn wrong_key_fails() {
        let dek1 = test_dek();
        let mut dek2 = test_dek();
        dek2[0] ^= 0xFF;

        let nonce = [4u8; NONCE_SIZE];
        let plaintext = b"Secret data";

        let ciphertext = encrypt(&dek1, &nonce, plaintext, &[]).unwrap();
        let result = decrypt(&dek2, &nonce, &ciphertext, &[]);
        assert!(matches!(result, Err(ChaCha20Error::DecryptionFailed)));
    }

    #[test]
    fn wrong_nonce_fails() {
        let dek = test_dek();
        let nonce1 = [5u8; NONCE_SIZE];
        let nonce2 = [6u8; NONCE_SIZE];
        let plaintext = b"Secret data";

        let ciphertext = encrypt(&dek, &nonce1, plaintext, &[]).unwrap();
        let result = decrypt(&dek, &nonce2, &ciphertext, &[]);
        assert!(matches!(result, Err(ChaCha20Error::DecryptionFailed)));
    }

    #[test]
    fn derive_nonce_deterministic() {
        let dek = test_dek();
        let nonce1 = derive_nonce(&dek, 0, 0);
        let nonce2 = derive_nonce(&dek, 0, 0);
        assert_eq!(nonce1, nonce2);
    }

    #[test]
    fn derive_nonce_different_chunk_idx() {
        let dek = test_dek();
        let nonce1 = derive_nonce(&dek, 0, 0);
        let nonce2 = derive_nonce(&dek, 1, 0);
        assert_ne!(nonce1, nonce2);
    }

    #[test]
    fn derive_nonce_different_version() {
        let dek = test_dek();
        let nonce1 = derive_nonce(&dek, 0, 0);
        let nonce2 = derive_nonce(&dek, 0, 1);
        assert_ne!(nonce1, nonce2);
    }

    #[test]
    fn derive_nonce_different_key() {
        let dek1 = test_dek();
        let mut dek2 = test_dek();
        dek2[0] ^= 0xFF;

        let nonce1 = derive_nonce(&dek1, 0, 0);
        let nonce2 = derive_nonce(&dek2, 0, 0);
        assert_ne!(nonce1, nonce2);
    }

    #[test]
    fn chunk_encrypt_decrypt_roundtrip() {
        let dek = test_dek();
        let chunk_idx = 42;
        let version = 1;
        let plaintext = b"HDF5 chunk data goes here";

        let ciphertext = encrypt_chunk(&dek, chunk_idx, version, plaintext).unwrap();
        let decrypted = decrypt_chunk(&dek, chunk_idx, version, &ciphertext).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn chunk_wrong_version_fails() {
        let dek = test_dek();
        let chunk_idx = 42;
        let plaintext = b"HDF5 chunk data";

        let ciphertext = encrypt_chunk(&dek, chunk_idx, 1, plaintext).unwrap();
        let result = decrypt_chunk(&dek, chunk_idx, 2, &ciphertext);
        assert!(matches!(result, Err(ChaCha20Error::DecryptionFailed)));
    }

    #[test]
    fn chunk_wrong_index_fails() {
        let dek = test_dek();
        let version = 1;
        let plaintext = b"HDF5 chunk data";

        let ciphertext = encrypt_chunk(&dek, 0, version, plaintext).unwrap();
        let result = decrypt_chunk(&dek, 1, version, &ciphertext);
        assert!(matches!(result, Err(ChaCha20Error::DecryptionFailed)));
    }

    #[test]
    fn empty_plaintext() {
        let dek = test_dek();
        let nonce = [7u8; NONCE_SIZE];

        let ciphertext = encrypt(&dek, &nonce, &[], &[]).unwrap();
        assert_eq!(ciphertext.len(), TAG_SIZE); // Only the tag

        let decrypted = decrypt(&dek, &nonce, &ciphertext, &[]).unwrap();
        assert!(decrypted.is_empty());
    }

    #[test]
    fn large_plaintext() {
        let dek = test_dek();
        let nonce = [8u8; NONCE_SIZE];
        let plaintext: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();

        let ciphertext = encrypt(&dek, &nonce, &plaintext, &[]).unwrap();
        assert_eq!(ciphertext.len(), plaintext.len() + TAG_SIZE);

        let decrypted = decrypt(&dek, &nonce, &ciphertext, &[]).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn error_display() {
        let err = ChaCha20Error::EncryptionFailed;
        assert_eq!(err.to_string(), "ChaCha20-Poly1305 encryption failed");

        let err = ChaCha20Error::DecryptionFailed;
        assert!(err.to_string().contains("authentication tag mismatch"));

        let err = ChaCha20Error::InvalidKeyLength {
            expected: 32,
            actual: 16,
        };
        assert!(err.to_string().contains("32") && err.to_string().contains("16"));
    }

    // ---- EncryptedChunkWriter tests ----

    use std::io::Cursor;

    #[test]
    fn encrypted_chunk_writer_basic() {
        let dek = test_dek();
        let wal_buf = Cursor::new(Vec::new());
        let wal = VersionWal::new(wal_buf, 100).unwrap();
        let versions = VersionCounterStore::new();

        let mut writer = EncryptedChunkWriter::new(dek, wal, versions);

        // Encrypt chunk 0
        let plaintext = b"Hello, encrypted world!";
        let result = writer.encrypt_chunk(0, plaintext).unwrap();

        assert_eq!(result.chunk_idx, 0);
        assert_eq!(result.version, 1); // First write = version 1
        assert_eq!(result.ciphertext.len(), plaintext.len() + TAG_SIZE);

        // Verify we can decrypt with the same version
        let decrypted = decrypt_chunk(&dek, 0, 1, &result.ciphertext).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn encrypted_chunk_writer_commit_updates_version() {
        let dek = test_dek();
        let wal_buf = Cursor::new(Vec::new());
        let wal = VersionWal::new(wal_buf, 100).unwrap();
        let versions = VersionCounterStore::new();

        let mut writer = EncryptedChunkWriter::new(dek, wal, versions);

        // Initial version is 0
        assert_eq!(writer.get_version(0), 0);

        // Encrypt and commit
        let result = writer.encrypt_chunk(0, b"data").unwrap();
        assert_eq!(result.version, 1);

        writer.commit(0, result.version).unwrap();
        assert_eq!(writer.get_version(0), 1);

        // Second write should use version 2
        let result2 = writer.encrypt_chunk(0, b"updated data").unwrap();
        assert_eq!(result2.version, 2);

        writer.commit(0, result2.version).unwrap();
        assert_eq!(writer.get_version(0), 2);
    }

    #[test]
    fn encrypted_chunk_writer_multiple_chunks() {
        let dek = test_dek();
        let wal_buf = Cursor::new(Vec::new());
        let wal = VersionWal::new(wal_buf, 100).unwrap();
        let versions = VersionCounterStore::new();

        let mut writer = EncryptedChunkWriter::new(dek, wal, versions);

        // Write to multiple chunks
        for chunk_idx in 0..10 {
            let plaintext = format!("chunk {chunk_idx} data");
            let result = writer
                .encrypt_chunk(chunk_idx, plaintext.as_bytes())
                .unwrap();
            assert_eq!(result.chunk_idx, chunk_idx);
            assert_eq!(result.version, 1);
            writer.commit(chunk_idx, result.version).unwrap();
        }

        // All chunks should be at version 1
        for chunk_idx in 0..10 {
            assert_eq!(writer.get_version(chunk_idx), 1);
        }

        // Max version should be 1
        assert_eq!(writer.max_version(), 1);
    }

    #[test]
    fn encrypted_chunk_writer_in_place_update() {
        let dek = test_dek();
        let wal_buf = Cursor::new(Vec::new());
        let wal = VersionWal::new(wal_buf, 100).unwrap();
        let versions = VersionCounterStore::new();

        let mut writer = EncryptedChunkWriter::new(dek, wal, versions);

        // Initial write to chunk 7
        let result1 = writer.encrypt_chunk(7, b"original data").unwrap();
        assert_eq!(result1.version, 1);
        writer.commit(7, result1.version).unwrap();

        // In-place update to chunk 7
        let result2 = writer.encrypt_chunk(7, b"updated data").unwrap();
        assert_eq!(result2.version, 2);
        writer.commit(7, result2.version).unwrap();

        // Verify the ciphertexts are different (different nonces)
        assert_ne!(result1.ciphertext, result2.ciphertext);

        // Verify both can be decrypted with correct versions
        let dec1 = decrypt_chunk(&dek, 7, 1, &result1.ciphertext).unwrap();
        let dec2 = decrypt_chunk(&dek, 7, 2, &result2.ciphertext).unwrap();
        assert_eq!(dec1, b"original data");
        assert_eq!(dec2, b"updated data");

        // Verify wrong version fails
        assert!(decrypt_chunk(&dek, 7, 1, &result2.ciphertext).is_err());
        assert!(decrypt_chunk(&dek, 7, 2, &result1.ciphertext).is_err());
    }

    #[test]
    fn encrypted_chunk_writer_recover_uncommitted() {
        let dek = test_dek();
        let mut wal_buf = Cursor::new(Vec::new());

        // Simulate a crash: encrypt but don't commit
        {
            let wal = VersionWal::new(&mut wal_buf, 100).unwrap();
            let versions = VersionCounterStore::new();
            let mut writer = EncryptedChunkWriter::new(dek, wal, versions);

            // Encrypt chunk 5 but "crash" before commit
            let _result = writer.encrypt_chunk(5, b"data").unwrap();
            // No commit - simulating crash
        }

        // Recovery: reopen and check for uncommitted
        wal_buf.set_position(0);
        let wal = VersionWal::new(&mut wal_buf, 100).unwrap();
        let versions = VersionCounterStore::new();
        let mut writer = EncryptedChunkWriter::new(dek, wal, versions);

        let uncommitted = writer.recover().unwrap();
        assert_eq!(uncommitted.len(), 1);
        assert_eq!(uncommitted[0], (5, 1)); // chunk 5, version 1
    }

    #[test]
    fn encrypt_chunk_twice_without_commit_uses_distinct_nonces() {
        // Regression: two encrypt_chunk calls on the same chunk with no commit in
        // between must NOT reuse the version/nonce. encrypt_chunk reserves the
        // version in-memory, so the second call advances to the next one.
        let dek = test_dek();
        let wal = VersionWal::new(Cursor::new(Vec::new()), 100).unwrap();
        let mut writer = EncryptedChunkWriter::new(dek, wal, VersionCounterStore::new());

        let a = writer.encrypt_chunk(5, b"plaintext A").unwrap();
        let b = writer.encrypt_chunk(5, b"plaintext B DIFFERENT").unwrap();

        assert_eq!(a.version, 1);
        assert_eq!(b.version, 2, "second encrypt must reserve a fresh version");
        assert_ne!(
            derive_nonce(&dek, 5, a.version),
            derive_nonce(&dek, 5, b.version),
            "nonces must differ across two uncommitted encrypts of the same chunk"
        );

        // Each ciphertext decrypts only under its own reserved version.
        assert_eq!(decrypt_chunk(&dek, 5, 1, &a.ciphertext).unwrap(), b"plaintext A");
        assert_eq!(
            decrypt_chunk(&dek, 5, 2, &b.ciphertext).unwrap(),
            b"plaintext B DIFFERENT"
        );
    }
}
