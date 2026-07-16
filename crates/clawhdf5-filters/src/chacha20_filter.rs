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
//! # Key Separation (security-review follow-up, Remark A.5)
//!
//! The master DEK is never used directly for either encryption or nonce
//! derivation. Two domain-separated subkeys are derived from it via BLAKE3's
//! KDF mode with distinct context strings:
//!
//! ```text
//! DEK_enc = BLAKE3-derive-key("clawhdf5 chacha20 encryption key v1", DEK)
//! DEK_kdf = BLAKE3-derive-key("clawhdf5 chacha20 nonce-kdf key v1",  DEK)
//! ```
//!
//! `DEK_enc` keys ChaCha20-Poly1305; `DEK_kdf` keys the nonce KDF. This
//! implements the key-separation follow-up from the S2-D2-Yr2 security
//! review (`docs/security-review-notes.md`, Remark A.5): the security proof
//! models the AEAD and the nonce KDF as independently keyed oracles, and the
//! subkey split makes the implementation match the model instead of relying
//! on the (heuristic) independence of the two primitives under one key.
//! Functionally equivalent to the review's suggested `HKDF-Expand(DEK, info)`
//! with distinct `info` tags, using the codebase's BLAKE3 throughout.
//!
//! # Nonce Derivation
//!
//! To safely support in-place chunk updates, nonces are derived using BLAKE3:
//!
//! ```text
//! nonce = BLAKE3-derive(DEK_kdf, chunk_idx || version_counter)
//! ```
//!
//! This guarantees a unique nonce per (key, chunk, version) triple, preventing
//! keystream reuse even when chunks are updated in place.
//!
//! # Key Separation
//!
//! [`encrypt_chunk`]/[`decrypt_chunk`] (the combined API used by
//! [`EncryptedChunkWriter`] and the filter pipeline) never key the AEAD
//! cipher and the nonce KDF from the same raw DEK bytes. [`derive_subkeys`]
//! splits the DEK into an encryption subkey and a nonce-KDF subkey via
//! domain-separated BLAKE3 derivation before either is used, so a raw DEK is
//! never handed to two different cryptographic primitives. This closes a
//! key-separation gap flagged during the P2.3 security review (see
//! `docs/security-review-notes.md` in the S2-D2-Yr2 spec repo, Remark A.5 of
//! the security appendix). The low-level [`derive_nonce`]/[`encrypt`]/
//! [`decrypt`] primitives are unchanged and still accept a raw key directly;
//! callers building on those primitives directly (rather than through
//! `encrypt_chunk`/`decrypt_chunk`) are responsible for their own key
//! separation.
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

/// Derive the two domain-separated subkeys from the master DEK
/// (security-review follow-up, Remark A.5).
///
/// Returns `(DEK_enc, DEK_kdf)`: the ChaCha20-Poly1305 encryption key and the
/// nonce-KDF key. The master DEK itself must key neither primitive directly —
/// [`encrypt_chunk`] / [`decrypt_chunk`] apply this split internally.
#[inline]
#[must_use]
pub fn derive_subkeys(dek: &Dek) -> (Dek, Dek) {
    let enc = blake3::derive_key("clawhdf5 chacha20 encryption key v1", dek);
    let kdf = blake3::derive_key("clawhdf5 chacha20 nonce-kdf key v1", dek);
    (enc, kdf)
}

/// Derive a nonce from a nonce-KDF key, chunk index, and version counter
/// using BLAKE3.
///
/// This implements the version-counter nonce derivation from S2-D2-Yr2 §7.3:
///
/// ```text
/// nonce = BLAKE3-derive(DEK_kdf, chunk_idx || version_counter)
/// ```
///
/// The derived nonce is 12 bytes (96 bits) as required by ChaCha20-Poly1305.
///
/// # Arguments
///
/// * `dek` - The 32-byte nonce-KDF key. In the chunk APIs this is `DEK_kdf`
///   from [`derive_subkeys`], never the master DEK (Remark A.5 key
///   separation).
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
/// It automatically derives a unique nonce from the chunk index and version
/// counter, and internally splits `dek` into independent encryption/nonce-KDF
/// subkeys via [`derive_subkeys`] so the raw DEK never keys two different
/// cryptographic primitives.
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
    // Remark A.5 key separation: the AEAD and the nonce KDF are keyed by
    // distinct subkeys, never the master DEK directly.
    let (enc_key, kdf_key) = derive_subkeys(dek);
    let nonce = derive_nonce(&kdf_key, chunk_idx, version);
    encrypt(&enc_key, &nonce, plaintext, &[])
}

/// Decrypt a chunk with automatic nonce derivation.
///
/// This is the primary API for chunk decryption in the HDF5 filter pipeline.
/// See [`encrypt_chunk`] for the key-separation details.
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
    // Remark A.5 key separation, mirroring `encrypt_chunk`.
    let (enc_key, kdf_key) = derive_subkeys(dek);
    let nonce = derive_nonce(&kdf_key, chunk_idx, version);
    decrypt(&enc_key, &nonce, ciphertext, &[])
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
    /// Dataset-level version counter, strictly incremented once per committed
    /// mutation (P2.2b step 1, Remark A.13). This — not `max_version()` — is
    /// what `commit_with_write_order` binds into the root attribute.
    dataset_version: u64,
}

impl<W: Read + Write + Seek> EncryptedChunkWriter<W> {
    /// Create a new encrypted chunk writer for a fresh dataset (dataset
    /// version starts at 0; the first commit persists version 1).
    ///
    /// When reopening an existing dataset, use
    /// [`with_dataset_version`](Self::with_dataset_version) to seed the
    /// counter from the persisted `merkle_version` attribute instead.
    pub fn new(dek: Dek, wal: VersionWal<W>, versions: VersionCounterStore) -> Self {
        Self::with_dataset_version(dek, wal, versions, 0)
    }

    /// Create a writer seeded with the dataset version read from the persisted
    /// `merkle_version` attribute on open. Commits continue strictly upward
    /// from `dataset_version`, so the persisted counter never regresses or
    /// ties across commits (Remark A.13).
    pub fn with_dataset_version(
        dek: Dek,
        wal: VersionWal<W>,
        versions: VersionCounterStore,
        dataset_version: u64,
    ) -> Self {
        Self {
            dek,
            wal,
            versions,
            dataset_version,
        }
    }

    /// The high-water mark of dataset-level versions presented to storage.
    ///
    /// Strictly incremented by every
    /// [`commit_with_write_order`](Self::commit_with_write_order) that reaches
    /// the root-attribute step — including attempts that then fail, since a
    /// version that may be on disk can never be reused (gaps are allowed,
    /// ties are not). This is the value bound into the root attribute and the
    /// signed payload — **not** `max_version()` (the max of per-chunk
    /// counters), which is only non-decreasing and can tie or regress across
    /// commits touching different chunks.
    #[must_use]
    pub fn dataset_version(&self) -> u64 {
        self.dataset_version
    }

    /// Burn a dataset version the moment it is presented to storage, whether
    /// or not the commit then succeeds.
    ///
    /// Internal seam for `commit_with_write_order` (in `write_order.rs`).
    pub(crate) fn advance_dataset_version(&mut self, presented: u64) {
        debug_assert!(
            presented > self.dataset_version,
            "dataset version must strictly increase: current={}, presented={presented}",
            self.dataset_version
        );
        self.dataset_version = presented;
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

        // Step 2: Journal to WAL BEFORE deriving nonce (crash safety). The
        // plaintext hash travels with the record so a crash-recovery replay
        // can be verified against it rather than trusted blindly (Remark A.9).
        let plaintext_hash = crate::version_wal::WalRecord::hash_plaintext(plaintext);
        self.wal
            .journal_version(chunk_idx, version, plaintext_hash)?;

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

    /// Get the maximum version across all per-chunk counters.
    ///
    /// **Not the dataset-level version.** `max_k v_k` is only non-decreasing —
    /// two different committed states can tie on it, and a commit to a
    /// low-version chunk does not advance it — which is exactly the stateful-
    /// verifier blind spot Remark A.13 (S2-D2-Yr2 security review) rules out.
    /// Use [`dataset_version`](Self::dataset_version) for the persisted,
    /// signed, strictly-per-commit counter.
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
    /// Returns the list of `(chunk_idx, version, plaintext_hash)` triples that were
    /// pending. For each triple the caller **must** call
    /// [`WalRecord::verify_replay_plaintext`](crate::version_wal::WalRecord::verify_replay_plaintext)
    /// against its candidate replay plaintext and refuse to replay on mismatch
    /// (Remark A.9 — see [`version_wal::WAL_RECORD_SIZE`](crate::version_wal::WAL_RECORD_SIZE)'s
    /// doc comment for why the bare `(chunk_idx, version)` pair is not enough
    /// to make replay safe on its own). Once verified, promote the companion
    /// dataset to `version` and call [`commit`](Self::commit) to finalize the
    /// record.
    pub fn recover(&mut self) -> Result<Vec<(u64, u64, [u8; 32])>, EncryptedWriteError> {
        let uncommitted = self.wal.recover()?;

        // Seed the version store so journaled versions are treated as already consumed.
        // Without this, next_version() could hand back a version that was already used
        // to derive a nonce before the crash -> catastrophic keystream reuse.
        for &(chunk_idx, version, _) in &uncommitted {
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

    /// Remark A.5 key separation: the chunk APIs must key the AEAD and the
    /// nonce KDF with distinct subkeys, never the raw master DEK.
    #[test]
    fn chunk_apis_use_domain_separated_subkeys() {
        let dek = test_dek();
        let (enc_key, kdf_key) = derive_subkeys(&dek);

        // The subkeys differ from the master DEK and from each other.
        assert_ne!(enc_key, dek);
        assert_ne!(kdf_key, dek);
        assert_ne!(enc_key, kdf_key);

        // encrypt_chunk is NOT raw-DEK encryption: ciphertext produced by
        // keying the primitives with the master DEK directly must differ.
        let raw_nonce = derive_nonce(&dek, 3, 1);
        let raw_ct = encrypt(&dek, &raw_nonce, b"payload", &[]).unwrap();
        let sep_ct = encrypt_chunk(&dek, 3, 1, b"payload").unwrap();
        assert_ne!(raw_ct, sep_ct);

        // And the separated path round-trips.
        assert_eq!(decrypt_chunk(&dek, 3, 1, &sep_ct).unwrap(), b"payload");
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
    fn derive_subkeys_are_independent_of_each_other_and_the_dek() {
        let dek = test_dek();
        let (dek_enc, dek_kdf) = derive_subkeys(&dek);

        assert_ne!(dek_enc, dek, "encryption subkey must not equal the raw DEK");
        assert_ne!(dek_kdf, dek, "KDF subkey must not equal the raw DEK");
        assert_ne!(
            dek_enc, dek_kdf,
            "encryption and nonce-KDF subkeys must differ from each other"
        );
    }

    #[test]
    fn derive_subkeys_deterministic() {
        let dek = test_dek();
        assert_eq!(derive_subkeys(&dek), derive_subkeys(&dek));
    }

    #[test]
    fn derive_subkeys_differ_across_deks() {
        let dek1 = test_dek();
        let mut dek2 = test_dek();
        dek2[0] ^= 0xFF;

        let (enc1, kdf1) = derive_subkeys(&dek1);
        let (enc2, kdf2) = derive_subkeys(&dek2);
        assert_ne!(enc1, enc2);
        assert_ne!(kdf1, kdf2);
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
        assert_eq!(
            uncommitted[0],
            (5, 1, crate::version_wal::WalRecord::hash_plaintext(b"data"))
        ); // chunk 5, version 1
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
        assert_eq!(
            decrypt_chunk(&dek, 5, 1, &a.ciphertext).unwrap(),
            b"plaintext A"
        );
        assert_eq!(
            decrypt_chunk(&dek, 5, 2, &b.ciphertext).unwrap(),
            b"plaintext B DIFFERENT"
        );
    }
}
