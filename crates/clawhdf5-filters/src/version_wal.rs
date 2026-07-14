//! Write-Ahead Log (WAL) for crash-safe version counter management.
//!
//! This module implements the version-counter nonce derivation protocol from
//! S2-D2-Yr2 P2.2 step 3. The key security requirement is that nonces must
//! never be reused under the same DEK, even after a crash.
//!
//! # WAL Protocol (from spec §7.3)
//!
//! 1. Journal `(chunk_idx, v_new)` to a crash-durable WAL **before** deriving the nonce
//! 2. Derive the nonce using `v_new`, encrypt the chunk, and write the ciphertext
//! 3. Promote the journal record into the companion dataset (update the leaf's stored
//!    version counter to `v_new`) and recompute the Merkle path to the root
//! 4. Mark the journal record as committed
//!
//! On crash recovery, any uncommitted journal record indicates an in-progress write:
//! the system replays the chunk encryption using the journaled `v_new` and completes
//! the companion update before declaring the file consistent.
//!
//! # Nonce Reuse Prevention
//!
//! Nonce reuse requires `v_chunk` rollback, which is prevented by:
//! - The WAL protocol ensuring `v_new` is durably recorded before encryption
//! - Monotonic version counters that only increment
//! - Recovery replaying uncommitted journal records with their journaled versions
//!   (see [`crate::chacha20_filter::EncryptedChunkWriter::recover`], which seeds the
//!   version store so post-crash writes always advance past any journaled version)
//!
//! # Durability Caveat
//!
//! Journal writes currently use [`Write::flush`], **not** `File::sync_all`/`fsync`.
//! `flush` only pushes buffered bytes into the OS page cache; it does not guarantee
//! they have reached stable storage. A power loss between `flush` and the kernel
//! writing back the page can therefore lose the most recent journal record. This is
//! acceptable for the P2.2 prototype (the in-memory writer and `Cursor`-backed tests
//! have no such gap), but a production, file-backed WAL must call `sync_all()` after
//! each journal append for true crash durability. Tracked as `TODO(P3)` in
//! [`VersionWal::journal_version`].
//!
//! # Spec Reference
//!
//! S2-D2-Yr2 P2.2 step 3: "Implement version-counter nonce derivation. The nonce
//! for encrypting chunk i at write number v is:
//! nonce = BLAKE3-derive(DEK, chunk_idx || v_chunk)"

use std::collections::HashMap;
use std::io::{self, Read, Seek, SeekFrom, Write};

/// Size of a single WAL record in bytes.
/// Format: status (1) + chunk_idx (8) + version (8) + checksum (4) = 21 bytes
pub const WAL_RECORD_SIZE: usize = 21;

/// Magic bytes for WAL file header.
pub const WAL_MAGIC: [u8; 4] = *b"CWAL";

/// WAL file format version.
pub const WAL_VERSION: u8 = 1;

/// WAL header size in bytes.
/// Format: magic (4) + version (1) + reserved (3) = 8 bytes
pub const WAL_HEADER_SIZE: usize = 8;

/// Status byte values for WAL records.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalRecordStatus {
    /// Record slot is empty/unused.
    Empty = 0x00,
    /// Record is pending (write in progress).
    Pending = 0x01,
    /// Record is committed (write completed successfully).
    Committed = 0x02,
    /// Record was aborted (write failed, needs recovery).
    Aborted = 0x03,
}

impl TryFrom<u8> for WalRecordStatus {
    type Error = WalError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x00 => Ok(WalRecordStatus::Empty),
            0x01 => Ok(WalRecordStatus::Pending),
            0x02 => Ok(WalRecordStatus::Committed),
            0x03 => Ok(WalRecordStatus::Aborted),
            _ => Err(WalError::InvalidRecordStatus(value)),
        }
    }
}

/// A single WAL record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalRecord {
    /// Status of this record.
    pub status: WalRecordStatus,
    /// Chunk index being written.
    pub chunk_idx: u64,
    /// New version counter for the chunk.
    pub version: u64,
}

impl WalRecord {
    /// Create a new pending WAL record.
    #[must_use]
    pub fn new_pending(chunk_idx: u64, version: u64) -> Self {
        Self {
            status: WalRecordStatus::Pending,
            chunk_idx,
            version,
        }
    }

    /// Serialize this record to bytes.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; WAL_RECORD_SIZE] {
        let mut buf = [0u8; WAL_RECORD_SIZE];
        buf[0] = self.status as u8;
        buf[1..9].copy_from_slice(&self.chunk_idx.to_le_bytes());
        buf[9..17].copy_from_slice(&self.version.to_le_bytes());
        // Compute CRC32 checksum over the first 17 bytes
        let checksum = crc32_checksum(&buf[..17]);
        buf[17..21].copy_from_slice(&checksum.to_le_bytes());
        buf
    }

    /// Deserialize a record from bytes.
    pub fn from_bytes(buf: &[u8; WAL_RECORD_SIZE]) -> Result<Self, WalError> {
        // Verify checksum
        let stored_checksum = u32::from_le_bytes([buf[17], buf[18], buf[19], buf[20]]);
        let computed_checksum = crc32_checksum(&buf[..17]);
        if stored_checksum != computed_checksum {
            return Err(WalError::ChecksumMismatch {
                expected: computed_checksum,
                actual: stored_checksum,
            });
        }

        let status = WalRecordStatus::try_from(buf[0])?;
        let chunk_idx = u64::from_le_bytes([
            buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], buf[8],
        ]);
        let version = u64::from_le_bytes([
            buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15], buf[16],
        ]);

        Ok(Self {
            status,
            chunk_idx,
            version,
        })
    }
}

/// Errors that can occur during WAL operations.
#[derive(Debug)]
pub enum WalError {
    /// I/O error.
    Io(io::Error),
    /// Invalid WAL magic bytes.
    InvalidMagic([u8; 4]),
    /// Unsupported WAL version.
    UnsupportedVersion(u8),
    /// Invalid record status byte.
    InvalidRecordStatus(u8),
    /// Checksum mismatch in WAL record.
    ChecksumMismatch { expected: u32, actual: u32 },
    /// WAL is full (too many pending records).
    WalFull,
    /// Record not found.
    RecordNotFound { chunk_idx: u64 },
    /// Buffer too short for deserialization.
    BufferTooShort { expected: usize, actual: usize },
}

impl std::fmt::Display for WalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WalError::Io(e) => write!(f, "WAL I/O error: {e}"),
            WalError::InvalidMagic(m) => {
                write!(f, "invalid WAL magic: {:02x?}", m)
            }
            WalError::UnsupportedVersion(v) => {
                write!(f, "unsupported WAL version: {v}")
            }
            WalError::InvalidRecordStatus(s) => {
                write!(f, "invalid WAL record status: 0x{s:02x}")
            }
            WalError::ChecksumMismatch { expected, actual } => {
                write!(
                    f,
                    "WAL record checksum mismatch: expected 0x{expected:08x}, got 0x{actual:08x}"
                )
            }
            WalError::WalFull => write!(f, "WAL is full"),
            WalError::RecordNotFound { chunk_idx } => {
                write!(f, "WAL record not found for chunk {chunk_idx}")
            }
            WalError::BufferTooShort { expected, actual } => {
                write!(
                    f,
                    "buffer too short: expected {expected} bytes, got {actual}"
                )
            }
        }
    }
}

impl std::error::Error for WalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WalError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for WalError {
    fn from(e: io::Error) -> Self {
        WalError::Io(e)
    }
}

/// Simple CRC32 checksum (IEEE polynomial).
fn crc32_checksum(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// Write-Ahead Log for version counters.
///
/// This WAL ensures crash-safe version counter management. Before encrypting
/// a chunk with a new version counter, the new version must be durably recorded
/// in the WAL. This prevents nonce reuse even if a crash occurs between
/// encryption and companion dataset update.
pub struct VersionWal<W> {
    /// The underlying writer (file or buffer).
    writer: W,
    /// Maximum number of records in the WAL.
    max_records: usize,
    /// Current number of records written.
    record_count: usize,
}

impl<W: Read + Write + Seek> VersionWal<W> {
    /// Create a new WAL with the given writer and maximum record capacity.
    ///
    /// Initializes the WAL with a header if it's empty, or validates the
    /// existing header if it contains data.
    pub fn new(mut writer: W, max_records: usize) -> Result<Self, WalError> {
        // Check if the writer has existing content
        let pos = writer.seek(SeekFrom::End(0))?;

        if pos == 0 {
            // New WAL - write header
            writer.seek(SeekFrom::Start(0))?;
            let mut header = [0u8; WAL_HEADER_SIZE];
            header[..4].copy_from_slice(&WAL_MAGIC);
            header[4] = WAL_VERSION;
            // bytes 5-7 reserved
            writer.write_all(&header)?;
            writer.flush()?;

            Ok(Self {
                writer,
                max_records,
                record_count: 0,
            })
        } else {
            // Existing WAL - validate header
            writer.seek(SeekFrom::Start(0))?;
            let mut header = [0u8; WAL_HEADER_SIZE];
            writer.read_exact(&mut header)?;

            let magic: [u8; 4] = [header[0], header[1], header[2], header[3]];
            if magic != WAL_MAGIC {
                return Err(WalError::InvalidMagic(magic));
            }
            if header[4] != WAL_VERSION {
                return Err(WalError::UnsupportedVersion(header[4]));
            }

            // Count existing records
            let data_size = pos as usize - WAL_HEADER_SIZE;
            let record_count = data_size / WAL_RECORD_SIZE;

            Ok(Self {
                writer,
                max_records,
                record_count,
            })
        }
    }

    /// Journal a new version counter before encryption.
    ///
    /// This must be called BEFORE deriving the nonce and encrypting the chunk.
    /// The record is written with `Pending` status and synced to disk.
    pub fn journal_version(&mut self, chunk_idx: u64, version: u64) -> Result<(), WalError> {
        if self.record_count >= self.max_records {
            return Err(WalError::WalFull);
        }

        let record = WalRecord::new_pending(chunk_idx, version);
        let offset = WAL_HEADER_SIZE + self.record_count * WAL_RECORD_SIZE;

        self.writer.seek(SeekFrom::Start(offset as u64))?;
        self.writer.write_all(&record.to_bytes())?;
        self.writer.flush()?;
        // TODO(P3): call sync_all() for true crash durability

        self.record_count += 1;
        Ok(())
    }

    /// Mark a journaled version as committed.
    ///
    /// This should be called AFTER the chunk is written and the companion
    /// dataset is updated. The record status is changed to `Committed`.
    pub fn commit_version(&mut self, chunk_idx: u64) -> Result<(), WalError> {
        // Find the pending record for this chunk
        for i in (0..self.record_count).rev() {
            let offset = WAL_HEADER_SIZE + i * WAL_RECORD_SIZE;
            self.writer.seek(SeekFrom::Start(offset as u64))?;

            let mut buf = [0u8; WAL_RECORD_SIZE];
            self.writer.read_exact(&mut buf)?;

            if let Ok(record) = WalRecord::from_bytes(&buf)
                && record.chunk_idx == chunk_idx
                && record.status == WalRecordStatus::Pending
            {
                // Update status to committed
                self.writer.seek(SeekFrom::Start(offset as u64))?;
                self.writer.write_all(&[WalRecordStatus::Committed as u8])?;
                self.writer.flush()?;
                return Ok(());
            }
        }

        Err(WalError::RecordNotFound { chunk_idx })
    }

    /// Recover uncommitted records from the WAL.
    ///
    /// Returns a list of (chunk_idx, version) pairs for records that were
    /// journaled but not committed. These chunks need to have their writes
    /// replayed with the journaled version.
    pub fn recover(&mut self) -> Result<Vec<(u64, u64)>, WalError> {
        let mut uncommitted = Vec::new();

        for i in 0..self.record_count {
            let offset = WAL_HEADER_SIZE + i * WAL_RECORD_SIZE;
            self.writer.seek(SeekFrom::Start(offset as u64))?;

            let mut buf = [0u8; WAL_RECORD_SIZE];
            if self.writer.read_exact(&mut buf).is_err() {
                break;
            }

            if let Ok(record) = WalRecord::from_bytes(&buf)
                && record.status == WalRecordStatus::Pending
            {
                uncommitted.push((record.chunk_idx, record.version));
            }
        }

        Ok(uncommitted)
    }

    /// Clear all committed records from the WAL.
    ///
    /// This compacts the WAL by removing all committed records and keeping
    /// only pending ones. Should be called periodically to prevent WAL growth.
    ///
    /// # Limitation
    ///
    /// The generic `W: Write + Seek` bound provides no way to truncate the
    /// underlying storage (there is no `set_len` in `std::io`), so the file's byte
    /// length does not shrink. The freed tail slots are zeroed instead; because a
    /// zeroed record fails its CRC check, [`recover`](Self::recover) and
    /// [`commit_version`](Self::commit_version) skip them, so reopening the WAL is
    /// still correct — only the on-disk footprint stays at its high-water mark until
    /// a `File`-backed truncation path is added.
    pub fn compact(&mut self) -> Result<(), WalError> {
        let old_count = self.record_count;
        let mut pending_records = Vec::new();

        // Collect all pending records
        for i in 0..old_count {
            let offset = WAL_HEADER_SIZE + i * WAL_RECORD_SIZE;
            self.writer.seek(SeekFrom::Start(offset as u64))?;

            let mut buf = [0u8; WAL_RECORD_SIZE];
            if self.writer.read_exact(&mut buf).is_err() {
                break;
            }

            if let Ok(record) = WalRecord::from_bytes(&buf)
                && record.status == WalRecordStatus::Pending
            {
                pending_records.push(record);
            }
        }

        // Rewrite WAL with only pending records at the front.
        self.writer.seek(SeekFrom::Start(WAL_HEADER_SIZE as u64))?;
        for record in &pending_records {
            self.writer.write_all(&record.to_bytes())?;
        }

        // Zero only the slots that previously held data (pending_len..old_count) so
        // stale committed records can't be mistaken for live ones. We intentionally
        // do NOT extend to `max_records`: that would grow the file rather than
        // compact it. `max_records` still bounds future journaling via record_count.
        let empty_record = [0u8; WAL_RECORD_SIZE];
        for _ in pending_records.len()..old_count {
            self.writer.write_all(&empty_record)?;
        }

        let new_len = WAL_HEADER_SIZE + pending_records.len() * WAL_RECORD_SIZE;
        self.record_count = pending_records.len();
        self.writer.flush()?;
        self.writer.seek(SeekFrom::Start(new_len as u64))?;

        Ok(())
    }
}

/// In-memory version counter store.
///
/// Tracks per-chunk version counters. In the full implementation, this would
/// be backed by the companion dataset. For P2.2 step 3, we use an in-memory
/// store with WAL for crash safety.
#[derive(Debug, Clone)]
pub struct VersionCounterStore {
    /// Version counters indexed by chunk index.
    counters: HashMap<u64, u64>,
    /// Maximum version seen (for dataset-level version).
    max_version: u64,
}

impl Default for VersionCounterStore {
    fn default() -> Self {
        Self::new()
    }
}

impl VersionCounterStore {
    /// Create a new empty version counter store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            counters: HashMap::new(),
            max_version: 0,
        }
    }

    /// Create a version counter store with pre-populated counters.
    #[must_use]
    pub fn with_counters(counters: HashMap<u64, u64>) -> Self {
        let max_version = counters.values().copied().max().unwrap_or(0);
        Self {
            counters,
            max_version,
        }
    }

    /// Get the current version counter for a chunk.
    ///
    /// Returns 0 if the chunk has never been written.
    #[must_use]
    pub fn get(&self, chunk_idx: u64) -> u64 {
        self.counters.get(&chunk_idx).copied().unwrap_or(0)
    }

    /// Get the next version counter for a chunk (current + 1).
    ///
    /// This is the version that should be used for the next write.
    #[must_use]
    pub fn next_version(&self, chunk_idx: u64) -> u64 {
        self.get(chunk_idx) + 1
    }

    /// Update the version counter for a chunk.
    ///
    /// This should only be called after the chunk write is committed.
    ///
    /// # Panics
    ///
    /// Panics if the new version is not greater than the current version
    /// (version counters must be monotonically increasing).
    pub fn update(&mut self, chunk_idx: u64, version: u64) {
        let current = self.get(chunk_idx);
        assert!(
            version > current,
            "version counter must increase: current={current}, new={version}"
        );
        self.counters.insert(chunk_idx, version);
        if version > self.max_version {
            self.max_version = version;
        }
    }

    /// Seed the version counter for a chunk to at least `version`, without panicking.
    ///
    /// Unlike [`update`](Self::update), this is **monotonic and idempotent**: it sets
    /// the counter to `max(current, version)` and never regresses or panics. It is the
    /// crash-recovery primitive — replaying a journaled `(chunk_idx, version)` record
    /// marks that version as already consumed so that the next call to
    /// [`next_version`](Self::next_version) returns a strictly higher value.
    ///
    /// This is what makes nonce reuse impossible after a crash: once a journaled
    /// version is seeded, no subsequent write can re-derive the same
    /// `(DEK, chunk_idx, version)` nonce.
    pub fn seed(&mut self, chunk_idx: u64, version: u64) {
        let current = self.get(chunk_idx);
        if version > current {
            self.counters.insert(chunk_idx, version);
        }
        if version > self.max_version {
            self.max_version = version;
        }
    }

    /// Get the maximum version across all chunks.
    ///
    /// This is used as the dataset-level version in the signed root.
    #[must_use]
    pub fn max_version(&self) -> u64 {
        self.max_version
    }

    /// Get the number of chunks with version counters.
    #[must_use]
    pub fn len(&self) -> usize {
        self.counters.len()
    }

    /// Check if the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.counters.is_empty()
    }

    /// Iterate over all (chunk_idx, version) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&u64, &u64)> {
        self.counters.iter()
    }

    /// Serialize version counters to bytes.
    ///
    /// Format: count (8 bytes) + (chunk_idx (8) + version (8)) * count
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let count = self.counters.len() as u64;
        let mut buf = Vec::with_capacity(8 + self.counters.len() * 16);
        buf.extend_from_slice(&count.to_le_bytes());
        for (&chunk_idx, &version) in &self.counters {
            buf.extend_from_slice(&chunk_idx.to_le_bytes());
            buf.extend_from_slice(&version.to_le_bytes());
        }
        buf
    }

    /// Deserialize version counters from bytes.
    ///
    /// # Errors
    ///
    /// Returns [`WalError::BufferTooShort`] if the buffer is too small.
    pub fn from_bytes(buf: &[u8]) -> Result<Self, WalError> {
        if buf.len() < 8 {
            return Err(WalError::BufferTooShort {
                expected: 8,
                actual: buf.len(),
            });
        }
        let count = u64::from_le_bytes([
            buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
        ]) as usize;

        // `count` is attacker-controlled: compute the required length with checked
        // arithmetic so a huge value can't overflow past the bounds check below or
        // trigger an unbounded pre-allocation.
        let expected_len = count
            .checked_mul(16)
            .and_then(|n| n.checked_add(8))
            .ok_or(WalError::BufferTooShort {
                expected: usize::MAX,
                actual: buf.len(),
            })?;
        if buf.len() < expected_len {
            return Err(WalError::BufferTooShort {
                expected: expected_len,
                actual: buf.len(),
            });
        }

        // Safe now: expected_len <= buf.len(), so count <= (buf.len() - 8) / 16.
        let mut counters = HashMap::with_capacity(count);
        let mut offset = 8;
        for _ in 0..count {
            let chunk_idx = u64::from_le_bytes([
                buf[offset],
                buf[offset + 1],
                buf[offset + 2],
                buf[offset + 3],
                buf[offset + 4],
                buf[offset + 5],
                buf[offset + 6],
                buf[offset + 7],
            ]);
            let version = u64::from_le_bytes([
                buf[offset + 8],
                buf[offset + 9],
                buf[offset + 10],
                buf[offset + 11],
                buf[offset + 12],
                buf[offset + 13],
                buf[offset + 14],
                buf[offset + 15],
            ]);
            counters.insert(chunk_idx, version);
            offset += 16;
        }

        Ok(Self::with_counters(counters))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn wal_record_roundtrip() {
        let record = WalRecord::new_pending(42, 7);
        let bytes = record.to_bytes();
        let decoded = WalRecord::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, record);
    }

    #[test]
    fn wal_record_checksum_detects_corruption() {
        let record = WalRecord::new_pending(42, 7);
        let mut bytes = record.to_bytes();
        // Corrupt a byte
        bytes[5] ^= 0xFF;
        assert!(WalRecord::from_bytes(&bytes).is_err());
    }

    #[test]
    fn wal_create_and_journal() {
        let buf = Cursor::new(Vec::new());
        let mut wal = VersionWal::new(buf, 100).unwrap();

        wal.journal_version(0, 1).unwrap();
        wal.journal_version(1, 1).unwrap();
        wal.journal_version(0, 2).unwrap();

        // Check that records are recoverable
        let uncommitted = wal.recover().unwrap();
        assert_eq!(uncommitted.len(), 3);
        assert!(uncommitted.contains(&(0, 1)));
        assert!(uncommitted.contains(&(1, 1)));
        assert!(uncommitted.contains(&(0, 2)));
    }

    #[test]
    fn wal_commit_removes_from_uncommitted() {
        let buf = Cursor::new(Vec::new());
        let mut wal = VersionWal::new(buf, 100).unwrap();

        wal.journal_version(0, 1).unwrap();
        wal.journal_version(1, 1).unwrap();
        wal.commit_version(0).unwrap();

        let uncommitted = wal.recover().unwrap();
        assert_eq!(uncommitted.len(), 1);
        assert!(uncommitted.contains(&(1, 1)));
    }

    #[test]
    fn wal_reopen_existing() {
        let mut buf = Cursor::new(Vec::new());

        // Create and populate WAL
        {
            let mut wal = VersionWal::new(&mut buf, 100).unwrap();
            wal.journal_version(0, 1).unwrap();
            wal.journal_version(1, 1).unwrap();
        }

        // Reopen and verify
        buf.seek(SeekFrom::Start(0)).unwrap();
        let mut wal = VersionWal::new(&mut buf, 100).unwrap();
        let uncommitted = wal.recover().unwrap();
        assert_eq!(uncommitted.len(), 2);
    }

    #[test]
    fn version_counter_store_basic() {
        let mut store = VersionCounterStore::new();

        assert_eq!(store.get(0), 0);
        assert_eq!(store.next_version(0), 1);

        store.update(0, 1);
        assert_eq!(store.get(0), 1);
        assert_eq!(store.next_version(0), 2);
        assert_eq!(store.max_version(), 1);

        store.update(1, 5);
        assert_eq!(store.max_version(), 5);
    }

    #[test]
    #[should_panic(expected = "version counter must increase")]
    fn version_counter_store_rejects_non_monotonic() {
        let mut store = VersionCounterStore::new();
        store.update(0, 5);
        store.update(0, 3); // Should panic
    }

    #[test]
    fn version_counter_store_serialization() {
        let mut store = VersionCounterStore::new();
        store.update(0, 1);
        store.update(5, 3);
        store.update(10, 7);

        let bytes = store.to_bytes();
        let restored = VersionCounterStore::from_bytes(&bytes).unwrap();

        assert_eq!(restored.get(0), 1);
        assert_eq!(restored.get(5), 3);
        assert_eq!(restored.get(10), 7);
        assert_eq!(restored.max_version(), 7);
    }

    #[test]
    fn compact_drops_committed_and_keeps_pending_recoverable() {
        let buf = Cursor::new(Vec::new());
        let mut wal = VersionWal::new(buf, 100).unwrap();

        wal.journal_version(0, 1).unwrap();
        wal.journal_version(1, 1).unwrap();
        wal.journal_version(2, 1).unwrap();

        // Commit chunk 1; it should no longer be part of the pending set.
        wal.commit_version(1).unwrap();

        // Compaction physically drops the committed record and relocates the
        // remaining pending records to the front of the log.
        wal.compact().unwrap();

        // recover() must still report exactly the two pending records.
        let after = wal.recover().unwrap();
        assert_eq!(after.len(), 2);
        assert!(after.contains(&(0, 1)));
        assert!(after.contains(&(2, 1)));
        assert!(
            !after.iter().any(|&(k, _)| k == 1),
            "committed record must not reappear after compaction"
        );

        // The relocated pending records remain intact and committable.
        wal.commit_version(0).unwrap();
        let remaining = wal.recover().unwrap();
        assert_eq!(remaining, vec![(2, 1)]);

        // The already-committed chunk is gone: it cannot be committed again.
        assert!(matches!(
            wal.commit_version(1),
            Err(WalError::RecordNotFound { chunk_idx: 1 })
        ));
    }

    #[test]
    fn compact_then_reopen_recovers_only_pending() {
        let mut buf = Cursor::new(Vec::new());

        // Journal three, commit one, then compact — all on one handle.
        {
            let mut wal = VersionWal::new(&mut buf, 100).unwrap();
            wal.journal_version(0, 1).unwrap();
            wal.journal_version(1, 1).unwrap();
            wal.journal_version(2, 1).unwrap();
            wal.commit_version(1).unwrap();
            wal.compact().unwrap();
        }

        // Reopen from the same bytes: the zeroed (freed) slot must not resurrect a
        // record, and recovery must still yield exactly the two pending entries.
        buf.seek(SeekFrom::Start(0)).unwrap();
        let mut wal = VersionWal::new(&mut buf, 100).unwrap();
        let recovered = wal.recover().unwrap();
        assert_eq!(recovered.len(), 2);
        assert!(recovered.contains(&(0, 1)));
        assert!(recovered.contains(&(2, 1)));
        assert!(!recovered.iter().any(|&(k, _)| k == 1));
    }

    #[test]
    fn deserializers_never_panic_on_arbitrary_bytes() {
        // Deterministic byte sweep standing in for the cargo-fuzz targets so the
        // never-panic property is exercised under `cargo test` in CI. A simple
        // xorshift PRNG generates structured and random inputs; the test passing
        // (i.e. not aborting) is the assertion.
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for _ in 0..10_000 {
            let len = (next() % 80) as usize;
            let mut bytes = Vec::with_capacity(len);
            for _ in 0..len {
                bytes.push((next() & 0xff) as u8);
            }

            // Occasionally plant a plausible-looking count in the first 8 bytes
            // to reach deeper into the parser, including huge/overflowing values.
            if len >= 8 && next() & 1 == 0 {
                let planted = 1u64 << (next() % 64);
                bytes[..8].copy_from_slice(&planted.to_le_bytes());
            }

            // Must return Ok/Err, never panic.
            let _ = VersionCounterStore::from_bytes(&bytes);

            if bytes.len() >= WAL_RECORD_SIZE {
                let mut rec = [0u8; WAL_RECORD_SIZE];
                rec.copy_from_slice(&bytes[..WAL_RECORD_SIZE]);
                let _ = WalRecord::from_bytes(&rec);
            }
        }
    }

    #[test]
    fn version_counter_store_seed_is_monotonic_and_idempotent() {
        let mut store = VersionCounterStore::new();

        // Seeding from 0 sets the value.
        store.seed(3, 5);
        assert_eq!(store.get(3), 5);
        assert_eq!(store.max_version(), 5);
        // next_version now hands back a strictly higher value.
        assert_eq!(store.next_version(3), 6);

        // Seeding an equal or lower value is a no-op (no panic, no regression).
        store.seed(3, 5);
        store.seed(3, 2);
        assert_eq!(store.get(3), 5);

        // Seeding higher advances it.
        store.seed(3, 9);
        assert_eq!(store.get(3), 9);
        assert_eq!(store.max_version(), 9);
    }

    #[test]
    fn from_bytes_rejects_overflowing_count_without_panicking() {
        // Malicious header: count = 2^60, buffer far too small. The length
        // computation 8 + count*16 would overflow usize if unchecked.
        let buf = (1u64 << 60).to_le_bytes();
        let result = VersionCounterStore::from_bytes(&buf);
        assert!(matches!(result, Err(WalError::BufferTooShort { .. })));
    }

    #[test]
    fn from_bytes_rejects_truncated_body() {
        // count = 2 but only one (chunk_idx, version) pair present.
        let mut buf = Vec::new();
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        let result = VersionCounterStore::from_bytes(&buf);
        assert!(matches!(
            result,
            Err(WalError::BufferTooShort {
                expected: 40,
                actual: 24
            })
        ));
    }

    #[test]
    fn crc32_basic() {
        // Known test vector
        let data = b"123456789";
        let checksum = crc32_checksum(data);
        // IEEE CRC32 of "123456789" is 0xCBF43926
        assert_eq!(checksum, 0xCBF4_3926);
    }
}
