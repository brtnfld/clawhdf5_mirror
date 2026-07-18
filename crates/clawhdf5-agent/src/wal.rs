//! Write-Ahead Log (WAL) for edgehdf5 agent memory.
//!
//! Binary WAL format alongside the main .h5 file enables fast append-only
//! writes without rewriting the entire HDF5 file on every save.
//!
//! # Integrity (P2.4 red-team finding)
//!
//! The main .h5 file's `_merkle_root` attribute (see the `integrity` module)
//! only covers content that made it into a flush — it says nothing about
//! entries sitting in the `.h5.wal` sidecar, which `HDF5Memory::open` replays
//! into the live cache *after* that root verifies. A version-1 WAL had no
//! integrity protection at all (just a magic/version/count header), so an
//! attacker with write access to the sidecar could append a forged `Save` or
//! `Tombstone` record and have it silently accepted on the next open — with
//! the poisoned state then laundered into a fresh, valid root on the next
//! flush.
//!
//! Version 2 closes the naive case: every entry is followed by a 32-byte
//! SHA-256 hash chained over the previous entry's hash and this entry's own
//! bytes (`chain_i = SHA256(chain_{i-1} || entry_bytes_i)`, `chain_0` derived
//! from the header). An appended or edited entry that doesn't carry the
//! correct chain value is rejected with [`MemoryError::Integrity`] rather than
//! silently replayed. This is distinguished from ordinary crash truncation (a
//! partially written last entry, which `read_entries` has always tolerated
//! for crash safety): rejection only fires when a *complete* entry plus its
//! trailer hash were read but the hash doesn't match, never when bytes are
//! simply missing at EOF.
//!
//! **Documented limitation (unsigned, like the main file's Merkle root before
//! P2.1 signing).** This chain hash is unkeyed: the algorithm is public and
//! the current chain tip is derivable by anyone who can read the file. It
//! stops an attacker who appends or edits an entry naively (without also
//! recomputing the chain over everything before it) — the realistic case for
//! "an attacker with write access to the sidecar" absent WAL-specific
//! reverse-engineering. It does **not** stop a fully capable adversary who
//! reads the whole WAL, recomputes the correct chain, and appends a
//! self-consistent forged entry — the same class of gap T6b/T1d document for
//! an unsigned Merkle tree. Closing that would need a MAC or signature keyed
//! by material the write side alone controls, which no infrastructure in this
//! crate currently provides for the WAL.
//!
//! Version 1 files are still read (unauthenticated, for backward
//! compatibility with WALs written before this change); every freshly created
//! or truncated WAL is written as version 2.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::MemoryError;

const WAL_MAGIC: [u8; 4] = [0x45, 0x48, 0x57, 0x4C]; // "EHWL"
/// Legacy, unauthenticated format: no trailing hash after each entry. Still
/// read for backward compatibility; never written by this code.
const WAL_VERSION_V1: u8 = 1;
/// Current format: each entry followed by a 32-byte chained BLAKE3 hash. See
/// the module doc for the chaining scheme. Always written for new/truncated
/// WAL files.
const WAL_VERSION_V2: u8 = 2;
const WAL_VERSION: u8 = WAL_VERSION_V2;
const CHAIN_HASH_SIZE: usize = 32;

/// The chain's starting value, derived from the header rather than a fixed
/// all-zero seed so two WALs with different magic/version don't share a chain
/// origin (a cosmetic hardening; the header itself isn't secret).
fn initial_chain() -> [u8; CHAIN_HASH_SIZE] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(WAL_MAGIC);
    hasher.update([WAL_VERSION_V2]);
    hasher.finalize().into()
}

fn chain_next(prev: &[u8; CHAIN_HASH_SIZE], entry_bytes: &[u8]) -> [u8; CHAIN_HASH_SIZE] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(prev);
    hasher.update(entry_bytes);
    hasher.finalize().into()
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalEntryType {
    Save = 0x01,
    Tombstone = 0x02,
    ActivationUpdate = 0x03,
}

impl WalEntryType {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x01 => Some(Self::Save),
            0x02 => Some(Self::Tombstone),
            0x03 => Some(Self::ActivationUpdate),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WalEntry {
    pub entry_type: WalEntryType,
    pub timestamp: f64,
    pub chunk: String,
    pub embedding: Vec<f32>,
    pub source_channel: String,
    pub session_id: String,
    pub tags: String,
    /// For tombstone entries: the index of the entry to delete.
    pub tombstone_index: Option<usize>,
}

/// How many entries to accumulate before updating the header entry_count.
///
/// The header count is only needed for replay; `read_entries` already handles
/// stale counts by reading until EOF.  Updating every N entries rather than
/// every entry eliminates 3 lseek() + 1 write() per entry — see arXiv:2507.13062.
const GROUP_COMMIT_SIZE: u32 = 8;

#[derive(Debug)]
pub struct WalFile {
    path: PathBuf,
    file: Option<File>,
    entry_count: u32,
    /// Entries written since the last header count update.
    pending_header_sync: u32,
    /// Format version of this file: `WAL_VERSION_V1` (legacy, no chaining) or
    /// `WAL_VERSION_V2` (chained hash trailer per entry).
    version: u8,
    /// Running chain tip after the last verified entry, used to compute the
    /// next entry's trailer when `version == WAL_VERSION_V2`. Unused for v1.
    chain_tip: [u8; CHAIN_HASH_SIZE],
}

impl WalFile {
    /// Open or create a WAL file. If it exists, read the header and entry
    /// count, and — for a version-2 file — replay existing entries to
    /// recover the current chain tip so further appends continue the chain
    /// correctly. A genuine chain-hash mismatch during that replay (as
    /// opposed to ordinary crash truncation) is propagated as
    /// [`MemoryError::Integrity`], refusing to append onto a WAL whose
    /// history can't be trusted.
    pub fn open(path: &Path) -> Result<Self, MemoryError> {
        if path.exists() {
            // Read existing header
            let mut f = OpenOptions::new()
                .read(true)
                .write(true)
                .append(false)
                .open(path)?;
            let mut magic = [0u8; 4];
            f.read_exact(&mut magic)?;
            if magic != WAL_MAGIC {
                return Err(MemoryError::Schema("invalid WAL magic bytes".into()));
            }
            let mut ver = [0u8; 1];
            f.read_exact(&mut ver)?;
            let version = ver[0];
            if version != WAL_VERSION_V1 && version != WAL_VERSION_V2 {
                return Err(MemoryError::Schema(format!(
                    "unsupported WAL version {version}"
                )));
            }
            let mut count_buf = [0u8; 4];
            f.read_exact(&mut count_buf)?;
            let entry_count = u32::from_le_bytes(count_buf);

            // Replay to recover the chain tip (v2 only; v1 has none). This
            // also surfaces a genuine tamper/corruption as an error here,
            // before any further append trusts a broken chain.
            let chain_tip = if version == WAL_VERSION_V2 {
                parse_entries(&mut f, version)?.1
            } else {
                initial_chain()
            };

            // Seek to end for appending
            f.seek(SeekFrom::End(0))?;
            Ok(Self {
                path: path.to_path_buf(),
                file: Some(f),
                entry_count,
                pending_header_sync: 0,
                version,
                chain_tip,
            })
        } else {
            // Create new WAL, always in the current (v2) format.
            let mut f = File::create(path)?;
            f.write_all(&WAL_MAGIC)?;
            f.write_all(&[WAL_VERSION])?;
            f.write_all(&0u32.to_le_bytes())?;
            f.flush()?;
            Ok(Self {
                path: path.to_path_buf(),
                file: Some(f),
                entry_count: 0,
                pending_header_sync: 0,
                version: WAL_VERSION,
                chain_tip: initial_chain(),
            })
        }
    }

    /// Append a save entry to the WAL.
    ///
    /// Serializes the entry into a single buffer before writing to minimize
    /// syscall count (1 write() vs ~8 previously). The header entry_count is
    /// updated every GROUP_COMMIT_SIZE entries rather than on every write,
    /// eliminating 3 lseek() + 1 write() per entry (arXiv:2507.13062).
    ///
    /// Crash safety: `read_entries` reads until EOF and handles stale header
    /// counts, so deferred header updates do not compromise recovery.
    pub fn append_save(&mut self, entry: &WalEntry) -> Result<(), MemoryError> {
        let emb_len = entry.embedding.len();
        let mut buf = Vec::with_capacity(
            1 + 8 + // type + timestamp
            4 + entry.chunk.len() +
            4 + emb_len * 4 +
            4 + entry.source_channel.len() +
            4 + entry.session_id.len() +
            4 + entry.tags.len(),
        );
        buf.push(WalEntryType::Save as u8);
        buf.extend_from_slice(&entry.timestamp.to_le_bytes());
        serialize_str(&mut buf, &entry.chunk);
        buf.extend_from_slice(&(emb_len as u32).to_le_bytes());
        for &val in &entry.embedding {
            buf.extend_from_slice(&val.to_le_bytes());
        }
        serialize_str(&mut buf, &entry.source_channel);
        serialize_str(&mut buf, &entry.session_id);
        serialize_str(&mut buf, &entry.tags);
        self.write_entry_bytes(&buf)?;

        self.entry_count += 1;
        self.pending_header_sync += 1;
        if self.pending_header_sync >= GROUP_COMMIT_SIZE {
            self.write_entry_count()?;
        }
        Ok(())
    }

    /// Append a tombstone entry (deletion).
    pub fn append_tombstone(&mut self, index: usize, timestamp: f64) -> Result<(), MemoryError> {
        let mut buf = [0u8; 1 + 8 + 4]; // type + timestamp + index
        buf[0] = WalEntryType::Tombstone as u8;
        buf[1..9].copy_from_slice(&timestamp.to_le_bytes());
        buf[9..13].copy_from_slice(&(index as u32).to_le_bytes());
        self.write_entry_bytes(&buf)?;

        self.entry_count += 1;
        self.pending_header_sync += 1;
        if self.pending_header_sync >= GROUP_COMMIT_SIZE {
            self.write_entry_count()?;
        }
        Ok(())
    }

    /// Write one entry's already-serialized bytes, appending the chained hash
    /// trailer when this file is in the current (v2) format.
    fn write_entry_bytes(&mut self, buf: &[u8]) -> Result<(), MemoryError> {
        let next_tip = if self.version == WAL_VERSION_V2 {
            Some(chain_next(&self.chain_tip, buf))
        } else {
            None
        };
        let f = self
            .file
            .as_mut()
            .ok_or_else(|| MemoryError::Io(std::io::Error::other("WAL file not open")))?;
        f.write_all(buf)?;
        if let Some(tip) = next_tip {
            f.write_all(&tip)?;
            self.chain_tip = tip;
        }
        Ok(())
    }

    /// Read all entries from the WAL (for replay on open).
    ///
    /// Reads until EOF — the header `entry_count` is used only for pre-allocation
    /// (and may be stale if written with deferred group-commit updates). This
    /// tolerates truncated files (crash mid-write) and stale header counts
    /// (crash before the next group-commit header sync). For a version-2 file,
    /// a *complete* entry whose chain hash doesn't match is real tampering or
    /// corruption, not truncation, and fails closed with
    /// [`MemoryError::Integrity`] — see the module doc's documented limitation
    /// on what this chain hash does and doesn't defend against.
    pub fn read_entries(path: &Path) -> Result<Vec<WalEntry>, MemoryError> {
        if !path.exists() {
            return Ok(Vec::new());
        }
        let mut f = File::open(path)?;
        // Read header
        let mut header = [0u8; 9];
        f.read_exact(&mut header)?;
        if header[0..4] != WAL_MAGIC {
            return Err(MemoryError::Schema("invalid WAL magic bytes".into()));
        }
        let version = header[4];
        if version != WAL_VERSION_V1 && version != WAL_VERSION_V2 {
            return Err(MemoryError::Schema(format!(
                "unsupported WAL version {version}"
            )));
        }
        Ok(parse_entries(&mut f, version)?.0)
    }

    /// Truncate the WAL (after merge into .h5). Always recreates as the
    /// current (v2) format, so a legacy v1 file naturally upgrades the next
    /// time its entries are merged and the WAL is rotated.
    pub fn truncate(&mut self) -> Result<(), MemoryError> {
        // Close existing handle and recreate
        self.file = None;
        let mut f = File::create(&self.path)?;
        f.write_all(&WAL_MAGIC)?;
        f.write_all(&[WAL_VERSION])?;
        f.write_all(&0u32.to_le_bytes())?;
        f.flush()?;
        self.file = Some(f);
        self.entry_count = 0;
        self.pending_header_sync = 0;
        self.version = WAL_VERSION;
        self.chain_tip = initial_chain();
        Ok(())
    }

    /// Number of pending entries.
    pub fn pending_count(&self) -> u32 {
        self.entry_count
    }

    /// Is the WAL empty?
    pub fn is_empty(&self) -> bool {
        self.entry_count == 0
    }

    /// Update the entry_count in the header (seek to offset 5, write u32 LE).
    fn write_entry_count(&mut self) -> Result<(), MemoryError> {
        let f = self
            .file
            .as_mut()
            .ok_or_else(|| MemoryError::Io(std::io::Error::other("WAL file not open")))?;
        let pos = f.stream_position()?;
        f.seek(SeekFrom::Start(5))?;
        f.write_all(&self.entry_count.to_le_bytes())?;
        f.seek(SeekFrom::Start(pos))?;
        self.pending_header_sync = 0;
        Ok(())
    }
}

/// Parse entries from a freshly-opened WAL file positioned right after the
/// 9-byte header. Shared by [`WalFile::read_entries`] (discards the returned
/// chain tip) and [`WalFile::open`] (needs the tip to resume appending onto
/// an existing v2 file's chain).
///
/// Tolerates a partially written last entry (crash mid-write) exactly as
/// before: any read failure while parsing an entry's fields — including, for
/// v2, its trailing chain hash — stops the loop and returns what was
/// successfully parsed so far, with no error. A entry whose fields (and, for
/// v2, trailing hash) were read *in full* but whose chain hash doesn't match
/// is a different case: those bytes are genuinely present and malformed, not
/// missing, so it is reported as [`MemoryError::Integrity`] instead of being
/// silently dropped like a truncation would be.
fn parse_entries(
    f: &mut File,
    version: u8,
) -> Result<(Vec<WalEntry>, [u8; CHAIN_HASH_SIZE]), MemoryError> {
    let mut entries = Vec::new();
    let mut chain_tip = initial_chain();

    // A field read can fail two structurally different ways: an I/O error
    // (`read_exact` ran out of bytes -- genuine crash-mid-write truncation,
    // always tolerable), or a successful read whose bytes don't decode (only
    // possible for `read_len_prefixed_str`'s UTF-8 check, since a genuine
    // write always emits valid UTF-8 -- this means the bytes are fully
    // present but corrupt/tampered, not missing). For v2, only the latter is
    // reported as `Integrity`; v1 has no such promise and keeps the original
    // fully-tolerant behavior for backward compatibility.
    macro_rules! field {
        ($result:expr, $entry_start:expr) => {
            match $result {
                Ok(v) => v,
                Err(MemoryError::Io(_)) => break,
                Err(e) if version == WAL_VERSION_V2 => {
                    return Err(MemoryError::Integrity(format!(
                        "WAL entry at offset {} is corrupt (not truncated -- the bytes were \
                         fully present but failed to decode): {e}",
                        $entry_start
                    )));
                }
                Err(_) => break,
            }
        };
    }

    loop {
        let entry_start = f.stream_position()?;

        // Read entry type — EOF here is normal end-of-log, not an error.
        let mut type_buf = [0u8; 1];
        if f.read_exact(&mut type_buf).is_err() {
            break;
        }
        let Some(entry_type) = WalEntryType::from_u8(type_buf[0]) else {
            break;
        };

        let mut ts_buf = [0u8; 8];
        if f.read_exact(&mut ts_buf).is_err() {
            break;
        }
        let timestamp = f64::from_le_bytes(ts_buf);

        let entry = match entry_type {
            WalEntryType::Save => {
                let chunk = field!(read_len_prefixed_str(f), entry_start);
                let embedding = field!(read_embedding(f), entry_start);
                let source_channel = field!(read_len_prefixed_str(f), entry_start);
                let session_id = field!(read_len_prefixed_str(f), entry_start);
                let tags = field!(read_len_prefixed_str(f), entry_start);
                WalEntry {
                    entry_type,
                    timestamp,
                    chunk,
                    embedding,
                    source_channel,
                    session_id,
                    tags,
                    tombstone_index: None,
                }
            }
            WalEntryType::Tombstone => {
                let mut idx_buf = [0u8; 4];
                if f.read_exact(&mut idx_buf).is_err() {
                    break;
                }
                let idx = u32::from_le_bytes(idx_buf) as usize;
                WalEntry {
                    entry_type,
                    timestamp,
                    chunk: String::new(),
                    embedding: Vec::new(),
                    source_channel: String::new(),
                    session_id: String::new(),
                    tags: String::new(),
                    tombstone_index: Some(idx),
                }
            }
            WalEntryType::ActivationUpdate => {
                // Reserved for future use; no additional fields, no cache effect.
                continue;
            }
        };

        if version == WAL_VERSION_V2 {
            let entry_end = f.stream_position()?;
            let entry_len = (entry_end - entry_start) as usize;
            let mut trailer = [0u8; CHAIN_HASH_SIZE];
            if f.read_exact(&mut trailer).is_err() {
                // Fields were complete but the trailer wasn't -- crash mid-
                // trailer-write. Same tolerance class as any other partial
                // write: drop this entry, stop here, no error.
                break;
            }
            let mut raw = vec![0u8; entry_len];
            f.seek(SeekFrom::Start(entry_start))?;
            f.read_exact(&mut raw)?;
            f.seek(SeekFrom::Start(entry_end + CHAIN_HASH_SIZE as u64))?;

            let expected = chain_next(&chain_tip, &raw);
            if expected != trailer {
                return Err(MemoryError::Integrity(format!(
                    "WAL entry at offset {entry_start} failed chain-hash verification -- \
                     the .h5.wal sidecar was tampered with or corrupted"
                )));
            }
            chain_tip = expected;
        }

        entries.push(entry);
    }

    Ok((entries, chain_tip))
}

/// Replay WAL entries into a MemoryCache.
pub fn replay_into_cache(entries: &[WalEntry], cache: &mut crate::cache::MemoryCache) {
    for entry in entries {
        match entry.entry_type {
            WalEntryType::Save => {
                cache.push(
                    entry.chunk.clone(),
                    entry.embedding.clone(),
                    entry.source_channel.clone(),
                    entry.timestamp,
                    entry.session_id.clone(),
                    entry.tags.clone(),
                );
            }
            WalEntryType::Tombstone => {
                if let Some(idx) = entry.tombstone_index {
                    cache.mark_deleted(idx);
                }
            }
            WalEntryType::ActivationUpdate => {}
        }
    }
}

// --- Binary helpers ---

/// Serialize a length-prefixed string into an in-memory buffer (zero syscalls).
fn serialize_str(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(bytes);
}

fn read_len_prefixed_str(f: &mut File) -> Result<String, MemoryError> {
    let mut len_buf = [0u8; 4];
    f.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| MemoryError::Schema(format!("invalid UTF-8 in WAL: {e}")))
}

fn read_embedding(f: &mut File) -> Result<Vec<f32>, MemoryError> {
    let mut len_buf = [0u8; 4];
    f.read_exact(&mut len_buf)?;
    let count = u32::from_le_bytes(len_buf) as usize;
    let mut vals = Vec::with_capacity(count);
    for _ in 0..count {
        let mut val_buf = [0u8; 4];
        f.read_exact(&mut val_buf)?;
        vals.push(f32::from_le_bytes(val_buf));
    }
    Ok(vals)
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_wal_entry(chunk: &str, embedding: &[f32]) -> WalEntry {
        WalEntry {
            entry_type: WalEntryType::Save,
            timestamp: 1234567.89,
            chunk: chunk.to_string(),
            embedding: embedding.to_vec(),
            source_channel: "test-channel".to_string(),
            session_id: "sess-001".to_string(),
            tags: "tag1,tag2".to_string(),
            tombstone_index: None,
        }
    }

    #[test]
    fn test_wal_create_and_header() {
        let dir = TempDir::new().unwrap();
        let wal_path = dir.path().join("test.h5.wal");
        let wal = WalFile::open(&wal_path).unwrap();
        assert_eq!(wal.pending_count(), 0);
        assert!(wal.is_empty());
        drop(wal);

        // Verify raw bytes on disk
        let bytes = std::fs::read(&wal_path).unwrap();
        assert_eq!(&bytes[0..4], &WAL_MAGIC);
        assert_eq!(bytes[4], WAL_VERSION);
        assert_eq!(&bytes[5..9], &0u32.to_le_bytes());
    }

    #[test]
    fn test_wal_append_and_read() {
        let dir = TempDir::new().unwrap();
        let wal_path = dir.path().join("test.h5.wal");
        {
            let mut wal = WalFile::open(&wal_path).unwrap();
            wal.append_save(&make_wal_entry("first", &[1.0, 2.0]))
                .unwrap();
            wal.append_save(&make_wal_entry("second", &[3.0, 4.0]))
                .unwrap();
            wal.append_save(&make_wal_entry("third", &[5.0, 6.0]))
                .unwrap();
            assert_eq!(wal.pending_count(), 3);
        }

        let entries = WalFile::read_entries(&wal_path).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].chunk, "first");
        assert_eq!(entries[0].embedding, vec![1.0, 2.0]);
        assert_eq!(entries[1].chunk, "second");
        assert_eq!(entries[2].chunk, "third");
        assert_eq!(entries[2].embedding, vec![5.0, 6.0]);
    }

    #[test]
    fn test_wal_truncate() {
        let dir = TempDir::new().unwrap();
        let wal_path = dir.path().join("test.h5.wal");
        let mut wal = WalFile::open(&wal_path).unwrap();
        for i in 0..5 {
            wal.append_save(&make_wal_entry(&format!("entry {i}"), &[i as f32]))
                .unwrap();
        }
        assert_eq!(wal.pending_count(), 5);

        wal.truncate().unwrap();
        assert_eq!(wal.pending_count(), 0);
        assert!(wal.is_empty());

        let entries = WalFile::read_entries(&wal_path).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_wal_append_tombstone() {
        let dir = TempDir::new().unwrap();
        let wal_path = dir.path().join("test.h5.wal");
        {
            let mut wal = WalFile::open(&wal_path).unwrap();
            wal.append_tombstone(42, 9999.0).unwrap();
            assert_eq!(wal.pending_count(), 1);
        }

        let entries = WalFile::read_entries(&wal_path).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].entry_type, WalEntryType::Tombstone);
        assert_eq!(entries[0].tombstone_index, Some(42));
        assert!((entries[0].timestamp - 9999.0).abs() < 1e-6);
    }

    #[test]
    fn test_wal_binary_roundtrip() {
        let dir = TempDir::new().unwrap();
        let wal_path = dir.path().join("test.h5.wal");
        let unicode_chunk = "Hello 世界! 🌍 émojis & ünïcödé";
        let embedding = vec![0.1, -0.2, 3.14159, f32::MAX, f32::MIN_POSITIVE];
        {
            let mut wal = WalFile::open(&wal_path).unwrap();
            let entry = WalEntry {
                entry_type: WalEntryType::Save,
                timestamp: std::f64::consts::PI,
                chunk: unicode_chunk.to_string(),
                embedding: embedding.clone(),
                source_channel: "channel/with/slashes".to_string(),
                session_id: "sess-öö-123".to_string(),
                tags: "α,β,γ".to_string(),
                tombstone_index: None,
            };
            wal.append_save(&entry).unwrap();
        }

        let entries = WalFile::read_entries(&wal_path).unwrap();
        assert_eq!(entries.len(), 1);
        let e = &entries[0];
        assert_eq!(e.entry_type, WalEntryType::Save);
        assert!((e.timestamp - std::f64::consts::PI).abs() < 1e-15);
        assert_eq!(e.chunk, unicode_chunk);
        assert_eq!(e.embedding, embedding);
        assert_eq!(e.source_channel, "channel/with/slashes");
        assert_eq!(e.session_id, "sess-öö-123");
        assert_eq!(e.tags, "α,β,γ");
    }

    #[test]
    fn test_wal_empty_on_create() {
        let dir = TempDir::new().unwrap();
        let wal_path = dir.path().join("test.h5.wal");
        let wal = WalFile::open(&wal_path).unwrap();
        assert_eq!(wal.pending_count(), 0);
        assert!(wal.is_empty());
    }

    // --- Integration tests (WAL + HDF5Memory) ---

    use crate::{AgentMemory, HDF5Memory, MemoryConfig, MemoryEntry};

    fn make_config(dir: &TempDir) -> MemoryConfig {
        let mut config = MemoryConfig::new(dir.path().join("test.h5"), "agent-test", 4);
        config.wal_enabled = true;
        config
    }

    fn make_entry(chunk: &str, embedding: &[f32]) -> MemoryEntry {
        MemoryEntry {
            chunk: chunk.to_string(),
            embedding: embedding.to_vec(),
            source_channel: "test".to_string(),
            timestamp: 1000000.0,
            session_id: "session-1".to_string(),
            tags: "tag1,tag2".to_string(),
        }
    }

    #[test]
    fn test_save_with_wal() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let h5_path = config.path.clone();
        let mut mem = HDF5Memory::create(config).unwrap();

        // Get initial .h5 size (empty file)
        let initial_size = std::fs::metadata(&h5_path).unwrap().len();

        mem.save(make_entry("a", &[1.0, 0.0, 0.0, 0.0])).unwrap();
        mem.save(make_entry("b", &[0.0, 1.0, 0.0, 0.0])).unwrap();
        mem.save(make_entry("c", &[0.0, 0.0, 1.0, 0.0])).unwrap();

        // Cache has 3 entries
        assert_eq!(mem.count(), 3);

        // .h5 file should NOT have been updated (still initial size)
        let after_size = std::fs::metadata(&h5_path).unwrap().len();
        assert_eq!(
            initial_size, after_size,
            ".h5 should not grow with WAL enabled"
        );

        // .wal file should exist
        let wal_path = h5_path.with_extension("h5.wal");
        assert!(wal_path.exists(), ".wal file should exist");
        assert_eq!(mem.wal_pending_count(), 3);
    }

    #[test]
    fn test_wal_auto_merge() {
        let dir = TempDir::new().unwrap();
        let mut config = make_config(&dir);
        config.wal_max_entries = 5;
        let h5_path = config.path.clone();
        let mut mem = HDF5Memory::create(config).unwrap();

        // Save 5 entries (at threshold but not over)
        for i in 0..5 {
            mem.save(make_entry(
                &format!("entry {i}"),
                &[i as f32, 0.0, 0.0, 0.0],
            ))
            .unwrap();
        }
        // WAL should still have 5 pending (not yet merged, threshold is >=)
        assert_eq!(mem.wal_pending_count(), 5);

        // 6th entry triggers auto-merge (pending > wal_max_entries)
        mem.save(make_entry("entry 5", &[5.0, 0.0, 0.0, 0.0]))
            .unwrap();

        // After auto-merge: WAL should be empty, cache still has all entries
        assert_eq!(mem.wal_pending_count(), 0);
        assert_eq!(mem.count(), 6);

        // WAL file should be truncated (only header)
        let wal_path = h5_path.with_extension("h5.wal");
        let entries = WalFile::read_entries(&wal_path).unwrap();
        assert!(entries.is_empty(), "WAL should be empty after auto-merge");
    }

    #[test]
    fn test_wal_flush_explicit() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let h5_path = config.path.clone();
        let mut mem = HDF5Memory::create(config).unwrap();

        mem.save(make_entry("a", &[1.0, 0.0, 0.0, 0.0])).unwrap();
        mem.save(make_entry("b", &[0.0, 1.0, 0.0, 0.0])).unwrap();
        mem.save(make_entry("c", &[0.0, 0.0, 1.0, 0.0])).unwrap();

        assert_eq!(mem.wal_pending_count(), 3);

        mem.flush_wal().unwrap();

        // WAL should be empty after explicit flush
        assert_eq!(mem.wal_pending_count(), 0);
        // Cache should still have 3
        assert_eq!(mem.count(), 3);
        // WAL file on disk should be empty
        let wal_path = h5_path.with_extension("h5.wal");
        let entries = WalFile::read_entries(&wal_path).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_wal_replay_on_open() {
        // Test WAL replay using read_entries + replay_into_cache directly,
        // since the HDF5 read path is independent of WAL functionality.
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let h5_path = config.path.clone();

        {
            let mut mem = HDF5Memory::create(config).unwrap();
            mem.save(make_entry("replay-a", &[1.0, 0.0, 0.0, 0.0]))
                .unwrap();
            mem.save(make_entry("replay-b", &[0.0, 1.0, 0.0, 0.0]))
                .unwrap();
            mem.save(make_entry("replay-c", &[0.0, 0.0, 1.0, 0.0]))
                .unwrap();
            assert_eq!(mem.wal_pending_count(), 3);
            // Drop without flushing — WAL has 3 entries
        }

        // Verify WAL file has the entries
        let wal_path = h5_path.with_extension("h5.wal");
        assert!(wal_path.exists());
        let entries = WalFile::read_entries(&wal_path).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].chunk, "replay-a");
        assert_eq!(entries[1].chunk, "replay-b");
        assert_eq!(entries[2].chunk, "replay-c");

        // Replay into a fresh cache (simulates what open() does)
        let mut cache = crate::cache::MemoryCache::new(4);
        super::replay_into_cache(&entries, &mut cache);
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.chunks[0], "replay-a");
        assert_eq!(cache.chunks[1], "replay-b");
        assert_eq!(cache.chunks[2], "replay-c");
        assert_eq!(cache.count_active(), 3);
    }

    #[test]
    fn test_tick_session_merges_wal() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let h5_path = config.path.clone();
        let mut mem = HDF5Memory::create(config).unwrap();

        mem.save(make_entry("tick-a", &[1.0, 0.0, 0.0, 0.0]))
            .unwrap();
        mem.save(make_entry("tick-b", &[0.0, 1.0, 0.0, 0.0]))
            .unwrap();
        mem.save(make_entry("tick-c", &[0.0, 0.0, 1.0, 0.0]))
            .unwrap();

        assert_eq!(mem.wal_pending_count(), 3);

        mem.tick_session().unwrap();

        // WAL should be empty after tick_session merges
        assert_eq!(mem.wal_pending_count(), 0);
        // Cache should still have 3 entries
        assert_eq!(mem.count(), 3);
        // WAL file on disk should be empty
        let wal_path = h5_path.with_extension("h5.wal");
        let entries = WalFile::read_entries(&wal_path).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_wal_header_only_with_nonzero_count() {
        // Simulate crash: header says 5 entries but file is only 9 bytes (header only).
        // This is the exact scenario from the bug report — process exits before WAL
        // flushes, leaving a stale entry_count in the header.
        let dir = TempDir::new().unwrap();
        let wal_path = dir.path().join("corrupted.h5.wal");
        {
            let mut f = File::create(&wal_path).unwrap();
            f.write_all(&WAL_MAGIC).unwrap();
            f.write_all(&[WAL_VERSION]).unwrap();
            f.write_all(&5u32.to_le_bytes()).unwrap(); // claims 5 entries
            f.flush().unwrap();
        }

        // Should NOT error — should return empty vec
        let entries = WalFile::read_entries(&wal_path).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_wal_partial_truncation() {
        // Write 2 valid entries, then corrupt the header to claim 5.
        // read_entries should return the 2 valid entries, not error.
        let dir = TempDir::new().unwrap();
        let wal_path = dir.path().join("partial.h5.wal");
        {
            let mut wal = WalFile::open(&wal_path).unwrap();
            wal.append_save(&make_wal_entry("first", &[1.0, 2.0]))
                .unwrap();
            wal.append_save(&make_wal_entry("second", &[3.0, 4.0]))
                .unwrap();
            assert_eq!(wal.pending_count(), 2);
        }

        // Corrupt the header: overwrite entry_count to 5
        {
            let mut f = OpenOptions::new().write(true).open(&wal_path).unwrap();
            f.seek(SeekFrom::Start(5)).unwrap();
            f.write_all(&5u32.to_le_bytes()).unwrap();
            f.flush().unwrap();
        }

        // Should recover the 2 valid entries, not fail
        let entries = WalFile::read_entries(&wal_path).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].chunk, "first");
        assert_eq!(entries[1].chunk, "second");
    }

    #[test]
    fn test_wal_invalid_version_rejected() {
        let dir = TempDir::new().unwrap();
        let wal_path = dir.path().join("badversion.h5.wal");
        {
            let mut f = File::create(&wal_path).unwrap();
            f.write_all(&WAL_MAGIC).unwrap();
            f.write_all(&[0xFF]).unwrap(); // bad version
            f.write_all(&0u32.to_le_bytes()).unwrap();
            f.flush().unwrap();
        }

        let result = WalFile::read_entries(&wal_path);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("unsupported WAL version"), "got: {err}");
    }

    #[test]
    fn test_wal_disabled() {
        let dir = TempDir::new().unwrap();
        let mut config = make_config(&dir);
        config.wal_enabled = false;
        let h5_path = config.path.clone();
        let mut mem = HDF5Memory::create(config).unwrap();

        mem.save(make_entry("no-wal", &[1.0, 0.0, 0.0, 0.0]))
            .unwrap();

        // With WAL disabled, save goes through flush() directly (old behavior)
        assert_eq!(mem.count(), 1);
        // No WAL file should exist
        let wal_path = h5_path.with_extension("h5.wal");
        assert!(!wal_path.exists(), "no .wal file when WAL disabled");
        assert_eq!(mem.wal_pending_count(), 0);
    }

    // --- P2.4 red-team follow-up: WAL chain-hash integrity (v2) ---

    #[test]
    fn test_wal_v2_new_files_are_chain_hashed() {
        let dir = TempDir::new().unwrap();
        let wal_path = dir.path().join("test.h5.wal");
        {
            let mut wal = WalFile::open(&wal_path).unwrap();
            wal.append_save(&make_wal_entry("a", &[1.0])).unwrap();
            wal.append_save(&make_wal_entry("b", &[2.0])).unwrap();
        }
        let bytes = std::fs::read(&wal_path).unwrap();
        assert_eq!(bytes[4], WAL_VERSION_V2, "new WAL files must be v2");

        // Honest round-trip still verifies and returns both entries.
        let entries = WalFile::read_entries(&wal_path).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].chunk, "a");
        assert_eq!(entries[1].chunk, "b");
    }

    #[test]
    fn test_wal_v2_detects_tampered_entry() {
        let dir = TempDir::new().unwrap();
        let wal_path = dir.path().join("tampered.h5.wal");
        {
            let mut wal = WalFile::open(&wal_path).unwrap();
            wal.append_save(&make_wal_entry("first", &[1.0, 2.0]))
                .unwrap();
            wal.append_save(&make_wal_entry("second", &[3.0, 4.0]))
                .unwrap();
        }

        // Raw storage-level tamper: flip a byte inside the first entry's
        // chunk text (well past the 9-byte header), leaving its trailing
        // chain hash untouched -- exactly the T1a-style move, against the WAL
        // instead of the .h5.
        let mut bytes = std::fs::read(&wal_path).unwrap();
        let needle = b"first";
        let pos = bytes
            .windows(needle.len())
            .position(|w| w == needle)
            .expect("chunk text present");
        bytes[pos] ^= 0xFF;
        std::fs::write(&wal_path, &bytes).unwrap();

        let result = WalFile::read_entries(&wal_path);
        assert!(
            matches!(result, Err(MemoryError::Integrity(_))),
            "tampered entry must fail closed, got {result:?}"
        );
    }

    #[test]
    fn test_wal_v2_forged_entry_without_chain_is_not_replayed() {
        // The exact attack scenario from the P2.4 red-team sweep: an attacker
        // with write access to the sidecar appends a well-formed Save entry
        // (correct type byte, timestamp, length-prefixed fields) hoping it
        // gets replayed as a genuine memory. Without also computing the
        // correct chain hash (which requires reading and re-hashing every
        // prior entry), the forged entry must never reach the returned
        // entries -- whether that surfaces as an explicit rejection or a
        // silent drop, the forged content must not be replayed into the cache.
        let dir = TempDir::new().unwrap();
        let wal_path = dir.path().join("forged.h5.wal");
        {
            let mut wal = WalFile::open(&wal_path).unwrap();
            wal.append_save(&make_wal_entry("genuine", &[1.0])).unwrap();
        }

        // Manually append a well-formed Save entry's wire bytes with NO
        // trailing chain hash -- the attacker doesn't know the chaining
        // scheme exists, so they reproduce only the pre-existing (v1) format.
        let forged = make_wal_entry("forged-memory", &[9.9]);
        let mut buf = Vec::new();
        buf.push(WalEntryType::Save as u8);
        buf.extend_from_slice(&forged.timestamp.to_le_bytes());
        serialize_str(&mut buf, &forged.chunk);
        buf.extend_from_slice(&(forged.embedding.len() as u32).to_le_bytes());
        for &v in &forged.embedding {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        serialize_str(&mut buf, &forged.source_channel);
        serialize_str(&mut buf, &forged.session_id);
        serialize_str(&mut buf, &forged.tags);
        {
            let mut f = OpenOptions::new().append(true).open(&wal_path).unwrap();
            f.write_all(&buf).unwrap();
        }

        let entries = WalFile::read_entries(&wal_path).unwrap();
        assert!(
            entries.iter().all(|e| e.chunk != "forged-memory"),
            "forged entry without a valid chain hash must never be replayed"
        );
        assert_eq!(entries.len(), 1, "only the genuine entry survives");
        assert_eq!(entries[0].chunk, "genuine");
    }

    #[test]
    fn test_wal_v2_reopen_continues_chain() {
        // open() on an existing v2 file must recover the chain tip so further
        // appends verify correctly, and so a later tamper is still detected.
        let dir = TempDir::new().unwrap();
        let wal_path = dir.path().join("reopen.h5.wal");
        {
            let mut wal = WalFile::open(&wal_path).unwrap();
            wal.append_save(&make_wal_entry("first-entry-payload", &[1.0]))
                .unwrap();
        }
        {
            let mut wal = WalFile::open(&wal_path).unwrap();
            wal.append_save(&make_wal_entry("second-entry-payload", &[2.0]))
                .unwrap();
        }

        let entries = WalFile::read_entries(&wal_path).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].chunk, "first-entry-payload");
        assert_eq!(entries[1].chunk, "second-entry-payload");

        // Tamper the second entry after both appends; chain verification
        // must still catch it even though it was written across two opens.
        let mut bytes = std::fs::read(&wal_path).unwrap();
        let needle = b"second-entry-payload";
        let pos = bytes
            .windows(needle.len())
            .position(|w| w == needle)
            .expect("chunk text present");
        bytes[pos] ^= 0xFF;
        std::fs::write(&wal_path, &bytes).unwrap();

        assert!(matches!(
            WalFile::read_entries(&wal_path),
            Err(MemoryError::Integrity(_))
        ));
    }

    #[test]
    fn test_wal_v1_legacy_file_still_reads_unauthenticated() {
        // A pre-existing v1 WAL (no chain hash) must still be readable for
        // backward compatibility -- old sidecars aren't rejected outright.
        let dir = TempDir::new().unwrap();
        let wal_path = dir.path().join("legacy.h5.wal");
        {
            let mut f = File::create(&wal_path).unwrap();
            f.write_all(&WAL_MAGIC).unwrap();
            f.write_all(&[WAL_VERSION_V1]).unwrap();
            f.write_all(&0u32.to_le_bytes()).unwrap();
            let mut buf = Vec::new();
            let e = make_wal_entry("legacy-entry", &[1.0, 2.0]);
            buf.push(WalEntryType::Save as u8);
            buf.extend_from_slice(&e.timestamp.to_le_bytes());
            serialize_str(&mut buf, &e.chunk);
            buf.extend_from_slice(&(e.embedding.len() as u32).to_le_bytes());
            for &v in &e.embedding {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            serialize_str(&mut buf, &e.source_channel);
            serialize_str(&mut buf, &e.session_id);
            serialize_str(&mut buf, &e.tags);
            f.write_all(&buf).unwrap();
        }

        let entries = WalFile::read_entries(&wal_path).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].chunk, "legacy-entry");
    }
}
