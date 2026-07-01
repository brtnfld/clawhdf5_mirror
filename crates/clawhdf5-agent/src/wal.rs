//! Write-Ahead Log (WAL) for edgehdf5 agent memory.
//!
//! Binary WAL format alongside the main .h5 file enables fast append-only
//! writes without rewriting the entire HDF5 file on every save.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::MemoryError;

const WAL_MAGIC: [u8; 4] = [0x45, 0x48, 0x57, 0x4C]; // "EHWL"
const WAL_VERSION: u8 = 1;

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
}

impl WalFile {
    /// Open or create a WAL file. If it exists, read the header and entry count.
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
            if ver[0] != WAL_VERSION {
                return Err(MemoryError::Schema(format!(
                    "unsupported WAL version {}",
                    ver[0]
                )));
            }
            let mut count_buf = [0u8; 4];
            f.read_exact(&mut count_buf)?;
            let entry_count = u32::from_le_bytes(count_buf);
            // Seek to end for appending
            f.seek(SeekFrom::End(0))?;
            Ok(Self {
                path: path.to_path_buf(),
                file: Some(f),
                entry_count,
                pending_header_sync: 0,
            })
        } else {
            // Create new WAL
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

        let f = self
            .file
            .as_mut()
            .ok_or_else(|| MemoryError::Io(std::io::Error::other("WAL file not open")))?;
        f.write_all(&buf)?;

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

        let f = self
            .file
            .as_mut()
            .ok_or_else(|| MemoryError::Io(std::io::Error::other("WAL file not open")))?;
        f.write_all(&buf)?;

        self.entry_count += 1;
        self.pending_header_sync += 1;
        if self.pending_header_sync >= GROUP_COMMIT_SIZE {
            self.write_entry_count()?;
        }
        Ok(())
    }

    /// Read all entries from the WAL (for replay on open).
    ///
    /// Reads until EOF — the header `entry_count` is used only for pre-allocation
    /// (and may be stale if written with deferred group-commit updates). This
    /// tolerates both truncated files (crash mid-write) and stale header counts
    /// (crash before the next group-commit header sync).
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
        if header[4] != WAL_VERSION {
            return Err(MemoryError::Schema(format!(
                "unsupported WAL version {}",
                header[4]
            )));
        }
        // entry_count is a pre-allocation hint only — we read until EOF.
        let entry_count_hint = u32::from_le_bytes([header[5], header[6], header[7], header[8]]);
        let mut entries = Vec::with_capacity(entry_count_hint as usize);

        loop {
            // Read entry type — EOF here is normal end-of-log, not an error
            let mut type_buf = [0u8; 1];
            if f.read_exact(&mut type_buf).is_err() {
                break;
            }
            let entry_type = match WalEntryType::from_u8(type_buf[0]) {
                Some(et) => et,
                None => break,
            };

            let mut ts_buf = [0u8; 8];
            if f.read_exact(&mut ts_buf).is_err() {
                break;
            }
            let timestamp = f64::from_le_bytes(ts_buf);

            match entry_type {
                WalEntryType::Save => {
                    let Ok(chunk) = read_len_prefixed_str(&mut f) else {
                        break;
                    };
                    let Ok(embedding) = read_embedding(&mut f) else {
                        break;
                    };
                    let Ok(source_channel) = read_len_prefixed_str(&mut f) else {
                        break;
                    };
                    let Ok(session_id) = read_len_prefixed_str(&mut f) else {
                        break;
                    };
                    let Ok(tags) = read_len_prefixed_str(&mut f) else {
                        break;
                    };
                    entries.push(WalEntry {
                        entry_type,
                        timestamp,
                        chunk,
                        embedding,
                        source_channel,
                        session_id,
                        tags,
                        tombstone_index: None,
                    });
                }
                WalEntryType::Tombstone => {
                    let mut idx_buf = [0u8; 4];
                    if f.read_exact(&mut idx_buf).is_err() {
                        break;
                    }
                    let idx = u32::from_le_bytes(idx_buf) as usize;
                    entries.push(WalEntry {
                        entry_type,
                        timestamp,
                        chunk: String::new(),
                        embedding: Vec::new(),
                        source_channel: String::new(),
                        session_id: String::new(),
                        tags: String::new(),
                        tombstone_index: Some(idx),
                    });
                }
                WalEntryType::ActivationUpdate => {
                    // Reserved for future use
                }
            }
        }
        Ok(entries)
    }

    /// Truncate the WAL (after merge into .h5).
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
}
