//! A real, `std::fs`-backed [`MerkleWriteSink`].
//!
//! Every existing implementation of this trait (`CrashSink` in
//! `clawhdf5-filters/tests/crash_vs_tamper_matrix.rs`, `RecordingSink` in
//! `write_order.rs`'s own tests) is a test double that never touches durable
//! storage. This is the first implementation that writes to real files and
//! calls `File::sync_all()` — so Stage 5 of the demo exercises the P2.2b
//! three-step write order against actual fsync'd storage, not a mock.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use clawhdf5_filters::{MerkleWriteSink, WriteStep};

/// Error from [`FileMerkleSink`]: either a real I/O failure, or a
/// deliberately injected "crash" at a chosen [`WriteStep`] (see
/// [`FileMerkleSink::with_fail_at`]) — the write is refused *before* touching
/// disk, standing in for a process death at that exact point.
#[derive(Debug)]
pub enum FileSinkError {
    Io(std::io::Error),
    SimulatedCrash(WriteStep),
}

impl std::fmt::Display for FileSinkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FileSinkError::Io(e) => write!(f, "I/O error: {e}"),
            FileSinkError::SimulatedCrash(step) => {
                write!(f, "simulated crash at {step} step")
            }
        }
    }
}

impl std::error::Error for FileSinkError {}

/// Writes the three P2.2b artifacts (chunk data, companion nodes, root
/// attribute) as three separate files in a directory, syncing each to stable
/// storage before the next step is allowed to proceed.
///
/// `root.bin`'s layout is `root(32) || companion_hash(32) || dataset_version
/// (8, little-endian)` — a minimal stand-in for the real `merkle_root` HDF5
/// attribute, since `MerkleWriteSink` only hands us the three raw fields.
pub struct FileMerkleSink {
    dir: PathBuf,
    fail_at: Option<WriteStep>,
    /// The file handle from the most recent `write_*` call, still open so
    /// `sync()` can call `sync_all()` on the exact bytes just written.
    pending: Option<File>,
}

impl FileMerkleSink {
    #[must_use]
    pub fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            fail_at: None,
            pending: None,
        }
    }

    /// A sink that refuses the write at `step` (simulating a crash there)
    /// instead of performing it — used to test the write-order's
    /// crash-consistency guarantee.
    #[must_use]
    pub fn with_fail_at(dir: PathBuf, step: WriteStep) -> Self {
        Self {
            dir,
            fail_at: Some(step),
            pending: None,
        }
    }

    #[must_use]
    pub fn chunk_path(dir: &Path, chunk_idx: u64) -> PathBuf {
        dir.join(format!("chunk_{chunk_idx}.bin"))
    }

    #[must_use]
    pub fn companion_path(dir: &Path) -> PathBuf {
        dir.join("companion.bin")
    }

    #[must_use]
    pub fn root_path(dir: &Path) -> PathBuf {
        dir.join("root.bin")
    }

    fn write_and_hold(&mut self, path: &Path, data: &[u8]) -> Result<(), FileSinkError> {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(FileSinkError::Io)?;
        file.write_all(data).map_err(FileSinkError::Io)?;
        self.pending = Some(file);
        Ok(())
    }
}

impl MerkleWriteSink for FileMerkleSink {
    type Error = FileSinkError;

    fn write_chunk_data(&mut self, chunk_idx: u64, ciphertext: &[u8]) -> Result<(), Self::Error> {
        if self.fail_at == Some(WriteStep::ChunkData) {
            return Err(FileSinkError::SimulatedCrash(WriteStep::ChunkData));
        }
        let path = Self::chunk_path(&self.dir, chunk_idx);
        self.write_and_hold(&path, ciphertext)
    }

    fn write_companion_nodes(&mut self, nodes: &[u8]) -> Result<(), Self::Error> {
        if self.fail_at == Some(WriteStep::CompanionNodes) {
            return Err(FileSinkError::SimulatedCrash(WriteStep::CompanionNodes));
        }
        let path = Self::companion_path(&self.dir);
        self.write_and_hold(&path, nodes)
    }

    fn write_root_attribute(
        &mut self,
        root: &[u8],
        companion_hash: &[u8],
        dataset_version: u64,
    ) -> Result<(), Self::Error> {
        if self.fail_at == Some(WriteStep::RootAttribute) {
            return Err(FileSinkError::SimulatedCrash(WriteStep::RootAttribute));
        }
        let mut buf = Vec::with_capacity(root.len() + companion_hash.len() + 8);
        buf.extend_from_slice(root);
        buf.extend_from_slice(companion_hash);
        buf.extend_from_slice(&dataset_version.to_le_bytes());
        let path = Self::root_path(&self.dir);
        self.write_and_hold(&path, &buf)
    }

    fn sync(&mut self) -> Result<(), Self::Error> {
        if let Some(file) = self.pending.take() {
            file.sync_all().map_err(FileSinkError::Io)?;
        }
        Ok(())
    }
}

/// Read back `root.bin`'s three fields, for reopening after a crash/restart.
///
/// Returns `None` if the file doesn't exist yet (root attribute was never
/// durably written — step 3 never completed).
#[must_use]
pub fn read_root_file(dir: &Path) -> Option<(Vec<u8>, Vec<u8>, u64)> {
    let bytes = std::fs::read(FileMerkleSink::root_path(dir)).ok()?;
    if bytes.len() != 72 {
        return None;
    }
    let root = bytes[0..32].to_vec();
    let companion_hash = bytes[32..64].to_vec();
    let dataset_version = u64::from_le_bytes(bytes[64..72].try_into().unwrap());
    Some((root, companion_hash, dataset_version))
}
