//! Three-step crash-consistent write order for committed Merkle mutations.
//!
//! S2-D2-Yr2 §4.4 ("Crash consistency") and P2.2b step 2 mandate a strict
//! ordering when committing a chunk mutation to a Merkle-protected dataset:
//!
//! 1. Write the chunk data (via the filter pipeline).
//! 2. Write the updated companion Merkle nodes (leaf hash, then path to root).
//! 3. Write the root attribute (root hash, companion-integrity hash, version).
//!
//! Each step is followed by an explicit sync so that, on a crash between any two
//! steps, the root attribute never references companion nodes that never reached
//! disk (and vice versa). The per-chunk WAL from P2.2 brackets the whole
//! sequence: the `(chunk_idx, version)` record is journaled *before* the chunk is
//! encrypted (in [`EncryptedChunkWriter::encrypt_chunk`]) and is marked committed
//! only *after* step 3 has synced. A crash in between therefore leaves the WAL
//! entry uncommitted, and [`recover`](EncryptedChunkWriter::recover) will report
//! the chunk — which `clawhdf5-format`'s `verify_chunk_with_pending` then rejects
//! with `MerkleError::NoncePending` until the write is completed.
//!
//! `clawhdf5-filters` does not depend on `clawhdf5-format`, so the Merkle
//! artifacts (companion nodes, root hash, companion-integrity hash) are passed as
//! opaque byte slices; the caller supplies them from the reconstructed tree.

use crate::chacha20_filter::{EncryptedChunkResult, EncryptedChunkWriter, EncryptedWriteError};
use std::io::{Read, Seek, Write};

/// The three ordered, durability-separated write steps of a committed mutation.
///
/// Identifies which step failed in a [`WriteOrderError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteStep {
    /// Step 1: the chunk's (encrypted) data.
    ChunkData,
    /// Step 2: the companion Merkle node array.
    CompanionNodes,
    /// Step 3: the root attribute (root hash, companion hash, version).
    RootAttribute,
}

impl core::fmt::Display for WriteStep {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            WriteStep::ChunkData => "chunk data",
            WriteStep::CompanionNodes => "companion nodes",
            WriteStep::RootAttribute => "root attribute",
        };
        f.write_str(s)
    }
}

/// A sink that performs the three ordered writes and the syncs between them.
///
/// Implementors map each method onto their storage backend (an HDF5 file, a
/// test double, etc.). All payloads are opaque bytes; ordering and syncing are
/// enforced by [`EncryptedChunkWriter::commit_with_write_order`], not by the
/// implementor.
pub trait MerkleWriteSink {
    /// Error type returned by the sink's backend.
    type Error;

    /// Step 1: persist the (encrypted) chunk data for `chunk_idx`.
    fn write_chunk_data(&mut self, chunk_idx: u64, ciphertext: &[u8]) -> Result<(), Self::Error>;

    /// Step 2: persist the flattened companion Merkle node array.
    fn write_companion_nodes(&mut self, nodes: &[u8]) -> Result<(), Self::Error>;

    /// Step 3: persist the root attribute — root hash, companion-integrity hash,
    /// and the **dataset-level version**.
    ///
    /// `dataset_version` is the strictly-per-commit counter
    /// ([`EncryptedChunkWriter::dataset_version`]), *not* the mutated chunk's
    /// per-chunk WAL version and *not* `max_version()`: per-chunk versions can
    /// regress across commits touching different chunks, and the max can tie
    /// (Remark A.13) — either would give the stateful verifier's T4 rollback
    /// check false positives or a blind spot.
    fn write_root_attribute(
        &mut self,
        root: &[u8],
        companion_hash: &[u8],
        dataset_version: u64,
    ) -> Result<(), Self::Error>;

    /// Flush all preceding writes durably to storage.
    ///
    /// Called after each of the three steps. A real implementation must reach
    /// stable storage (e.g. `File::sync_all`), not merely a userspace buffer.
    fn sync(&mut self) -> Result<(), Self::Error>;
}

/// Error from committing a chunk through the three-step write order.
#[derive(Debug)]
pub enum WriteOrderError<E> {
    /// A sink write or sync failed. `step` identifies where, so the caller can
    /// tell how far the durable state advanced before the failure.
    Sink {
        /// The step whose write or subsequent sync failed.
        step: WriteStep,
        /// The underlying sink error.
        source: E,
    },
    /// Every durable step succeeded but marking the WAL entry committed failed.
    /// The on-disk data is consistent; the WAL still shows the entry pending and
    /// recovery will (harmlessly, idempotently) re-report it.
    Wal(EncryptedWriteError),
    /// The dataset-level version counter reached `u64::MAX` and cannot be
    /// incremented without wrapping (which would break the strict-monotonicity
    /// invariant the T4 rollback check depends on). Checked before any sink
    /// write, so no durable state has changed.
    DatasetVersionExhausted,
}

impl<E: core::fmt::Display> core::fmt::Display for WriteOrderError<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            WriteOrderError::Sink { step, source } => {
                write!(f, "write-order failure at {step} step: {source}")
            }
            WriteOrderError::Wal(e) => write!(f, "WAL commit failed after write order: {e}"),
            WriteOrderError::DatasetVersionExhausted => {
                write!(f, "dataset version counter exhausted (u64::MAX commits)")
            }
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for WriteOrderError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WriteOrderError::Sink { source, .. } => Some(source),
            WriteOrderError::Wal(e) => Some(e),
            WriteOrderError::DatasetVersionExhausted => None,
        }
    }
}

impl<W: Read + Write + Seek> EncryptedChunkWriter<W> {
    /// Commit a chunk mutation using the mandated three-step write order.
    ///
    /// Given the [`EncryptedChunkResult`] from [`encrypt_chunk`](Self::encrypt_chunk)
    /// (which already journaled the `(chunk_idx, version)` WAL record), this drives
    /// the sink through the ordered sequence — chunk data → sync → companion nodes
    /// → sync → root attribute → sync — and only then marks the WAL entry
    /// committed. If any step fails, the WAL entry is left uncommitted so the
    /// chunk is recoverable and reported as pending by verification.
    ///
    /// The root attribute is written with the **next dataset version**
    /// (`self.dataset_version() + 1`), so the persisted counter advances
    /// exactly once per committed mutation (P2.2b step 1) and never regresses
    /// or ties across commits, regardless of which chunk was touched
    /// (Remark A.13). The version is **burned the moment it is presented to
    /// the sink**: the in-memory counter advances just before the step-3
    /// write, even if that write, its sync, or the final WAL commit then
    /// fails. Burning on attempt is what keeps the invariant airtight — after
    /// a step-3 or WAL failure the caller cannot tell whether the root
    /// attribute reached the platter, and reusing the number for a different
    /// state could persist two distinct roots under one version (the
    /// Remark A.13 tie). A failed commit therefore leaves a version gap;
    /// gaps are harmless, ties and regressions are not.
    ///
    /// # Arguments
    ///
    /// * `result` - the result returned by `encrypt_chunk` for this mutation
    /// * `companion_nodes` - flattened companion Merkle node array (step 2)
    /// * `root` - the new Merkle root hash (step 3)
    /// * `companion_hash` - the companion-integrity hash (step 3)
    /// * `sink` - the storage backend performing the writes and syncs
    ///
    /// # Errors
    ///
    /// - [`WriteOrderError::Sink`] if any write or sync fails (with the step)
    /// - [`WriteOrderError::Wal`] if the final WAL commit fails
    /// - [`WriteOrderError::DatasetVersionExhausted`] if the dataset version
    ///   counter is at `u64::MAX` (checked before any durable write)
    pub fn commit_with_write_order<S: MerkleWriteSink>(
        &mut self,
        result: &EncryptedChunkResult,
        companion_nodes: &[u8],
        root: &[u8],
        companion_hash: &[u8],
        sink: &mut S,
    ) -> Result<(), WriteOrderError<S::Error>> {
        // Reserve the next dataset version up front — the strictly-per-commit
        // counter bound into the root attribute at step 3. NOT result.version
        // (the per-chunk WAL version): a first write to chunk B after three
        // writes to chunk A carries per-chunk version 1, and persisting that
        // would regress the dataset version from 3 to 1 — tripping the
        // stateful verifier's T4 rollback check on an honest file.
        let next_dataset_version = self
            .dataset_version()
            .checked_add(1)
            .ok_or(WriteOrderError::DatasetVersionExhausted)?;

        // Step 1: chunk data, then sync.
        sink.write_chunk_data(result.chunk_idx, &result.ciphertext)
            .map_err(|e| WriteOrderError::Sink {
                step: WriteStep::ChunkData,
                source: e,
            })?;
        sink.sync().map_err(|e| WriteOrderError::Sink {
            step: WriteStep::ChunkData,
            source: e,
        })?;

        // Step 2: companion Merkle nodes, then sync.
        sink.write_companion_nodes(companion_nodes)
            .map_err(|e| WriteOrderError::Sink {
                step: WriteStep::CompanionNodes,
                source: e,
            })?;
        sink.sync().map_err(|e| WriteOrderError::Sink {
            step: WriteStep::CompanionNodes,
            source: e,
        })?;

        // Step 3: root attribute (root, companion hash, dataset version), then
        // sync. Burn the version BEFORE presenting it to the sink: if the write
        // or its sync fails we cannot know whether the attribute reached stable
        // storage, and if the WAL commit below fails the attribute definitely
        // did. Reusing the number for a later commit could then persist two
        // different roots under the same version — the Remark A.13 tie the
        // strict counter exists to rule out. A burned-but-failed commit only
        // costs a gap in the version sequence.
        self.advance_dataset_version(next_dataset_version);
        sink.write_root_attribute(root, companion_hash, next_dataset_version)
            .map_err(|e| WriteOrderError::Sink {
                step: WriteStep::RootAttribute,
                source: e,
            })?;
        sink.sync().map_err(|e| WriteOrderError::Sink {
            step: WriteStep::RootAttribute,
            source: e,
        })?;

        // All three steps are durable: only now mark the WAL entry committed.
        self.commit(result.chunk_idx, result.version)
            .map_err(WriteOrderError::Wal)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock_keystore::Dek;
    use crate::version_wal::{VersionCounterStore, VersionWal};
    use std::io::Cursor;

    fn test_dek() -> Dek {
        let mut dek = [0u8; 32];
        for (i, b) in dek.iter_mut().enumerate() {
            *b = i as u8;
        }
        dek
    }

    /// Records the exact sequence of sink operations, optionally failing at a
    /// chosen step to simulate a crash mid-write.
    #[derive(Default)]
    struct RecordingSink {
        log: Vec<Op>,
        fail_at: Option<WriteStep>,
    }

    #[derive(Debug, PartialEq, Eq)]
    enum Op {
        Chunk(u64),
        Nodes,
        Root(u64),
        Sync,
    }

    impl MerkleWriteSink for RecordingSink {
        type Error = String;

        fn write_chunk_data(&mut self, idx: u64, _ct: &[u8]) -> Result<(), String> {
            if self.fail_at == Some(WriteStep::ChunkData) {
                return Err("chunk write failed".into());
            }
            self.log.push(Op::Chunk(idx));
            Ok(())
        }

        fn write_companion_nodes(&mut self, _nodes: &[u8]) -> Result<(), String> {
            if self.fail_at == Some(WriteStep::CompanionNodes) {
                return Err("nodes write failed".into());
            }
            self.log.push(Op::Nodes);
            Ok(())
        }

        fn write_root_attribute(
            &mut self,
            _root: &[u8],
            _companion_hash: &[u8],
            version: u64,
        ) -> Result<(), String> {
            if self.fail_at == Some(WriteStep::RootAttribute) {
                return Err("root write failed".into());
            }
            self.log.push(Op::Root(version));
            Ok(())
        }

        fn sync(&mut self) -> Result<(), String> {
            self.log.push(Op::Sync);
            Ok(())
        }
    }

    fn new_writer() -> EncryptedChunkWriter<Cursor<Vec<u8>>> {
        let wal = VersionWal::new(Cursor::new(Vec::new()), 100).unwrap();
        EncryptedChunkWriter::new(test_dek(), wal, VersionCounterStore::new())
    }

    #[test]
    fn write_order_is_enforced_and_wal_commits_last() {
        let mut writer = new_writer();
        let result = writer.encrypt_chunk(7, b"payload").unwrap();

        let mut sink = RecordingSink::default();
        writer
            .commit_with_write_order(&result, b"nodes", b"root", b"companion", &mut sink)
            .unwrap();

        // Exact ordering: chunk -> sync -> nodes -> sync -> root -> sync.
        assert_eq!(
            sink.log,
            vec![
                Op::Chunk(7),
                Op::Sync,
                Op::Nodes,
                Op::Sync,
                Op::Root(1),
                Op::Sync,
            ]
        );

        // The WAL entry is committed only after all three durable steps, so
        // nothing remains to recover.
        assert!(writer.recover().unwrap().is_empty());
        assert_eq!(writer.dataset_version(), 1);
    }

    /// Regression for the Remark A.13 / P2.2b step 1 requirement: the root
    /// attribute's version is the strictly-per-commit dataset counter, not the
    /// mutated chunk's per-chunk WAL version. Three commits to chunk 5
    /// followed by a first commit to chunk 2 must persist versions 1,2,3,4 —
    /// never regressing to chunk 2's per-chunk version 1, which would trip the
    /// stateful verifier's T4 rollback check on an honest file.
    #[test]
    fn dataset_version_strictly_increases_across_chunks() {
        let mut writer = new_writer();
        let mut sink = RecordingSink::default();

        // Three commits to chunk 5 (per-chunk versions 1, 2, 3)...
        for expected_chunk_version in 1..=3 {
            let result = writer.encrypt_chunk(5, b"payload").unwrap();
            assert_eq!(result.version, expected_chunk_version);
            writer
                .commit_with_write_order(&result, b"nodes", b"root", b"companion", &mut sink)
                .unwrap();
        }
        // ...then a first commit to chunk 2 (per-chunk version 1).
        let result = writer.encrypt_chunk(2, b"payload").unwrap();
        assert_eq!(result.version, 1);
        writer
            .commit_with_write_order(&result, b"nodes", b"root", b"companion", &mut sink)
            .unwrap();

        // The persisted dataset versions are 1, 2, 3, 4 — strictly increasing,
        // with the fourth commit NOT regressing to chunk 2's per-chunk version.
        let roots: Vec<u64> = sink
            .log
            .iter()
            .filter_map(|op| match op {
                Op::Root(v) => Some(*v),
                _ => None,
            })
            .collect();
        assert_eq!(roots, vec![1, 2, 3, 4]);
        assert_eq!(writer.dataset_version(), 4);
    }

    /// A writer reopened on an existing dataset continues strictly upward from
    /// the persisted dataset version.
    #[test]
    fn reopened_writer_continues_from_seeded_dataset_version() {
        let wal = VersionWal::new(Cursor::new(Vec::new()), 100).unwrap();
        let mut writer = EncryptedChunkWriter::with_dataset_version(
            test_dek(),
            wal,
            VersionCounterStore::new(),
            7,
        );

        let result = writer.encrypt_chunk(0, b"payload").unwrap();
        let mut sink = RecordingSink::default();
        writer
            .commit_with_write_order(&result, b"nodes", b"root", b"companion", &mut sink)
            .unwrap();

        assert!(sink.log.contains(&Op::Root(8)));
        assert_eq!(writer.dataset_version(), 8);
    }

    /// A version presented at the root-attribute step is burned even when the
    /// step fails: the caller cannot know whether the attribute reached stable
    /// storage, so the retry must use the NEXT version — reuse could persist
    /// two different roots under one version (the Remark A.13 tie). Gaps are
    /// allowed, ties and regressions are not.
    #[test]
    fn failed_commit_burns_the_presented_version() {
        let mut writer = new_writer();
        let result = writer.encrypt_chunk(0, b"payload").unwrap();

        let mut sink = RecordingSink {
            fail_at: Some(WriteStep::RootAttribute),
            ..Default::default()
        };
        writer
            .commit_with_write_order(&result, b"nodes", b"root", b"companion", &mut sink)
            .unwrap_err();
        // Version 1 was presented to the sink and is burned.
        assert_eq!(writer.dataset_version(), 1);

        // The retry persists version 2 — never re-presenting 1.
        let mut sink = RecordingSink::default();
        writer
            .commit_with_write_order(&result, b"nodes", b"root", b"companion", &mut sink)
            .unwrap();
        assert!(sink.log.contains(&Op::Root(2)));
        assert_eq!(writer.dataset_version(), 2);
    }

    /// The tie regression from review round 2: steps 1–3 all durably succeed
    /// (the root attribute is on disk with version N), then the WAL commit
    /// fails. The writer must NOT hand version N to a later commit of a
    /// different state.
    #[test]
    fn wal_failure_after_durable_root_write_does_not_reuse_version() {
        let mut writer = new_writer();
        let result = writer.encrypt_chunk(0, b"payload").unwrap();

        let mut sink = RecordingSink::default();
        writer
            .commit_with_write_order(&result, b"nodes", b"root", b"companion", &mut sink)
            .unwrap();

        // Drive the same result through again: every sink step succeeds (the
        // root attribute is durably rewritten, now with version 2), but the
        // WAL entry was already committed, so the final WAL commit fails.
        let err = writer
            .commit_with_write_order(&result, b"nodes", b"root", b"companion", &mut sink)
            .unwrap_err();
        assert!(matches!(err, WriteOrderError::Wal(_)));

        // Version 2 reached the sink durably and is burned despite the error.
        assert_eq!(writer.dataset_version(), 2);

        // The next commit persists version 3 — version 2 is never re-presented
        // for a different root.
        let result = writer.encrypt_chunk(1, b"other payload").unwrap();
        writer
            .commit_with_write_order(&result, b"nodes", b"root2", b"companion", &mut sink)
            .unwrap();

        let roots: Vec<u64> = sink
            .log
            .iter()
            .filter_map(|op| match op {
                Op::Root(v) => Some(*v),
                _ => None,
            })
            .collect();
        assert_eq!(roots, vec![1, 2, 3]);
    }

    #[test]
    fn failure_mid_order_leaves_wal_entry_pending() {
        let mut writer = new_writer();
        let result = writer.encrypt_chunk(5, b"payload").unwrap();

        // Crash while writing the companion nodes (step 2).
        let mut sink = RecordingSink {
            fail_at: Some(WriteStep::CompanionNodes),
            ..Default::default()
        };
        let err = writer
            .commit_with_write_order(&result, b"nodes", b"root", b"companion", &mut sink)
            .unwrap_err();

        assert!(matches!(
            err,
            WriteOrderError::Sink {
                step: WriteStep::CompanionNodes,
                ..
            }
        ));

        // Only step 1 (chunk data + its sync) reached the sink.
        assert_eq!(sink.log, vec![Op::Chunk(5), Op::Sync]);

        // The WAL still shows the chunk pending, so verification (via
        // clawhdf5-format's verify_chunk_with_pending) will report NoncePending
        // until the write is retried and completed.
        let pending = writer.recover().unwrap();
        assert_eq!(pending, vec![(5, result.version)]);
    }

    #[test]
    fn failure_at_root_step_still_leaves_entry_pending() {
        let mut writer = new_writer();
        let result = writer.encrypt_chunk(3, b"payload").unwrap();

        let mut sink = RecordingSink {
            fail_at: Some(WriteStep::RootAttribute),
            ..Default::default()
        };
        let err = writer
            .commit_with_write_order(&result, b"nodes", b"root", b"companion", &mut sink)
            .unwrap_err();

        assert!(matches!(
            err,
            WriteOrderError::Sink {
                step: WriteStep::RootAttribute,
                ..
            }
        ));
        // Steps 1 and 2 completed and synced; step 3 write failed before its sync.
        assert_eq!(sink.log, vec![Op::Chunk(3), Op::Sync, Op::Nodes, Op::Sync]);
        assert_eq!(writer.recover().unwrap(), vec![(3, result.version)]);
    }
}
