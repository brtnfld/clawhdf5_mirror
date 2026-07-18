//! Disk I/O operations for HDF5 memory files.
//!
//! Uses memory-mapped I/O via `clawhdf5_io::MmapReader` for efficient
//! file reading with OS-managed paging.

use std::path::Path;

use crate::MemoryConfig;
use crate::MemoryError;
use crate::cache::MemoryCache;
use crate::knowledge::KnowledgeCache;
use crate::schema;
use crate::session::SessionCache;

/// Write all in-memory state to an HDF5 file on disk.
pub fn write_to_disk(
    path: &Path,
    config: &MemoryConfig,
    cache: &MemoryCache,
    sessions: &SessionCache,
    knowledge: &KnowledgeCache,
) -> Result<(), MemoryError> {
    let bytes = schema::build_hdf5_file(config, cache, sessions, knowledge)?;

    if bytes.is_empty() {
        return Err(MemoryError::Hdf5("build_hdf5_file produced 0 bytes".into()));
    }

    // Write to a temp file first, then rename for atomicity
    let tmp_path = path.with_extension("h5.tmp");
    std::fs::write(&tmp_path, &bytes).map_err(MemoryError::Io)?;
    std::fs::rename(&tmp_path, path).map_err(MemoryError::Io)?;

    Ok(())
}

/// Read an HDF5 file and return all state.
///
/// Uses memory-mapped I/O via `clawhdf5_io::MmapReader` for efficient
/// file access. The OS pages in data on demand rather than reading the
/// entire file into a contiguous buffer upfront.
pub fn read_from_disk(
    path: &Path,
) -> Result<(MemoryConfig, MemoryCache, SessionCache, KnowledgeCache), MemoryError> {
    read_from_disk_inner(path, false)
}

/// Like [`read_from_disk`], but fail-closed when the file carries no content
/// Merkle root (P2.4 Finding 1 strict mode). Rejects a file whose
/// `_merkle_root` attribute was stripped to force an unverified load.
pub fn read_from_disk_strict(
    path: &Path,
) -> Result<(MemoryConfig, MemoryCache, SessionCache, KnowledgeCache), MemoryError> {
    read_from_disk_inner(path, true)
}

fn read_from_disk_inner(
    path: &Path,
    strict: bool,
) -> Result<(MemoryConfig, MemoryCache, SessionCache, KnowledgeCache), MemoryError> {
    let mmap = clawhdf5_io::MmapReader::open(path).map_err(MemoryError::Io)?;

    // Advise the OS we'll need the whole file for parsing
    mmap.advise_willneed(0, mmap.len());

    // Parse the HDF5 file from the mmap'd bytes
    let file = clawhdf5::File::from_bytes(mmap.as_bytes().to_vec())
        .map_err(|e| MemoryError::Hdf5(format!("cannot open {}: {e}", path.display())))?;

    let (mut config, cache, sessions, knowledge) = if strict {
        schema::validate_and_load_strict(&file)?
    } else {
        schema::validate_and_load(&file)?
    };
    config.path = path.to_path_buf();

    Ok((config, cache, sessions, knowledge))
}

/// Copy an HDF5 file atomically to a destination.
pub fn snapshot_file(src: &Path, dest: &Path) -> Result<std::path::PathBuf, MemoryError> {
    let dest_file = if dest.is_dir() {
        let filename = src.file_name().ok_or_else(|| {
            MemoryError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "source has no filename",
            ))
        })?;
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        dest.join(format!("snapshot_{ts}_{}", filename.to_string_lossy()))
    } else {
        dest.to_path_buf()
    };

    // Atomic copy: write to temp, then rename
    let tmp_path = dest_file.with_extension("h5.tmp");
    std::fs::copy(src, &tmp_path).map_err(MemoryError::Io)?;
    std::fs::rename(&tmp_path, &dest_file).map_err(MemoryError::Io)?;

    Ok(dest_file)
}

/// Snapshot `src` into `dest` and journal the snapshot against the certified
/// Merkle root (P2.2b step 3).
///
/// This is the binding between the full-file snapshot primitive
/// ([`snapshot_file`]) and the certified state: it creates the snapshot, then
/// appends a [`ProvenanceRecord`] tying the snapshot's path to the exact
/// `(version, signed_root, hybrid_sig, timestamp)` it certifies. After this
/// call, the returned path is a valid rollback target
/// (`journal.is_valid_rollback_target(path)` is `true`), whereas any snapshot
/// taken *without* journaling is not — enforcing the rule that "a snapshot with
/// no journaled root is not a valid rollback target."
///
/// The `hybrid_sig` is the serialized P2.1 hybrid signature (opaque here). The
/// journal is the caller's durable log; persist it (e.g. to the `/_merkle/journal`
/// dataset via [`ProvenanceJournal::pack`]) after appending.
///
/// `version` must be the **dataset-level commit counter** — strictly
/// incremented once per committed mutation, i.e.
/// `EncryptedChunkWriter::dataset_version` after the commit — not a per-chunk
/// WAL version (which regresses across chunks) and not `max_version()` (which
/// can tie across commits; Remark A.13). The journal's append-only check
/// enforces strict increase and will reject anything else.
///
/// # Errors
///
/// - [`MemoryError::Io`] if the snapshot copy fails.
/// - [`MemoryError::Provenance`] if the journal rejects the record (e.g. a
///   non-increasing version).
#[cfg(feature = "merkle-provenance")]
pub fn snapshot_and_journal(
    src: &Path,
    dest: &Path,
    journal: &mut clawhdf5_format::merkle_journal::ProvenanceJournal,
    version: u64,
    signed_root: [u8; 32],
    hybrid_sig: Vec<u8>,
    timestamp: u64,
) -> Result<std::path::PathBuf, MemoryError> {
    use clawhdf5_format::merkle_journal::ProvenanceRecord;

    let snapshot_path = snapshot_file(src, dest)?;
    let snapshot_ref = snapshot_path.to_string_lossy().into_owned();

    journal
        .append(ProvenanceRecord {
            version,
            signed_root,
            hybrid_sig,
            timestamp,
            snapshot_ref,
        })
        .map_err(|e| MemoryError::Provenance(format!("{e}")))?;

    Ok(snapshot_path)
}

/// Revert to a journaled snapshot for a target version, accepting the restored
/// state only after both the signature and dataset gates pass (P2.2b step 4).
///
/// This ties the format-side recovery decision to real file I/O:
///
/// 1. [`select_restore_record`] picks the target — an explicit version or the
///    default "last known-good" — and enforces the **signature gate**.
/// 2. The record's snapshot is copied to a *temporary* path next to `live_path`,
///    so the live file is never touched until the restore is accepted.
/// 3. `verify_restored` is called with the temp path and the selected record; it
///    must load the restored dataset and run
///    [`clawhdf5_format::merkle_recovery::verify_restored_dataset`] (the
///    **dataset gate**), returning an error to reject the restore.
/// 4. Only if the dataset gate passes is the temp atomically renamed over
///    `live_path`. On any failure the temp file is removed and the live file is
///    left exactly as it was.
///
/// Returns the version that was restored.
///
/// # Errors
///
/// - [`MemoryError::Provenance`] if selection/signature verification fails.
/// - [`MemoryError::Io`] if the snapshot copy or rename fails.
/// - Whatever error `verify_restored` returns if the dataset gate fails.
#[cfg(feature = "merkle-provenance")]
pub fn restore_to_version<V, F>(
    live_path: &Path,
    journal: &clawhdf5_format::merkle_journal::ProvenanceJournal,
    target: clawhdf5_format::merkle_recovery::RestoreTarget,
    verifier: &V,
    verify_restored: F,
) -> Result<u64, MemoryError>
where
    V: clawhdf5_format::merkle_recovery::SignatureVerifier,
    F: FnOnce(&Path, &clawhdf5_format::merkle_journal::ProvenanceRecord) -> Result<(), MemoryError>,
{
    use clawhdf5_format::merkle_recovery::select_restore_record;

    // 1. Selection + signature gate (format).
    let record = select_restore_record(journal, target, verifier)
        .map_err(|e| MemoryError::Provenance(format!("{e}")))?;

    // 2. Revert to a temp path so the live file is untouched until accepted.
    // The name includes the pid so two processes restoring the same file
    // cannot clobber each other's temp copy.
    let tmp_path = restore_tmp_path(live_path);
    std::fs::copy(&record.snapshot_ref, &tmp_path).map_err(MemoryError::Io)?;

    // 3. Dataset gate at the temp path (caller loads + verify_restored_dataset).
    if let Err(e) = verify_restored(&tmp_path, record) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e);
    }

    // 4. Accept: flush the verified bytes to stable storage, then atomically
    // swap over the live file. Without the sync, a power loss right after the
    // rename could surface a live file whose contents never left the page
    // cache. (Directory-entry durability of the rename itself is not synced —
    // the same prototype gap as the WAL's flush-not-fsync, TODO(P3).)
    let sync_result = std::fs::OpenOptions::new()
        .write(true)
        .open(&tmp_path)
        .and_then(|f| f.sync_all());
    if let Err(e) = sync_result {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(MemoryError::Io(e));
    }
    if let Err(e) = std::fs::rename(&tmp_path, live_path) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(MemoryError::Io(e));
    }
    Ok(record.version)
}

/// A unique temp path `restore_to_version` stages a snapshot at before the
/// atomic swap. Includes the pid and a process-wide sequence number so
/// concurrent restores — across processes or threads — never collide.
#[cfg(feature = "merkle-provenance")]
fn restore_tmp_path(live_path: &Path) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static RESTORE_SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = RESTORE_SEQ.fetch_add(1, Ordering::Relaxed);
    live_path.with_extension(format!("restore.{}.{seq}.tmp", std::process::id()))
}

/// Outcome of [`handle_verify_error`]: which branch of the halt/quarantine/alert
/// policy (or the rebuild-by-rehash repair) was actually applied.
#[cfg(feature = "merkle-provenance")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandledOutcome {
    /// Access refused; the caller must surface a hard error.
    Halted,
    /// The file was marked unverified; writes are denied.
    Quarantined,
    /// The failure was logged and the caller may proceed read-only, without
    /// the integrity guarantee.
    AlertedDegraded,
    /// The Merkle tree was rebuilt by rehashing on-disk chunk data
    /// (unsigned datasets only).
    Repaired,
}

/// Context identifying *where* a detected `MerkleError` occurred, for the
/// logging requirement in Sec. merkle-storage, "Error response and recovery":
/// "the error must be logged with the affected file path, dataset name,
/// chunk index, and the full `MerkleError` variant before any further action
/// is taken."
#[cfg(feature = "merkle-provenance")]
#[derive(Debug, Clone, Copy)]
pub struct VerifyErrorContext<'a> {
    /// Path to the file the error was detected in.
    pub file_path: &'a Path,
    /// Name of the affected dataset.
    pub dataset_name: &'a str,
}

/// Wire the P2.2b step 5 error-response decision into the calling path:
/// log the required context, resolve the halt/quarantine/alert policy via
/// [`clawhdf5_format::merkle::resolve_response`], and, when a rebuild is
/// offered, run the caller-supplied `rebuild` step.
///
/// This is the "calling path" side of
/// [`clawhdf5_format::merkle::resolve_response`] — `clawhdf5-format` has no
/// file I/O or logging, so the actual log line and the decision of *how* to
/// rebuild (calling `extend_merkle`/`update_merkle` and persisting the
/// result) belong here, next to `restore_to_version` and
/// `snapshot_and_journal`.
///
/// `rebuild` is invoked if and only if the resolved recovery action is
/// `RebuildByRehash`, which [`resolve_response`] only ever returns for
/// [`SigningContext::Unsigned`] — a signed dataset always fails closed and
/// `rebuild` is never called, so the runtime never auto-rehashes and
/// re-signs on-disk data.
///
/// # Errors
///
/// Propagates whatever [`MemoryError`] `rebuild` returns.
///
/// [`SigningContext::Unsigned`]: clawhdf5_format::merkle::SigningContext::Unsigned
/// [`resolve_response`]: clawhdf5_format::merkle::resolve_response
#[cfg(feature = "merkle-provenance")]
pub fn handle_verify_error<F>(
    ctx: &VerifyErrorContext<'_>,
    error: &clawhdf5_format::merkle::MerkleError,
    policy: clawhdf5_format::merkle::ResponsePolicy,
    signing: clawhdf5_format::merkle::SigningContext,
    rebuild: F,
) -> Result<HandledOutcome, MemoryError>
where
    F: FnOnce() -> Result<(), MemoryError>,
{
    use clawhdf5_format::merkle::{RecoveryAction, VerifyResponse, resolve_response};

    let chunk_idx = match error {
        clawhdf5_format::merkle::MerkleError::HashMismatch { chunk_idx } => Some(*chunk_idx),
        _ => None,
    };

    // Logging requirement: file path, dataset name, chunk index, variant —
    // before any further action is taken.
    eprintln!(
        "[clawhdf5-agent] merkle verification error: file={} dataset={} chunk_idx={} variant={error}",
        ctx.file_path.display(),
        ctx.dataset_name,
        chunk_idx.map_or_else(|| "-".to_string(), |i| i.to_string()),
    );

    let resolved = resolve_response(error, policy, signing);

    if resolved.recovery == RecoveryAction::RebuildByRehash {
        rebuild()?;
        return Ok(HandledOutcome::Repaired);
    }

    Ok(match resolved.response {
        VerifyResponse::Quarantine => HandledOutcome::Quarantined,
        VerifyResponse::Alert => HandledOutcome::AlertedDegraded,
        // VerifyResponse::Halt, and any future non_exhaustive variant: fail closed.
        _ => HandledOutcome::Halted,
    })
}

#[cfg(all(test, feature = "merkle-provenance"))]
mod provenance_tie_tests {
    use super::*;
    use clawhdf5_format::merkle_journal::ProvenanceJournal;

    /// Whether `dir` contains any leftover `.tmp` staging file. The restore
    /// temp name includes a sequence number, so tests scan for the suffix
    /// rather than predicting the exact path.
    fn has_tmp_file(dir: &Path) -> bool {
        std::fs::read_dir(dir)
            .unwrap()
            .any(|e| e.unwrap().file_name().to_string_lossy().ends_with(".tmp"))
    }

    #[test]
    fn snapshot_and_journal_makes_snapshot_a_valid_rollback_target() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("data.h5");
        std::fs::write(&src, b"some hdf5 bytes").unwrap();

        let mut journal = ProvenanceJournal::new();

        let snap = snapshot_and_journal(
            &src,
            dir.path(),
            &mut journal,
            1,
            [0xAB; 32],
            vec![0xCD; 3374],
            1_700_000_000,
        )
        .unwrap();

        // The snapshot file exists and is now a journaled, valid rollback target.
        assert!(snap.exists());
        let snap_ref = snap.to_string_lossy();
        assert!(journal.is_valid_rollback_target(&snap_ref));
        assert_eq!(
            journal.record_for_version(1).unwrap().signed_root,
            [0xAB; 32]
        );

        // An arbitrary backup that was never journaled is NOT a valid target.
        assert!(!journal.is_valid_rollback_target("/tmp/some-random-backup.h5"));
    }

    #[test]
    fn snapshot_and_journal_rejects_version_regression() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("data.h5");
        std::fs::write(&src, b"bytes").unwrap();

        let mut journal = ProvenanceJournal::new();
        snapshot_and_journal(&src, dir.path(), &mut journal, 5, [0; 32], vec![], 1).unwrap();

        // A lower version must be refused (append-only history).
        let err = snapshot_and_journal(&src, dir.path(), &mut journal, 4, [0; 32], vec![], 2)
            .unwrap_err();
        assert!(matches!(err, MemoryError::Provenance(_)));
    }

    // ----- restore_to_version (P2.2b step 4) -----

    use clawhdf5_format::merkle_journal::ProvenanceRecord;
    use clawhdf5_format::merkle_recovery::{RestoreTarget, SignatureVerifier};

    /// Verifier accepting only an allow-listed set of versions.
    struct AllowList(Vec<u64>);
    impl SignatureVerifier for AllowList {
        fn verify(&self, r: &ProvenanceRecord) -> bool {
            self.0.contains(&r.version)
        }
    }

    #[test]
    fn restore_to_version_reverts_live_file_to_journaled_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let live = dir.path().join("live.h5");
        std::fs::write(&live, b"good-v1").unwrap();

        // Snapshot the good state as version 1, then journal it.
        let mut journal = ProvenanceJournal::new();
        snapshot_and_journal(&live, dir.path(), &mut journal, 1, [1; 32], vec![], 100).unwrap();

        // Corrupt the live file.
        std::fs::write(&live, b"TAMPERED").unwrap();

        // Restore to last known-good (version 1). The dataset gate here just
        // confirms the temp holds the certified snapshot bytes.
        let restored = restore_to_version(
            &live,
            &journal,
            RestoreTarget::LastKnownGood,
            &AllowList(vec![1]),
            |tmp, _record| {
                let bytes = std::fs::read(tmp).map_err(MemoryError::Io)?;
                if bytes == b"good-v1" {
                    Ok(())
                } else {
                    Err(MemoryError::Provenance("unexpected restored bytes".into()))
                }
            },
        )
        .unwrap();

        assert_eq!(restored, 1);
        // The live file was reverted to the good snapshot.
        assert_eq!(std::fs::read(&live).unwrap(), b"good-v1");
        // No temp file left behind.
        assert!(!has_tmp_file(dir.path()));
    }

    #[test]
    fn restore_to_version_leaves_live_untouched_when_dataset_gate_fails() {
        let dir = tempfile::tempdir().unwrap();
        let live = dir.path().join("live.h5");
        std::fs::write(&live, b"current-state").unwrap();

        let mut journal = ProvenanceJournal::new();
        snapshot_and_journal(&live, dir.path(), &mut journal, 1, [1; 32], vec![], 100).unwrap();

        // Overwrite the live file with something we must NOT lose if restore fails.
        std::fs::write(&live, b"do-not-clobber").unwrap();

        // The dataset gate rejects the restore.
        let err = restore_to_version(
            &live,
            &journal,
            RestoreTarget::Version(1),
            &AllowList(vec![1]),
            |_tmp, _record| Err(MemoryError::Provenance("dataset gate failed".into())),
        )
        .unwrap_err();

        assert!(matches!(err, MemoryError::Provenance(_)));
        // Live file is untouched, temp is cleaned up.
        assert_eq!(std::fs::read(&live).unwrap(), b"do-not-clobber");
        assert!(!has_tmp_file(dir.path()));
    }

    #[test]
    fn restore_to_version_refuses_when_signature_gate_fails() {
        let dir = tempfile::tempdir().unwrap();
        let live = dir.path().join("live.h5");
        std::fs::write(&live, b"state").unwrap();

        let mut journal = ProvenanceJournal::new();
        snapshot_and_journal(&live, dir.path(), &mut journal, 1, [1; 32], vec![], 100).unwrap();
        std::fs::write(&live, b"live-untouched").unwrap();

        // No version's signature verifies -> no known-good target, apply never runs.
        let mut applied = false;
        let err = restore_to_version(
            &live,
            &journal,
            RestoreTarget::LastKnownGood,
            &AllowList(vec![]),
            |_tmp, _record| {
                applied = true;
                Ok(())
            },
        )
        .unwrap_err();

        assert!(matches!(err, MemoryError::Provenance(_)));
        assert!(!applied);
        assert_eq!(std::fs::read(&live).unwrap(), b"live-untouched");
    }

    // ----- handle_verify_error (P2.2b step 5) -----

    use clawhdf5_format::merkle::{MerkleError, ResponsePolicy, SigningContext};

    #[test]
    fn handle_verify_error_signed_hash_mismatch_halts_and_never_rebuilds() {
        let path = Path::new("live.h5");
        let ctx = VerifyErrorContext {
            file_path: path,
            dataset_name: "temperature",
        };
        let mut rebuilt = false;

        let outcome = handle_verify_error(
            &ctx,
            &MerkleError::HashMismatch { chunk_idx: 42 },
            ResponsePolicy::Halt,
            SigningContext::Signed,
            || {
                rebuilt = true;
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(outcome, HandledOutcome::Halted);
        assert!(
            !rebuilt,
            "signed dataset must never auto-rehash and re-sign"
        );
    }

    #[test]
    fn handle_verify_error_signed_companion_tampered_respects_quarantine_policy() {
        let path = Path::new("live.h5");
        let ctx = VerifyErrorContext {
            file_path: path,
            dataset_name: "temperature",
        };

        let outcome = handle_verify_error(
            &ctx,
            &MerkleError::CompanionTampered,
            ResponsePolicy::Quarantine,
            SigningContext::Signed,
            || panic!("must not rebuild a signed dataset"),
        )
        .unwrap();

        assert_eq!(outcome, HandledOutcome::Quarantined);
    }

    #[test]
    fn handle_verify_error_signed_signature_invalid_respects_alert_policy() {
        let path = Path::new("live.h5");
        let ctx = VerifyErrorContext {
            file_path: path,
            dataset_name: "temperature",
        };

        let outcome = handle_verify_error(
            &ctx,
            &MerkleError::SignatureInvalid,
            ResponsePolicy::AlertAndContinue,
            SigningContext::Signed,
            || panic!("must not rebuild a signed dataset"),
        )
        .unwrap();

        assert_eq!(outcome, HandledOutcome::AlertedDegraded);
    }

    #[test]
    fn handle_verify_error_unsigned_hash_mismatch_rebuilds() {
        let path = Path::new("live.h5");
        let ctx = VerifyErrorContext {
            file_path: path,
            dataset_name: "temperature",
        };
        let mut rebuilt = false;

        // Even with a Halt policy selected, an unsigned dataset repairs by
        // rehashing instead of halting -- no authenticity guarantee is at risk.
        let outcome = handle_verify_error(
            &ctx,
            &MerkleError::HashMismatch { chunk_idx: 0 },
            ResponsePolicy::Halt,
            SigningContext::Unsigned,
            || {
                rebuilt = true;
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(outcome, HandledOutcome::Repaired);
        assert!(rebuilt);
    }

    #[test]
    fn handle_verify_error_propagates_rebuild_failure() {
        let path = Path::new("live.h5");
        let ctx = VerifyErrorContext {
            file_path: path,
            dataset_name: "temperature",
        };

        let err = handle_verify_error(
            &ctx,
            &MerkleError::CompanionTampered,
            ResponsePolicy::Halt,
            SigningContext::Unsigned,
            || Err(MemoryError::Io(std::io::Error::other("disk full"))),
        )
        .unwrap_err();

        assert!(matches!(err, MemoryError::Io(_)));
    }

    #[test]
    fn handle_verify_error_out_of_bounds_error_always_halts() {
        let path = Path::new("live.h5");
        let ctx = VerifyErrorContext {
            file_path: path,
            dataset_name: "temperature",
        };

        // Not a content-inconsistency variant: always Halt, regardless of
        // policy or signing context, and never triggers a rebuild.
        let outcome = handle_verify_error(
            &ctx,
            &MerkleError::HyperslabOutOfBounds { idx: 99 },
            ResponsePolicy::AlertAndContinue,
            SigningContext::Unsigned,
            || panic!("must not rebuild for a non-content-inconsistency error"),
        )
        .unwrap();

        assert_eq!(outcome, HandledOutcome::Halted);
    }
}
