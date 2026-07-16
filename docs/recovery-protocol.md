# Recovery protocol (P2.2b)

Crash-consistency recovery and signed-root rollback for Merkle-protected
datasets. Covers the write order that keeps a crash detectable, the
provenance journal that makes rollback possible, and the halt/quarantine/alert
policy that decides what happens when tampering or a crash inconsistency is
detected. See S2-D2-Yr2 §`sec:merkle-storage` ("Crash consistency", "Error
response and recovery") and §7 task P2.2b for the full specification this
implements.

## Write order

Every committed chunk mutation goes through three durability-separated steps,
each followed by an explicit sync (`clawhdf5-filters::write_order`):

1. **Chunk data** — write the (encrypted) chunk payload.
2. **Companion Merkle nodes** — write the updated leaf hash and the
   recomputed path to the root into the companion node array.
3. **Root attribute** — write the root hash, companion-integrity hash, and
   dataset version.

The dataset version written in step 3 is a **strictly-per-commit counter**
(`EncryptedChunkWriter::dataset_version`), incremented exactly once per
committed mutation and seeded from the persisted `merkle_version` attribute on
reopen (`EncryptedChunkWriter::with_dataset_version`). It is deliberately
*not* the mutated chunk's per-chunk WAL version (which regresses across
commits touching different chunks) and *not* `max_k v_k` (which can tie when
a mutation doesn't touch the currently-maximal chunk) — either definition
would give the stateful verifier's T4 rollback check false positives or a
tie-based blind spot (security review Remark A.13). A version is **burned the
moment it is presented at the root-attribute step**, even if that write, its
sync, or the WAL commit then fails — after such a failure the writer cannot
know whether the attribute reached the platter, and reusing the number for a
different state could persist two distinct roots under one version. Failed
commits therefore leave gaps in the version sequence; gaps are allowed,
regressions and ties are not.

The per-chunk WAL (`clawhdf5-filters::version_wal`) brackets the whole
sequence: `(chunk_idx, version)` is journaled *before* the nonce is derived
and the chunk is encrypted, and is marked committed only *after* step 3 has
synced. A crash between any two steps therefore leaves the WAL entry pending,
so on reopen the affected chunk is reported rather than silently trusted.

If the crash lands between steps 2 and 3, the root attribute still names the
pre-mutation companion state while the companion nodes have already moved —
`verify_root` (`clawhdf5-format::merkle`) detects this as
`MerkleError::CompanionTampered`, because it recomputes the companion hash
from the (now newer) node bytes and compares it against the (still older)
hash recorded in the root attribute. This is the same detection path used for
genuine companion tampering; recovery is what tells the two apart (see
below).

## Provenance journal

`clawhdf5-format::merkle_journal::ProvenanceJournal` is an append-only,
strictly-increasing-by-version log. One `ProvenanceRecord` is appended per
commit:

```
ProvenanceRecord {
    version:      u64,        // dataset version this record certifies
    signed_root:  [u8; 32],   // the Merkle root that was signed
    hybrid_sig:   Vec<u8>,    // opaque serialized P2.1 hybrid signature
    timestamp:    u64,        // commit time (seconds since Unix epoch)
    snapshot_ref: String,     // opaque handle to a full-file snapshot
}
```

On-disk layout (suitable for a `/_merkle/journal` dataset or a sidecar file):

```
[magic "MJRN":4][format version:1][reserved:3, must be zero][record_count:u32 BE]
then record_count records, each:
  [version:u64 BE][signed_root:32][timestamp:u64 BE]
  [sig_len:u32 BE][hybrid_sig:sig_len]
  [ref_len:u32 BE][snapshot_ref:ref_len UTF-8]
```

`ProvenanceJournal::unpack` validates every length against the remaining
buffer with checked arithmetic, so malformed or truncated input is rejected
(`MerkleError::JournalCorrupt`) rather than panicking.

`clawhdf5-agent::storage::snapshot_and_journal` is the binding between the
journal and the full-file snapshot primitive: it snapshots the file, then
appends a record tying that exact snapshot path to the version, root,
signature, and timestamp it certifies. A snapshot with no journaled record is
not a valid rollback target — `ProvenanceJournal::is_valid_rollback_target`
enforces this structurally, not by convention.

## restore_to_version

`clawhdf5-format::merkle_recovery::restore_to_version` reverts to a journaled
snapshot through two gates, run in order:

1. **Signature gate** (`select_restore_record`) — picks the target record
   (an explicit version, or `RestoreTarget::LastKnownGood`: the highest
   version whose signature verifies) and requires the record's hybrid
   signature to verify. A stale or forged signature is rejected here
   (`RestoreError::SignatureInvalid`) before anything is touched.
2. **Dataset gate** (`verify_restored_dataset`) — after the caller physically
   reverts to the record's snapshot, the restored dataset must pass
   `verify_dataset` *and* its Merkle root must equal the record's
   `signed_root`.

Only when both gates pass is the restore accepted.
`clawhdf5-agent::storage::restore_to_version` is the file-I/O counterpart:
it copies the journaled snapshot to a per-process temp path, runs the dataset
gate against the temp file, syncs the verified bytes to stable storage, and
only then atomically renames it over the live file — any failure leaves the
live file untouched and removes the temp copy.

**Durability caveat (prototype):** the restored file's *contents* are fsynced
before the rename, but the rename's directory entry is not, and the version
WAL still uses `flush` rather than `sync_all` (tracked as `TODO(P3)` in
`version_wal.rs`). A power loss at exactly the wrong instant can therefore
undo an apparently-completed restore or lose the newest WAL record; it cannot
produce a half-restored or unverified live file.

## Halt / quarantine / alert policy

`clawhdf5-format::merkle::resolve_response(error, policy, signing)` is the
full error-response decision (replacing the Phase 1 `default_response`
fail-closed stub for the three content-inconsistency errors):

| `MerkleError`                                    | Response                                    | Recovery                                      |
|---------------------------------------------------|----------------------------------------------|------------------------------------------------|
| `HashMismatch`, `CompanionTampered`, `SignatureInvalid` | operator's `ResponsePolicy` (`Halt` / `Quarantine` / `AlertAndContinue`) | `RebuildByRehash` iff `SigningContext::Unsigned`; `None` for `Signed` |
| everything else (bounds errors, `NoncePending`, T4 `VersionRollback`, `JournalCorrupt`, malformed input) | `Halt`                                       | `None`, unconditionally                        |

- **`Halt`** refuses all further access — the correct default for automated
  pipelines and archival ingest.
- **`Quarantine`** marks the file unverified and denies writes, allowing
  read-only access only under explicit operator acknowledgement — for
  forensic inspection before disposal.
- **`AlertAndContinue`** logs and proceeds read-only without the integrity
  guarantee — acceptable only for interactive, non-critical exploration.
- **Signed datasets always fail closed**: `resolve_response` never returns
  `RecoveryAction::RebuildByRehash` for `SigningContext::Signed`, so the
  runtime can never auto-rehash and re-sign on-disk data — doing so over
  tampered chunks would launder the corruption under a freshly, validly
  signed root.
- **Unsigned datasets may rebuild**: no authenticity guarantee is at risk, so
  rehashing the on-disk chunk data and recomputing the tree is a safe repair.

`clawhdf5-agent::storage::handle_verify_error` is the calling-path wiring:
it logs the file path, dataset name, chunk index, and error variant (the
logging requirement from "Error response and recovery") *before* acting,
calls `resolve_response`, and runs the caller-supplied rebuild step only when
offered.

## Regression fixture

`crates/clawhdf5-filters/tests/crash_vs_tamper_matrix.rs` is the crash-vs-tamper
test matrix (P2.2b step 6): named tests for a crash between write-order steps
2 and 3, a single flipped chunk byte, a corrupted companion node, a stale
signature, and a chunk-plus-version rollback — each confirming the expected
`MerkleError` variant and, for every tampering case, that `restore_to_version`
round-trips the file back to a state that passes full verification.
