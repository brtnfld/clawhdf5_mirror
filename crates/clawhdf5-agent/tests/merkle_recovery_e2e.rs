//! End-to-end P2.2b recovery flow against a real HDF5 file.
//!
//! The unit tests in `storage.rs` exercise `snapshot_and_journal`,
//! `handle_verify_error`, and `restore_to_version` with raw byte files and
//! stub closures. This test drives the same flow through the agent's real
//! machinery: an actual HDF5 memory file written by [`HDF5Memory`], tampered
//! via raw file I/O (bypassing ClawHDF5), detected by
//! `clawhdf5_format::merkle::verify_dataset`, routed through the
//! halt/quarantine/alert policy, and restored from a journaled snapshot —
//! after which the file must both pass Merkle verification *and* still parse
//! as a valid HDF5 memory file.
//!
//! Run with: `cargo test -p clawhdf5-agent --features merkle-provenance`
#![cfg(feature = "merkle-provenance")]

use std::collections::BTreeMap;
use std::path::Path;

use clawhdf5_agent::storage::{
    HandledOutcome, VerifyErrorContext, handle_verify_error, restore_to_version,
    snapshot_and_journal,
};
use clawhdf5_agent::{AgentMemory, HDF5Memory, MemoryConfig, MemoryEntry, MemoryError};
use clawhdf5_format::merkle::{
    Dataset, HashAlg, MerkleAttr, MerkleError, MerkleTree, ResponsePolicy, SigningContext,
    verify_dataset,
};
use clawhdf5_format::merkle_journal::ProvenanceRecord;
use clawhdf5_format::merkle_recovery::{RestoreTarget, SignatureVerifier, verify_restored_dataset};

/// Fixed chunk size the Merkle layer slices the file into.
const CHUNK: usize = 256;

/// Slice `bytes` into fixed-size chunks (last one may be short).
fn chunks_of(bytes: &[u8]) -> Vec<&[u8]> {
    bytes.chunks(CHUNK).collect()
}

/// Build the Merkle protection state (attribute + flattened companion nodes)
/// over a file's bytes.
fn merkle_state_for(bytes: &[u8]) -> (MerkleAttr, Vec<u8>) {
    let tree = MerkleTree::from_chunks(&chunks_of(bytes), HashAlg::Blake3);
    let mut nodes = Vec::with_capacity(tree.nodes().len() * 32);
    for n in tree.nodes() {
        nodes.extend_from_slice(n);
    }
    (MerkleAttr::from_tree(&tree), nodes)
}

/// P2.1 stand-in: accepts exactly the known genuine signature per version.
struct KnownGoodSignatures(BTreeMap<u64, Vec<u8>>);
impl SignatureVerifier for KnownGoodSignatures {
    fn verify(&self, r: &ProvenanceRecord) -> bool {
        self.0.get(&r.version) == Some(&r.hybrid_sig)
    }
}

#[test]
fn tamper_detect_halt_restore_roundtrip_on_real_hdf5_file() {
    let dir = tempfile::tempdir().unwrap();
    let live = dir.path().join("live.h5");

    // Phase 1: create a real HDF5 memory file with one saved entry. WAL is
    // disabled so `save` flushes the entry to the file immediately and the
    // on-disk bytes are the complete, self-contained state.
    let mut config = MemoryConfig::new(live.clone(), "e2e-recovery", 4);
    config.wal_enabled = false;
    let mut mem = HDF5Memory::create(config).unwrap();
    mem.save(MemoryEntry {
        chunk: "the good state worth restoring".to_string(),
        embedding: vec![0.1, 0.2, 0.3, 0.4],
        source_channel: "e2e".to_string(),
        timestamp: 1_700_000_000.0,
        session_id: "s1".to_string(),
        tags: String::new(),
    })
    .unwrap();
    drop(mem);

    // Phase 2: build the Merkle state over the good file and journal a
    // snapshot certifying it (version 1, "signed" by the P2.1 stand-in).
    let good_bytes = std::fs::read(&live).unwrap();
    assert!(
        good_bytes.len() > 2 * CHUNK,
        "file must span multiple chunks for a mid-file tamper to be meaningful"
    );
    let (good_attr, good_nodes) = merkle_state_for(&good_bytes);

    let mut journal = clawhdf5_format::merkle_journal::ProvenanceJournal::new();
    let snapshot = snapshot_and_journal(
        &live,
        dir.path(),
        &mut journal,
        1,
        good_attr.root,
        b"genuine-sig-v1".to_vec(),
        1_700_000_000,
    )
    .unwrap();
    assert!(snapshot.exists());

    // Phase 3: tamper mid-file via raw I/O, bypassing ClawHDF5 entirely.
    let mut tampered = good_bytes.clone();
    let victim = good_bytes.len() / 2;
    tampered[victim] ^= 0xFF;
    std::fs::write(&live, &tampered).unwrap();

    // Phase 4: detection. Re-hash the live file against the committed tree.
    let live_bytes = std::fs::read(&live).unwrap();
    let ds = Dataset::from_owned(
        good_attr.clone(),
        good_nodes.clone(),
        chunks_of(&live_bytes),
    );
    let err = verify_dataset(&ds).unwrap_err();
    let expected_idx = victim / CHUNK;
    assert!(
        matches!(err, MerkleError::HashMismatch { chunk_idx } if chunk_idx == expected_idx),
        "expected HashMismatch at chunk {expected_idx}, got {err:?}"
    );

    // Phase 5: the error-response policy. Signed dataset, halt policy — the
    // runtime must halt and must never auto-rehash and re-sign.
    let outcome = handle_verify_error(
        &VerifyErrorContext {
            file_path: &live,
            dataset_name: "agent-memory",
        },
        &err,
        ResponsePolicy::Halt,
        SigningContext::Signed,
        || panic!("rebuild must never run for a signed dataset"),
    )
    .unwrap();
    assert_eq!(outcome, HandledOutcome::Halted);

    // Phase 6: recovery — restore to the last known-good signed version. The
    // dataset gate re-verifies the staged snapshot bytes against the
    // journaled record before the live file is touched.
    let mut genuine = BTreeMap::new();
    genuine.insert(1u64, b"genuine-sig-v1".to_vec());
    let verifier = KnownGoodSignatures(genuine);

    let restored_version = restore_to_version(
        &live,
        &journal,
        RestoreTarget::LastKnownGood,
        &verifier,
        |tmp: &Path, record: &ProvenanceRecord| {
            let staged = std::fs::read(tmp).map_err(MemoryError::Io)?;
            let (attr, nodes) = merkle_state_for(&staged);
            let ds = Dataset::from_owned(attr, nodes, chunks_of(&staged));
            verify_restored_dataset(record, &ds).map_err(|e| MemoryError::Provenance(e.to_string()))
        },
    )
    .unwrap();
    assert_eq!(restored_version, 1);

    // Phase 7: the restored live file passes full Merkle verification against
    // the journaled state...
    let restored_bytes = std::fs::read(&live).unwrap();
    let ds = Dataset::from_owned(good_attr, good_nodes, chunks_of(&restored_bytes));
    assert!(verify_dataset(&ds).unwrap());

    // ...and is still a valid HDF5 memory file with the original entry intact.
    let mem = HDF5Memory::open(&live).unwrap();
    assert_eq!(mem.count(), 1);
}

#[test]
fn restore_refuses_stale_signature_on_real_file() {
    let dir = tempfile::tempdir().unwrap();
    let live = dir.path().join("live.h5");

    let mut config = MemoryConfig::new(live.clone(), "e2e-stale-sig", 4);
    config.wal_enabled = false;
    let mem = HDF5Memory::create(config).unwrap();
    drop(mem);

    let good_bytes = std::fs::read(&live).unwrap();
    let (good_attr, _) = merkle_state_for(&good_bytes);

    // Journal version 1 with its genuine signature, then version 2 whose
    // record carries version 1's signature bytes (a stale-signature splice).
    let mut journal = clawhdf5_format::merkle_journal::ProvenanceJournal::new();
    snapshot_and_journal(
        &live,
        dir.path(),
        &mut journal,
        1,
        good_attr.root,
        b"genuine-sig-v1".to_vec(),
        100,
    )
    .unwrap();
    snapshot_and_journal(
        &live,
        dir.path(),
        &mut journal,
        2,
        [0xEE; 32],
        b"genuine-sig-v1".to_vec(),
        200,
    )
    .unwrap();

    let mut genuine = BTreeMap::new();
    genuine.insert(1u64, b"genuine-sig-v1".to_vec());
    genuine.insert(2u64, b"genuine-sig-v2".to_vec());
    let verifier = KnownGoodSignatures(genuine);

    // An explicit restore to version 2 must fail at the signature gate,
    // before the staging closure ever runs.
    let err = restore_to_version(
        &live,
        &journal,
        RestoreTarget::Version(2),
        &verifier,
        |_tmp: &Path, _rec: &ProvenanceRecord| {
            panic!("staging must not run when the signature gate fails")
        },
    )
    .unwrap_err();
    assert!(matches!(err, MemoryError::Provenance(_)));

    // Last-known-good skips the stale v2 record and lands on v1.
    let restored = restore_to_version(
        &live,
        &journal,
        RestoreTarget::LastKnownGood,
        &verifier,
        |tmp: &Path, record: &ProvenanceRecord| {
            let staged = std::fs::read(tmp).map_err(MemoryError::Io)?;
            let (attr, nodes) = merkle_state_for(&staged);
            let ds = Dataset::from_owned(attr, nodes, chunks_of(&staged));
            verify_restored_dataset(record, &ds).map_err(|e| MemoryError::Provenance(e.to_string()))
        },
    )
    .unwrap();
    assert_eq!(restored, 1);
}
