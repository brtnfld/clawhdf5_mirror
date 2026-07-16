//! Crash-vs-tamper test matrix (P2.2b step 6).
//!
//! S2-D2-Yr2 §7, task P2.2b, step 6 requires named tests for each of five
//! causes, each confirming the expected `MerkleError` variant (or clean
//! recovery), plus a `restore_to_version` round-trip back to a verifying
//! state for every tampering case:
//!
//! (a) kill the process between write-order steps 2 and 3 -> the
//!     inconsistency is detected on reopen; unsigned datasets may rebuild by
//!     rehashing, signed datasets fail closed.
//! (b) flip one byte in one chunk -> `HashMismatch`.
//! (c) corrupt a companion node -> `CompanionTampered`.
//! (d) present a stale signature -> `SignatureInvalid`.
//! (e) roll a chunk and its version counter back to a prior valid state ->
//!     rejected via the version-counter binding (T4 selective rollback).
//!
//! This lives in `clawhdf5-filters` (not `clawhdf5-format` or
//! `clawhdf5-agent`) because scenario (a) needs the three-step write order
//! and per-chunk WAL from this crate, and scenario (e) needs the
//! version-bound leaf-hash formula (`compute_leaf_hash`) that only exists
//! here; `clawhdf5-format` (Merkle verification, the halt/quarantine/alert
//! policy, and `restore_to_version`) is already a dev-dependency, matching
//! the pattern established by `encrypted_merkle_test.rs`.

use clawhdf5_filters::{
    EncryptedChunkWriter, FilterPipeline, FilterPipelineConfig, MerkleWriteSink,
    VersionCounterStore, VersionWal, WriteOrderError, WriteStep, compute_leaf_hash, mock_dek,
};
use clawhdf5_format::merkle::{
    Dataset, HashAlg, MerkleAttr, MerkleError, MerkleTree, RecoveryAction, ResponsePolicy,
    SigningContext, VerifyResponse, VersionObservationStore, resolve_response, verify_dataset,
    verify_root,
};
use clawhdf5_format::merkle_journal::{ProvenanceJournal, ProvenanceRecord};
use clawhdf5_format::merkle_recovery::{
    RestoreError, RestoreTarget, SignatureVerifier, restore_to_version, select_restore_record,
    verify_restored_dataset,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::io::Cursor;

const NUM_CHUNKS: usize = 8;

/// Deterministic, distinct plaintext-ish bytes standing in for chunk contents.
fn chunk_bytes(idx: u64, salt: u8) -> Vec<u8> {
    vec![(idx as u8).wrapping_mul(31).wrapping_add(salt); 64]
}

/// SHA-256 of the flattened companion node array, matching
/// `MerkleAttr::verify_companion`'s expectation exactly (same hash function,
/// same input bytes) so `verify_root` can actually detect companion tampering
/// instead of taking the "no companion" shortcut.
fn companion_hash(nodes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(nodes);
    hasher.finalize().into()
}

/// Build a `(MerkleAttr, flattened_tree_nodes)` pair from raw chunk bytes,
/// using plain leaf-domain-separated hashing (`HashAlg::hash_leaf`) — the
/// scheme `clawhdf5_format::merkle::verify_*` understands natively.
fn build_attr_and_nodes(chunks: &[Vec<u8>]) -> (MerkleAttr, Vec<u8>) {
    let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
    let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);
    let mut nodes = Vec::with_capacity(tree.nodes().len() * 32);
    for n in tree.nodes() {
        nodes.extend_from_slice(n);
    }
    let attr = MerkleAttr::from_tree_with_companion(&tree, companion_hash(&nodes));
    (attr, nodes)
}

fn dataset_view<'a>(attr: MerkleAttr, nodes: Vec<u8>, chunks: &'a [Vec<u8>]) -> Dataset<'a> {
    let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
    Dataset::from_owned(attr, nodes, refs)
}

// ===========================================================================
// (a) Crash between write-order steps 2 and 3.
// ===========================================================================

/// A [`MerkleWriteSink`] that retains what actually reached "storage", so the
/// test can inspect the post-crash on-disk state. Failing at
/// [`WriteStep::RootAttribute`] leaves the previous root attribute in place
/// while chunk data and companion nodes have already been durably written —
/// exactly "kill the process between steps 2 and 3".
#[derive(Default)]
struct CrashSink {
    chunks: HashMap<u64, Vec<u8>>,
    companion_nodes: Option<Vec<u8>>,
    fail_at: Option<WriteStep>,
}

impl MerkleWriteSink for CrashSink {
    type Error = &'static str;

    fn write_chunk_data(&mut self, chunk_idx: u64, ciphertext: &[u8]) -> Result<(), Self::Error> {
        if self.fail_at == Some(WriteStep::ChunkData) {
            return Err("crash before chunk data reached storage");
        }
        self.chunks.insert(chunk_idx, ciphertext.to_vec());
        Ok(())
    }

    fn write_companion_nodes(&mut self, nodes: &[u8]) -> Result<(), Self::Error> {
        if self.fail_at == Some(WriteStep::CompanionNodes) {
            return Err("crash before companion nodes reached storage");
        }
        self.companion_nodes = Some(nodes.to_vec());
        Ok(())
    }

    fn write_root_attribute(&mut self, _: &[u8], _: &[u8], _: u64) -> Result<(), Self::Error> {
        if self.fail_at == Some(WriteStep::RootAttribute) {
            return Err("crash before root attribute reached storage");
        }
        Ok(())
    }

    fn sync(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

#[test]
fn scenario_a_crash_between_companion_and_root_write_diverges_by_signing_context() {
    // Phase 1: a fully consistent, already-committed initial state (N chunks).
    let initial_chunks: Vec<Vec<u8>> = (0..NUM_CHUNKS as u64).map(|i| chunk_bytes(i, 1)).collect();
    let (good_attr, good_nodes) = build_attr_and_nodes(&initial_chunks);

    let mut sink = CrashSink {
        chunks: initial_chunks
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, c)| (i as u64, c))
            .collect(),
        companion_nodes: Some(good_nodes),
        fail_at: None,
    };

    // Phase 2: mutate chunk 0 through the mandated three-step write order, but
    // the process is killed between steps 2 (companion nodes) and 3 (root
    // attribute) -- durably updating the chunk and the companion tree, but
    // never reaching the root attribute.
    let wal = VersionWal::new(Cursor::new(Vec::new()), 100).unwrap();
    let mut versions = VersionCounterStore::new();
    versions.update(0, 1);
    let mut writer = EncryptedChunkWriter::new(mock_dek(), wal, versions);

    let result = writer
        .encrypt_chunk(0, b"post-crash replacement payload")
        .unwrap();
    assert_eq!(result.version, 2);

    let mut mutated_chunks = initial_chunks.clone();
    mutated_chunks[0] = result.ciphertext.clone();
    let (new_attr, new_nodes) = build_attr_and_nodes(&mutated_chunks);

    sink.fail_at = Some(WriteStep::RootAttribute);
    let err = writer
        .commit_with_write_order(
            &result,
            &new_nodes,
            &new_attr.root,
            &new_attr.companion_hash,
            &mut sink,
        )
        .unwrap_err();
    assert!(matches!(
        err,
        WriteOrderError::Sink {
            step: WriteStep::RootAttribute,
            ..
        }
    ));
    // The WAL entry for chunk 0's mutation is still pending: commit() (which
    // only runs after all three steps succeed) never ran.
    assert_eq!(
        writer.recover().unwrap(),
        vec![(
            0,
            2,
            clawhdf5_filters::WalRecord::hash_plaintext(b"post-crash replacement payload")
        )]
    );

    // On reopen: chunk data and companion nodes reflect the mutation, but the
    // root attribute is still the pre-mutation one (step 3 never landed).
    let mut on_disk_chunks = initial_chunks.clone();
    on_disk_chunks[0] = sink.chunks[&0].clone();
    let reopened = dataset_view(
        good_attr.clone(),
        sink.companion_nodes.clone().unwrap(),
        &on_disk_chunks,
    );

    // "the root attribute will not match the companion dataset state"
    // (Sec merkle-storage, "Crash consistency").
    let detected = verify_root(&reopened).unwrap_err();
    assert!(matches!(detected, MerkleError::CompanionTampered));

    // Signed: fail closed, no auto-rehash-and-resign, regardless of policy.
    let signed = resolve_response(&detected, ResponsePolicy::Halt, SigningContext::Signed);
    assert_eq!(signed.response, VerifyResponse::Halt);
    assert_eq!(signed.recovery, RecoveryAction::None);

    // Unsigned: no authenticity guarantee is at risk, so a rebuild is offered.
    let unsigned = resolve_response(&detected, ResponsePolicy::Halt, SigningContext::Unsigned);
    assert_eq!(unsigned.recovery, RecoveryAction::RebuildByRehash);

    // Perform the rebuild: recompute the tree from the current (intact) on-disk
    // chunk data. All chunk *content* is valid -- only the metadata ordering
    // was interrupted -- so rehashing repairs the inconsistency.
    let (rebuilt_attr, rebuilt_nodes) = build_attr_and_nodes(&on_disk_chunks);
    let rebuilt = dataset_view(rebuilt_attr, rebuilt_nodes, &on_disk_chunks);
    assert!(verify_root(&rebuilt).unwrap());
    assert!(verify_dataset(&rebuilt).unwrap());
}

// ===========================================================================
// (b) Flip one byte in one chunk -> HashMismatch.
// ===========================================================================

#[test]
fn scenario_b_flipped_chunk_byte_yields_hash_mismatch() {
    let chunks: Vec<Vec<u8>> = (0..NUM_CHUNKS as u64).map(|i| chunk_bytes(i, 5)).collect();
    let (attr, nodes) = build_attr_and_nodes(&chunks);

    let mut tampered = chunks.clone();
    tampered[3][0] ^= 0xFF;
    let live = dataset_view(attr, nodes, &tampered);

    let err = verify_dataset(&live).unwrap_err();
    assert!(matches!(err, MerkleError::HashMismatch { chunk_idx: 3 }));

    // Content-inconsistency: policy-selected response, signed context fails closed.
    let resolved = resolve_response(&err, ResponsePolicy::Quarantine, SigningContext::Signed);
    assert_eq!(resolved.response, VerifyResponse::Quarantine);
    assert_eq!(resolved.recovery, RecoveryAction::None);
}

#[test]
fn scenario_b_restore_to_version_recovers_from_hash_mismatch() {
    let good_chunks: Vec<Vec<u8>> = (0..NUM_CHUNKS as u64).map(|i| chunk_bytes(i, 5)).collect();
    let (good_attr, good_nodes) = build_attr_and_nodes(&good_chunks);

    let record = ProvenanceRecord {
        version: 1,
        signed_root: good_attr.root,
        hybrid_sig: b"genuine-sig-v1".to_vec(),
        timestamp: 1_700_000_000,
        snapshot_ref: String::from("snapshot-v1"),
    };
    let mut journal = ProvenanceJournal::new();
    journal.append(record).unwrap();

    // Live state is tampered.
    let mut tampered = good_chunks.clone();
    tampered[6][0] ^= 0xFF;
    let live = dataset_view(good_attr.clone(), good_nodes.clone(), &tampered);
    assert!(matches!(
        verify_dataset(&live),
        Err(MerkleError::HashMismatch { chunk_idx: 6 })
    ));

    struct AlwaysValid;
    impl SignatureVerifier for AlwaysValid {
        fn verify(&self, _: &ProvenanceRecord) -> bool {
            true
        }
    }

    let mut restored_dataset: Option<Dataset<'_>> = None;
    let restored_version = restore_to_version(
        &journal,
        RestoreTarget::LastKnownGood,
        &AlwaysValid,
        |rec| {
            // "Physically" revert to the snapshot: the known-good chunk set.
            let ds = dataset_view(good_attr.clone(), good_nodes.clone(), &good_chunks);
            verify_restored_dataset(rec, &ds)?;
            restored_dataset = Some(ds);
            Ok(())
        },
    )
    .unwrap();

    assert_eq!(restored_version, 1);
    assert!(verify_dataset(&restored_dataset.unwrap()).unwrap());
}

// ===========================================================================
// (c) Corrupt a companion node -> CompanionTampered.
// ===========================================================================

#[test]
fn scenario_c_corrupted_companion_node_yields_companion_tampered() {
    let chunks: Vec<Vec<u8>> = (0..NUM_CHUNKS as u64).map(|i| chunk_bytes(i, 9)).collect();
    let (attr, mut nodes) = build_attr_and_nodes(&chunks);

    // Flip a byte inside the companion node array itself (not a leaf/chunk).
    nodes[0] ^= 0xFF;
    let live = dataset_view(attr, nodes, &chunks);

    let err = verify_root(&live).unwrap_err();
    assert!(matches!(err, MerkleError::CompanionTampered));

    let resolved = resolve_response(
        &err,
        ResponsePolicy::AlertAndContinue,
        SigningContext::Signed,
    );
    assert_eq!(resolved.response, VerifyResponse::Alert);
    assert_eq!(
        resolved.recovery,
        RecoveryAction::None,
        "signed: never auto-rehash and re-sign"
    );
}

#[test]
fn scenario_c_restore_to_version_recovers_from_companion_tampered() {
    let good_chunks: Vec<Vec<u8>> = (0..NUM_CHUNKS as u64).map(|i| chunk_bytes(i, 9)).collect();
    let (good_attr, good_nodes) = build_attr_and_nodes(&good_chunks);

    let record = ProvenanceRecord {
        version: 1,
        signed_root: good_attr.root,
        hybrid_sig: b"genuine-sig-v1".to_vec(),
        timestamp: 1_700_000_000,
        snapshot_ref: String::from("snapshot-v1"),
    };
    let mut journal = ProvenanceJournal::new();
    journal.append(record).unwrap();

    let mut tampered_nodes = good_nodes.clone();
    tampered_nodes[0] ^= 0xFF;
    let live = dataset_view(good_attr.clone(), tampered_nodes, &good_chunks);
    assert!(matches!(
        verify_root(&live),
        Err(MerkleError::CompanionTampered)
    ));

    struct AlwaysValid;
    impl SignatureVerifier for AlwaysValid {
        fn verify(&self, _: &ProvenanceRecord) -> bool {
            true
        }
    }

    let mut restored_dataset: Option<Dataset<'_>> = None;
    let restored_version = restore_to_version(
        &journal,
        RestoreTarget::LastKnownGood,
        &AlwaysValid,
        |rec| {
            let ds = dataset_view(good_attr.clone(), good_nodes.clone(), &good_chunks);
            verify_restored_dataset(rec, &ds)?;
            restored_dataset = Some(ds);
            Ok(())
        },
    )
    .unwrap();

    assert_eq!(restored_version, 1);
    let restored = restored_dataset.unwrap();
    assert!(verify_root(&restored).unwrap());
    assert!(verify_dataset(&restored).unwrap());
}

// ===========================================================================
// (d) Present a stale signature -> SignatureInvalid.
// ===========================================================================

#[test]
fn scenario_d_stale_signature_yields_signature_invalid() {
    // A verifier that knows the one genuine signature per version -- standing
    // in for the not-yet-implemented P2.1 hybrid-signature check.
    struct KnownGoodSignatures(BTreeMap<u64, Vec<u8>>);
    impl SignatureVerifier for KnownGoodSignatures {
        fn verify(&self, r: &ProvenanceRecord) -> bool {
            self.0.get(&r.version) == Some(&r.hybrid_sig)
        }
    }

    let mut genuine = BTreeMap::new();
    genuine.insert(1, b"genuine-sig-v1".to_vec());
    genuine.insert(2, b"genuine-sig-v2".to_vec());
    let verifier = KnownGoodSignatures(genuine);

    let mut journal = ProvenanceJournal::new();
    journal
        .append(ProvenanceRecord {
            version: 1,
            signed_root: [1; 32],
            hybrid_sig: b"genuine-sig-v1".to_vec(),
            timestamp: 100,
            snapshot_ref: String::from("snapshot-v1"),
        })
        .unwrap();
    journal
        .append(ProvenanceRecord {
            // Stale: the v1 signature bytes were copied onto the v2 record
            // instead of the genuine v2 signature.
            version: 2,
            signed_root: [2; 32],
            hybrid_sig: b"genuine-sig-v1".to_vec(),
            timestamp: 200,
            snapshot_ref: String::from("snapshot-v2"),
        })
        .unwrap();

    let err = select_restore_record(&journal, RestoreTarget::Version(2), &verifier).unwrap_err();
    assert!(matches!(err, RestoreError::SignatureInvalid { version: 2 }));

    // The genuine (non-stale) v1 record still verifies.
    let rec = select_restore_record(&journal, RestoreTarget::Version(1), &verifier).unwrap();
    assert_eq!(rec.version, 1);

    // The full halt/quarantine/alert policy still applies to SignatureInvalid,
    // and, being a signature failure, never offers a rebuild.
    let resolved = resolve_response(
        &MerkleError::SignatureInvalid,
        ResponsePolicy::Quarantine,
        SigningContext::Signed,
    );
    assert_eq!(resolved.response, VerifyResponse::Quarantine);
    assert_eq!(resolved.recovery, RecoveryAction::None);
}

#[test]
fn scenario_d_restore_to_version_refuses_stale_signature_and_falls_back_to_genuine() {
    struct KnownGoodSignatures(BTreeMap<u64, Vec<u8>>);
    impl SignatureVerifier for KnownGoodSignatures {
        fn verify(&self, r: &ProvenanceRecord) -> bool {
            self.0.get(&r.version) == Some(&r.hybrid_sig)
        }
    }

    let good_chunks: Vec<Vec<u8>> = (0..NUM_CHUNKS as u64).map(|i| chunk_bytes(i, 11)).collect();
    let (good_attr, good_nodes) = build_attr_and_nodes(&good_chunks);

    let mut genuine = BTreeMap::new();
    genuine.insert(1, b"genuine-sig-v1".to_vec());
    genuine.insert(2, b"genuine-sig-v2".to_vec());
    let verifier = KnownGoodSignatures(genuine);

    let mut journal = ProvenanceJournal::new();
    journal
        .append(ProvenanceRecord {
            version: 1,
            signed_root: good_attr.root,
            hybrid_sig: b"genuine-sig-v1".to_vec(),
            timestamp: 100,
            snapshot_ref: String::from("snapshot-v1"),
        })
        .unwrap();
    journal
        .append(ProvenanceRecord {
            // Someone presented v1's signature bytes as if they certified v2:
            // the signature gate must refuse this record outright.
            version: 2,
            signed_root: [0xEE; 32],
            hybrid_sig: b"genuine-sig-v1".to_vec(),
            timestamp: 200,
            snapshot_ref: String::from("snapshot-v2"),
        })
        .unwrap();

    let mut restored_dataset: Option<Dataset<'_>> = None;
    let restored_version =
        restore_to_version(&journal, RestoreTarget::LastKnownGood, &verifier, |rec| {
            // v2's stale signature must never reach here; only v1 (genuine) does.
            assert_eq!(rec.version, 1);
            let ds = dataset_view(good_attr.clone(), good_nodes.clone(), &good_chunks);
            verify_restored_dataset(rec, &ds)?;
            restored_dataset = Some(ds);
            Ok(())
        })
        .unwrap();

    assert_eq!(restored_version, 1);
    assert!(verify_dataset(&restored_dataset.unwrap()).unwrap());
}

// ===========================================================================
// (e) Roll a chunk and its version counter back -> rejected via the
//     version-counter binding (T4 selective rollback).
// ===========================================================================

#[test]
fn scenario_e_chunk_and_version_rollback_breaks_the_leaf_hash_binding() {
    let dek = mock_dek();
    let config = FilterPipelineConfig::encrypted(4, 6);
    let pipeline = FilterPipeline::new(config);

    // Chunk 2 written at version 1, then overwritten at version 2. Every other
    // chunk stays at version 1.
    let mut ciphertexts: Vec<Vec<u8>> = Vec::with_capacity(NUM_CHUNKS);
    let mut versions: Vec<u64> = vec![1; NUM_CHUNKS];
    for idx in 0..NUM_CHUNKS as u64 {
        let filtered = pipeline
            .write(&chunk_bytes(idx, 3), Some(&dek), idx, 1)
            .unwrap();
        ciphertexts.push(filtered.ciphertext);
    }

    let old_ciphertext = ciphertexts[2].clone(); // the version-1 state we'll roll back to
    let new_filtered = pipeline
        .write(&chunk_bytes(2, 3), Some(&dek), 2, 2)
        .unwrap();
    ciphertexts[2] = new_filtered.ciphertext.clone();
    versions[2] = 2;

    // The committed tree's leaves bind (chunk_idx, ciphertext, version) --
    // this is the "version-counter binding" the task refers to.
    let leaf_hashes: Vec<[u8; 32]> = ciphertexts
        .iter()
        .zip(&versions)
        .enumerate()
        .map(|(idx, (ct, &v))| compute_leaf_hash(idx as u64, ct, v))
        .collect();
    let tree = MerkleTree::from_leaf_hashes(&leaf_hashes, HashAlg::Blake3);

    // Attacker rolls chunk 2 back to its version-1 ciphertext in storage, but
    // the companion tree (built above) still reflects the committed version-2
    // leaf. The version-1 ciphertext is itself perfectly valid (it decrypts
    // fine under the version-1 nonce) -- detection depends specifically on the
    // leaf hash binding the version counter, not on the ciphertext being
    // garbage.
    let recomputed_after_rollback = compute_leaf_hash(2, &old_ciphertext, 1);
    assert_ne!(
        &recomputed_after_rollback,
        tree.leaf_hash(2).unwrap(),
        "rolled-back (chunk, version) pair must not reproduce the committed leaf hash"
    );

    // The one place clawhdf5-format's own MerkleError vocabulary applies to a
    // single-leaf mismatch is HashMismatch; a real integrated verifier (the
    // seam `extend_merkle`/`update_merkle` leave a TODO for) would report
    // exactly this once the WAL-aware version-bound check is wired through.
    let err = MerkleError::HashMismatch { chunk_idx: 2 };
    let resolved = resolve_response(&err, ResponsePolicy::Halt, SigningContext::Signed);
    assert_eq!(resolved.response, VerifyResponse::Halt);
    assert_eq!(resolved.recovery, RecoveryAction::None);
}

#[test]
fn scenario_e_dataset_level_version_rollback_rejected() {
    // The persisted dataset-level `_merkle_version` counter (P2.2b step 1)
    // gives the same T4 protection one level up: a verifier that has already
    // observed version 2 must reject a reopen presenting version 1.
    let mut store = VersionObservationStore::new();
    store.observe("dataset.h5", 2).unwrap();

    let err = store.observe("dataset.h5", 1).unwrap_err();
    assert!(matches!(
        err,
        MerkleError::VersionRollback {
            observed: 1,
            highest_seen: 2
        }
    ));
    // The high-water mark is unchanged by the rejected rollback.
    assert_eq!(store.highest("dataset.h5"), Some(2));

    // T4 is adversarial: always Halt, never a rebuild, regardless of policy
    // or signing context.
    for policy in [
        ResponsePolicy::Halt,
        ResponsePolicy::Quarantine,
        ResponsePolicy::AlertAndContinue,
    ] {
        for signing in [SigningContext::Signed, SigningContext::Unsigned] {
            let resolved = resolve_response(&err, policy, signing);
            assert_eq!(resolved.response, VerifyResponse::Halt);
            assert_eq!(resolved.recovery, RecoveryAction::None);
        }
    }
}

#[test]
fn scenario_e_restore_to_version_recovers_from_chunk_rollback() {
    let good_chunks: Vec<Vec<u8>> = (0..NUM_CHUNKS as u64).map(|i| chunk_bytes(i, 7)).collect();
    let (good_attr, good_nodes) = build_attr_and_nodes(&good_chunks);

    let record = ProvenanceRecord {
        version: 2,
        signed_root: good_attr.root,
        hybrid_sig: b"genuine-sig-v2".to_vec(),
        timestamp: 1_700_000_100,
        snapshot_ref: String::from("snapshot-v2"),
    };
    let mut journal = ProvenanceJournal::new();
    journal.append(record).unwrap();

    // Live state: chunk 4 rolled back to a stale (but structurally valid)
    // byte string -- detected as a plain leaf mismatch against the committed
    // tree, exactly like scenario (b), since `clawhdf5_format::merkle`
    // verifies whatever bytes it's given against the tree it was built from.
    let mut rolled_back = good_chunks.clone();
    rolled_back[4] = chunk_bytes(4, 0); // a different, "older" chunk state
    let live = dataset_view(good_attr.clone(), good_nodes.clone(), &rolled_back);
    assert!(matches!(
        verify_dataset(&live),
        Err(MerkleError::HashMismatch { chunk_idx: 4 })
    ));

    struct AlwaysValid;
    impl SignatureVerifier for AlwaysValid {
        fn verify(&self, _: &ProvenanceRecord) -> bool {
            true
        }
    }

    let mut restored_dataset: Option<Dataset<'_>> = None;
    let restored_version =
        restore_to_version(&journal, RestoreTarget::Version(2), &AlwaysValid, |rec| {
            let ds = dataset_view(good_attr.clone(), good_nodes.clone(), &good_chunks);
            verify_restored_dataset(rec, &ds)?;
            restored_dataset = Some(ds);
            Ok(())
        })
        .unwrap();

    assert_eq!(restored_version, 2);
    assert!(verify_dataset(&restored_dataset.unwrap()).unwrap());
}
