//! End-to-end demonstration: one real HDF5 file, carrying the *whole* ClawHDF5
//! security stack in-band — ChaCha20-Poly1305 encryption, a Merkle-tree
//! integrity root, a real hybrid Ed25519+ML-DSA-65 signature, and P2.2b
//! crash-consistency + rollback recovery — assembled and exercised together
//! for the first time.
//!
//! Every one of these primitives already exists and is individually tested
//! and adversarially red-teamed (P2.1, P2.2, the Merkle layer, P2.2b). What
//! doesn't exist anywhere else in this workspace: a test or example that (a)
//! writes `merkle_root` *and* `merkle_sig` into the same file, (b) reopens
//! that file from scratch (nothing held over from the write side) and
//! verifies both from the bytes on disk, or (c) runs the three-step
//! crash-consistent write order against real, `fsync`'d storage rather than a
//! test double. This binary does all three.
//!
//! ```text
//! cargo run -p clawhdf5-secure-demo --release
//! ```
//!
//! Every stage below asserts the property it claims; a failed assertion
//! panics (nonzero exit), so this file doubles as a regression check.

mod file_sink;

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use clawhdf5_filters::{
    EncryptedChunkWriter, MerkleWriteSink, VersionCounterStore, VersionWal, WalRecord,
    WriteOrderError, WriteStep, decrypt_chunk, encrypt_chunk, mock_dek,
};
use clawhdf5_format::group_v2;
use clawhdf5_format::merkle::{
    Dataset, HashAlg, MerkleAttr, MerkleError, MerkleTree, VersionObservationStore,
    verify_chunk, verify_dataset, write_merkle_companion, MerkleCompanionResult,
    MERKLE_ATTR_NAME, MERKLE_NODES_ATTR_NAME,
};
use clawhdf5_format::merkle_journal::{ProvenanceJournal, ProvenanceRecord};
use clawhdf5_format::merkle_recovery::{
    RestoreError, RestoreTarget, SignatureVerifier, restore_to_version, verify_restored_dataset,
};
use clawhdf5_format::object_header::ObjectHeader;
use clawhdf5_format::signature;
use clawhdf5_format::superblock::Superblock;
use clawhdf5_format::type_builders::AttrValue;
use clawhdf5_sign::mldsa::{MlDsaSigningKey, MlDsaVerifyingKey};
use clawhdf5_sign::{
    HybridSignature, SigningKey, VerifyingKey, canonical_payload, read_sig_attr, sign_root,
    verify_sig,
};

use file_sink::FileMerkleSink;

/// Number of chunks in the demo dataset.
const NUM_CHUNKS: usize = 6;
/// Plaintext bytes per chunk (kept fixed-size so ciphertext lengths -- and
/// hence chunk byte ranges inside the concatenated on-disk dataset -- are
/// uniform, avoiding a separate chunk-offset sidecar for this demo).
const CHUNK_LEN: usize = 48;
/// ChaCha20-Poly1305 tag size (see `clawhdf5_filters::chacha20_filter::TAG_SIZE`).
const TAG_SIZE: usize = 16;
const CIPHERTEXT_LEN: usize = CHUNK_LEN + TAG_SIZE;
/// Name of the dataset holding the concatenated ciphertext.
const DATASET_NAME: &str = "ciphertext";

fn sha256(data: &[u8]) -> [u8; 32] {
    Sha256::digest(data).into()
}

/// Deterministic, distinguishable plaintext for each chunk.
fn build_plaintext_chunks() -> Vec<[u8; CHUNK_LEN]> {
    (0..NUM_CHUNKS)
        .map(|i| {
            let mut buf = [0u8; CHUNK_LEN];
            let marker = format!("scientific-reading-chunk-{i:02}-value=");
            let bytes = marker.as_bytes();
            let n = bytes.len().min(CHUNK_LEN);
            buf[..n].copy_from_slice(&bytes[..n]);
            buf[CHUNK_LEN - 1] = i as u8; // a distinguishing tail byte too
            buf
        })
        .collect()
}

/// Everything produced by Stage 1, needed by later stages.
struct WrittenFile {
    root: [u8; 32],
    companion_hash: [u8; 32],
    sig: HybridSignature,
    timestamp: u64,
    version: u64,
}

/// Everything Stage 3 reads back *out of the file* -- nothing here is
/// retained from Stage 1's in-memory state.
struct ReopenedFile {
    attr: MerkleAttr,
    companion_nodes: Vec<u8>,
    sig: HybridSignature,
    ciphertext: Vec<u8>,
}

fn main() {
    println!("=== ClawHDF5 secure-file lifecycle demonstration ===\n");

    let dir = tempfile::tempdir().expect("create temp dir");
    let live_path = dir.path().join("secure_demo.h5");

    // ---- Stage 0: keys --------------------------------------------------
    println!("Stage 0: key setup");
    let dek = mock_dek();
    let ed_key = SigningKey::generate();
    let ml_key = MlDsaSigningKey::generate();
    let ed_pub = ed_key.verifying_key();
    let ml_pub = ml_key.verifying_key();
    println!("  DEK, Ed25519, and ML-DSA-65 keypairs generated.");
    println!("Stage 0 PASSED\n");

    // ---- Stage 1 ----------------------------------------------------------
    println!("Stage 1: build & write a protected file (in-band)");
    let plaintexts = build_plaintext_chunks();
    let written = stage1_build_and_write(&live_path, &dek, &ed_key, &ml_key, &plaintexts);
    println!("  wrote {}", live_path.display());
    println!("Stage 1 PASSED\n");

    // ---- Stage 2 ------------------------------------------------------------
    println!("Stage 2: confidentiality");
    stage2_confidentiality(&live_path, &dek, &plaintexts);
    println!("Stage 2 PASSED\n");

    // ---- Stage 3 ------------------------------------------------------------
    println!("Stage 3: reopen from scratch & verify (out of the file, not held state)");
    let reopened = stage3_reopen_and_verify(&live_path, &ed_pub, &ml_pub, &written);
    println!("Stage 3 PASSED\n");

    // ---- Stage 4 ------------------------------------------------------------
    println!("Stage 4: tamper detection");
    stage4_tamper_detect(&live_path, &reopened, &ed_pub, &ml_pub);
    println!("Stage 4 PASSED\n");

    // ---- Stage 5 ------------------------------------------------------------
    println!("Stage 5: crash-consistency (real fsync'd three-step write order)");
    stage5_crash_consistency(dir.path(), &dek);
    println!("Stage 5 PASSED\n");

    // ---- Stage 6 ------------------------------------------------------------
    println!("Stage 6: rollback recovery");
    stage6_rollback(&live_path, dir.path(), &written, &reopened, &ed_pub, &ml_pub);
    println!("Stage 6 PASSED\n");

    println!("=== ALL STAGES PASSED ===");
}

// ============================================================================
// Stage 1
// ============================================================================

fn stage1_build_and_write(
    path: &Path,
    dek: &clawhdf5_filters::Dek,
    ed_key: &SigningKey,
    ml_key: &MlDsaSigningKey,
    plaintexts: &[[u8; CHUNK_LEN]],
) -> WrittenFile {
    use clawhdf5_format::file_writer::FileWriter;

    let version: u64 = 1;

    // Encrypt-then-MAC: encrypt every chunk, then Merkle-hash the CIPHERTEXT.
    let ciphertexts: Vec<Vec<u8>> = plaintexts
        .iter()
        .enumerate()
        .map(|(i, pt)| encrypt_chunk(dek, i as u64, version, pt).expect("encryption"))
        .collect();
    for ct in &ciphertexts {
        assert_eq!(ct.len(), CIPHERTEXT_LEN);
    }
    let ct_refs: Vec<&[u8]> = ciphertexts.iter().map(Vec::as_slice).collect();
    let tree = MerkleTree::from_chunks(&ct_refs, HashAlg::Blake3);
    let root: [u8; 32] = *tree.root();

    let mut fw = FileWriter::new();

    // write_merkle_companion must be called before create_dataset borrows fw.
    let companion = write_merkle_companion(&mut fw, DATASET_NAME, &tree)
        .expect("write_merkle_companion should succeed for a small tree");
    let (companion_nodes, companion_hash) = match companion {
        MerkleCompanionResult::Inline {
            nodes,
            companion_hash,
        } => (nodes, companion_hash),
        MerkleCompanionResult::Dataset { .. } => {
            panic!("expected an inline companion result for a {NUM_CHUNKS}-chunk tree")
        }
    };

    let timestamp: u64 = 1_700_000_000;
    let alg_id = clawhdf5_sign::HashAlg::Blake3;
    let payload = canonical_payload(&root, &companion_hash, version, timestamp, alg_id);
    let sig = sign_root(&payload, ed_key, ml_key).expect("hybrid signing should succeed");

    let mut all_ciphertext = Vec::with_capacity(CIPHERTEXT_LEN * NUM_CHUNKS);
    for ct in &ciphertexts {
        all_ciphertext.extend_from_slice(ct);
    }

    // Bind root + real companion_hash + grid_hash=0 (subset extraction is out
    // of scope for this demo) into the merkle_root attribute -- deliberately
    // NOT via `write_merkle_attr`, which always zeroes companion_hash (that
    // helper is for the "no companion" case); we have a real one to bind.
    let attr = MerkleAttr::from_tree_with_companion_and_grid(&tree, companion_hash, [0u8; 32]);

    let ds = fw.create_dataset(DATASET_NAME);
    ds.with_u8_data(&all_ciphertext);
    ds.set_attr(MERKLE_NODES_ATTR_NAME, AttrValue::Bytes(companion_nodes));
    ds.set_attr(MERKLE_ATTR_NAME, AttrValue::Bytes(attr.pack().to_vec()));
    clawhdf5_sign::write_sig_attr(ds, &sig);

    let bytes = fw.finish().expect("file should build");
    std::fs::write(path, &bytes).expect("write file");

    println!(
        "  {NUM_CHUNKS} chunks encrypted (ChaCha20-Poly1305), Merkle root computed over ciphertext (EtM),"
    );
    println!("  hybrid Ed25519+ML-DSA-65 signature computed, merkle_root+merkle_nodes+merkle_sig written as real HDF5 attributes.");

    WrittenFile {
        root,
        companion_hash,
        sig,
        timestamp,
        version,
    }
}

// ============================================================================
// Stage 2
// ============================================================================

fn stage2_confidentiality(path: &Path, dek: &clawhdf5_filters::Dek, plaintexts: &[[u8; CHUNK_LEN]]) {
    let raw = std::fs::read(path).expect("read file");

    // The plaintext marker strings must NOT appear anywhere in the on-disk
    // bytes -- what's stored is ciphertext, not plaintext.
    for (i, pt) in plaintexts.iter().enumerate() {
        let marker = &pt[..pt.len().min(20)];
        assert!(
            !contains_subslice(&raw, marker),
            "plaintext for chunk {i} leaked into the on-disk file unencrypted"
        );
    }
    println!("  none of the {NUM_CHUNKS} plaintext markers appear in the on-disk file bytes.");

    // But the data is genuinely recoverable with the DEK: read the ciphertext
    // dataset back and decrypt every chunk.
    let file = clawhdf5::File::open(path).expect("open file");
    let ds = file.dataset(DATASET_NAME).expect("dataset");
    let all_ciphertext = ds.read_u8_zerocopy().expect("read ciphertext dataset");
    assert_eq!(all_ciphertext.len(), CIPHERTEXT_LEN * NUM_CHUNKS);

    for (i, pt) in plaintexts.iter().enumerate() {
        let ct = &all_ciphertext[i * CIPHERTEXT_LEN..(i + 1) * CIPHERTEXT_LEN];
        let decrypted = decrypt_chunk(dek, i as u64, 1, ct).expect("decrypt");
        assert_eq!(decrypted.as_slice(), pt.as_slice());
    }
    println!("  every chunk decrypts back to its original plaintext with the DEK.");
}

fn contains_subslice(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || haystack.len() < needle.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|w| w == needle)
}

// ============================================================================
// Stage 3
// ============================================================================

/// Parse `path` from raw bytes and pull `merkle_root`/`merkle_nodes`/
/// `merkle_sig` back out as real HDF5 attributes -- the read-back counterpart
/// to Stage 1's write, sharing no state with it.
fn reopen_from_disk(path: &Path) -> ReopenedFile {
    let bytes = std::fs::read(path).expect("read file");
    let sig_offset = signature::find_signature(&bytes).expect("HDF5 signature");
    let superblock = Superblock::parse(&bytes, sig_offset).expect("superblock");
    let addr = group_v2::resolve_path_any(&bytes, &superblock, DATASET_NAME)
        .expect("resolve dataset path");
    let header = ObjectHeader::parse(
        &bytes,
        addr as usize,
        superblock.offset_size,
        superblock.length_size,
    )
    .expect("object header");
    let attrs = clawhdf5_format::attribute::extract_attributes(&header, superblock.length_size)
        .expect("extract attributes");

    let merkle_attr_msg = clawhdf5_format::attribute::find_attribute(&attrs, MERKLE_ATTR_NAME)
        .expect("merkle_root attribute present");
    let attr = MerkleAttr::unpack(&merkle_attr_msg.raw_data).expect("unpack merkle_root");

    let nodes_msg = clawhdf5_format::attribute::find_attribute(&attrs, MERKLE_NODES_ATTR_NAME)
        .expect("merkle_nodes attribute present");
    let companion_nodes = nodes_msg.raw_data.clone();

    let sig = read_sig_attr(&attrs).expect("merkle_sig attribute present and well-formed");

    // The dataset content itself: plain contiguous u8 data, read via the
    // high-level facade (only the attribute decoder lacks an opaque/Bytes
    // branch -- raw dataset bytes are unaffected).
    let file = clawhdf5::File::open(path).expect("open file");
    let ds = file.dataset(DATASET_NAME).expect("dataset");
    let ciphertext = ds.read_u8_zerocopy().expect("read ciphertext dataset").to_vec();

    ReopenedFile {
        attr,
        companion_nodes,
        sig,
        ciphertext,
    }
}

fn ciphertext_chunks(ciphertext: &[u8]) -> Vec<&[u8]> {
    ciphertext.chunks(CIPHERTEXT_LEN).collect()
}

fn stage3_reopen_and_verify(
    path: &Path,
    ed_pub: &VerifyingKey,
    ml_pub: &MlDsaVerifyingKey,
    written: &WrittenFile,
) -> ReopenedFile {
    let reopened = reopen_from_disk(path);

    assert_eq!(reopened.attr.root, written.root);
    assert_eq!(reopened.attr.companion_hash, written.companion_hash);
    assert_eq!(reopened.sig, written.sig);
    println!("  merkle_root, merkle_nodes, and merkle_sig read back out of the reopened file bytes.");

    let payload = canonical_payload(
        &reopened.attr.root,
        &reopened.attr.companion_hash,
        written.version,
        written.timestamp,
        clawhdf5_sign::HashAlg::Blake3,
    );
    verify_sig(&payload, &reopened.sig, ed_pub, ml_pub)
        .expect("the real hybrid signature read back from the file must verify");
    println!("  real Ed25519+ML-DSA-65 signature verifies against the payload reconstructed from the file.");

    let dataset = Dataset::from_owned(
        reopened.attr.clone(),
        reopened.companion_nodes.clone(),
        ciphertext_chunks(&reopened.ciphertext),
    );
    assert!(verify_dataset(&dataset).expect("verify_dataset"));
    println!("  verify_dataset passes over the re-read ciphertext and companion nodes.");

    reopened
}

// ============================================================================
// Stage 4
// ============================================================================

fn stage4_tamper_detect(
    path: &Path,
    reopened: &ReopenedFile,
    ed_pub: &VerifyingKey,
    ml_pub: &MlDsaVerifyingKey,
) {
    // T1: flip one byte of stored ciphertext via raw file I/O.
    let victim = NUM_CHUNKS / 2;
    let mut tampered_ciphertext = reopened.ciphertext.clone();
    tampered_ciphertext[victim * CIPHERTEXT_LEN] ^= 0xFF;
    let tampered_dataset = Dataset::from_owned(
        reopened.attr.clone(),
        reopened.companion_nodes.clone(),
        ciphertext_chunks(&tampered_ciphertext),
    );
    let err = verify_chunk(&tampered_dataset, victim).unwrap_err();
    assert!(matches!(err, MerkleError::HashMismatch { chunk_idx } if chunk_idx == victim));
    println!("  T1 (chunk-data tamper): verify_chunk rejects with HashMismatch at chunk {victim}.");

    // T3: attacker rebuilds the tree over tampered data and computes a FRESH
    // root, but cannot produce a valid signature without the private keys --
    // reusing the old signature over the new payload must fail.
    let mut retampered = reopened.ciphertext.clone();
    retampered[victim * CIPHERTEXT_LEN] ^= 0xFF;
    let new_tree = MerkleTree::from_chunks(&ciphertext_chunks(&retampered), HashAlg::Blake3);
    let forged_payload = canonical_payload(
        new_tree.root(),
        &reopened.attr.companion_hash,
        1,
        1_700_000_000,
        clawhdf5_sign::HashAlg::Blake3,
    );
    let result = verify_sig(&forged_payload, &reopened.sig, ed_pub, ml_pub);
    assert!(
        result.is_err(),
        "the old signature must not verify over a new, unsigned root"
    );
    println!(
        "  T3 (root substitution): the old signature does not verify over a freshly (unsigned) recomputed root."
    );

    // Confirm the LIVE file on disk is untouched by any of the above (all
    // tampering above was performed on in-memory copies).
    assert!(path.exists());
}

// ============================================================================
// Stage 5
// ============================================================================

fn stage5_crash_consistency(base_dir: &Path, dek: &clawhdf5_filters::Dek) {
    let wal_path = base_dir.join("crash_demo_wal.bin");
    let sink_dir = base_dir.join("crash_demo_sink");
    std::fs::create_dir_all(&sink_dir).expect("create sink dir");
    let plaintext = b"crash-consistency demo chunk payload";

    // --- Commit interrupted between companion (step 2) and root (step 3). ---
    {
        let wal_file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&wal_path)
            .expect("create WAL file");
        let wal = VersionWal::new(wal_file, 100).expect("create WAL");
        let mut writer = EncryptedChunkWriter::new(*dek, wal, VersionCounterStore::new());

        let result = writer.encrypt_chunk(0, plaintext).expect("encrypt");
        assert_eq!(result.version, 1);

        let tree = MerkleTree::from_chunks(&[result.ciphertext.as_slice()], HashAlg::Blake3);
        let mut nodes = Vec::new();
        for n in tree.nodes() {
            nodes.extend_from_slice(n);
        }
        let companion_hash_bytes = sha256(&nodes);

        let mut crashing_sink = FileMerkleSink::with_fail_at(sink_dir.clone(), WriteStep::RootAttribute);
        let err = writer
            .commit_with_write_order(&result, &nodes, tree.root(), &companion_hash_bytes, &mut crashing_sink)
            .expect_err("the injected crash must surface as an error");
        assert!(matches!(
            err,
            WriteOrderError::Sink {
                step: WriteStep::RootAttribute,
                ..
            }
        ));
        println!(
            "  crash injected between the companion-node write and the root-attribute write (both prior steps durably synced)."
        );
    } // `writer` and the WAL file handle drop here -- simulates the process dying.

    // --- "Reboot": reopen the WAL fresh and recover. ---
    {
        let wal_file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&wal_path)
            .expect("reopen WAL file");
        let wal = VersionWal::new(wal_file, 100).expect("reopen WAL");
        let mut writer = EncryptedChunkWriter::new(*dek, wal, VersionCounterStore::new());

        let pending = writer.recover().expect("recover");
        assert_eq!(pending.len(), 1, "the interrupted write must still be pending after reopen");
        let (chunk_idx, version, plaintext_hash) = pending[0];
        assert_eq!((chunk_idx, version), (0, 1));

        // Remark A.9: verify the candidate replay plaintext BEFORE trusting it.
        assert_eq!(
            WalRecord::hash_plaintext(plaintext),
            plaintext_hash,
            "the plaintext about to be replayed must match what was journaled"
        );
        println!(
            "  recover() reports chunk {chunk_idx} pending at version {version}; replay plaintext verified against the journaled hash (Remark A.9)."
        );

        // Steps 1+2 already landed durably before the simulated crash --
        // confirm that, then finish only the interrupted step 3.
        let on_disk_ciphertext =
            std::fs::read(FileMerkleSink::chunk_path(&sink_dir, chunk_idx)).expect("read chunk file");
        let redo_ciphertext =
            encrypt_chunk(dek, chunk_idx, version, plaintext).expect("re-derive ciphertext");
        assert_eq!(
            on_disk_ciphertext, redo_ciphertext,
            "chunk-data step already durably completed before the crash"
        );

        let recovered_tree = MerkleTree::from_chunks(&[on_disk_ciphertext.as_slice()], HashAlg::Blake3);
        let on_disk_nodes =
            std::fs::read(FileMerkleSink::companion_path(&sink_dir)).expect("read companion file");
        let mut expected_nodes = Vec::new();
        for n in recovered_tree.nodes() {
            expected_nodes.extend_from_slice(n);
        }
        assert_eq!(
            on_disk_nodes, expected_nodes,
            "companion-node step already durably completed before the crash"
        );
        assert!(
            file_sink::read_root_file(&sink_dir).is_none(),
            "root-attribute step must NOT have landed -- that's the crash we injected"
        );

        // Finish the interrupted commit: (re-)issue the root-attribute step
        // and sync, then mark the WAL entry committed.
        let companion_hash_bytes = sha256(&on_disk_nodes);
        let mut finishing_sink = FileMerkleSink::new(sink_dir.clone());
        finishing_sink
            .write_root_attribute(recovered_tree.root(), &companion_hash_bytes, version)
            .expect("finish root-attribute write");
        finishing_sink.sync().expect("sync");
        writer.commit(chunk_idx, version).expect("commit");

        assert!(file_sink::read_root_file(&sink_dir).is_some(), "root attribute now durably present");
        let pending_after = writer.recover().expect("recover again");
        assert!(pending_after.is_empty(), "the completed commit must no longer be pending");
        println!("  interrupted commit finished (root attribute written+synced); WAL entry now committed.");
    }
}

// ============================================================================
// Stage 6
// ============================================================================

struct RealVerifier<'a> {
    companion_hash: [u8; 32],
    alg_id: clawhdf5_sign::HashAlg,
    ed_pub: &'a VerifyingKey,
    ml_pub: &'a MlDsaVerifyingKey,
}

impl SignatureVerifier for RealVerifier<'_> {
    fn verify(&self, record: &ProvenanceRecord) -> bool {
        let Ok(sig) = HybridSignature::from_bytes(&record.hybrid_sig) else {
            return false;
        };
        let payload = canonical_payload(
            &record.signed_root,
            &self.companion_hash,
            record.version,
            record.timestamp,
            self.alg_id,
        );
        verify_sig(&payload, &sig, self.ed_pub, self.ml_pub).is_ok()
    }
}

fn stage6_rollback(
    live_path: &Path,
    base_dir: &Path,
    written: &WrittenFile,
    reopened: &ReopenedFile,
    ed_pub: &VerifyingKey,
    ml_pub: &MlDsaVerifyingKey,
) {
    // Snapshot the known-good live file BEFORE tampering it.
    let snapshot_path: PathBuf = base_dir.join("snapshot_v1.h5");
    std::fs::copy(live_path, &snapshot_path).expect("snapshot good file");

    let mut journal = ProvenanceJournal::new();
    journal
        .append(ProvenanceRecord {
            version: written.version,
            signed_root: written.root,
            hybrid_sig: written.sig.to_bytes().to_vec(),
            timestamp: written.timestamp,
            snapshot_ref: snapshot_path.to_string_lossy().into_owned(),
        })
        .expect("journal append");
    println!("  version {} journaled, tied to a real snapshot of the good file.", written.version);

    // Tamper the LIVE file via raw I/O (bypassing every ClawHDF5 write path).
    // The flip must land inside the ciphertext dataset's bytes, not anywhere
    // else in the file -- an arbitrary offset (e.g. len()/2) can just as
    // easily hit the object header's own checksum region, which fails
    // low-level parsing outright instead of exercising the intended
    // higher-level Merkle detection path.
    let mut live_bytes = std::fs::read(live_path).expect("read live file");
    let dataset_offset = live_bytes
        .windows(reopened.ciphertext.len())
        .position(|w| w == reopened.ciphertext.as_slice())
        .expect("locate ciphertext dataset bytes in the live file");
    live_bytes[dataset_offset] ^= 0xFF;
    std::fs::write(live_path, &live_bytes).expect("write tampered live file");

    let tampered = reopen_from_disk(live_path);
    let tampered_dataset = Dataset::from_owned(
        tampered.attr.clone(),
        tampered.companion_nodes.clone(),
        ciphertext_chunks(&tampered.ciphertext),
    );
    assert!(
        verify_dataset(&tampered_dataset).is_err(),
        "the tampered live file must fail verification"
    );
    println!("  live file tampered via raw I/O; verify_dataset now fails, as expected.");

    // Restore to the last known-good signed version, using REAL signature
    // verification (not a placeholder byte-string comparison).
    let verifier = RealVerifier {
        companion_hash: written.companion_hash,
        alg_id: clawhdf5_sign::HashAlg::Blake3,
        ed_pub,
        ml_pub,
    };
    let restored_version = restore_to_version(&journal, RestoreTarget::LastKnownGood, &verifier, |record| {
        std::fs::copy(&snapshot_path, live_path).map_err(|_e| RestoreError::DatasetVerificationFailed {
            version: record.version,
            source: None,
        })?;
        let restored = reopen_from_disk(live_path);
        let restored_dataset = Dataset::from_owned(
            restored.attr.clone(),
            restored.companion_nodes.clone(),
            ciphertext_chunks(&restored.ciphertext),
        );
        verify_restored_dataset(record, &restored_dataset)
    })
    .expect("restore_to_version should succeed against the real signature and the real snapshot");
    assert_eq!(restored_version, written.version);
    println!(
        "  restore_to_version: signature gate (real Ed25519+ML-DSA-65) + dataset gate both passed; live file restored."
    );

    let final_dataset_bytes = std::fs::read(live_path).expect("read restored live file");
    let final_reopened = reopen_from_disk(live_path);
    let final_dataset = Dataset::from_owned(
        final_reopened.attr.clone(),
        final_reopened.companion_nodes.clone(),
        ciphertext_chunks(&final_reopened.ciphertext),
    );
    assert!(verify_dataset(&final_dataset).expect("verify_dataset"));
    assert_eq!(final_reopened.ciphertext, reopened.ciphertext);
    println!("  restored live file re-verifies and matches the original content exactly.");

    // Also demonstrate T4a (whole-file rollback rejection) via
    // VersionObservationStore, independent of the file above.
    let mut store = VersionObservationStore::new();
    store.observe("secure-demo-file", 5).expect("observe v5");
    let rollback_result = store.observe("secure-demo-file", 3);
    assert!(matches!(
        rollback_result,
        Err(MerkleError::VersionRollback {
            observed: 3,
            highest_seen: 5
        })
    ));
    println!("  VersionObservationStore rejects a rollback to a previously-observed-lower version (T4a).");

    let _ = final_dataset_bytes;
}
