//! The P2.4 white-box red-team attack suite (S2-D2-Yr2 §`sec:adversarial`).
//!
//! Every attack function follows the same shape mandated by the spec: open
//! (or build) a test HDF5-derived dataset, perform **one** targeted
//! modification using raw file/byte I/O (never a ClawHDF5 write API), then
//! call the verification API and assert the expected result.
//!
//! Attacks are split into two groups:
//!
//! - **Dataset attacks** (T1a, T1b, T2, T6a, subset-a/b/c) run against the
//!   [`HarnessDataset`] — either the real GOES-18 file's actual on-disk HDF5
//!   chunks, or the synthetic fallback — because they only need "a chunked
//!   Merkle-protected byte blob" and get more authentic coverage from real
//!   satellite chunk boundaries and filter-compressed sizes when available.
//! - **Mechanism attacks** (T3, T4a, T4b, T5, T6b, T7, T8) target a specific
//!   primitive (the provenance journal, the version counter, the AEAD leaf
//!   formula, the null-sentinel padding) that isn't meaningfully expressed in
//!   terms of "a chunk in the dataset," so each builds its own minimal fixture.

use std::time::Instant;

use sha2::{Digest, Sha256};

use clawhdf5_filters::{compute_leaf_hash, encrypt_chunk};
use clawhdf5_format::merkle::{
    Dataset, HashAlg, MerkleAttr, MerkleError, MerkleTree, VersionObservationStore, verify_chunk,
    verify_dataset, verify_root,
};
use clawhdf5_format::merkle_journal::{ProvenanceJournal, ProvenanceRecord};
use clawhdf5_format::merkle_recovery::{
    RestoreError, RestoreTarget, SignatureVerifier, select_restore_record,
};
use clawhdf5_format::selection::Selection;
use clawhdf5_format::subset_proof::{
    ChunkData, ChunkGridParams, LeafOrder, extract_subset, verify_subset,
};

use crate::fixture::HarnessDataset;
use crate::report::AttackResult;

fn companion_hash(nodes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(nodes);
    hasher.finalize().into()
}

/// Build the Merkle-protection state (tree, packed attribute, companion node
/// array) over a [`HarnessDataset`]'s current chunk boundaries.
fn build_merkle_state(ds: &HarnessDataset) -> (MerkleTree, MerkleAttr, Vec<u8>) {
    let chunks = ds.all_chunks(&ds.bytes);
    let tree = MerkleTree::from_chunks(&chunks, HashAlg::Blake3);
    let mut nodes = Vec::with_capacity(tree.nodes().len() * 32);
    for n in tree.nodes() {
        nodes.extend_from_slice(n);
    }
    let attr = MerkleAttr::from_tree_with_companion(&tree, companion_hash(&nodes));
    (tree, attr, nodes)
}

fn dataset_view<'a>(
    attr: MerkleAttr,
    nodes: Vec<u8>,
    ds: &'a HarnessDataset,
    bytes: &'a [u8],
) -> Dataset<'a> {
    Dataset::from_owned(attr, nodes, ds.all_chunks(bytes))
}

// ============================================================================
// Dataset attacks: T1a, T1b, T2, T6a, and the three subset-extraction attacks.
// ============================================================================

/// T1a — storage-level tampering (data): overwrite raw chunk bytes using a
/// hex-editor-style single-byte flip. `verify_chunk` must detect it.
pub fn t1a_chunk_data_tamper(ds: &HarnessDataset) -> AttackResult {
    let (_, attr, nodes) = build_merkle_state(ds);
    let victim = ds.chunk_count() / 2;

    // Raw file I/O: flip one byte inside the victim chunk's on-disk range,
    // bypassing every ClawHDF5 write path.
    let mut tampered = ds.bytes.clone();
    let (start, _) = ds.chunk_ranges[victim];
    tampered[start] ^= 0xFF;

    let start_time = Instant::now();
    let view = dataset_view(attr, nodes, ds, &tampered);
    let result = verify_chunk(&view, victim);
    let latency = start_time.elapsed();

    let detected =
        matches!(result, Err(MerkleError::HashMismatch { chunk_idx }) if chunk_idx == victim);
    AttackResult {
        threat_class: "T1",
        attack_id: "T1a",
        dataset: ds.label,
        detected,
        verifier_fn: "verify_chunk",
        latency,
        root_cause: None,
    }
}

/// T1b — storage-level tampering (internal tree nodes): modify a companion
/// node without altering the root. `verify_root` must detect it via the
/// companion-integrity hash.
pub fn t1b_companion_node_tamper(ds: &HarnessDataset) -> AttackResult {
    let (_, attr, nodes) = build_merkle_state(ds);

    // Raw modification of the companion node array itself (not chunk data,
    // not the root attribute): flip a byte in an internal node.
    let mut tampered_nodes = nodes.clone();
    tampered_nodes[0] ^= 0xFF;

    let start_time = Instant::now();
    let view = dataset_view(attr, tampered_nodes, ds, &ds.bytes);
    let result = verify_root(&view);
    let latency = start_time.elapsed();

    AttackResult {
        threat_class: "T1",
        attack_id: "T1b",
        dataset: ds.label,
        detected: matches!(result, Err(MerkleError::CompanionTampered)),
        verifier_fn: "verify_root",
        latency,
        root_cause: None,
    }
}

/// T2 — silent data corruption: a single-bit flip standing in for a hardware
/// bit-rot / firmware fault on an otherwise-untouched chunk. `verify_dataset`
/// (the full O(N) rehash) must detect it and name the right chunk.
///
/// **Spec-mapping note.** This follows `sec:threat`'s T2 definition ("silent
/// modifications to stored *chunks*") and `sec:adversarial`'s elaboration
/// ("single-bit and burst-error injection... on a known-good chunk") --
/// which is what Table 5 and the CSV `threat_class` schema are keyed on.
/// Section 7's own P2.4 item-2 bullet list describes "T2" differently
/// ("flip one bit in the *companion dataset*; `verify_root` must detect
/// it") -- that scenario (a companion-node bit flip, not a chunk-data bit
/// flip) is what `sec:adversarial` calls **T1b**, implemented above as
/// `t1b_companion_node_tamper`. The two sections' prose disagree on which
/// letter names which attack; this file follows the canonical
/// `sec:threat`/`sec:adversarial` taxonomy since that's what the rest of
/// the paper (and this harness's own CSV output) is organized around.
pub fn t2_single_bit_corruption(ds: &HarnessDataset) -> AttackResult {
    let (_, attr, nodes) = build_merkle_state(ds);
    let victim = ds.chunk_count().saturating_sub(1);

    let mut tampered = ds.bytes.clone();
    let (start, _) = ds.chunk_ranges[victim];
    tampered[start] ^= 0x01; // single bit, not a full byte

    let start_time = Instant::now();
    let view = dataset_view(attr, nodes, ds, &tampered);
    let result = verify_dataset(&view);
    let latency = start_time.elapsed();

    let detected =
        matches!(result, Err(MerkleError::HashMismatch { chunk_idx }) if chunk_idx == victim);
    AttackResult {
        threat_class: "T2",
        attack_id: "T2",
        dataset: ds.label,
        detected,
        verifier_fn: "verify_dataset",
        latency,
        root_cause: None,
    }
}

/// T6a — security stripping: zero out the `merkle_root` attribute bytes in
/// place (simulating deletion/overwrite via raw attribute access) and confirm
/// the parser fails closed instead of silently treating the dataset as
/// unprotected.
pub fn t6a_root_attribute_stripped(ds: &HarnessDataset) -> AttackResult {
    let (_, attr, _) = build_merkle_state(ds);

    // Raw overwrite of the packed attribute bytes with zeros, as a
    // storage-level adversary stripping `_merkle_root` would produce if the
    // attribute slot is zeroed rather than removed.
    let mut corrupted = attr.pack();
    corrupted.fill(0);

    let start_time = Instant::now();
    let result = MerkleAttr::unpack(&corrupted);
    let latency = start_time.elapsed();

    AttackResult {
        threat_class: "T6",
        attack_id: "T6a",
        dataset: ds.label,
        detected: result.is_err(),
        verifier_fn: "MerkleAttr::unpack",
        latency,
        root_cause: None,
    }
}

/// A flat 1D re-indexing of `ds`'s chunks (`dims = [chunk_count]`,
/// `chunk_shape = [1]`), used as the "chunk grid" for the subset-extraction
/// attacks below.
///
/// This is independent of the real dataset's actual N-dimensional chunk grid
/// (e.g. the GOES-18 `Rad` dataset's real 4x4 grid of 2D chunks): the subset
/// attacks exercise `verify_subset`'s soundness properties (omission,
/// substitution, wrong-coverage detection) generically over "chunk `i` of
/// N," not real 2D hyperslab coverage against the satellite image's true grid.
fn whole_dataset_grid(ds: &HarnessDataset) -> ChunkGridParams {
    ChunkGridParams::new(vec![ds.chunk_count() as u64], vec![1], HashAlg::Blake3)
}

/// A contiguous 1D hyperslab `start..end`, built directly (rather than via
/// `Selection::slice(&[range])`, which clippy flags as an ambiguous
/// single-element array-of-`Range`) since every selection this harness needs
/// is over the flat, one-dimensional chunk grid.
fn range_1d(range: std::ops::Range<u64>) -> Selection {
    Selection::Hyperslab {
        start: vec![range.start],
        stride: vec![1],
        count: vec![range.end.saturating_sub(range.start)],
        block: vec![1],
    }
}

/// Subset-(a) — omission: the delivered chunk set is missing an entry the
/// proof claims to cover. `verify_subset` must reject it.
pub fn subset_a_omitted_chunk(ds: &HarnessDataset) -> AttackResult {
    let (tree, _, _) = build_merkle_state(ds);
    let grid = whole_dataset_grid(ds);
    let sel = range_1d(0..ds.chunk_count() as u64);
    let proof = extract_subset(&tree, &grid, &sel, LeafOrder::RowMajor).unwrap();

    // Deliver every claimed chunk except the last one (saturating so an
    // empty chunk set, however unlikely for a real dataset, degrades to "no
    // chunks delivered" instead of underflowing the slice bound).
    let keep = proof.chunk_indices.len().saturating_sub(1);
    let delivered: Vec<ChunkData<'_>> = proof.chunk_indices[..keep]
        .iter()
        .map(|&idx| ChunkData {
            index: idx,
            data: ds.chunk(&ds.bytes, idx),
        })
        .collect();

    let start_time = Instant::now();
    let result = verify_subset(
        tree.root(),
        HashAlg::Blake3,
        &delivered,
        &proof,
        &grid,
        &sel,
        LeafOrder::RowMajor,
    );
    let latency = start_time.elapsed();

    AttackResult {
        threat_class: "subset",
        attack_id: "subset-a",
        dataset: ds.label,
        detected: matches!(result, Err(MerkleError::CompanionTampered)),
        verifier_fn: "verify_subset",
        latency,
        root_cause: None,
    }
}

/// Subset-(b) — substitution: one delivered chunk's bytes are swapped for a
/// different chunk's content. `verify_subset` must reject it via the
/// leaf-hash check.
pub fn subset_b_substituted_chunk(ds: &HarnessDataset) -> AttackResult {
    let (tree, _, _) = build_merkle_state(ds);
    let grid = whole_dataset_grid(ds);
    let sel = range_1d(0..ds.chunk_count() as u64);
    let proof = extract_subset(&tree, &grid, &sel, LeafOrder::RowMajor).unwrap();

    let swap_a = 0usize;
    // saturating_sub, consistent with subset_a: a 0-chunk dataset has no
    // meaningful "substitute chunk 0 with chunk N-1" to perform either way,
    // but this avoids underflowing to usize::MAX and failing with a far more
    // confusing out-of-bounds message than the plain "no chunk 0" case would.
    // `main` refuses to run the suite at all on a 0-chunk dataset (see the
    // `chunk_count() == 0` guard there), which is the actual protection.
    let swap_b = ds.chunk_count().saturating_sub(1);
    let mut delivered: Vec<ChunkData<'_>> = proof
        .chunk_indices
        .iter()
        .map(|&idx| ChunkData {
            index: idx,
            data: ds.chunk(&ds.bytes, idx),
        })
        .collect();
    // Substitute chunk swap_a's delivered payload with chunk swap_b's actual
    // content — a different, validly-existing chunk from elsewhere in the
    // same dataset, standing in for "a chunk from a different version."
    delivered[swap_a].data = ds.chunk(&ds.bytes, swap_b);

    let start_time = Instant::now();
    let result = verify_subset(
        tree.root(),
        HashAlg::Blake3,
        &delivered,
        &proof,
        &grid,
        &sel,
        LeafOrder::RowMajor,
    );
    let latency = start_time.elapsed();

    AttackResult {
        threat_class: "subset",
        attack_id: "subset-b",
        dataset: ds.label,
        detected: matches!(result, Err(MerkleError::HashMismatch { chunk_idx }) if chunk_idx == swap_a),
        verifier_fn: "verify_subset",
        latency,
        root_cause: None,
    }
}

/// Subset-(c) — bad coverage: the prover delivers a proof for a different
/// region than the verifier actually requested. `verify_subset` must reject
/// it via the independently-recomputed expected chunk set.
pub fn subset_c_wrong_coverage(ds: &HarnessDataset) -> AttackResult {
    let (tree, _, _) = build_merkle_state(ds);
    let grid = whole_dataset_grid(ds);
    let half = (ds.chunk_count() as u64) / 2;

    // Prover extracts (and delivers) a proof for the FIRST half...
    let proved_sel = range_1d(0..half);
    let proof = extract_subset(&tree, &grid, &proved_sel, LeafOrder::RowMajor).unwrap();
    let delivered: Vec<ChunkData<'_>> = proof
        .chunk_indices
        .iter()
        .map(|&idx| ChunkData {
            index: idx,
            data: ds.chunk(&ds.bytes, idx),
        })
        .collect();

    // ...but the verifier actually asked for the SECOND half.
    let requested_sel = range_1d(half..ds.chunk_count() as u64);

    let start_time = Instant::now();
    let result = verify_subset(
        tree.root(),
        HashAlg::Blake3,
        &delivered,
        &proof,
        &grid,
        &requested_sel,
        LeafOrder::RowMajor,
    );
    let latency = start_time.elapsed();

    AttackResult {
        threat_class: "subset",
        attack_id: "subset-c",
        dataset: ds.label,
        detected: matches!(result, Err(MerkleError::SelectionMismatch)),
        verifier_fn: "verify_subset",
        latency,
        root_cause: None,
    }
}

// ============================================================================
// Mechanism attacks: T3, T4a, T4b, T5, T6b, T7, T8.
// ============================================================================

/// P2.1 hybrid-signature stand-in: accepts only the exact byte string it was
/// told is genuine for a given version. `clawhdf5-format`/`clawhdf5-agent`
/// consume signature verification through exactly this trait
/// ([`SignatureVerifier`]); this repo does not yet have the P2.1 hybrid-signing
/// crate merged, so attacks that need "a signature" use this instead of a real
/// Ed25519/ML-DSA-65 check. That substitution is disclosed inline wherever
/// it's load-bearing (T3) and called out as an explicit gap where it would
/// change the result (T5).
struct KnownGoodSignatures(std::collections::BTreeMap<u64, Vec<u8>>);
impl SignatureVerifier for KnownGoodSignatures {
    fn verify(&self, r: &ProvenanceRecord) -> bool {
        self.0.get(&r.version) == Some(&r.hybrid_sig)
    }
}

/// T3 — provenance forgery: a stale signature is spliced onto a newer
/// provenance record (raw substitution of the `hybrid_sig`/`signed_root`
/// bytes an attacker with storage access could perform directly on the
/// journal). `select_restore_record` must reject it.
///
/// **Spec-mapping note.** This attack -- reusing an old, genuine signature
/// on different/newer data, rejected because the version doesn't match --
/// is literally what Section 7's P2.4 item-2 bullet list describes as
/// *"T4 (replay): copy the `_merkle_sig` from an older file version onto
/// the current file."* It's labeled T3 here instead, following
/// `sec:threat`'s T3 ("provenance forgery... requires non-repudiable
/// authorship via digital signatures") and `sec:adversarial`'s T3
/// ("key-confusion attacks on the hybrid signature"), which is the
/// canonical taxonomy this harness's `threat_class` column follows (see
/// the note on `t2_single_bit_corruption` for the parallel T2 discrepancy).
/// `sec:threat`'s own T4 is **rollback** (substituting an entire prior,
/// internally-consistent file state) -- a different attack, implemented
/// below as `t4a_whole_file_rollback` / `t4b_selective_chunk_rollback`.
pub fn t3_provenance_forgery() -> AttackResult {
    let mut genuine = std::collections::BTreeMap::new();
    genuine.insert(1u64, b"genuine-sig-v1".to_vec());
    genuine.insert(2u64, b"genuine-sig-v2".to_vec());
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
    // Raw tamper: splice version 1's signature bytes onto the version-2
    // record instead of the genuine (unknown-to-the-attacker) v2 signature.
    journal
        .append(ProvenanceRecord {
            version: 2,
            signed_root: [2; 32],
            hybrid_sig: b"genuine-sig-v1".to_vec(),
            timestamp: 200,
            snapshot_ref: String::from("snapshot-v2"),
        })
        .unwrap();

    let start_time = Instant::now();
    let result = select_restore_record(&journal, RestoreTarget::Version(2), &verifier);
    let latency = start_time.elapsed();

    AttackResult {
        threat_class: "T3",
        attack_id: "T3",
        dataset: "n/a",
        detected: matches!(result, Err(RestoreError::SignatureInvalid { version: 2 })),
        verifier_fn: "select_restore_record",
        latency,
        root_cause: None,
    }
}

/// T4a — rollback: an entire prior, previously-observed-valid file version is
/// substituted back in. `VersionObservationStore` (the persisted
/// `merkle_version` high-water mark, P2.2b step 1) must reject the reopen.
pub fn t4a_whole_file_rollback() -> AttackResult {
    let mut store = VersionObservationStore::new();
    // The verifier has previously opened this file at version 2...
    store.observe("live.h5", 2).unwrap();

    let start_time = Instant::now();
    // ...raw storage-level substitution restores the version-1 bytes (data,
    // companion, root attribute together) — a valid, properly signed OLDER
    // state of the same file.
    let result = store.observe("live.h5", 1);
    let latency = start_time.elapsed();

    AttackResult {
        threat_class: "T4",
        attack_id: "T4a",
        dataset: "n/a",
        detected: matches!(
            result,
            Err(MerkleError::VersionRollback {
                observed: 1,
                highest_seen: 2
            })
        ),
        verifier_fn: "VersionObservationStore::observe",
        latency,
        root_cause: None,
    }
}

/// T4b — selective per-chunk rollback: one chunk and its version counter are
/// both restored to a prior, individually-valid state. This targets the
/// version-counter binding into the leaf hash (`compute_leaf_hash`), not the
/// dataset-level counter.
///
/// **Caveat (disclosed):** `compute_leaf_hash`'s version-bound formula is not
/// yet wired into `clawhdf5-format`'s public `verify_chunk`/`verify_dataset`
/// (see the `TODO(P2.4)` on `compute_leaf_hash` in
/// `clawhdf5-filters/src/filter_pipeline.rs`), so this attack is exercised at
/// the leaf-hash-formula level — the same recomputation a real verifier would
/// perform — rather than through a `verify_*` wrapper that doesn't exist yet.
pub fn t4b_selective_chunk_rollback() -> AttackResult {
    let dek = clawhdf5_filters::mock_dek();

    // Chunk 2 written at version 1, then overwritten at version 2.
    let old_plaintext = b"pre-rollback state";
    let new_plaintext = b"post-rollback state, committed";
    let old_ciphertext = encrypt_chunk(&dek, 2, 1, old_plaintext).unwrap();
    let new_ciphertext = encrypt_chunk(&dek, 2, 2, new_plaintext).unwrap();

    // The committed tree's leaf binds (chunk_idx, ciphertext, version) for
    // the CURRENT (version-2) state.
    let committed_leaf = compute_leaf_hash(2, &new_ciphertext, 2);

    let start_time = Instant::now();
    // Raw storage-level attack: roll chunk 2's on-disk ciphertext back to its
    // version-1 bytes, together with its version counter (both mutually
    // consistent, decrypt fine under the version-1 nonce — this isn't
    // garbage data, it's a stale-but-valid prior state).
    let rolled_back_leaf = compute_leaf_hash(2, &old_ciphertext, 1);
    let latency = start_time.elapsed();

    // Sanity: the rolled-back ciphertext really does decrypt under its own
    // (chunk_idx, version) — this is a *valid old state*, not corruption.
    debug_assert_eq!(
        clawhdf5_filters::decrypt_chunk(&dek, 2, 1, &old_ciphertext).unwrap(),
        old_plaintext
    );

    AttackResult {
        threat_class: "T4",
        attack_id: "T4b",
        dataset: "n/a",
        detected: rolled_back_leaf != committed_leaf,
        verifier_fn: "compute_leaf_hash (leaf-hash recomputation)",
        latency,
        root_cause: None,
    }
}

/// T5 — harvest-now, forge-later: simulated forgery of the Ed25519 component
/// of the hybrid signature, to confirm the ML-DSA-65 (post-quantum) component
/// alone still blocks acceptance.
///
/// **Out of scope — documented limitation.** The P2.1 hybrid-signing crate
/// (Ed25519 + ML-DSA-65) is not merged into this workspace (only a stray,
/// unwired `crates/clawhdf5-sign/` fuzz directory exists), so there is no
/// dual-component signature to partially forge. Fabricating a fake "hybrid
/// signature" object here would test nothing real. This is reported as
/// **undetected** with the CSV's required `root_cause`, matching the P2.4
/// artifact's own schema for attacks that are a genuine, disclosed gap rather
/// than a silent skip.
pub fn t5_post_quantum_forgery() -> AttackResult {
    AttackResult {
        threat_class: "T5",
        attack_id: "T5",
        dataset: "n/a",
        detected: false,
        verifier_fn: "n/a",
        latency: std::time::Duration::ZERO,
        root_cause: Some(
            "P2.1 hybrid signing (Ed25519 + ML-DSA-65) is not merged into this workspace; \
             there is no dual-component signature to partially forge against. Requires P2.1.",
        ),
    }
}

/// T6b — algorithm downgrade: an attacker with full storage access rehashes
/// an *unsigned* dataset's entire tree under a weaker algorithm and rewrites
/// the root attribute, companion nodes, and integrity hash to match —
/// entirely self-consistent, with no forged bytes anywhere.
///
/// **Documented limitation.** Nothing in this workspace binds the *original*
/// algorithm choice into an unforgeable payload for unsigned data: the
/// in-band `integrity` hash only proves internal self-consistency of
/// whatever `(root, alg_id)` pair is currently on disk, and a full attacker
/// rebuild trivially reproduces it. `resolve_response`'s fail-closed rule
/// only helps *signed* datasets (P2.1). This is genuinely undetected for
/// unsigned data with today's code, so it's reported as such rather than
/// silently omitted.
pub fn t6b_algorithm_downgrade() -> AttackResult {
    let chunks: Vec<&[u8]> = vec![b"chunk-a", b"chunk-b", b"chunk-c", b"chunk-d"];
    let honest_tree = MerkleTree::from_chunks(&chunks, HashAlg::Blake3);
    let mut honest_nodes = Vec::new();
    for n in honest_tree.nodes() {
        honest_nodes.extend_from_slice(n);
    }
    let honest_attr =
        MerkleAttr::from_tree_with_companion(&honest_tree, companion_hash(&honest_nodes));

    let start_time = Instant::now();
    // Full attacker-side rebuild under a weaker algorithm, over the SAME
    // (untampered) chunk content -- no signature exists to catch this for an
    // unsigned dataset.
    let downgraded_tree = MerkleTree::from_chunks(&chunks, HashAlg::Sha256);
    let mut downgraded_nodes = Vec::new();
    for n in downgraded_tree.nodes() {
        downgraded_nodes.extend_from_slice(n);
    }
    let downgraded_attr =
        MerkleAttr::from_tree_with_companion(&downgraded_tree, companion_hash(&downgraded_nodes));
    // Captured before `downgraded_attr` moves into `Dataset::from_owned` below,
    // so the "a real downgrade occurred" check compares the actual rebuilt
    // algorithm against the honest one, rather than checking a stale/wrong value.
    let downgraded_algorithm = downgraded_attr.algorithm;
    // Both the packed attribute's own integrity check AND a full
    // verify_dataset/verify_root against the rebuilt state pass cleanly.
    let attr_self_consistent = MerkleAttr::unpack(&downgraded_attr.pack()).is_ok();
    let view = Dataset::from_owned(downgraded_attr, downgraded_nodes, chunks.clone());
    let dataset_verifies = verify_dataset(&view).unwrap_or(false);
    let latency = start_time.elapsed();

    // Confirms a real downgrade occurred (the rebuilt tree's algorithm really
    // does differ from the honest one) before crediting the attacker.
    let downgrade_succeeded =
        attr_self_consistent && dataset_verifies && downgraded_algorithm != honest_attr.algorithm;

    AttackResult {
        threat_class: "T6",
        attack_id: "T6b",
        dataset: "n/a",
        detected: !downgrade_succeeded,
        verifier_fn: "verify_dataset / MerkleAttr::unpack",
        latency,
        root_cause: if downgrade_succeeded {
            Some(
                "No signature binds the hash-algorithm identifier into an unforgeable payload \
                 for unsigned datasets; the in-band integrity hash only proves self-consistency, \
                 which a full attacker rebuild trivially reproduces. Requires P2.1's signed \
                 canonical_payload to cover alg_id (per Sec. merkle-storage, threat T6).",
            )
        } else {
            None
        },
    }
}

/// T7 — verification resource exhaustion (DoS): a crafted chunk-grid
/// (`dims`/`chunk_shape`) claims an astronomically large chunk count from a
/// tiny selection. `verify_subset` must reject it before doing any
/// `O(total_chunks)` work or allocation.
pub fn t7_verification_dos() -> AttackResult {
    // A hostile grid: one dimension of size u64::MAX with chunk_shape 1,
    // implying ~2^64 chunks -- the "tiny hyperslab maps to astronomically
    // many chunks" scenario from Sec. threat, T7.
    let hostile_grid = ChunkGridParams::new(vec![u64::MAX], vec![1], HashAlg::Blake3);
    let sel = Selection::All;
    let proof = clawhdf5_format::subset_proof::SubsetProof {
        chunk_indices: Vec::new(),
        leaf_hashes: Vec::new(),
        proof_nodes: std::collections::BTreeMap::new(),
        grid_params: hostile_grid.clone(),
        coverage_cert: [0u8; 32],
    };

    let start_time = Instant::now();
    let result = verify_subset(
        &[0u8; 32],
        HashAlg::Blake3,
        &[],
        &proof,
        &hostile_grid,
        &sel,
        LeafOrder::RowMajor,
    );
    let latency = start_time.elapsed();

    AttackResult {
        threat_class: "T7",
        attack_id: "T7",
        dataset: "n/a",
        detected: matches!(result, Err(MerkleError::TreeTooDeep { .. })),
        verifier_fn: "verify_subset (checked_padded_leaf_count)",
        latency,
        root_cause: None,
    }
}

/// T8 — structural metadata leakage: for a sparse dataset, the companion
/// tree's null-leaf positions reveal the physical allocation pattern to any
/// reader, without decrypting a single chunk.
///
/// **Documented limitation.** The structure-hiding, PRF-masked null sentinel
/// (Sec. merkle-storage: `H(0x02 || PRF(Structure Key, "null"))`) is scoped as
/// a Phase-2 design option and is not implemented — the current null sentinel
/// `H(0x02 || "null")` is a fixed public constant. This attack demonstrates
/// the leak directly (an outside observer needs no secret to identify
/// unallocated slots) and reports it as undetected by design.
pub fn t8_structural_leakage() -> AttackResult {
    // A sparse dataset: leaves 1 and 3 are unallocated (padded with the
    // public null sentinel); leaves 0 and 2 hold real data.
    let alg = HashAlg::Blake3;
    let null = alg.null_sentinel();
    let leaves = [
        alg.hash_leaf(b"real-chunk-0"),
        null,
        alg.hash_leaf(b"real-chunk-2"),
        null,
    ];
    let tree = MerkleTree::from_leaf_hashes(&leaves, alg);

    let start_time = Instant::now();
    // The "attack": an observer with only the companion node array (no DEK,
    // no chunk plaintext) recomputes the PUBLIC null constant and checks
    // which leaves match it.
    let inferred_sparsity: Vec<bool> = (0..4).map(|i| tree.leaf_hash(i) == Some(&null)).collect();
    let latency = start_time.elapsed();

    let leak_succeeded = inferred_sparsity == vec![false, true, false, true];

    AttackResult {
        threat_class: "T8",
        attack_id: "T8",
        dataset: "n/a",
        detected: !leak_succeeded,
        verifier_fn: "n/a (structure-hiding not implemented)",
        latency,
        root_cause: if leak_succeeded {
            Some(
                "Null sentinel H(0x02 || \"null\") is a fixed public constant, not PRF-masked \
                 with a Structure/Data Encryption Key; sparsity pattern is directly recoverable \
                 without decryption. Structure-hiding padded-binary-tree mode (Sec. \
                 merkle-storage) is scoped as future Phase-2 work, not yet implemented.",
            )
        } else {
            None
        },
    }
}
