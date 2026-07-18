//! The P2.4 white-box red-team attack suite (S2-D2-Yr2 §`sec:adversarial`).
//!
//! Every attack function follows the same shape mandated by the spec: open
//! (or build) a test HDF5-derived dataset, perform **one** targeted
//! modification using raw file/byte I/O (never a ClawHDF5 write API), then
//! call the verification API and assert the expected result.
//!
//! Attacks are split into two groups:
//!
//! - **Dataset attacks** (T1a, T1c, T1b, T2a, T2b, T6a, subset-a/b/c/d) run
//!   against the [`HarnessDataset`] — either the real GOES-18 file's actual
//!   on-disk HDF5 chunks, or the synthetic fallback — because they only need
//!   "a chunked Merkle-protected byte blob" and get more authentic coverage
//!   from real satellite chunk boundaries and filter-compressed sizes when
//!   available. T1a/T2a are the "easy" storage-tamper/bit-flip baselines;
//!   T1c (a valid-after-pipeline compressed-stream substitution) and T2b
//!   (a burst error) are the harder, directed-adversary variants
//!   §`sec:adversarial` explicitly calls for.
//! - **Mechanism attacks** (T1d, T1e, T1f, T3, T4a, T4b, T5, T6b, T6c, T7, T8)
//!   target a specific primitive (the companion-integrity self-check, the
//!   leaf/internal hash domain separation, the signed-root binding, the
//!   provenance journal, the version counter, the AEAD leaf formula, the
//!   null-sentinel padding) that isn't meaningfully expressed in terms of "a
//!   chunk in the dataset," so each builds its own minimal fixture. T1d/T6b are
//!   the UNSIGNED directed forgery/downgrade attacks (undetected by design);
//!   T1f/T6c are their SIGNED counterparts, detected via
//!   `clawhdf5-format::verify_signed_root` (P2.4). T1e is the second-preimage /
//!   node-as-leaf domain-separation regression.

use std::time::Instant;

use sha2::{Digest, Sha256};

use clawhdf5_filters::{compute_leaf_hash, encrypt_chunk};
use clawhdf5_format::merkle::{
    Dataset, HashAlg, MerkleAttr, MerkleError, MerkleTree, SignedRootVerifier,
    VersionObservationStore, verify_chunk, verify_dataset, verify_root, verify_signed_root,
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

/// T1c — storage-level tampering of a *compressed* payload, crafted to remain
/// valid after the filter pipeline. This is the specific harder case
/// §`sec:adversarial` calls out under T1a: *"modifications of compressed
/// payloads designed to remain valid after the filter pipeline."*
///
/// The distinction from T1a matters. T1a flips a raw byte; if the stored chunk
/// were a real deflate stream, a random flip would almost always make it fail
/// to decode, so the *pipeline itself* rejects it at the decompression stage
/// before the integrity layer is even consulted — the integrity check never
/// has to do any work. A directed adversary who wants to change the actual
/// scientific values, not just trigger a decode error, instead substitutes a
/// *different, still-well-formed* deflate stream that decompresses cleanly to
/// attacker-chosen data. That modification survives the filter pipeline
/// intact; only the Merkle leaf hash over the stored (compressed) bytes catches
/// it. This attack builds that exact substitution and confirms `verify_chunk`
/// rejects it.
pub fn t1c_compressed_payload_tamper(ds: &HarnessDataset) -> AttackResult {
    use clawhdf5_filters::{deflate_compress, deflate_decompress};

    // The stored (post-filter) representation is the deflate-compressed chunk.
    let plaintext_chunks: Vec<&[u8]> = ds.all_chunks(&ds.bytes);
    let compressed: Vec<Vec<u8>> = plaintext_chunks
        .iter()
        .map(|c| deflate_compress(c, 6).expect("deflate compression should not fail"))
        .collect();

    // Merkle state is committed over the ORIGINAL compressed chunks (this is
    // the honest, signed-root state the attacker cannot recompute).
    let comp_refs: Vec<&[u8]> = compressed.iter().map(|c| c.as_slice()).collect();
    let tree = MerkleTree::from_chunks(&comp_refs, HashAlg::Blake3);
    let mut nodes = Vec::with_capacity(tree.nodes().len() * 32);
    for n in tree.nodes() {
        nodes.extend_from_slice(n);
    }
    let attr = MerkleAttr::from_tree_with_companion(&tree, companion_hash(&nodes));

    let victim = ds.chunk_count() / 2;

    // Craft a DIFFERENT but still-valid deflate stream: take the victim's
    // plaintext, change the actual data (the values a real attacker cares
    // about), and recompress. The result is a well-formed deflate stream that
    // decodes without error -- it "remains valid after the filter pipeline" --
    // but to different bytes than the committed chunk.
    let mut victim_plaintext = plaintext_chunks[victim].to_vec();
    if victim_plaintext.is_empty() {
        victim_plaintext.push(0x01);
    } else {
        victim_plaintext[0] ^= 0xFF;
    }
    let forged_stream =
        deflate_compress(&victim_plaintext, 6).expect("deflate compression should not fail");

    // Confirm the forged stream genuinely survives the pipeline: it decompresses
    // without error, to the attacker's chosen data, and differs from the honest
    // compressed chunk. If any of these fail this isn't the attack we claim.
    let survives_pipeline = matches!(
        deflate_decompress(&forged_stream, victim_plaintext.len().max(1)),
        Ok(ref d) if *d == victim_plaintext
    ) && forged_stream != compressed[victim];

    // Substitute the forged compressed stream into the STORED bytes, keeping the
    // committed attr/nodes -- exactly what a storage-level adversary can do.
    let mut tampered_compressed = compressed.clone();
    tampered_compressed[victim] = forged_stream;
    let tampered_refs: Vec<&[u8]> = tampered_compressed.iter().map(|c| c.as_slice()).collect();
    let view = Dataset::from_owned(attr, nodes, tampered_refs);

    let start_time = Instant::now();
    let result = verify_chunk(&view, victim);
    let latency = start_time.elapsed();

    let detected = survives_pipeline
        && matches!(result, Err(MerkleError::HashMismatch { chunk_idx }) if chunk_idx == victim);
    AttackResult {
        threat_class: "T1",
        attack_id: "T1c",
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
///
/// **Naive vs directed adversary.** This flips a companion node but leaves the
/// stored `companion_hash` alone, so `verify_root`'s self-consistency check
/// (`companion_hash == SHA-256(nodes)`) fires — the naive-adversary case, which
/// is genuinely caught. A *directed* adversary against an **unsigned** dataset
/// recomputes `companion_hash` over the tampered nodes and walks straight
/// through; that variant is [`t1d_directed_companion_forgery`], reported
/// undetected-by-design with the same root cause as T6b (no signature ⇒ only
/// self-consistency, which a full rebuild reproduces).
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

/// T1d — directed companion forgery on an **unsigned** dataset: the same
/// companion-node tamper as T1b, but the adversary also recomputes the stored
/// `companion_hash` over the tampered nodes, producing a fully self-consistent
/// attribute. This is the directed-adversary companion analogue of T1c/T2b.
///
/// **Documented limitation (undetected by design for unsigned data).**
/// `verify_root` on an unsigned dataset only checks self-consistency
/// (`companion_hash == SHA-256(nodes)`, and the attribute's own
/// `H(0x03 || root || alg_id)` integrity hash). None of that is bound to an
/// unforgeable value, so an attacker with storage access who rebuilds the
/// companion array and its hash together defeats it — the exact same root
/// cause as T6b (algorithm downgrade). A **signed** dataset is protected, and
/// this attack is detected against it: `clawhdf5_sign::canonical_payload` folds
/// the `companion_hash` into the signed payload, and `clawhdf5-format`'s
/// `verify_signed_root` (P2.4) now checks that binding — see [`t1f_signed_companion_forgery`],
/// the signed counterpart, which reports detected. This T1d stays undetected
/// only because it exercises the UNSIGNED path (`verify_root` alone). (Note this
/// is the *format-primitive* limitation; the P2.4 Finding-1 agent wiring
/// rehashes content from disk and never trusts stored companion nodes, so it is
/// not affected.)
pub fn t1d_directed_companion_forgery() -> AttackResult {
    // Minimal own fixture (mechanism attack): a small unsigned chunk set.
    let chunks: Vec<&[u8]> = vec![b"chunk-a", b"chunk-b", b"chunk-c", b"chunk-d"];
    let honest_tree = MerkleTree::from_chunks(&chunks, HashAlg::Blake3);
    let mut honest_nodes = Vec::new();
    for n in honest_tree.nodes() {
        honest_nodes.extend_from_slice(n);
    }

    let start_time = Instant::now();
    // Directed tamper: flip a companion node AND recompute the companion hash
    // over the tampered nodes, so the attribute is internally self-consistent.
    let mut tampered_nodes = honest_nodes.clone();
    tampered_nodes[0] ^= 0xFF;
    let forged_attr =
        MerkleAttr::from_tree_with_companion(&honest_tree, companion_hash(&tampered_nodes));
    let view = Dataset::from_owned(forged_attr, tampered_nodes.clone(), chunks.clone());
    let result = verify_root(&view);
    let latency = start_time.elapsed();

    // Confirm a real directed forgery: the nodes genuinely changed AND the
    // recomputed companion hash still matches them (self-consistent).
    let forgery_is_self_consistent = tampered_nodes != honest_nodes;
    let bypassed = forgery_is_self_consistent && matches!(result, Ok(true));

    AttackResult {
        threat_class: "T1",
        attack_id: "T1d",
        dataset: "n/a",
        detected: !bypassed,
        verifier_fn: "verify_root",
        latency,
        root_cause: if bypassed {
            Some(
                "By design for an UNSIGNED dataset: verify_root only checks companion \
                 self-consistency (companion_hash == SHA-256(nodes)) and the attribute's own \
                 H(0x03 || root || alg_id) integrity hash, neither bound to an unforgeable \
                 value -- a full attacker rebuild of the companion array and its hash \
                 reproduces both. Same root cause as T6b. This is the UNSIGNED path (verify_root \
                 alone). A SIGNED dataset IS protected and this attack is detected against it: \
                 clawhdf5_sign::canonical_payload folds companion_hash into the signed payload, \
                 and clawhdf5-format::verify_signed_root now checks that binding -- see T1f, the \
                 signed counterpart of this attack, which reports detected=yes.",
            )
        } else {
            None
        },
    }
}

/// T1e — Merkle second-preimage / node-as-leaf confusion. An attacker tries to
/// collapse a multi-chunk dataset into a single crafted chunk whose *leaf* hash
/// equals an *internal* node, so a shorter forged tree reproduces the honest
/// root. This works only if leaf and internal hashes share a hash domain.
/// clawhdf5 domain-separates them (`0x00` leaf prefix, `0x01` internal prefix),
/// so `H_leaf(child0 || child1) != H_internal(child0, child1)` and the collapse
/// fails. Regression lock-in for that domain separation (a defense a crypto
/// reviewer always checks, previously exercised by no attack).
pub fn t1e_second_preimage_node_as_leaf() -> AttackResult {
    let alg = HashAlg::Blake3;
    let leaf0 = alg.hash_leaf(b"chunk-0");
    let leaf1 = alg.hash_leaf(b"chunk-1");
    let internal = alg.hash_pair(&leaf0, &leaf1);

    let start_time = Instant::now();
    // Craft a single chunk X = child0 || child1 and hope that hashing X as a
    // leaf reproduces the internal node (it would, without domain separation).
    let mut forged_chunk = Vec::with_capacity(64);
    forged_chunk.extend_from_slice(&leaf0);
    forged_chunk.extend_from_slice(&leaf1);
    let forged_leaf = alg.hash_leaf(&forged_chunk);
    let collapsed = forged_leaf == internal;
    let latency = start_time.elapsed();

    AttackResult {
        threat_class: "T1",
        attack_id: "T1e",
        dataset: "n/a",
        // Domain separation must make the collapse impossible.
        detected: !collapsed,
        verifier_fn: "HashAlg::hash_leaf / hash_pair (domain separation)",
        latency,
        root_cause: if collapsed {
            Some(
                "LEAF/INTERNAL DOMAIN SEPARATION BROKEN: hash_leaf(child0||child1) equals \
                 hash_pair(child0, child1), so a multi-chunk tree can be collapsed to a single \
                 forged chunk with the same root. This must never happen.",
            )
        } else {
            None
        },
    }
}

/// T2a — silent data corruption, single-bit: a lone bit flip standing in for a
/// hardware bit-rot / firmware fault on an otherwise-untouched chunk.
/// `verify_dataset` (the full O(N) rehash) must detect it and name the right
/// chunk. `sec:adversarial` asks for *"single-bit **and** burst-error
/// injection"*; this is the single-bit half, with the burst half in
/// [`t2b_burst_error_corruption`].
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
pub fn t2a_single_bit_corruption(ds: &HarnessDataset) -> AttackResult {
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
        attack_id: "T2a",
        dataset: ds.label,
        detected,
        verifier_fn: "verify_dataset",
        latency,
        root_cause: None,
    }
}

/// T2b — silent data corruption, burst error: a contiguous run of bytes is
/// corrupted at once, standing in for a multi-byte fault (a failing disk
/// sector, a DMA glitch, a torn write) rather than a lone bit flip. This is
/// the burst-error half of `sec:adversarial`'s *"single-bit and burst-error
/// injection"*; `verify_dataset` must still detect it and name the right chunk.
pub fn t2b_burst_error_corruption(ds: &HarnessDataset) -> AttackResult {
    let (_, attr, nodes) = build_merkle_state(ds);
    let victim = ds.chunk_count().saturating_sub(1);

    let mut tampered = ds.bytes.clone();
    let (start, end) = ds.chunk_ranges[victim];
    // Corrupt a contiguous 16-byte burst inside the victim chunk (clamped to
    // the chunk's own length so a short final chunk can't overrun into the
    // next one). XOR 0xAA guarantees every touched byte actually changes.
    let burst_end = (start + 16).min(end);
    for b in &mut tampered[start..burst_end] {
        *b ^= 0xAA;
    }

    let start_time = Instant::now();
    let view = dataset_view(attr, nodes, ds, &tampered);
    let result = verify_dataset(&view);
    let latency = start_time.elapsed();

    let detected =
        matches!(result, Err(MerkleError::HashMismatch { chunk_idx }) if chunk_idx == victim);
    AttackResult {
        threat_class: "T2",
        attack_id: "T2b",
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
    // `grid.grid_hash` stands in for the caller's already-verified
    // MerkleAttr::grid_hash(): this harness builds `grid` itself (there's no
    // real signed file here), so it's trivially self-consistent -- the
    // omission attack this function tests is unrelated to grid-hash binding.
    let result = verify_subset(
        tree.root(),
        HashAlg::Blake3,
        &delivered,
        &proof,
        &grid,
        &grid.grid_hash,
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
        &grid.grid_hash,
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
        &grid.grid_hash,
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

/// Subset-(d) — malformed proof, mismatched array lengths: a crafted
/// `SubsetProof` whose `leaf_hashes` is shorter than its `chunk_indices`
/// (and the delivered chunk set). Without an up-front length check,
/// `verify_subset`'s per-chunk loop would index `leaf_hashes[i]` out of bounds
/// and panic — a denial-of-service on the verifier from attacker-controlled
/// wire data. `verify_subset` must reject the length mismatch cleanly instead.
/// Regression lock-in for that guard.
pub fn subset_d_malformed_proof_lengths(ds: &HarnessDataset) -> AttackResult {
    let (tree, _, _) = build_merkle_state(ds);
    let grid = whole_dataset_grid(ds);
    let sel = range_1d(0..ds.chunk_count() as u64);
    let mut proof = extract_subset(&tree, &grid, &sel, LeafOrder::RowMajor).unwrap();

    // Attacker truncates leaf_hashes so the proof's parallel arrays disagree in
    // length. Deliver a full chunk set (matching chunk_indices) so the ONLY
    // inconsistency is the short leaf_hashes vector.
    let delivered: Vec<ChunkData<'_>> = proof
        .chunk_indices
        .iter()
        .map(|&idx| ChunkData {
            index: idx,
            data: ds.chunk(&ds.bytes, idx),
        })
        .collect();
    proof.leaf_hashes.pop();

    let start_time = Instant::now();
    let result = verify_subset(
        tree.root(),
        HashAlg::Blake3,
        &delivered,
        &proof,
        &grid,
        &grid.grid_hash,
        &sel,
        LeafOrder::RowMajor,
    );
    let latency = start_time.elapsed();

    AttackResult {
        threat_class: "subset",
        attack_id: "subset-d",
        dataset: ds.label,
        // Clean rejection (no panic) via the length guard.
        detected: matches!(result, Err(MerkleError::CompanionTampered)),
        verifier_fn: "verify_subset (length guard)",
        latency,
        root_cause: None,
    }
}

// ============================================================================
// Mechanism attacks: T3, T4a, T4b, T5, T6b, T7, T8.
// ============================================================================

/// Hybrid-signature stand-in for [`SignatureVerifier`], the trait
/// `select_restore_record` (`clawhdf5-format::merkle_recovery`, P2.2b)
/// actually consumes: accepts only the exact byte string it was told is
/// genuine for a given version. The real `clawhdf5-sign` crate (Ed25519 +
/// ML-DSA-65 hybrid signing, P2.1) is merged into this workspace -- see
/// `t5_post_quantum_forgery`, which uses it directly -- but nothing yet
/// implements `SignatureVerifier` backed by a real `HybridSignature`, so T3
/// (which needs to plug into `select_restore_record`'s trait-based API) still
/// uses this stand-in rather than real Ed25519/ML-DSA-65 verification.
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

/// T5 — harvest-now, forge-later: an adversary who has recovered the Ed25519
/// private key (simulating a future CRQC breaking classical ECC via Shor's
/// algorithm) tampers the root and re-signs it with the compromised Ed25519
/// key, but cannot produce a matching ML-DSA-65 (post-quantum) signature
/// without the ML-DSA private key. `verify_sig`'s strict-AND policy must
/// reject the forgery on the ML-DSA-65 mismatch alone.
///
/// Uses the real `clawhdf5-sign` crate (Ed25519 + ML-DSA-65 hybrid signing,
/// P2.1) directly -- that crate is merged into this workspace, just not yet
/// wired into `clawhdf5-format`'s Merkle verification flow (no `MerkleAttr`/
/// `Dataset` caller uses it), so this attack exercises `verify_sig` standalone
/// rather than through an integrated file-level API.
pub fn t5_post_quantum_forgery() -> AttackResult {
    use clawhdf5_sign::mldsa::MlDsaSigningKey;
    use clawhdf5_sign::{
        HashAlg as SignHashAlg, SigningKey, canonical_payload, sign_root, verify_sig,
    };

    // Legitimate signer: genuine Ed25519 + ML-DSA-65 keypairs, genuine payload.
    let ed_key = SigningKey::generate();
    let ml_key = MlDsaSigningKey::generate();
    let ed_pub = ed_key.verifying_key();
    let ml_pub = ml_key.verifying_key();

    let honest_root = [0x11u8; 32];
    let companion_hash = [0x22u8; 32];
    let honest_payload = canonical_payload(
        &honest_root,
        &companion_hash,
        1,
        1_700_000_000,
        SignHashAlg::Blake3,
    );
    let genuine_sig = sign_root(&honest_payload, &ed_key, &ml_key).unwrap();
    // Sanity: the genuine signature verifies before we attack anything.
    debug_assert!(verify_sig(&honest_payload, &genuine_sig, &ed_pub, &ml_pub).is_ok());

    let start_time = Instant::now();
    // The adversary has recovered ed_key (simulated CRQC break), tampers the
    // root, and re-signs the NEW payload with the compromised Ed25519 key --
    // a perfectly valid Ed25519 signature over tampered data. They do not
    // have ml_key, so they can only carry over the OLD ML-DSA-65 signature,
    // which was computed over the honest payload, not this tampered one.
    let tampered_root = [0xEEu8; 32];
    let tampered_payload = canonical_payload(
        &tampered_root,
        &companion_hash,
        1,
        1_700_000_000,
        SignHashAlg::Blake3,
    );
    let forged_ed25519_sig = ed_key.sign_payload(&tampered_payload);
    let forged_sig =
        clawhdf5_sign::HybridSignature::new(forged_ed25519_sig, genuine_sig.mldsa65_sig);

    let result = verify_sig(&tampered_payload, &forged_sig, &ed_pub, &ml_pub);
    let latency = start_time.elapsed();

    AttackResult {
        threat_class: "T5",
        attack_id: "T5",
        dataset: "n/a",
        detected: result.is_err(),
        verifier_fn: "clawhdf5_sign::verify_sig",
        latency,
        root_cause: None,
    }
}

/// T6b — algorithm downgrade: an attacker with full storage access rehashes
/// an *unsigned* dataset's entire tree under a weaker algorithm and rewrites
/// the root attribute, companion nodes, and integrity hash to match —
/// entirely self-consistent, with no forged bytes anywhere.
///
/// **Documented limitation (unsigned only).** Nothing binds the *original*
/// algorithm choice into an unforgeable payload for unsigned data: the in-band
/// `integrity` hash only proves internal self-consistency of whatever
/// `(root, alg_id)` pair is on disk, and a full attacker rebuild reproduces it.
/// A *signed* dataset IS protected, and this attack is detected against it —
/// see [`t6c_signed_algorithm_downgrade`], which routes the same rebuild
/// through `clawhdf5-format`'s `verify_signed_root` (P2.4) and is rejected on
/// the `alg`-bound signature. This T6b stays undetected only for the unsigned
/// path, so it's reported as such rather than silently omitted.
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
                "By design for an UNSIGNED dataset: no signature is in force, so the in-band \
                 integrity hash only proves self-consistency, which a full attacker rebuild \
                 trivially reproduces -- expected, not a gap. This is the UNSIGNED path. A SIGNED \
                 dataset IS protected and this attack is detected against it: \
                 clawhdf5_sign::canonical_payload binds alg_id into the signed payload, and \
                 clawhdf5-format::verify_signed_root now checks that binding -- see T6c, the \
                 signed counterpart of this attack, which reports detected=yes.",
            )
        } else {
            None
        },
    }
}

/// Injectable [`SignedRootVerifier`] backed by the real P2.1 hybrid signer
/// (Ed25519 + ML-DSA-65). Holds one genuine signature and its public keys, and
/// reports a root as authentic only if that signature validates over the
/// canonical payload binding `(root, companion_hash, alg, version, timestamp)`.
/// This is the injection point that wires `clawhdf5-sign` into
/// `clawhdf5-format`'s `verify_signed_root` without a dependency cycle.
struct HybridRootVerifier {
    sig: clawhdf5_sign::HybridSignature,
    ed_pub: clawhdf5_sign::VerifyingKey,
    ml_pub: clawhdf5_sign::mldsa::MlDsaVerifyingKey,
}

impl SignedRootVerifier for HybridRootVerifier {
    fn verify_signed_root(
        &self,
        root: &[u8; 32],
        companion_hash: &[u8; 32],
        alg: HashAlg,
        version: u64,
        timestamp: u64,
    ) -> bool {
        // clawhdf5-sign re-exports clawhdf5-format's HashAlg, so `alg` feeds
        // canonical_payload directly — the payload binds every field.
        let payload =
            clawhdf5_sign::canonical_payload(root, companion_hash, version, timestamp, alg);
        clawhdf5_sign::verify_sig(&payload, &self.sig, &self.ed_pub, &self.ml_pub).is_ok()
    }
}

/// Sign the honest `(root, companion_hash, alg, version, timestamp)` with fresh
/// hybrid keys and return a verifier bound to that signature. Shared by the
/// signed-dataset attacks (T6c, T1f).
fn sign_honest_root(attr: &MerkleAttr, version: u64, timestamp: u64) -> HybridRootVerifier {
    use clawhdf5_sign::{SigningKey, canonical_payload, mldsa::MlDsaSigningKey, sign_root};
    let ed_key = SigningKey::generate();
    let ml_key = MlDsaSigningKey::generate();
    let payload = canonical_payload(
        &attr.root,
        &attr.companion_hash,
        version,
        timestamp,
        attr.algorithm,
    );
    let sig = sign_root(&payload, &ed_key, &ml_key).expect("honest signing succeeds");
    HybridRootVerifier {
        sig,
        ed_pub: ed_key.verifying_key(),
        ml_pub: ml_key.verifying_key(),
    }
}

/// T6c — algorithm downgrade against a **signed** dataset: the signed
/// counterpart to T6b, showing the gap T6b documents is closed once a
/// signature is in force. A legitimate signer signs the honest BLAKE3 root;
/// the attacker rebuilds a fully self-consistent tree under the weaker SHA-256,
/// which passes `verify_root`'s self-consistency check — but cannot produce a
/// signature over the downgraded payload. `verify_signed_root` binds `alg` into
/// the checked payload, so the carried-over signature fails and it is detected.
pub fn t6c_signed_algorithm_downgrade() -> AttackResult {
    let chunks: Vec<&[u8]> = vec![b"chunk-a", b"chunk-b", b"chunk-c", b"chunk-d"];
    let honest_tree = MerkleTree::from_chunks(&chunks, HashAlg::Blake3);
    let mut honest_nodes = Vec::new();
    for n in honest_tree.nodes() {
        honest_nodes.extend_from_slice(n);
    }
    let honest_attr =
        MerkleAttr::from_tree_with_companion(&honest_tree, companion_hash(&honest_nodes));

    let version = 1u64;
    let timestamp = 1_700_000_000u64;
    let verifier = sign_honest_root(&honest_attr, version, timestamp);

    let start_time = Instant::now();
    // Attacker rebuilds the tree under SHA-256 (self-consistent, no forged bytes).
    let downgraded_tree = MerkleTree::from_chunks(&chunks, HashAlg::Sha256);
    let mut down_nodes = Vec::new();
    for n in downgraded_tree.nodes() {
        down_nodes.extend_from_slice(n);
    }
    let downgraded_attr =
        MerkleAttr::from_tree_with_companion(&downgraded_tree, companion_hash(&down_nodes));
    let view = Dataset::from_owned(downgraded_attr, down_nodes, chunks.clone());
    let result = verify_signed_root(&view, version, timestamp, &verifier);
    let latency = start_time.elapsed();

    AttackResult {
        threat_class: "T6",
        attack_id: "T6c",
        dataset: "n/a",
        detected: matches!(result, Err(MerkleError::SignatureInvalid)),
        verifier_fn: "verify_signed_root",
        latency,
        root_cause: None,
    }
}

/// T1f — directed companion forgery against a **signed** dataset: the signed
/// counterpart to T1d. The attacker flips a companion node and recomputes the
/// companion hash so the attribute is self-consistent (defeats bare
/// `verify_root`), but `verify_signed_root` binds `companion_hash` into the
/// signed payload, so the honest signature no longer validates — detected.
pub fn t1f_signed_companion_forgery() -> AttackResult {
    let chunks: Vec<&[u8]> = vec![b"chunk-a", b"chunk-b", b"chunk-c", b"chunk-d"];
    let honest_tree = MerkleTree::from_chunks(&chunks, HashAlg::Blake3);
    let mut honest_nodes = Vec::new();
    for n in honest_tree.nodes() {
        honest_nodes.extend_from_slice(n);
    }
    let honest_attr =
        MerkleAttr::from_tree_with_companion(&honest_tree, companion_hash(&honest_nodes));

    let version = 1u64;
    let timestamp = 1_700_000_000u64;
    let verifier = sign_honest_root(&honest_attr, version, timestamp);

    let start_time = Instant::now();
    // Directed forgery: flip a companion node AND recompute the companion hash.
    let mut tampered_nodes = honest_nodes.clone();
    tampered_nodes[0] ^= 0xFF;
    let forged_attr =
        MerkleAttr::from_tree_with_companion(&honest_tree, companion_hash(&tampered_nodes));
    let view = Dataset::from_owned(forged_attr, tampered_nodes, chunks.clone());
    let result = verify_signed_root(&view, version, timestamp, &verifier);
    let latency = start_time.elapsed();

    AttackResult {
        threat_class: "T1",
        attack_id: "T1f",
        dataset: "n/a",
        detected: matches!(result, Err(MerkleError::SignatureInvalid)),
        verifier_fn: "verify_signed_root",
        latency,
        root_cause: None,
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
    // The grid hash is self-consistent (built via ChunkGridParams::new) so
    // this check passes and the actual DoS guard being tested --
    // checked_padded_leaf_count -- is what's exercised, not GridHashMismatch.
    let result = verify_subset(
        &[0u8; 32],
        HashAlg::Blake3,
        &[],
        &proof,
        &hostile_grid,
        &hostile_grid.grid_hash,
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
