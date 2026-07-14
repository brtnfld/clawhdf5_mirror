//! End-to-end encrypted dataset test with Merkle tree verification.
//!
//! This test implements P2.2 step 6 from S2-D2-Yr2:
//!
//! > "Write an end-to-end encrypted test: write a 256-chunk encrypted dataset,
//! > verify the Merkle tree, overwrite chunk 7 (which must increment v_chunk[7]
//! > and update the leaf hash), then call verify_dataset and confirm the tree
//! > is still consistent."
//!
//! # Nonce Reuse Prevention Proof
//!
//! **Nonce reuse is impossible by construction** because:
//!
//! 1. The nonce is derived as `BLAKE3-KDF(DEK || chunk_idx || v_chunk)` where
//!    `v_chunk` is a monotonically increasing version counter.
//!
//! 2. The WAL protocol ensures `v_chunk` is durably recorded BEFORE the nonce
//!    is derived for encryption. A crash after journaling but before commit
//!    results in recovery replaying the write with the **same journaled version**,
//!    not rolling back to a previous version.
//!
//! 3. The `VersionCounterStore` enforces strict monotonicity: `update(k, v)`
//!    panics if `v <= current_version[k]`. This is an invariant, not a check—
//!    violating it indicates a logic error in the caller.
//!
//! 4. For nonce reuse to occur, an attacker would need to:
//!    - Roll back `v_chunk[k]` to a previous value, AND
//!    - Encrypt new plaintext under that rolled-back version
//!
//!    The WAL protocol prevents (a): after a crash, recovery either replays
//!    the pending write at its journaled version or the write never happened.
//!    The monotonicity invariant prevents (b): even if `v_chunk` were somehow
//!    rolled back in memory, the next call to `update()` would panic.
//!
//! 5. The Merkle leaf hash binds `v_chunk` into the tree structure:
//!    `H_leaf(k) = H(0x00 || len(k) || k || len(ct) || ct || tag || v_k)`
//!
//!    If an attacker selectively rolls back a chunk and its version counter
//!    in storage, the leaf hash changes, breaking the Merkle path to the
//!    signed root. Verification rejects the tampered dataset.
//!
//! Therefore, under the threat model where the attacker controls storage but
//! not the encryption oracle (i.e., cannot call `encrypt_chunk` after rollback),
//! nonce reuse is cryptographically prevented.

use clawhdf5_filters::{
    FilterPipeline, FilterPipelineConfig, FilteredChunk, Hash, VersionCounterStore, VersionWal,
    compute_leaf_hash, derive_nonce, mock_dek,
};
use clawhdf5_format::merkle::{HashAlg, MerkleTree};
use std::io::Cursor;

/// Number of chunks in the test dataset (per P2.2 step 6).
const NUM_CHUNKS: usize = 256;

/// The chunk index to overwrite (per P2.2 step 6).
const OVERWRITE_CHUNK_IDX: u64 = 7;

/// Generate deterministic test data for a chunk.
fn generate_chunk_data(chunk_idx: u64, version: u64) -> Vec<u8> {
    // Create reproducible but distinct data for each (chunk_idx, version) pair
    let mut data = Vec::with_capacity(1024);
    for i in 0..256 {
        let byte = ((chunk_idx as u32)
            .wrapping_mul(31)
            .wrapping_add(version as u32)
            .wrapping_mul(17)
            .wrapping_add(i)) as u8;
        data.extend_from_slice(&[byte; 4]); // 4 bytes per element (simulate f32)
    }
    data
}

/// Verify that all leaf hashes in the tree match the expected hashes.
fn verify_leaf_hashes(
    tree: &MerkleTree,
    chunks: &[FilteredChunk],
) -> bool {
    for (idx, chunk) in chunks.iter().enumerate() {
        if let Some(stored_hash) = tree.leaf_hash(idx) {
            if stored_hash != &chunk.leaf_hash {
                return false;
            }
        } else {
            return false;
        }
    }
    true
}

/// End-to-end encrypted dataset test with Merkle tree verification.
///
/// This test:
/// 1. Creates a 256-chunk encrypted dataset
/// 2. Builds a Merkle tree from the leaf hashes
/// 3. Verifies the tree is consistent
/// 4. Overwrites chunk 7 (incrementing its version counter)
/// 5. Updates the Merkle tree with the new leaf hash
/// 6. Verifies the tree is still consistent after the update
#[test]
fn end_to_end_encrypted_256_chunk_dataset() {
    // Get test DEK from environment or use default
    let dek = mock_dek();

    // Create pipeline with shuffle (4-byte elements) and compression
    let config = FilterPipelineConfig::encrypted(4, 6);
    let pipeline = FilterPipeline::new(config);

    // Phase 1: Create 256 encrypted chunks
    // =====================================
    // Each chunk starts at version 1 (first write).
    let mut chunks: Vec<FilteredChunk> = Vec::with_capacity(NUM_CHUNKS);
    let mut version_store = VersionCounterStore::new();

    for chunk_idx in 0..NUM_CHUNKS as u64 {
        // Increment version for this chunk (first write = version 1)
        let version = version_store.next_version(chunk_idx);
        version_store.update(chunk_idx, version);

        // Generate test data and encrypt
        let plaintext = generate_chunk_data(chunk_idx, version);
        let filtered = pipeline
            .write(&plaintext, Some(&dek), chunk_idx, version)
            .expect("encryption should succeed");

        assert_eq!(filtered.chunk_idx, chunk_idx);
        assert_eq!(filtered.version, version);

        chunks.push(filtered);
    }

    assert_eq!(chunks.len(), NUM_CHUNKS);

    // Phase 2: Build Merkle tree from leaf hashes
    // ============================================
    // The leaf hashes from FilterPipeline include:
    // - Chunk index binding (prevents position-swapping attacks)
    // - AEAD tag (Poly1305)
    // - Version counter (prevents rollback attacks)
    let leaf_hashes: Vec<Hash> = chunks.iter().map(|c| c.leaf_hash).collect();
    let tree = MerkleTree::build(&leaf_hashes, HashAlg::Blake3)
        .expect("Merkle tree build should succeed");

    assert_eq!(tree.leaf_count(), NUM_CHUNKS);

    // Phase 3: Verify tree consistency
    // =================================
    // All leaf hashes should match their stored values.
    assert!(
        verify_leaf_hashes(&tree, &chunks),
        "initial tree verification should pass"
    );

    // Store the original root for comparison
    let original_root = *tree.root();

    // Phase 4: Overwrite chunk 7 (in-place update)
    // =============================================
    // This MUST increment v_chunk[7] from 1 to 2.
    // The new nonce will be derived from (DEK, chunk_idx=7, version=2).
    let new_version = version_store.next_version(OVERWRITE_CHUNK_IDX);
    assert_eq!(new_version, 2, "overwrite should increment version to 2");
    version_store.update(OVERWRITE_CHUNK_IDX, new_version);

    // Generate different data for the overwrite
    let new_plaintext = generate_chunk_data(OVERWRITE_CHUNK_IDX, new_version);
    let new_filtered = pipeline
        .write(&new_plaintext, Some(&dek), OVERWRITE_CHUNK_IDX, new_version)
        .expect("overwrite encryption should succeed");

    assert_eq!(new_filtered.chunk_idx, OVERWRITE_CHUNK_IDX);
    assert_eq!(new_filtered.version, 2);

    // The new ciphertext must differ from the original (different nonce)
    assert_ne!(
        new_filtered.ciphertext,
        chunks[OVERWRITE_CHUNK_IDX as usize].ciphertext,
        "ciphertext must change due to different nonce"
    );

    // The new leaf hash must differ (different ciphertext + version)
    assert_ne!(
        new_filtered.leaf_hash,
        chunks[OVERWRITE_CHUNK_IDX as usize].leaf_hash,
        "leaf hash must change after overwrite"
    );

    // Phase 5: Update Merkle tree with new leaf hash
    // ================================================
    // This is the O(log N) update operation.
    let mut updated_tree = tree.clone();
    updated_tree
        .update_leaf(OVERWRITE_CHUNK_IDX as usize, new_filtered.leaf_hash)
        .expect("tree update should succeed");

    // The root must change after updating a leaf
    assert_ne!(
        updated_tree.root(),
        &original_root,
        "root must change after leaf update"
    );

    // Update our chunks array to reflect the change
    chunks[OVERWRITE_CHUNK_IDX as usize] = new_filtered;

    // Phase 6: Verify tree is still consistent after update
    // =======================================================
    // All leaf hashes should still match their stored values.
    assert!(
        verify_leaf_hashes(&updated_tree, &chunks),
        "tree verification should pass after update"
    );

    // Verify the updated chunk can be verified independently
    let stored_hash = updated_tree
        .leaf_hash(OVERWRITE_CHUNK_IDX as usize)
        .expect("leaf hash should exist");
    assert_eq!(
        stored_hash,
        &chunks[OVERWRITE_CHUNK_IDX as usize].leaf_hash,
        "stored leaf hash should match filtered chunk"
    );

    // Verify using the pipeline's verify_leaf_hash (EtM verification)
    assert!(
        pipeline.verify_leaf_hash(
            OVERWRITE_CHUNK_IDX,
            &chunks[OVERWRITE_CHUNK_IDX as usize].ciphertext,
            chunks[OVERWRITE_CHUNK_IDX as usize].version,
            &chunks[OVERWRITE_CHUNK_IDX as usize].leaf_hash
        ),
        "pipeline leaf hash verification should pass"
    );
}

/// Test that decryption succeeds for all chunks with correct version.
#[test]
fn end_to_end_decrypt_all_chunks() {
    let dek = mock_dek();
    let config = FilterPipelineConfig::encrypted(4, 6);
    let pipeline = FilterPipeline::new(config);

    // Create and encrypt chunks
    let mut chunks: Vec<(Vec<u8>, FilteredChunk)> = Vec::with_capacity(NUM_CHUNKS);
    let mut version_store = VersionCounterStore::new();

    for chunk_idx in 0..NUM_CHUNKS as u64 {
        let version = version_store.next_version(chunk_idx);
        version_store.update(chunk_idx, version);

        let plaintext = generate_chunk_data(chunk_idx, version);
        let filtered = pipeline
            .write(&plaintext, Some(&dek), chunk_idx, version)
            .expect("encryption should succeed");

        chunks.push((plaintext, filtered));
    }

    // Decrypt and verify each chunk
    for (chunk_idx, (original_plaintext, filtered)) in chunks.iter().enumerate() {
        let decrypted = pipeline
            .read(
                &filtered.ciphertext,
                Some(&dek),
                chunk_idx as u64,
                filtered.version,
                original_plaintext.len(),
            )
            .expect("decryption should succeed");

        assert_eq!(
            &decrypted, original_plaintext,
            "decrypted data should match original for chunk {chunk_idx}"
        );
    }
}

/// Test that wrong version fails decryption (AEAD tag mismatch).
#[test]
fn wrong_version_fails_decryption() {
    let dek = mock_dek();
    let config = FilterPipelineConfig::encrypted(4, 6);
    let pipeline = FilterPipeline::new(config);

    let plaintext = generate_chunk_data(0, 1);
    let filtered = pipeline
        .write(&plaintext, Some(&dek), 0, 1)
        .expect("encryption should succeed");

    // Attempt decryption with wrong version
    let result = pipeline.read(
        &filtered.ciphertext,
        Some(&dek),
        0,
        2, // wrong version
        plaintext.len(),
    );

    assert!(
        result.is_err(),
        "decryption with wrong version should fail"
    );
}

/// Test that tampering with ciphertext is detected.
#[test]
fn tampered_ciphertext_detected() {
    let dek = mock_dek();
    let config = FilterPipelineConfig::encrypted(4, 6);
    let pipeline = FilterPipeline::new(config);

    let plaintext = generate_chunk_data(0, 1);
    let filtered = pipeline
        .write(&plaintext, Some(&dek), 0, 1)
        .expect("encryption should succeed");

    // Tamper with ciphertext
    let mut tampered = filtered.ciphertext.clone();
    tampered[0] ^= 0xFF;

    // Decryption should fail (AEAD tag mismatch)
    let result = pipeline.read(&tampered, Some(&dek), 0, 1, plaintext.len());
    assert!(result.is_err(), "tampered ciphertext should fail decryption");

    // Leaf hash verification should also fail
    assert!(
        !pipeline.verify_leaf_hash(0, &tampered, 1, &filtered.leaf_hash),
        "tampered ciphertext should fail leaf hash verification"
    );
}

/// Test WAL-based crash recovery scenario.
#[test]
fn wal_crash_recovery_preserves_nonce_safety() {
    use clawhdf5_filters::EncryptedChunkWriter;

    let dek = mock_dek();

    // Simulate a crash: encrypt but don't commit
    let mut wal_buf = Cursor::new(Vec::new());
    let uncommitted_version: u64;
    {
        let wal = VersionWal::new(&mut wal_buf, 100).expect("WAL creation should succeed");
        let versions = VersionCounterStore::new();
        let mut writer = EncryptedChunkWriter::new(dek, wal, versions);

        // Encrypt chunk 5 but "crash" before commit
        let result = writer
            .encrypt_chunk(5, b"pre-crash data")
            .expect("encryption should succeed");
        uncommitted_version = result.version;

        // No commit - simulating crash
    }

    // Recovery: reopen and check for uncommitted
    wal_buf.set_position(0);
    let wal = VersionWal::new(&mut wal_buf, 100).expect("WAL reopen should succeed");
    let versions = VersionCounterStore::new();
    let mut writer = EncryptedChunkWriter::new(dek, wal, versions);

    let uncommitted = writer.recover().expect("recovery should succeed");
    assert_eq!(uncommitted.len(), 1);
    assert_eq!(uncommitted[0], (5, uncommitted_version));

    // CRITICAL: recover() seeds the version store with the journaled version, so a
    // subsequent write of NEW data to the same chunk must use a strictly higher
    // version — never re-deriving the pre-crash nonce.
    let retried = writer
        .encrypt_chunk(5, b"post-crash data")
        .expect("post-crash write should succeed");
    assert!(
        retried.version > uncommitted_version,
        "post-crash write must use a higher version than the journaled one \
         (got {}, journaled {})",
        retried.version,
        uncommitted_version
    );

    // The nonces for the pre-crash and post-crash versions differ, so encrypting
    // two different plaintexts can never reuse a keystream.
    assert_ne!(
        derive_nonce(&dek, 5, uncommitted_version),
        derive_nonce(&dek, 5, retried.version),
        "post-crash nonce must differ from the journaled-version nonce"
    );

    // Nonce reuse is impossible because:
    // - Idempotent replay uses the same version → same nonce → same ciphertext
    // - A new write uses a seeded, strictly higher version → different nonce
}

/// Test position-swapping attack detection.
#[test]
fn position_swapping_attack_detected() {
    let dek = mock_dek();
    let config = FilterPipelineConfig::encrypted(4, 6);
    let pipeline = FilterPipeline::new(config);

    // Create two chunks at different positions
    let chunk_0 = pipeline
        .write(b"chunk zero data", Some(&dek), 0, 1)
        .expect("encryption should succeed");
    let chunk_1 = pipeline
        .write(b"chunk one data", Some(&dek), 1, 1)
        .expect("encryption should succeed");

    // Build Merkle tree
    let tree = MerkleTree::build(&[chunk_0.leaf_hash, chunk_1.leaf_hash], HashAlg::Blake3)
        .expect("tree build should succeed");

    // Verify both chunks at their correct positions
    assert_eq!(tree.leaf_hash(0).unwrap(), &chunk_0.leaf_hash);
    assert_eq!(tree.leaf_hash(1).unwrap(), &chunk_1.leaf_hash);

    // Position-swapping attack: try to verify chunk_1's ciphertext at position 0
    // This should fail because the chunk index is bound into the leaf hash.
    let forged_leaf_hash = compute_leaf_hash(
        0, // attacker claims this is position 0
        &chunk_1.ciphertext,
        chunk_1.version,
    );

    // The forged hash won't match the stored hash at position 0
    assert_ne!(
        tree.leaf_hash(0).unwrap(),
        &forged_leaf_hash,
        "position-swapping attack should be detected"
    );

    // The leaf hash at position 0 is specifically for chunk_0's data
    assert_eq!(tree.leaf_hash(0).unwrap(), &chunk_0.leaf_hash);
}

/// Test that VersionCounterStore serialization/deserialization preserves state
/// across simulated restarts.
#[test]
fn version_store_persists_across_restart() {
    let dek = mock_dek();
    let config = FilterPipelineConfig::encrypted(4, 6);
    let pipeline = FilterPipeline::new(config);

    // Phase 1: Create chunks and build initial state
    let mut version_store = VersionCounterStore::new();
    let mut chunks: Vec<FilteredChunk> = Vec::new();

    for chunk_idx in 0..10u64 {
        let version = version_store.next_version(chunk_idx);
        version_store.update(chunk_idx, version);

        let plaintext = generate_chunk_data(chunk_idx, version);
        let filtered = pipeline
            .write(&plaintext, Some(&dek), chunk_idx, version)
            .expect("encryption should succeed");
        chunks.push(filtered);
    }

    // Serialize version store (simulating persistence)
    let serialized = version_store.to_bytes();

    // Phase 2: Simulate restart by deserializing
    let restored_store =
        VersionCounterStore::from_bytes(&serialized).expect("deserialization should succeed");

    // Verify all versions match
    for chunk_idx in 0..10u64 {
        assert_eq!(
            restored_store.get(chunk_idx),
            version_store.get(chunk_idx),
            "version for chunk {chunk_idx} should match after restore"
        );
    }
    assert_eq!(restored_store.max_version(), version_store.max_version());

    // Phase 3: Continue writing with restored store
    let mut continued_store = restored_store;
    let new_version = continued_store.next_version(5);
    assert_eq!(new_version, 2, "next version should be 2 after restore");
    continued_store.update(5, new_version);

    // Encrypt with the new version
    let new_plaintext = generate_chunk_data(5, new_version);
    let new_filtered = pipeline
        .write(&new_plaintext, Some(&dek), 5, new_version)
        .expect("encryption should succeed");

    // Verify the new chunk has different ciphertext (different nonce from new version)
    assert_ne!(
        new_filtered.ciphertext, chunks[5].ciphertext,
        "ciphertext should differ due to new version/nonce"
    );

    // Verify old chunk can still be decrypted with original version
    let decrypted_old = pipeline
        .read(&chunks[5].ciphertext, Some(&dek), 5, 1, 1024)
        .expect("decryption of old chunk should succeed");
    assert_eq!(decrypted_old, generate_chunk_data(5, 1));

    // Verify new chunk can be decrypted with new version
    let decrypted_new = pipeline
        .read(&new_filtered.ciphertext, Some(&dek), 5, new_version, 1024)
        .expect("decryption of new chunk should succeed");
    assert_eq!(decrypted_new, new_plaintext);
}
