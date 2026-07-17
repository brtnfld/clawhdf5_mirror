//! Content-integrity wiring for persisted agent memory (P2.4 Finding 1).
//!
//! The P2.4 attack harness proved the Merkle primitives in `clawhdf5-format`
//! detect chunk tampering — but it built the Merkle state itself and called
//! the verifiers directly. Nothing on the *production* write/read path
//! (`storage::write_to_disk` → `schema::build_hdf5_file`, and
//! `storage::read_from_disk` → `schema::validate_and_load`) computed a root or
//! verified one, so a tampered on-disk memory file was accepted unconditionally.
//!
//! This module closes that gap. On write we compute a Merkle root over the
//! logical memory content and store the packed [`MerkleAttr`] as a hex-encoded
//! `_merkle_root` attribute in `/meta`. On read we re-hash the content loaded
//! from disk and compare. A mismatch (or a corrupted attribute) is a
//! fail-closed [`MemoryError::Integrity`].
//!
//! **Scope.** The root covers the round-trip-stable memory fields (`chunks`,
//! `embeddings`, `source_channels`, `timestamps`, `session_ids`, `tags`,
//! `tombstones`) — the payload an attacker would tamper to alter what the agent
//! recalls. Derived fields recomputed on read (`norms`, `activation_weights`)
//! are deliberately excluded so an honest round-trip never spuriously fails.
//! This is unsigned integrity: it detects tampering/corruption of an existing
//! file, not an attacker who rebuilds the whole file (root included) from
//! scratch — that requires the P2.1 hybrid signature, still unwired here.
//! Files with no `_merkle_root` attribute (older writers, or `integrity`
//! disabled) load unverified for backward compatibility.

use clawhdf5_format::merkle::{Dataset, HashAlg, MerkleAttr, MerkleError, MerkleTree, verify_dataset};
use sha2::{Digest, Sha256};

use crate::MemoryError;
use crate::cache::MemoryCache;

/// The `/meta` attribute name holding the hex-encoded packed [`MerkleAttr`].
pub const MERKLE_ROOT_ATTR: &str = "_merkle_root";

/// Hash algorithm for the content Merkle tree. Blake3 matches the harness and
/// the rest of the P2.x Merkle work.
const ALG: HashAlg = HashAlg::Blake3;

/// SHA-256 over the packed companion-node array, matching the harness's
/// `companion_hash` helper so the stored attribute is a genuine
/// [`MerkleAttr::from_tree_with_companion`], self-checking on unpack (T6a).
fn companion_hash(nodes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(nodes);
    hasher.finalize().into()
}

fn push_len_prefixed(buf: &mut Vec<u8>, bytes: &[u8]) {
    buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(bytes);
}

/// Canonical per-entry leaf bytes. One leaf per memory entry, so the tree maps
/// onto the natural "chunk = one memory record" boundary and supports
/// per-entry verification later. Length-prefixing every variable field makes
/// the encoding unambiguous (no field boundary can be shifted without changing
/// the bytes). Returns `None` when there is nothing to protect (empty store).
fn content_leaves(cache: &MemoryCache) -> Option<Vec<Vec<u8>>> {
    let n = cache.chunks.len();
    if n == 0 {
        return None;
    }
    // Every parallel field must line up with `chunks`; if a caller ever hands
    // us a ragged cache, refuse rather than hash a misaligned view.
    if cache.embeddings.len() != n
        || cache.source_channels.len() != n
        || cache.timestamps.len() != n
        || cache.session_ids.len() != n
        || cache.tags.len() != n
        || cache.tombstones.len() != n
    {
        return None;
    }

    let mut leaves = Vec::with_capacity(n);
    for i in 0..n {
        let mut buf = Vec::new();
        push_len_prefixed(&mut buf, cache.chunks[i].as_bytes());
        // Embedding: raw f32 little-endian, length-prefixed by element count so
        // dimension changes are part of the hash.
        buf.extend_from_slice(&(cache.embeddings[i].len() as u64).to_le_bytes());
        for &v in &cache.embeddings[i] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        push_len_prefixed(&mut buf, cache.source_channels[i].as_bytes());
        buf.extend_from_slice(&cache.timestamps[i].to_le_bytes());
        push_len_prefixed(&mut buf, cache.session_ids[i].as_bytes());
        push_len_prefixed(&mut buf, cache.tags[i].as_bytes());
        buf.push(cache.tombstones[i]);
        leaves.push(buf);
    }
    Some(leaves)
}

/// Build the Merkle tree + packed companion attribute over a cache's content.
/// Returns `None` for an empty/ragged cache (no attribute is written then).
fn build_attr(cache: &MemoryCache) -> Option<(MerkleAttr, MerkleTree)> {
    let leaves = content_leaves(cache)?;
    let refs: Vec<&[u8]> = leaves.iter().map(|l| l.as_slice()).collect();
    let tree = MerkleTree::from_chunks(&refs, ALG);
    let mut nodes = Vec::with_capacity(tree.nodes().len() * 32);
    for node in tree.nodes() {
        nodes.extend_from_slice(node);
    }
    let attr = MerkleAttr::from_tree_with_companion(&tree, companion_hash(&nodes));
    Some((attr, tree))
}

/// Compute the `_merkle_root` attribute value (hex-encoded packed
/// [`MerkleAttr`]) for a cache's current content, or `None` when there's
/// nothing to protect.
pub fn content_root_hex(cache: &MemoryCache) -> Option<String> {
    let (attr, _) = build_attr(cache)?;
    Some(hex_encode(&attr.pack()))
}

/// Re-hash `cache` (freshly loaded from disk) and verify it against the stored
/// hex-encoded `_merkle_root` attribute.
///
/// Fail-closed: any decode failure, corrupted attribute, or root mismatch is a
/// [`MemoryError::Integrity`]. Verifies via `verify_dataset` (a full O(N)
/// rehash), the same path the harness's T2 attacks exercise.
pub fn verify_content(cache: &MemoryCache, stored_hex: &str) -> Result<(), MemoryError> {
    let packed = hex_decode(stored_hex)
        .ok_or_else(|| MemoryError::Integrity(format!("{MERKLE_ROOT_ATTR} is not valid hex")))?;
    // Unpack self-checks the attribute's own integrity hash, catching a zeroed
    // or truncated attribute (T6a) before we trust its root/algorithm.
    let attr = MerkleAttr::unpack(&packed)
        .map_err(|e| MemoryError::Integrity(format!("{MERKLE_ROOT_ATTR} is corrupt: {e:?}")))?;

    let Some((_, tree)) = build_attr(cache) else {
        // Attribute present but the loaded content is empty/ragged: the file
        // claimed protected content that isn't there.
        return Err(MemoryError::Integrity(
            "content is empty or ragged but a _merkle_root attribute is present".into(),
        ));
    };

    // Rebuild the companion node array from the freshly-hashed content and
    // verify the full dataset against the stored attribute.
    let mut nodes = Vec::with_capacity(tree.nodes().len() * 32);
    for node in tree.nodes() {
        nodes.extend_from_slice(node);
    }
    let leaves = content_leaves(cache).unwrap_or_default();
    let leaf_refs: Vec<&[u8]> = leaves.iter().map(|l| l.as_slice()).collect();
    let view = Dataset::from_owned(attr, nodes, leaf_refs);
    match verify_dataset(&view) {
        Ok(true) => Ok(()),
        Ok(false) => Err(MemoryError::Integrity(
            "content Merkle verification returned false".into(),
        )),
        Err(MerkleError::HashMismatch { chunk_idx }) => Err(MemoryError::Integrity(format!(
            "content tampering detected: memory entry {chunk_idx} does not match its stored Merkle leaf"
        ))),
        Err(e) => Err(MemoryError::Integrity(format!(
            "content Merkle verification failed: {e:?}"
        ))),
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(char::from_digit((b >> 4) as u32, 16).unwrap());
        s.push(char::from_digit((b & 0x0f) as u32, 16).unwrap());
    }
    s
}

fn hex_decode(s: &str) -> Option<Vec<u8>> {
    let s = s.as_bytes();
    if !s.len().is_multiple_of(2) {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    for pair in s.chunks_exact(2) {
        let hi = (pair[0] as char).to_digit(16)?;
        let lo = (pair[1] as char).to_digit(16)?;
        out.push(((hi << 4) | lo) as u8);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_cache() -> MemoryCache {
        let mut c = MemoryCache::new(3);
        c.chunks = vec!["first memory".into(), "second memory".into()];
        c.embeddings = vec![vec![0.1, 0.2, 0.3], vec![-1.0, 0.5, 2.5]];
        c.source_channels = vec!["slack".into(), "email".into()];
        c.timestamps = vec![100.0, 200.0];
        c.session_ids = vec!["s1".into(), "s2".into()];
        c.tags = vec!["a".into(), "b".into()];
        c.tombstones = vec![0, 0];
        c
    }

    #[test]
    fn honest_roundtrip_verifies() {
        let c = sample_cache();
        let hex = content_root_hex(&c).expect("non-empty cache produces a root");
        // A cache reconstructed with identical content must verify.
        let c2 = sample_cache();
        verify_content(&c2, &hex).expect("honest round-trip must verify");
    }

    #[test]
    fn tampered_chunk_is_detected() {
        let c = sample_cache();
        let hex = content_root_hex(&c).unwrap();
        let mut tampered = sample_cache();
        tampered.chunks[1] = "second memory (forged)".into();
        let err = verify_content(&tampered, &hex).expect_err("tamper must be detected");
        matches!(err, MemoryError::Integrity(_));
    }

    #[test]
    fn tampered_embedding_is_detected() {
        let c = sample_cache();
        let hex = content_root_hex(&c).unwrap();
        let mut tampered = sample_cache();
        tampered.embeddings[0][0] = 9.9;
        verify_content(&tampered, &hex).expect_err("embedding tamper must be detected");
    }

    #[test]
    fn corrupt_attribute_is_detected() {
        let c = sample_cache();
        verify_content(&c, "not-hex!!").expect_err("bad hex must fail closed");
        let zeroed = "00".repeat(129);
        verify_content(&c, &zeroed).expect_err("zeroed attribute must fail closed");
    }

    #[test]
    fn empty_cache_has_no_root() {
        let c = MemoryCache::new(3);
        assert!(content_root_hex(&c).is_none());
    }
}
