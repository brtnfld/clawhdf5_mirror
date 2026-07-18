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
//! logical persisted content and store the packed [`MerkleAttr`] as a
//! hex-encoded `_merkle_root` attribute in `/meta`. On read we re-hash the
//! content loaded from disk and compare. A mismatch (or a corrupted attribute)
//! is a fail-closed [`MemoryError::Integrity`].
//!
//! **Coverage.** The root spans every field that survives the write→read
//! round-trip across all three persisted groups:
//! - `/memory`: chunks, embeddings, source channels, timestamps, session ids,
//!   tags, tombstones.
//! - `/sessions`: id, start/end index, channel, timestamp, summary.
//! - `/knowledge_graph`: entity (id, name, type, embedding index), relation
//!   (src, tgt, type, weight, ts), and aliases (string, entity id).
//!
//! Fields the loader reconstructs as defaults rather than reading back (entity
//! `properties`/`embedding`/`created_at`/`updated_at`, relation `metadata`) and
//! fields recomputed on read (`norms`, `activation_weights`) are deliberately
//! excluded — hashing them would make an honest round-trip fail, since the
//! bytes on read differ from the bytes on write. Each leaf carries a section
//! tag byte so content can't be moved between sections undetected.
//!
//! **Threat model.** This is *unsigned* integrity: it detects tampering or
//! corruption of an existing file, not an attacker who rebuilds the whole file
//! (root included) from scratch — that requires the P2.1 hybrid signature,
//! still unwired here. Two policies are offered for the missing-attribute case:
//! [`verify_content`] fails *open* when there is no `_merkle_root` (older
//! writers, or `integrity` disabled), for backward compatibility; strict
//! callers (see `schema::validate_and_load_strict`) reject a file that carries
//! no root, closing the strip-to-downgrade attack.

use clawhdf5_format::merkle::{Dataset, HashAlg, MerkleAttr, MerkleError, MerkleTree, verify_dataset};
use sha2::{Digest, Sha256};

use crate::MemoryError;
use crate::cache::MemoryCache;
use crate::knowledge::KnowledgeCache;
use crate::session::SessionCache;

/// The `/meta` attribute name holding the hex-encoded packed [`MerkleAttr`].
pub const MERKLE_ROOT_ATTR: &str = "_merkle_root";

/// Hash algorithm for the content Merkle tree. Blake3 matches the harness and
/// the rest of the P2.x Merkle work.
const ALG: HashAlg = HashAlg::Blake3;

// Per-section leaf tags: content can never be reinterpreted across sections
// because the first byte of every leaf commits to which section it belongs to.
const TAG_MEMORY: u8 = b'M';
const TAG_SESSION: u8 = b'S';
const TAG_ENTITY: u8 = b'E';
const TAG_RELATION: u8 = b'R';
const TAG_ALIAS: u8 = b'A';

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

/// Canonical leaf bytes for every persisted record across the three groups, in
/// a fixed order (memory, sessions, entities, relations, aliases). Each leaf is
/// self-delimiting (length-prefixed variable fields) and section-tagged.
/// Returns `None` when there is nothing to protect (all sections empty).
fn all_leaves(
    memory: &MemoryCache,
    sessions: &SessionCache,
    knowledge: &KnowledgeCache,
) -> Option<Vec<Vec<u8>>> {
    let mut leaves: Vec<Vec<u8>> = Vec::new();

    // --- /memory ---
    for i in 0..memory.chunks.len() {
        let mut buf = vec![TAG_MEMORY];
        push_len_prefixed(&mut buf, memory.chunks[i].as_bytes());
        let emb = memory.embeddings.get(i).map(Vec::as_slice).unwrap_or(&[]);
        buf.extend_from_slice(&(emb.len() as u64).to_le_bytes());
        for &v in emb {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        push_len_prefixed(
            &mut buf,
            memory.source_channels.get(i).map_or(b"".as_slice(), |s| s.as_bytes()),
        );
        buf.extend_from_slice(&memory.timestamps.get(i).copied().unwrap_or(0.0).to_le_bytes());
        push_len_prefixed(
            &mut buf,
            memory.session_ids.get(i).map_or(b"".as_slice(), |s| s.as_bytes()),
        );
        push_len_prefixed(&mut buf, memory.tags.get(i).map_or(b"".as_slice(), |s| s.as_bytes()));
        buf.push(memory.tombstones.get(i).copied().unwrap_or(0));
        leaves.push(buf);
    }

    // --- /sessions ---
    for (i, e) in sessions.entries.iter().enumerate() {
        let mut buf = vec![TAG_SESSION];
        push_len_prefixed(&mut buf, e.id.as_bytes());
        buf.extend_from_slice(&e.start_idx.to_le_bytes());
        buf.extend_from_slice(&e.end_idx.to_le_bytes());
        push_len_prefixed(&mut buf, e.channel.as_bytes());
        buf.extend_from_slice(&e.ts.to_le_bytes());
        push_len_prefixed(
            &mut buf,
            sessions.summaries.get(i).map_or(b"".as_slice(), |s| s.as_bytes()),
        );
        leaves.push(buf);
    }

    // --- /knowledge_graph: entities ---
    for ent in &knowledge.entities {
        let mut buf = vec![TAG_ENTITY];
        buf.extend_from_slice(&ent.id.to_le_bytes());
        push_len_prefixed(&mut buf, ent.name.as_bytes());
        push_len_prefixed(&mut buf, ent.entity_type.as_bytes());
        buf.extend_from_slice(&ent.embedding_idx.to_le_bytes());
        leaves.push(buf);
    }

    // --- /knowledge_graph: relations ---
    for rel in &knowledge.relations {
        let mut buf = vec![TAG_RELATION];
        buf.extend_from_slice(&rel.src.to_le_bytes());
        buf.extend_from_slice(&rel.tgt.to_le_bytes());
        push_len_prefixed(&mut buf, rel.relation.as_bytes());
        buf.extend_from_slice(&rel.weight.to_le_bytes());
        buf.extend_from_slice(&rel.ts.to_le_bytes());
        leaves.push(buf);
    }

    // --- /knowledge_graph: aliases ---
    for i in 0..knowledge.alias_strings.len() {
        let mut buf = vec![TAG_ALIAS];
        push_len_prefixed(&mut buf, knowledge.alias_strings[i].as_bytes());
        buf.extend_from_slice(
            &knowledge.alias_entity_ids.get(i).copied().unwrap_or(0).to_le_bytes(),
        );
        leaves.push(buf);
    }

    if leaves.is_empty() { None } else { Some(leaves) }
}

/// Build the Merkle tree + packed companion attribute over all persisted
/// content. Returns `None` for an empty store (no attribute is written then).
fn build_attr(
    memory: &MemoryCache,
    sessions: &SessionCache,
    knowledge: &KnowledgeCache,
) -> Option<(MerkleAttr, MerkleTree, Vec<Vec<u8>>)> {
    let leaves = all_leaves(memory, sessions, knowledge)?;
    let refs: Vec<&[u8]> = leaves.iter().map(|l| l.as_slice()).collect();
    let tree = MerkleTree::from_chunks(&refs, ALG);
    let mut nodes = Vec::with_capacity(tree.nodes().len() * 32);
    for node in tree.nodes() {
        nodes.extend_from_slice(node);
    }
    let attr = MerkleAttr::from_tree_with_companion(&tree, companion_hash(&nodes));
    Some((attr, tree, leaves))
}

/// Compute the `_merkle_root` attribute value (hex-encoded packed
/// [`MerkleAttr`]) for the current persisted content, or `None` when there's
/// nothing to protect.
pub fn content_root_hex(
    memory: &MemoryCache,
    sessions: &SessionCache,
    knowledge: &KnowledgeCache,
) -> Option<String> {
    let (attr, _, _) = build_attr(memory, sessions, knowledge)?;
    Some(hex_encode(&attr.pack()))
}

/// Re-hash the content freshly loaded from disk and verify it against the
/// stored `_merkle_root` attribute.
///
/// `stored_hex` is `None` when the file carries no `_merkle_root`. When
/// `strict` is false (default), that is accepted (older writers / feature off).
/// When `strict` is true, a missing attribute is itself rejected, closing the
/// strip-to-downgrade attack. A present-but-mismatched or corrupt attribute is
/// always rejected.
pub fn verify_content(
    memory: &MemoryCache,
    sessions: &SessionCache,
    knowledge: &KnowledgeCache,
    stored_hex: Option<&str>,
    strict: bool,
) -> Result<(), MemoryError> {
    let Some(stored_hex) = stored_hex else {
        if strict {
            return Err(MemoryError::Integrity(format!(
                "strict mode: file has no {MERKLE_ROOT_ATTR} attribute (unprotected or stripped)"
            )));
        }
        return Ok(());
    };

    let packed = hex_decode(stored_hex)
        .ok_or_else(|| MemoryError::Integrity(format!("{MERKLE_ROOT_ATTR} is not valid hex")))?;
    // Unpack self-checks the attribute's own integrity hash, catching a zeroed
    // or truncated attribute (T6a) before we trust its root/algorithm.
    let attr = MerkleAttr::unpack(&packed)
        .map_err(|e| MemoryError::Integrity(format!("{MERKLE_ROOT_ATTR} is corrupt: {e:?}")))?;

    let Some((_, tree, leaves)) = build_attr(memory, sessions, knowledge) else {
        // Attribute present but the loaded content is empty: the file claimed
        // protected content that isn't there.
        return Err(MemoryError::Integrity(
            "content is empty but a _merkle_root attribute is present".into(),
        ));
    };

    let mut nodes = Vec::with_capacity(tree.nodes().len() * 32);
    for node in tree.nodes() {
        nodes.extend_from_slice(node);
    }
    let leaf_refs: Vec<&[u8]> = leaves.iter().map(|l| l.as_slice()).collect();
    let view = Dataset::from_owned(attr, nodes, leaf_refs);
    match verify_dataset(&view) {
        Ok(true) => Ok(()),
        Ok(false) => Err(MemoryError::Integrity(
            "content Merkle verification returned false".into(),
        )),
        Err(MerkleError::HashMismatch { chunk_idx }) => Err(MemoryError::Integrity(format!(
            "content tampering detected: persisted record {chunk_idx} does not match its stored Merkle leaf"
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
    use crate::knowledge::{Entity, Relation};
    use crate::session::SessionEntry;

    fn sample() -> (MemoryCache, SessionCache, KnowledgeCache) {
        let mut m = MemoryCache::new(3);
        m.chunks = vec!["first memory".into(), "second memory".into()];
        m.embeddings = vec![vec![0.1, 0.2, 0.3], vec![-1.0, 0.5, 2.5]];
        m.source_channels = vec!["slack".into(), "email".into()];
        m.timestamps = vec![100.0, 200.0];
        m.session_ids = vec!["s1".into(), "s2".into()];
        m.tags = vec!["a".into(), "b".into()];
        m.tombstones = vec![0, 0];

        let mut s = SessionCache::new();
        s.entries.push(SessionEntry {
            id: "sess-1".into(),
            start_idx: 0,
            end_idx: 2,
            channel: "slack".into(),
            ts: 100.0,
        });
        s.summaries.push("a chat about hashing".into());

        let mut k = KnowledgeCache::new();
        k.entities.push(Entity {
            id: 7,
            name: "Merkle".into(),
            entity_type: "concept".into(),
            embedding_idx: -1,
            ..Default::default()
        });
        k.relations.push(Relation {
            src: 7,
            tgt: 8,
            relation: "relates_to".into(),
            weight: 0.9,
            ts: 100.0,
            ..Default::default()
        });
        k.alias_strings = vec!["merkle-tree".into()];
        k.alias_entity_ids = vec![7];
        (m, s, k)
    }

    #[test]
    fn honest_roundtrip_verifies() {
        let (m, s, k) = sample();
        let hex = content_root_hex(&m, &s, &k).expect("non-empty store produces a root");
        let (m2, s2, k2) = sample();
        verify_content(&m2, &s2, &k2, Some(&hex), false).expect("honest round-trip must verify");
        verify_content(&m2, &s2, &k2, Some(&hex), true).expect("strict honest round-trip verifies");
    }

    #[test]
    fn tampered_memory_is_detected() {
        let (m, s, k) = sample();
        let hex = content_root_hex(&m, &s, &k).unwrap();
        let (mut m2, s2, k2) = sample();
        m2.chunks[1] = "second memory (forged)".into();
        verify_content(&m2, &s2, &k2, Some(&hex), false).expect_err("memory tamper detected");
    }

    #[test]
    fn tampered_session_summary_is_detected() {
        let (m, s, k) = sample();
        let hex = content_root_hex(&m, &s, &k).unwrap();
        let (m2, mut s2, k2) = sample();
        s2.summaries[0] = "a chat about NOTHING".into();
        verify_content(&m2, &s2, &k2, Some(&hex), false).expect_err("session tamper detected");
    }

    #[test]
    fn tampered_knowledge_relation_is_detected() {
        let (m, s, k) = sample();
        let hex = content_root_hex(&m, &s, &k).unwrap();
        let (m2, s2, mut k2) = sample();
        k2.relations[0].tgt = 999; // re-point an edge
        verify_content(&m2, &s2, &k2, Some(&hex), false).expect_err("knowledge tamper detected");
    }

    #[test]
    fn strict_mode_rejects_missing_attribute() {
        let (m, s, k) = sample();
        // Non-strict accepts a missing attribute (legacy); strict rejects it.
        verify_content(&m, &s, &k, None, false).expect("non-strict tolerates missing attr");
        verify_content(&m, &s, &k, None, true).expect_err("strict rejects missing attr");
    }

    #[test]
    fn corrupt_attribute_is_detected() {
        let (m, s, k) = sample();
        verify_content(&m, &s, &k, Some("not-hex!!"), false).expect_err("bad hex fails closed");
        let zeroed = "00".repeat(129);
        verify_content(&m, &s, &k, Some(&zeroed), false).expect_err("zeroed attr fails closed");
    }

    #[test]
    fn empty_store_has_no_root() {
        let m = MemoryCache::new(3);
        let s = SessionCache::new();
        let k = KnowledgeCache::new();
        assert!(content_root_hex(&m, &s, &k).is_none());
    }
}
