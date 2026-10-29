//! Regression tests for the 2026-10 red-team findings against the agent
//! persistence layer. Each test is a crafted on-disk state that previously
//! panicked, aborted, loaded under `open_strict` unverified, or bypassed the
//! WAL chain. The attacker model is read/write access to `<store>.h5` and
//! `<store>.h5.wal`, no access to the running process.

use std::io::Write;
use std::path::{Path, PathBuf};

use clawhdf5_agent::cache::MemoryCache;
use clawhdf5_agent::knowledge::{Entity, KnowledgeCache};
use clawhdf5_agent::session::{SessionCache, SessionEntry};
use clawhdf5_agent::wal::WalFile;
use clawhdf5_agent::{AgentMemory, HDF5Memory, MemoryConfig, MemoryError, integrity, storage};
use sha2::{Digest, Sha256};

const DIM: usize = 4;

fn config(path: PathBuf, dim: usize) -> MemoryConfig {
    let mut c = MemoryConfig::new(path, "agent", dim);
    c.wal_enabled = true;
    c
}

fn sample_cache(n: usize, dim: usize) -> MemoryCache {
    let mut m = MemoryCache::new(dim);
    for i in 0..n {
        m.push(
            format!("memory {i}"),
            vec![0.1 * (i as f32 + 1.0); dim],
            "chan".into(),
            100.0 + i as f64,
            "sess".into(),
            format!("tag{i}"),
        );
    }
    m
}

fn write_store(
    path: &Path,
    cfg: &MemoryConfig,
    m: &MemoryCache,
    s: &SessionCache,
    k: &KnowledgeCache,
) {
    storage::write_to_disk(path, cfg, m, s, k).expect("write_to_disk");
}

fn expect_schema_error(label: &str, r: Result<HDF5Memory, MemoryError>) {
    match r {
        Err(MemoryError::Schema(msg)) => {
            eprintln!("{label}: rejected with schema error: {msg}");
        }
        Err(e) => panic!("{label}: expected a Schema error, got {e}"),
        Ok(_) => panic!("{label}: expected rejection, but the file loaded"),
    }
}

// ---------------------------------------------------------------------------
// schema.rs: parallel-dataset lengths and untrusted /meta attributes
// ---------------------------------------------------------------------------

/// A `/memory/tombstones` dataset shortened from [n] to [1] hashes to the
/// honest content root (the leaf encoding defaults a missing tombstone to 0),
/// so it passed `open_strict` and then panicked in `compact()`/`get_chunk()`.
/// The loader must reject the length mismatch before any consumer indexes it.
#[test]
fn truncated_tombstones_dataset_is_rejected_not_loaded() {
    let dir = tempfile::tempdir().unwrap();
    let honest_path = dir.path().join("honest.h5");
    let tampered_path = dir.path().join("tampered.h5");
    let cfg = config(honest_path.clone(), DIM);
    let honest = sample_cache(3, DIM);
    let sessions = SessionCache::new();
    let knowledge = KnowledgeCache::new();
    write_store(&honest_path, &cfg, &honest, &sessions, &knowledge);

    let mut tampered = honest.clone();
    tampered.tombstones.truncate(1);
    // The attack's premise: the root does not see the shortened array.
    assert_eq!(
        integrity::content_root_hex(&honest, &sessions, &knowledge),
        integrity::content_root_hex(&tampered, &sessions, &knowledge),
        "the content root is identical for the shortened tombstones array"
    );
    write_store(&tampered_path, &cfg, &tampered, &sessions, &knowledge);

    // Sanity: the honest file loads strictly and compacts without panicking.
    let mut mem = HDF5Memory::open_strict(&honest_path).expect("honest file opens strictly");
    assert_eq!(mem.count(), 3);
    assert_eq!(mem.count_active(), 3);
    let _ = mem.compact();

    expect_schema_error("truncated tombstones (strict)", HDF5Memory::open_strict(&tampered_path));
    expect_schema_error("truncated tombstones (default)", HDF5Memory::open(&tampered_path));
}

/// `/meta/embedding_dim` is not covered by the content root; a value of 0
/// reached `slice::chunks(0)` and panicked before the integrity check ran.
#[test]
fn zero_embedding_dim_attribute_is_rejected_not_panic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dim0.h5");
    // The attribute comes from the config; the data from the cache.
    let cfg = config(path.clone(), 0);
    let cache = sample_cache(2, DIM);
    write_store(&path, &cfg, &cache, &SessionCache::new(), &KnowledgeCache::new());
    expect_schema_error("embedding_dim = 0", HDF5Memory::open(&path));
}

/// `/sessions/summaries` shorter than `/sessions/ids` was indexed out of
/// bounds on open.
#[test]
fn short_sessions_summaries_dataset_is_rejected_not_panic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sessions.h5");
    let cfg = config(path.clone(), DIM);
    let mut sessions = SessionCache::new();
    for i in 0..2u64 {
        sessions.entries.push(SessionEntry {
            id: format!("s{i}"),
            start_idx: 0,
            end_idx: 1,
            channel: "chan".into(),
            ts: 1.0,
        });
    }
    sessions.summaries.push("only one summary".into());
    write_store(&path, &cfg, &sample_cache(2, DIM), &sessions, &KnowledgeCache::new());
    expect_schema_error("short summaries", HDF5Memory::open(&path));
}

/// Entity ids are stored as `u64 as i64`; an id >= 2^63 reads back negative
/// and `max + 1` overflowed (debug panic, release wrap to next_id = 0).
#[test]
fn negative_entity_id_is_rejected_not_overflow() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("entity.h5");
    let cfg = config(path.clone(), DIM);
    let mut knowledge = KnowledgeCache::new();
    knowledge.entities.push(Entity {
        id: u64::MAX,
        name: "overflow".into(),
        entity_type: "concept".into(),
        embedding_idx: -1,
        ..Default::default()
    });
    write_store(&path, &cfg, &sample_cache(1, DIM), &SessionCache::new(), &knowledge);
    expect_schema_error("entity id >= 2^63", HDF5Memory::open(&path));
}

// ---------------------------------------------------------------------------
// wal.rs: strict mode, bounded allocation, full chain coverage
// ---------------------------------------------------------------------------

const WAL_MAGIC: [u8; 4] = *b"EHWL";

fn save_record(chunk: &str, embedding: &[f32]) -> Vec<u8> {
    let mut b = vec![0x01u8];
    b.extend_from_slice(&1234.5f64.to_le_bytes());
    for s in [chunk] {
        b.extend_from_slice(&(s.len() as u32).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    }
    b.extend_from_slice(&(embedding.len() as u32).to_le_bytes());
    for v in embedding {
        b.extend_from_slice(&v.to_le_bytes());
    }
    for s in ["chan", "sess", "tags"] {
        b.extend_from_slice(&(s.len() as u32).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    }
    b
}

fn tombstone_record(index: u32) -> Vec<u8> {
    let mut b = vec![0x02u8];
    b.extend_from_slice(&1234.5f64.to_le_bytes());
    b.extend_from_slice(&index.to_le_bytes());
    b
}

fn activation_record() -> Vec<u8> {
    let mut b = vec![0x03u8];
    b.extend_from_slice(&1234.5f64.to_le_bytes());
    b
}

fn header(version: u8, count: u32) -> Vec<u8> {
    let mut b = WAL_MAGIC.to_vec();
    b.push(version);
    b.extend_from_slice(&count.to_le_bytes());
    b
}

/// Build a version-2 WAL with a correct chain over `records`. Records in
/// `unchained` positions are spliced in WITHOUT a trailer and without
/// advancing the chain -- the forgery under test.
fn v2_file(records: &[Vec<u8>], unchained: &[usize]) -> Vec<u8> {
    let mut out = header(2, records.len() as u32);
    let mut tip: [u8; 32] = Sha256::new()
        .chain_update(WAL_MAGIC)
        .chain_update([2u8])
        .finalize()
        .into();
    for (i, r) in records.iter().enumerate() {
        out.extend_from_slice(r);
        if unchained.contains(&i) {
            continue;
        }
        tip = Sha256::new().chain_update(tip).chain_update(r).finalize().into();
        out.extend_from_slice(&tip);
    }
    out
}

fn write_bytes(path: &Path, bytes: &[u8]) {
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(bytes).unwrap();
}

/// Set up a flushed, integrity-protected `.h5` so that `open_strict` gets as
/// far as the sidecar.
fn protected_store(dir: &Path) -> PathBuf {
    let path = dir.join("mem.h5");
    let cfg = config(path.clone(), DIM);
    write_store(&path, &cfg, &sample_cache(2, DIM), &SessionCache::new(), &KnowledgeCache::new());
    path
}

/// Rewriting the header's version byte to 1 (and dropping trailers) removed
/// the chain entirely, and `open_strict` replayed the forged entries anyway,
/// then kept appending unchained records. Strict mode must refuse a v1
/// sidecar; the default open keeps its documented legacy tolerance.
#[test]
fn strict_open_rejects_version1_wal_sidecar() {
    let dir = tempfile::tempdir().unwrap();
    let path = protected_store(dir.path());
    let wal_path = path.with_extension("h5.wal");

    let mut v1 = header(1, 2);
    v1.extend_from_slice(&save_record("FORGED by attacker", &[0.5; DIM]));
    v1.extend_from_slice(&tombstone_record(0));
    write_bytes(&wal_path, &v1);

    match HDF5Memory::open_strict(&path) {
        Err(MemoryError::Integrity(msg)) => eprintln!("strict: {msg}"),
        Err(e) => panic!("expected Integrity error, got {e}"),
        Ok(_) => panic!("open_strict accepted a version-1 WAL sidecar"),
    }
    assert!(matches!(WalFile::read_entries_strict(&wal_path), Err(MemoryError::Integrity(_))));
    assert!(matches!(WalFile::open_strict(&wal_path), Err(MemoryError::Integrity(_))));

    // Legacy, non-strict behavior is unchanged and documented: the v1 file
    // still replays. (This is the disclosed unauthenticated path.)
    let mem = HDF5Memory::open(&path).expect("non-strict open tolerates v1");
    assert_eq!(mem.count(), 3);
}

/// A u32 length prefix of 0xFFFF_FFFF used to drive `vec![0u8; 4 GiB]` /
/// `Vec::<f32>::with_capacity(16 GiB)` before a single payload byte was read.
/// The prefix must be bounded by the bytes actually present, after which the
/// record is simply the truncation case.
#[test]
fn oversized_wal_length_prefix_is_bounded_by_file_size() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("mem.h5.wal");

    // Save record whose chunk length claims 4 GiB.
    let mut bad = header(2, 1);
    bad.push(0x01);
    bad.extend_from_slice(&1.0f64.to_le_bytes());
    bad.extend_from_slice(&u32::MAX.to_le_bytes());
    bad.extend_from_slice(b"short");
    write_bytes(&wal_path, &bad);
    let entries = WalFile::read_entries(&wal_path).expect("treated as truncation, not an abort");
    assert!(entries.is_empty());
    assert!(WalFile::open(&wal_path).is_ok());

    // Save record whose embedding count claims 2^32-1 floats (16 GiB).
    let mut bad = header(2, 1);
    bad.push(0x01);
    bad.extend_from_slice(&1.0f64.to_le_bytes());
    bad.extend_from_slice(&3u32.to_le_bytes());
    bad.extend_from_slice(b"abc");
    bad.extend_from_slice(&u32::MAX.to_le_bytes());
    write_bytes(&wal_path, &bad);
    let entries = WalFile::read_entries(&wal_path).expect("treated as truncation, not an abort");
    assert!(entries.is_empty());
}

/// `ActivationUpdate` (0x03) records skipped the v2 trailer check, so
/// unchained 9-byte blobs could be spliced anywhere in a chained file. They
/// must be chained like every other record.
#[test]
fn unchained_activation_record_in_v2_wal_fails_closed() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("mem.h5.wal");

    let records = vec![
        save_record("one", &[0.1; DIM]),
        activation_record(),
        save_record("two", &[0.2; DIM]),
    ];
    // Spliced without a trailer: must fail the chain.
    write_bytes(&wal_path, &v2_file(&records, &[1]));
    assert!(
        matches!(WalFile::read_entries(&wal_path), Err(MemoryError::Integrity(_))),
        "unchained 0x03 record must be a chain failure"
    );
    // Properly chained: accepted, and ignored by replay.
    write_bytes(&wal_path, &v2_file(&records, &[]));
    let entries = WalFile::read_entries(&wal_path).expect("chained 0x03 is fine");
    assert_eq!(entries.len(), 3);
    let mut cache = MemoryCache::new(DIM);
    clawhdf5_agent::wal::replay_into_cache(&entries, &mut cache);
    assert_eq!(cache.len(), 2);
}

/// A present-but-unrecognized type byte silently ended replay (dropping every
/// later entry) even in a chained file, contradicting the module's own
/// truncation-vs-corruption rule. In v2 it must fail closed.
#[test]
fn unknown_type_byte_in_v2_wal_fails_closed() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("mem.h5.wal");
    let records = vec![save_record("one", &[0.1; DIM]), save_record("two", &[0.2; DIM])];
    let mut bytes = v2_file(&records, &[]);
    // Corrupt the second record's type byte in place.
    let second_start = header(2, 0).len() + records[0].len() + 32;
    assert_eq!(bytes[second_start], 0x01);
    bytes[second_start] = 0xFF;
    write_bytes(&wal_path, &bytes);
    assert!(matches!(WalFile::read_entries(&wal_path), Err(MemoryError::Integrity(_))));
}

// ---------------------------------------------------------------------------
// Strict constructors on every public open path (follow-up round)
// ---------------------------------------------------------------------------

/// The OpenClaw backend (the integration point for other agents) and the
/// async wrapper had no strict constructor, so their users could not get the
/// stripped-root / WAL-downgrade protection. An empty store is written with
/// no content root, which stands in for an unprotected or stripped file.
#[test]
fn openclaw_backend_open_strict_rejects_an_unprotected_store() {
    use clawhdf5_agent::openclaw::ClawhdfBackend;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("oc.h5");
    drop(ClawhdfBackend::create(&path, DIM).unwrap());
    assert!(ClawhdfBackend::open(&path).is_ok(), "default open stays fail-open");
    let err = match ClawhdfBackend::open_strict(&path) {
        Err(e) => e,
        Ok(_) => panic!("open_strict accepted a store with no content root"),
    };
    assert!(err.contains("integrity"), "unexpected error: {err}");
}

#[cfg(feature = "async")]
#[test]
fn async_open_strict_rejects_an_unprotected_store() {
    use clawhdf5_agent::async_memory::AsyncHDF5Memory;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("async.h5");
    drop(HDF5Memory::create(config(path.clone(), DIM)).unwrap());
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
    rt.block_on(async {
        assert!(AsyncHDF5Memory::open(&path).await.is_ok());
        assert!(matches!(
            AsyncHDF5Memory::open_strict(&path).await,
            Err(MemoryError::Integrity(_))
        ));
    });
}
