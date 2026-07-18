//! End-to-end integrity check (P2.4 Finding 1): a memory `.h5` written through
//! the real production path (`HDF5Memory::create` → `flush` → `write_to_disk`)
//! must carry a `_merkle_root`, verify cleanly on reopen, and fail closed when
//! its on-disk bytes are tampered with — the guarantee the standalone attack
//! harness could not previously demonstrate on the production path.

use std::path::PathBuf;

use clawhdf5_agent::{AgentMemory, HDF5Memory, MemoryConfig, MemoryEntry, MemoryError};

fn unique_path(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("clawhdf5_integrity_{tag}_{nanos}.h5"));
    p
}

fn seed(path: &PathBuf) {
    let config = MemoryConfig {
        path: path.clone(),
        wal_enabled: false,
        ..MemoryConfig::new(path.clone(), "test-agent", 3)
    };
    let mut mem = HDF5Memory::create(config).expect("create");
    for (i, text) in ["alpha secret memory", "beta secret memory", "gamma secret memory"]
        .iter()
        .enumerate()
    {
        mem.save(MemoryEntry {
            chunk: (*text).to_string(),
            embedding: vec![i as f32, i as f32 + 0.5, i as f32 - 0.5],
            source_channel: "test".into(),
            timestamp: 1000.0 + i as f64,
            session_id: "s1".into(),
            tags: format!("tag{i}"),
        })
        .expect("save");
    }
    // With WAL disabled, each save flushes to disk via write_to_disk, so the
    // file on disk already reflects all three entries.
    drop(mem);
}

#[test]
fn honest_file_reopens_cleanly() {
    let path = unique_path("honest");
    seed(&path);
    let reopened = HDF5Memory::open(&path).expect("honest reopen must succeed");
    assert_eq!(reopened.count(), 3);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn written_file_carries_merkle_root() {
    let path = unique_path("hasroot");
    seed(&path);
    let bytes = std::fs::read(&path).expect("read raw");
    // The hex-encoded packed MerkleAttr attribute name appears verbatim in the
    // serialized attribute-message region.
    let needle = b"_merkle_root";
    assert!(
        bytes.windows(needle.len()).any(|w| w == needle),
        "written file must contain a _merkle_root attribute"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn tampered_content_fails_closed_on_reopen() {
    let path = unique_path("tamper");
    seed(&path);

    // Raw storage-level tamper: locate a chunk string in the on-disk bytes and
    // flip a byte, bypassing every write API — exactly the harness's T1a move,
    // but against a real file produced by the production path.
    let mut bytes = std::fs::read(&path).expect("read raw");
    let needle = b"beta secret memory";
    let pos = bytes
        .windows(needle.len())
        .position(|w| w == needle)
        .expect("chunk text must be present on disk");
    bytes[pos] ^= 0xFF;
    std::fs::write(&path, &bytes).expect("write tampered");

    match HDF5Memory::open(&path) {
        Err(MemoryError::Integrity(msg)) => {
            eprintln!("integrity check correctly rejected tampered file: {msg}");
        }
        Ok(_) => panic!("tampered file must NOT open cleanly — integrity check failed to fire"),
        Err(other) => panic!("expected MemoryError::Integrity, got {other:?}"),
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn strict_open_rejects_stripped_root() {
    let path = unique_path("stripped");
    seed(&path);

    // Strip the `_merkle_root` attribute value in place: overwrite the hex
    // string bytes with '0's (same length, so the file stays structurally
    // valid) — an attacker downgrading to force an unverified load. The zeroed
    // hex still decodes but the packed attribute's own integrity self-check
    // fails, and a wholesale-absent attribute is caught by strict mode.
    let mut bytes = std::fs::read(&path).expect("read raw");
    // Find the stored hex value: it follows the attribute name. Rather than
    // parse HDF5, just corrupt the first long hex run after the attr name.
    let name = b"_merkle_root";
    let name_pos = bytes
        .windows(name.len())
        .position(|w| w == name)
        .expect("_merkle_root present");
    // Zero a span well past the name to hit the value payload.
    for b in bytes.iter_mut().skip(name_pos + name.len()).take(64) {
        if b.is_ascii_hexdigit() {
            *b = b'0';
        }
    }
    std::fs::write(&path, &bytes).expect("write");

    // Strict open must reject; the corrupted/zeroed root can't verify.
    assert!(
        matches!(HDF5Memory::open_strict(&path), Err(MemoryError::Integrity(_))),
        "strict open must reject a downgraded/stripped root"
    );
    let _ = std::fs::remove_file(&path);
}
