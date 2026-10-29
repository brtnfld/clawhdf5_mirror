//! Regression test for the 2026-10 red-team finding that a writer reopened
//! after a clean commit reused a nonce. `EncryptedChunkWriter::recover`
//! seeded the version store only from *pending* WAL records, and
//! `VersionWal::compact` deleted committed ones, so following the documented
//! `new` + `recover` protocol restarted every committed chunk's counter at 1.
//! The next write to that chunk re-derived the same (DEK, chunk, version)
//! nonce: a two-time pad, C1 ^ C2 == P1 ^ P2.
#![cfg(feature = "chacha20")]

use clawhdf5_filters::{EncryptedChunkWriter, VersionCounterStore, VersionWal, decrypt_chunk};
use std::io::Cursor;

const DEK: [u8; 32] = [7u8; 32];
const P1: &[u8; 32] = b"attack at dawn, bring the maps!!";
const P2: &[u8; 32] = b"retreat at dusk, burn the papers";

fn first_session() -> (Vec<u8>, Vec<u8>) {
    let mut buf = Cursor::new(Vec::new());
    let wal = VersionWal::new(&mut buf, 100).unwrap();
    let mut w = EncryptedChunkWriter::new(DEK, wal, VersionCounterStore::new());
    let r = w.encrypt_chunk(0, P1).unwrap();
    assert_eq!(r.version, 1);
    w.commit(0, r.version).unwrap();
    (buf.into_inner(), r.ciphertext)
}

fn second_session(wal_bytes: Vec<u8>, compact_first: bool) -> u64 {
    let mut buf = Cursor::new(wal_bytes);
    if compact_first {
        VersionWal::new(&mut buf, 100).unwrap().compact().unwrap();
        buf.set_position(0);
    }
    let wal = VersionWal::new(&mut buf, 100).unwrap();
    let mut w = EncryptedChunkWriter::new(DEK, wal, VersionCounterStore::new());
    let pending = w.recover().unwrap();
    assert!(pending.is_empty(), "nothing was left pending");
    w.encrypt_chunk(0, P2).unwrap().version
}

#[test]
fn reopened_writer_never_reuses_a_committed_version() {
    let (wal_bytes, c1) = first_session();
    for compact_first in [false, true] {
        let v2 = second_session(wal_bytes.clone(), compact_first);
        assert_eq!(v2, 2, "compact_first={compact_first}: committed version 1 must stay burned");
    }
    // The first session's ciphertext still decrypts under its own version.
    assert_eq!(decrypt_chunk(&DEK, 0, 1, &c1).unwrap(), P1);
}

#[test]
fn compaction_keeps_one_high_water_record_per_chunk() {
    let mut buf = Cursor::new(Vec::new());
    let mut wal = VersionWal::new(&mut buf, 100).unwrap();
    let ph = [0u8; 32];
    // Chunk 3 is committed three times; chunk 4 once; chunk 5 left pending.
    for v in 1..=3 {
        wal.journal_version(3, v, ph).unwrap();
        wal.commit_version(3).unwrap();
    }
    wal.journal_version(4, 1, ph).unwrap();
    wal.commit_version(4).unwrap();
    wal.journal_version(5, 9, ph).unwrap();
    wal.compact().unwrap();

    let marks = wal.high_water_marks().unwrap();
    assert_eq!(marks.get(&3), Some(&3));
    assert_eq!(marks.get(&4), Some(&1));
    assert_eq!(marks.get(&5), Some(&9));
    // Pending-only recovery is unchanged.
    assert_eq!(wal.recover().unwrap(), vec![(5, 9, ph)]);

    // The high-water marks survive a reopen of the compacted bytes.
    drop(wal);
    buf.set_position(0);
    let mut reopened = VersionWal::new(&mut buf, 100).unwrap();
    assert_eq!(reopened.high_water_marks().unwrap().get(&3), Some(&3));
}
