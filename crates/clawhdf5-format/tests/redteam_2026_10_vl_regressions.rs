//! Regression tests for the 2026-10 red-team finding that variable-length
//! reads re-parsed the whole global heap collection, and then linearly
//! scanned it, for every element. N references into one S-byte collection
//! cost N x S; both are attacker-sized, so a large file meant hours of CPU
//! on read. Measured before the fix: ~40 us per reference into a 1 MiB
//! collection in release (4,096 references took 166 ms; after the fix,
//! 0.5 ms).

use clawhdf5_format::error::FormatError;
use clawhdf5_format::vl_data::{read_vl_bytes, read_vl_strings};

/// A GCOL at `base` holding objects `1..=nobj`, object `i` being `objsz`
/// copies of the byte `tag + i % 26`.
fn collection_at(file: &mut Vec<u8>, base: usize, nobj: usize, objsz: usize, tag: u8) {
    let mut objs = Vec::new();
    for i in 0..nobj {
        objs.extend_from_slice(&((i + 1) as u16).to_le_bytes());
        objs.extend_from_slice(&1u16.to_le_bytes());
        objs.extend_from_slice(&[0u8; 4]);
        objs.extend_from_slice(&(objsz as u64).to_le_bytes());
        objs.extend(std::iter::repeat_n(tag + (i % 26) as u8, objsz));
        objs.extend(std::iter::repeat_n(0u8, (8 - objsz % 8) % 8));
    }
    let size = 16 + objs.len() + 16;
    if file.len() < base + size {
        file.resize(base + size, 0);
    }
    let mut c = Vec::new();
    c.extend_from_slice(b"GCOL");
    c.extend_from_slice(&[1, 0, 0, 0]);
    c.extend_from_slice(&(size as u64).to_le_bytes());
    c.extend_from_slice(&objs);
    c.extend_from_slice(&[0u8; 16]);
    file[base..base + c.len()].copy_from_slice(&c);
}

fn vl_ref(len: u32, addr: u64, index: u32) -> [u8; 16] {
    let mut r = [0u8; 16];
    r[0..4].copy_from_slice(&len.to_le_bytes());
    r[4..12].copy_from_slice(&addr.to_le_bytes());
    r[12..16].copy_from_slice(&index.to_le_bytes());
    r
}

#[test]
fn many_references_into_one_large_collection_are_linear_time() {
    let mut file = Vec::new();
    collection_at(&mut file, 0, 1000, 1024, b'a'); // ~1 MiB
    let n = 65_536usize;
    let mut raw = Vec::with_capacity(n * 16);
    for i in 0..n {
        raw.extend_from_slice(&vl_ref(4, 0, (i % 1000 + 1) as u32));
    }
    let t = std::time::Instant::now();
    let out = read_vl_strings(&file, &raw, n as u64, 8, 8).unwrap();
    let elapsed = t.elapsed();
    assert_eq!(out.len(), n);
    assert_eq!(out[0], "aaaa");
    assert_eq!(out[1], "bbbb");
    // The old per-element re-parse needed about 65,536 x 40 us = 2.6 s in a
    // release build, and several times that in debug.
    assert!(elapsed < std::time::Duration::from_secs(2), "took {elapsed:?}");
}

#[test]
fn references_across_collections_resolve_correctly() {
    let mut file = Vec::new();
    collection_at(&mut file, 0, 3, 8, b'a');
    collection_at(&mut file, 4096, 3, 8, b'n');
    let mut raw = Vec::new();
    for (addr, idx) in [(0u64, 1u32), (4096, 1), (0, 3), (4096, 2), (0, 1)] {
        raw.extend_from_slice(&vl_ref(8, addr, idx));
    }
    let strings = read_vl_strings(&file, &raw, 5, 8, 8).unwrap();
    assert_eq!(strings, ["aaaaaaaa", "nnnnnnnn", "cccccccc", "oooooooo", "aaaaaaaa"]);
    let bytes = read_vl_bytes(&file, &raw, 5, 8, 8).unwrap();
    assert_eq!(bytes[1], b"nnnnnnnn");
}

#[test]
fn missing_object_index_is_still_reported() {
    let mut file = Vec::new();
    collection_at(&mut file, 0, 3, 8, b'a');
    let raw = vl_ref(8, 0, 9);
    let err = read_vl_strings(&file, &raw, 1, 8, 8).unwrap_err();
    assert!(
        matches!(err, FormatError::GlobalHeapObjectNotFound { collection_address: 0, index: 9 }),
        "got {err:?}"
    );
}
