//! Integration tests for the optional HNSW-accelerated vector search path.
//!
//! These only run when the crate is built with `--features hnsw`. They drive the
//! real `HDF5Memory` API (save / save_batch / delete / hybrid_search) and check
//! the approximate results against a brute-force cosine oracle, plus confirm that
//! deletions are honoured end-to-end.
#![cfg(feature = "hnsw")]

use clawhdf5_agent::{AgentMemory, HDF5Memory, MemoryConfig, MemoryEntry};
use tempfile::TempDir;

/// Deterministic splitmix64 so tests are reproducible without an RNG crate.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn make_vector(seed: &mut u64, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|_| (splitmix64(seed) >> 40) as f32 / 16_777_216.0 - 0.5)
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn entry(chunk: &str, embedding: Vec<f32>, tags: &str) -> MemoryEntry {
    MemoryEntry {
        chunk: chunk.to_string(),
        embedding,
        source_channel: "test".to_string(),
        timestamp: 0.0,
        session_id: "s".to_string(),
        tags: tags.to_string(),
    }
}

fn new_memory(dir: &TempDir, dim: usize) -> HDF5Memory {
    let config = MemoryConfig::new(dir.path().join("mem.h5"), "agent", dim);
    HDF5Memory::create(config).unwrap()
}

#[test]
fn hnsw_matches_bruteforce_oracle() {
    let dir = TempDir::new().unwrap();
    let dim = 16;
    let n = 250;
    let mut mem = new_memory(&dir, dim);

    let mut seed = 0xC0FF_EE12_3456_789A;
    let vectors: Vec<Vec<f32>> = (0..n).map(|_| make_vector(&mut seed, dim)).collect();
    for (i, v) in vectors.iter().enumerate() {
        mem.save(entry(&format!("chunk {i}"), v.clone(), &format!("k{i}")))
            .unwrap();
    }

    // Vector-only query: keyword weight 0 isolates the HNSW vector stage.
    let query = make_vector(&mut seed, dim);
    let k = 10;
    let results = mem.hybrid_search(&query, "", 1.0, 0.0, k);
    assert_eq!(results.len(), k, "should return k results");

    // Brute-force cosine top-k oracle.
    let mut oracle: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i, cosine(&query, v)))
        .collect();
    oracle.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let oracle_ids: std::collections::HashSet<usize> =
        oracle.iter().take(k).map(|(i, _)| *i).collect();
    let hnsw_ids: std::collections::HashSet<usize> =
        results.iter().map(|r| r.index).collect();

    let overlap = oracle_ids.intersection(&hnsw_ids).count();
    assert!(
        overlap >= 7,
        "HNSW recall too low vs brute force: {overlap}/{k} (hnsw={hnsw_ids:?})"
    );
}

#[test]
fn deleted_entry_excluded_from_search() {
    let dir = TempDir::new().unwrap();
    let dim = 8;
    let mut mem = new_memory(&dir, dim);

    let mut seed = 42;
    let vectors: Vec<Vec<f32>> = (0..60).map(|_| make_vector(&mut seed, dim)).collect();
    for (i, v) in vectors.iter().enumerate() {
        mem.save(entry(&format!("c{i}"), v.clone(), &format!("t{i}")))
            .unwrap();
    }

    // Query exactly equal to vector 5 — it must be the top hit.
    let query = vectors[5].clone();
    let top = mem.hybrid_search(&query, "", 1.0, 0.0, 1);
    assert_eq!(top[0].index, 5, "exact match should rank first");

    mem.delete(5).unwrap();

    let after = mem.hybrid_search(&query, "", 1.0, 0.0, 5);
    assert!(
        after.iter().all(|r| r.index != 5),
        "deleted entry must not appear in results"
    );
}

#[test]
fn incremental_inserts_after_search_are_found() {
    let dir = TempDir::new().unwrap();
    let dim = 8;
    let mut mem = new_memory(&dir, dim);

    let mut seed = 7;
    // First batch, then a search to force the index to build.
    for i in 0..40 {
        let v = make_vector(&mut seed, dim);
        mem.save(entry(&format!("a{i}"), v, &format!("a{i}"))).unwrap();
    }
    let _ = mem.hybrid_search(&make_vector(&mut seed, dim), "", 1.0, 0.0, 5);

    // Now insert a distinctive vector incrementally and confirm we can find it.
    let needle = vec![10.0f32; dim];
    let idx = mem
        .save(entry("needle", needle.clone(), "needle"))
        .unwrap();
    let hits = mem.hybrid_search(&needle, "", 1.0, 0.0, 1);
    assert_eq!(hits[0].index, idx, "incrementally inserted vector must be found");
}

#[test]
fn save_batch_then_search_is_consistent() {
    let dir = TempDir::new().unwrap();
    let dim = 8;
    let mut mem = new_memory(&dir, dim);

    let mut seed = 99;
    let vectors: Vec<Vec<f32>> = (0..50).map(|_| make_vector(&mut seed, dim)).collect();
    let entries: Vec<MemoryEntry> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| entry(&format!("b{i}"), v.clone(), &format!("b{i}")))
        .collect();
    mem.save_batch(entries).unwrap();

    // Exact-match queries should resolve to themselves after a batch insert.
    for probe in [0usize, 17, 49] {
        let hits = mem.hybrid_search(&vectors[probe], "", 1.0, 0.0, 1);
        assert_eq!(hits[0].index, probe, "batch-inserted vector {probe} not found");
    }
}
