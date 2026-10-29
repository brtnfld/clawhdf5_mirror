//! Vector search using cosine similarity.
//!
//! Provides SIMD-accelerated cosine similarity for f32 embeddings via
//! `clawhdf5_accel`, with optional float16 support via the `half` crate.
//! Supports pre-computed norms for eliminating redundant norm computations.

/// Compute cosine similarity between two f32 slices.
///
/// Returns 0.0 if either vector has zero magnitude.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have equal length");
    clawhdf5_accel::cosine_similarity(a, b)
}

// Dimension mismatches are skipped, not scored. The SIMD kernels in
// clawhdf5-accel assert equal lengths (that assertion is what keeps their
// unsafe loads in bounds), so a query whose dimension differed from a stored
// vector used to panic the whole search; through the Android C ABI or the
// Node binding a panic aborts the host process (2026-10 red-team finding).
// A mismatched vector cannot be compared meaningfully, so it simply
// contributes no vector score; BM25 still ranks it.

/// Compute cosine similarity between `query` and each vector in `vectors`,
/// skipping tombstoned entries (tombstone != 0).
///
/// Returns `(index, score)` pairs sorted by score descending.
pub fn cosine_similarity_batch(
    query: &[f32],
    vectors: &[Vec<f32>],
    tombstones: &[u8],
) -> Vec<(usize, f32)> {
    let query_norm = clawhdf5_accel::vector_norm(query);
    if query_norm == 0.0 {
        return Vec::new();
    }

    let n = vectors.len();
    let mut results: Vec<(usize, f32)> = Vec::with_capacity(n);

    // Process 4 vectors at a time where possible
    let chunks = n / 4;
    for chunk in 0..chunks {
        let base = chunk * 4;
        for j in 0..4 {
            let i = base + j;
            if i < tombstones.len() && tombstones[i] != 0 {
                continue;
            }
            if vectors[i].len() != query.len() {
                continue; // dimension mismatch: see module note
            }
            let vec_norm = clawhdf5_accel::vector_norm(&vectors[i]);
            let score = crate::cosine_similarity_prenorm(query, query_norm, &vectors[i], vec_norm);
            results.push((i, score));
        }
    }

    // Remainder
    for i in (chunks * 4)..n {
        if i < tombstones.len() && tombstones[i] != 0 {
            continue;
        }
        if vectors[i].len() != query.len() {
            continue; // dimension mismatch: see module note
        }
        let vec_norm = clawhdf5_accel::vector_norm(&vectors[i]);
        let score = crate::cosine_similarity_prenorm(query, query_norm, &vectors[i], vec_norm);
        results.push((i, score));
    }

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Compute cosine similarity batch with pre-computed norms.
///
/// Eliminates N norm computations per search — the main bottleneck for large
/// collections. Uses `score = dot(query, vec) / (query_norm * stored_norm)`.
pub fn cosine_similarity_batch_prenorm(
    query: &[f32],
    vectors: &[Vec<f32>],
    norms: &[f32],
    tombstones: &[u8],
) -> Vec<(usize, f32)> {
    let query_norm = clawhdf5_accel::vector_norm(query);
    if query_norm == 0.0 {
        return Vec::new();
    }

    let n = vectors.len();
    let mut results: Vec<(usize, f32)> = Vec::with_capacity(n);

    for i in 0..n {
        if i < tombstones.len() && tombstones[i] != 0 {
            continue;
        }
        if vectors[i].len() != query.len() {
            continue; // dimension mismatch: see module note
        }
        let vec_norm = norms[i];
        let score = crate::cosine_similarity_prenorm(query, query_norm, &vectors[i], vec_norm);
        results.push((i, score));
    }

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Return the top `k` entries from a pre-sorted score list.
pub fn top_k(scores: Vec<(usize, f32)>, k: usize) -> Vec<(usize, f32)> {
    scores.into_iter().take(k).collect()
}

/// Compute cosine similarity between an f32 query and float16-encoded vectors.
///
/// `vectors_f16` is a flat buffer of u16 values (IEEE 754 half-precision),
/// laid out as `num_vectors * dim` elements. Each consecutive `dim` values
/// form one vector.
///
/// Tombstoned entries (tombstone != 0) are skipped.
/// Returns `(index, score)` pairs sorted by score descending.
#[cfg(feature = "float16")]
pub fn cosine_similarity_f16(
    query: &[f32],
    vectors_f16: &[u16],
    dim: usize,
    tombstones: &[u8],
) -> Vec<(usize, f32)> {
    use half::f16;

    assert!(dim > 0, "dimension must be positive");
    assert_eq!(query.len(), dim, "query length must match dimension");
    assert_eq!(
        vectors_f16.len() % dim,
        0,
        "vectors_f16 length must be a multiple of dim"
    );

    let num_vectors = vectors_f16.len() / dim;
    let mut results = Vec::with_capacity(num_vectors);

    for i in 0..num_vectors {
        if i >= tombstones.len() || tombstones[i] != 0 {
            continue;
        }

        let offset = i * dim;
        let slice = &vectors_f16[offset..offset + dim];

        let mut dot = 0.0f32;
        let mut mag_a = 0.0f32;
        let mut mag_b = 0.0f32;

        for (j, &raw) in slice.iter().enumerate() {
            let bj = f16::from_bits(raw).to_f32();
            let aj = query[j];
            dot += aj * bj;
            mag_a += aj * aj;
            mag_b += bj * bj;
        }

        let denom = mag_a.sqrt() * mag_b.sqrt();
        let score = if denom == 0.0 { 0.0 } else { dot / denom };
        results.push((i, score));
    }

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Parallel cosine similarity batch using rayon.
///
/// Splits vectors into chunks across threads, each running SIMD cosine,
/// then merges top-k results. Falls back to sequential if rayon is not available.
#[cfg(feature = "parallel")]
pub fn parallel_cosine_batch(
    query: &[f32],
    vectors: &[Vec<f32>],
    tombstones: &[u8],
    k: usize,
) -> Vec<(usize, f32)> {
    use rayon::prelude::*;

    let query_norm = clawhdf5_accel::vector_norm(query);
    if query_norm == 0.0 {
        return Vec::new();
    }

    let num_cores = rayon::current_num_threads().max(1);
    let chunk_size = vectors.len().div_ceil(num_cores);
    if chunk_size == 0 {
        return Vec::new();
    }

    let mut all_results: Vec<(usize, f32)> = vectors
        .par_chunks(chunk_size)
        .enumerate()
        .flat_map(|(chunk_idx, chunk)| {
            let base = chunk_idx * chunk_size;
            let mut local: Vec<(usize, f32)> = Vec::with_capacity(chunk.len());
            for (j, vec) in chunk.iter().enumerate() {
                let i = base + j;
                if i < tombstones.len() && tombstones[i] != 0 {
                    continue;
                }
                if vec.len() != query.len() {
                    continue; // dimension mismatch: see module note
                }
                let vec_norm = clawhdf5_accel::vector_norm(vec);
                let score = crate::cosine_similarity_prenorm(query, query_norm, vec, vec_norm);
                local.push((i, score));
            }
            local.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            local.truncate(k);
            local
        })
        .collect();

    all_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    all_results.truncate(k);
    all_results
}

/// Parallel cosine similarity batch with pre-computed norms.
#[cfg(feature = "parallel")]
pub fn parallel_cosine_batch_prenorm(
    query: &[f32],
    vectors: &[Vec<f32>],
    norms: &[f32],
    tombstones: &[u8],
    k: usize,
) -> Vec<(usize, f32)> {
    use rayon::prelude::*;

    let query_norm = clawhdf5_accel::vector_norm(query);
    if query_norm == 0.0 {
        return Vec::new();
    }

    let num_cores = rayon::current_num_threads().max(1);
    let chunk_size = vectors.len().div_ceil(num_cores);
    if chunk_size == 0 {
        return Vec::new();
    }

    let mut all_results: Vec<(usize, f32)> = vectors
        .par_chunks(chunk_size)
        .enumerate()
        .flat_map(|(chunk_idx, chunk)| {
            let base = chunk_idx * chunk_size;
            let mut local: Vec<(usize, f32)> = Vec::with_capacity(chunk.len());
            for (j, vec) in chunk.iter().enumerate() {
                let i = base + j;
                if i < tombstones.len() && tombstones[i] != 0 {
                    continue;
                }
                if vec.len() != query.len() {
                    continue; // dimension mismatch: see module note
                }
                let score = crate::cosine_similarity_prenorm(query, query_norm, vec, norms[i]);
                local.push((i, score));
            }
            local.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            local.truncate(k);
            local
        })
        .collect();

    all_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    all_results.truncate(k);
    all_results
}

/// Compute batch cosine similarity using BLAS matrix-vector multiply.
///
/// When the `fast-math` feature is enabled, this delegates to
/// `blas_search::blas_cosine_batch` which uses cache-oblivious sgemm
/// for significantly faster batch dot products. Falls back to
/// `cosine_similarity_batch_prenorm` when the feature is disabled.
pub fn cosine_similarity_batch_blas(
    query: &[f32],
    vectors: &[Vec<f32>],
    norms: &[f32],
    tombstones: &[u8],
    k: usize,
) -> Vec<(usize, f32)> {
    #[cfg(feature = "fast-math")]
    {
        crate::blas_search::blas_cosine_batch(query, vectors, norms, tombstones, k)
    }
    #[cfg(not(feature = "fast-math"))]
    {
        let all = cosine_similarity_batch_prenorm(query, vectors, norms, tombstones);
        top_k(all, k)
    }
}

/// Compute the norm of a vector (for pre-computation).
pub fn compute_norm(v: &[f32]) -> f32 {
    clawhdf5_accel::vector_norm(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_similarity_is_one() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6, "expected ~1.0, got {sim}");
    }

    #[test]
    fn orthogonal_vectors_similarity_is_zero() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "expected ~0.0, got {sim}");
    }

    #[test]
    fn negative_correlation() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6, "expected ~-1.0, got {sim}");
    }

    #[test]
    fn zero_vector_returns_zero() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    #[should_panic(expected = "vectors must have equal length")]
    fn different_lengths_panics() {
        cosine_similarity(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    fn batch_cosine_with_tombstones() {
        let query = vec![1.0, 0.0, 0.0];
        let vectors = vec![
            vec![1.0, 0.0, 0.0], // idx 0: identical
            vec![0.0, 1.0, 0.0], // idx 1: orthogonal, tombstoned
            vec![0.5, 0.5, 0.0], // idx 2: partial match
        ];
        let tombstones = vec![0, 1, 0]; // idx 1 is tombstoned

        let results = cosine_similarity_batch(&query, &vectors, &tombstones);

        // Should only have idx 0 and idx 2
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // highest score
        assert_eq!(results[1].0, 2);
        // idx 1 must not appear
        assert!(results.iter().all(|(idx, _)| *idx != 1));
    }

    #[test]
    fn batch_all_tombstoned_returns_empty() {
        let query = vec![1.0, 0.0];
        let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let tombstones = vec![1, 1];
        let results = cosine_similarity_batch(&query, &vectors, &tombstones);
        assert!(results.is_empty());
    }

    #[test]
    fn top_k_selection() {
        let scores = vec![(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6), (4, 0.5)];
        let top = top_k(scores, 3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, 0);
        assert_eq!(top[2].0, 2);
    }

    #[test]
    fn top_k_larger_than_input() {
        let scores = vec![(0, 0.9), (1, 0.8)];
        let top = top_k(scores, 10);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn top_k_zero() {
        let scores = vec![(0, 0.9)];
        let top = top_k(scores, 0);
        assert!(top.is_empty());
    }

    #[cfg(feature = "float16")]
    #[test]
    fn f16_cosine_matches_f32_within_tolerance() {
        use half::f16;

        let query = vec![1.0, 2.0, 3.0, 4.0];
        let f32_vectors = [vec![4.0, 3.0, 2.0, 1.0]];
        let tombstones = vec![0u8];

        // Encode as f16
        let vectors_f16: Vec<u16> = f32_vectors[0]
            .iter()
            .map(|&v| f16::from_f32(v).to_bits())
            .collect();

        let f32_sim = cosine_similarity(&query, &f32_vectors[0]);
        let f16_results = cosine_similarity_f16(&query, &vectors_f16, 4, &tombstones);

        assert_eq!(f16_results.len(), 1);
        let f16_sim = f16_results[0].1;
        assert!(
            (f32_sim - f16_sim).abs() < 0.01,
            "f32={f32_sim}, f16={f16_sim}"
        );
    }

    #[cfg(feature = "float16")]
    #[test]
    fn f16_cosine_skips_tombstoned() {
        use half::f16;

        let query = vec![1.0, 0.0];
        let vectors_f16: Vec<u16> = [1.0f32, 0.0, 0.0, 1.0]
            .iter()
            .map(|&v| f16::from_f32(v).to_bits())
            .collect();
        let tombstones = vec![1, 0]; // first vector tombstoned

        let results = cosine_similarity_f16(&query, &vectors_f16, 2, &tombstones);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1); // only second vector
    }

    #[test]
    fn batch_cosine_ordering() {
        let query = vec![1.0, 0.0, 0.0];
        let vectors = vec![
            vec![0.0, 1.0, 0.0], // orthogonal = 0
            vec![0.7, 0.7, 0.0], // partial
            vec![1.0, 0.0, 0.0], // identical = 1.0
        ];
        let tombstones = vec![0, 0, 0];
        let results = cosine_similarity_batch(&query, &vectors, &tombstones);

        assert_eq!(results[0].0, 2); // highest
        assert_eq!(results[1].0, 1); // middle
        assert_eq!(results[2].0, 0); // lowest
    }

    #[test]
    fn prenorm_batch_matches_regular_batch() {
        let dim = 384;
        let n = 100;
        let mut seed: u32 = 42;
        let mut next_f32 = || -> f32 {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            ((seed >> 16) as f32) / 65536.0 - 0.5
        };

        let query: Vec<f32> = (0..dim).map(|_| next_f32()).collect();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| next_f32()).collect())
            .collect();
        let norms: Vec<f32> = vectors.iter().map(|v| compute_norm(v)).collect();
        let tombstones = vec![0u8; n];

        let regular = cosine_similarity_batch(&query, &vectors, &tombstones);
        let prenorm = cosine_similarity_batch_prenorm(&query, &vectors, &norms, &tombstones);

        assert_eq!(regular.len(), prenorm.len());
        for (r, p) in regular.iter().zip(&prenorm) {
            assert_eq!(r.0, p.0, "index mismatch");
            assert!(
                (r.1 - p.1).abs() < 1e-5,
                "score mismatch at idx {}: {} vs {}",
                r.0,
                r.1,
                p.1
            );
        }
    }

    #[test]
    fn prenorm_search_same_ranking() {
        let query = vec![1.0, 0.5, 0.0];
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.8, 0.6, 0.0],
        ];
        let norms: Vec<f32> = vectors.iter().map(|v| compute_norm(v)).collect();
        let tombstones = vec![0, 0, 0];

        let regular = cosine_similarity_batch(&query, &vectors, &tombstones);
        let prenorm = cosine_similarity_batch_prenorm(&query, &vectors, &norms, &tombstones);

        let regular_ids: Vec<usize> = regular.iter().map(|r| r.0).collect();
        let prenorm_ids: Vec<usize> = prenorm.iter().map(|r| r.0).collect();
        assert_eq!(regular_ids, prenorm_ids, "ranking should be identical");
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_search_matches_sequential() {
        let dim = 128;
        let n = 500;
        let mut seed: u32 = 42;
        let mut next_f32 = || -> f32 {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            ((seed >> 16) as f32) / 65536.0 - 0.5
        };

        let query: Vec<f32> = (0..dim).map(|_| next_f32()).collect();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| next_f32()).collect())
            .collect();
        let tombstones = vec![0u8; n];

        let sequential = cosine_similarity_batch(&query, &vectors, &tombstones);
        let seq_top10 = top_k(sequential, 10);
        let parallel = parallel_cosine_batch(&query, &vectors, &tombstones, 10);

        assert_eq!(seq_top10.len(), parallel.len());
        for (s, p) in seq_top10.iter().zip(&parallel) {
            assert_eq!(s.0, p.0, "index mismatch");
            assert!((s.1 - p.1).abs() < 1e-5, "score mismatch");
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_prenorm_matches_sequential() {
        let dim = 64;
        let n = 300;
        let mut seed: u32 = 77;
        let mut next_f32 = || -> f32 {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            ((seed >> 16) as f32) / 65536.0 - 0.5
        };

        let query: Vec<f32> = (0..dim).map(|_| next_f32()).collect();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| next_f32()).collect())
            .collect();
        let norms: Vec<f32> = vectors.iter().map(|v| compute_norm(v)).collect();
        let tombstones = vec![0u8; n];

        let sequential = cosine_similarity_batch_prenorm(&query, &vectors, &norms, &tombstones);
        let seq_top10 = top_k(sequential, 10);
        let parallel = parallel_cosine_batch_prenorm(&query, &vectors, &norms, &tombstones, 10);

        assert_eq!(seq_top10.len(), parallel.len());
        for (s, p) in seq_top10.iter().zip(&parallel) {
            assert_eq!(s.0, p.0, "index mismatch");
            assert!((s.1 - p.1).abs() < 1e-5, "score mismatch");
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_search_with_tombstones() {
        let dim = 32;
        let n = 100;
        let mut seed: u32 = 42;
        let mut next_f32 = || -> f32 {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            ((seed >> 16) as f32) / 65536.0 - 0.5
        };

        let query: Vec<f32> = (0..dim).map(|_| next_f32()).collect();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| next_f32()).collect())
            .collect();
        let mut tombstones = vec![0u8; n];
        // Tombstone every other vector
        for i in (0..n).step_by(2) {
            tombstones[i] = 1;
        }

        let results = parallel_cosine_batch(&query, &vectors, &tombstones, 10);
        assert!(
            results.iter().all(|r| r.0 % 2 != 0),
            "should skip tombstoned"
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_search_empty_vectors() {
        let query = vec![1.0, 0.0, 0.0];
        let vectors: Vec<Vec<f32>> = Vec::new();
        let tombstones: Vec<u8> = Vec::new();

        let results = parallel_cosine_batch(&query, &vectors, &tombstones, 10);
        assert!(results.is_empty());
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_search_zero_query() {
        let query = vec![0.0, 0.0, 0.0];
        let vectors = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let tombstones = vec![0u8; 2];

        let results = parallel_cosine_batch(&query, &vectors, &tombstones, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn performance_10k_vectors_384d() {
        let dim = 384;
        let n = 10_000;

        // Generate deterministic pseudo-random vectors
        let mut seed: u32 = 42;
        let mut next_f32 = || -> f32 {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            ((seed >> 16) as f32) / 65536.0 - 0.5
        };

        let query: Vec<f32> = (0..dim).map(|_| next_f32()).collect();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| next_f32()).collect())
            .collect();
        let tombstones = vec![0u8; n];

        let start = std::time::Instant::now();
        let results = cosine_similarity_batch(&query, &vectors, &tombstones);
        let elapsed = start.elapsed();

        assert_eq!(results.len(), n);
        assert!(
            elapsed.as_millis() < 500,
            "10K x 384 cosine search took {}ms, expected <500ms",
            elapsed.as_millis()
        );
    }
}
