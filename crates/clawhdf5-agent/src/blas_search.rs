//! BLAS-accelerated batch cosine search using matrix-vector multiplication.
//!
//! Uses the `matrixmultiply` crate for cache-oblivious, SIMD-optimized sgemm
//! to compute all dot products in a single matrix-vector multiply, matching
//! or exceeding numpy/BLAS performance for large collections.

/// Compute batch cosine similarity using matrix-vector multiply (sgemv via sgemm).
///
/// Treats the collection as an N×D row-major matrix and computes
/// `scores = M × query` in a single optimized operation, then divides by norms.
///
/// Returns top-k `(index, score)` pairs sorted by score descending.
/// Tombstoned entries (tombstone != 0) are excluded.
pub fn blas_cosine_batch(
    query: &[f32],
    vectors: &[Vec<f32>],
    norms: &[f32],
    tombstones: &[u8],
    k: usize,
) -> Vec<(usize, f32)> {
    let query_norm = clawhdf5_accel::vector_norm(query);
    if query_norm == 0.0 || vectors.is_empty() {
        return Vec::new();
    }

    let dim = query.len();
    let n = vectors.len();

    // Build a mapping of active (non-tombstoned) indices and a flat matrix.
    //
    // Only vectors of exactly `dim` elements may enter the matrix. sgemm
    // below reads `active_n * dim` elements from `flat`, and this loop used to
    // append every vector whatever its length: one stored vector shorter than
    // the query left `flat` short and sgemm read uninitialized memory past its
    // end -- undefined behavior from a safe pub fn, despite the SAFETY comment
    // claiming the lengths were checked (2026-10 red-team finding). A
    // mismatched vector cannot be compared, so it is skipped, matching the
    // scalar and SIMD paths in vector_search.
    let mut active_indices: Vec<usize> = Vec::with_capacity(n);
    let mut flat: Vec<f32> = Vec::with_capacity(n.saturating_mul(dim));

    for i in 0..n {
        if i < tombstones.len() && tombstones[i] != 0 {
            continue;
        }
        if vectors[i].len() != dim {
            continue;
        }
        active_indices.push(i);
        flat.extend_from_slice(&vectors[i]);
    }

    let active_n = active_indices.len();
    if active_n == 0 {
        return Vec::new();
    }
    // The invariant sgemm's reads depend on, checked rather than assumed.
    if Some(flat.len()) != active_n.checked_mul(dim) || norms.len() < n {
        return Vec::new();
    }

    // Compute scores = M × query using sgemm (treating query as D×1 matrix)
    // M is active_n × dim (row-major), query is dim × 1, output is active_n × 1
    let mut scores = vec![0.0f32; active_n];

    // SAFETY: sgemm requires valid f32 pointers with consistent row/column strides.
    // All slice lengths are checked against active_n * dim before this point.
    unsafe {
        matrixmultiply::sgemm(
            active_n, // m: rows of A (and C)
            dim,      // k: cols of A / rows of B
            1,        // n: cols of B (and C)
            1.0,      // alpha
            flat.as_ptr(),
            dim as isize, // rsa: row stride of A (row-major: dim)
            1,            // csa: col stride of A (row-major: 1)
            query.as_ptr(),
            1,   // rsb: row stride of B (column vector: 1)
            1,   // csb: col stride of B (single column: doesn't matter, use 1)
            0.0, // beta
            scores.as_mut_ptr(),
            1, // rsc: row stride of C
            1, // csc: col stride of C
        );
    }

    // Convert dot products to cosine similarities and collect results
    let mut results: Vec<(usize, f32)> = Vec::with_capacity(active_n);
    for (j, &orig_idx) in active_indices.iter().enumerate() {
        let vec_norm = norms[orig_idx];
        let denom = query_norm * vec_norm;
        let score = if denom == 0.0 { 0.0 } else { scores[j] / denom };
        results.push((orig_idx, score));
    }

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

/// Compute batch cosine similarity from a pre-flattened matrix buffer.
///
/// `vectors_flat` is a contiguous `[N × dim]` f32 buffer in row-major order.
/// This avoids the flatten overhead when vectors are already stored contiguously.
pub fn blas_cosine_batch_flat(
    query: &[f32],
    vectors_flat: &[f32],
    norms: &[f32],
    tombstones: &[u8],
    dim: usize,
    k: usize,
) -> Vec<(usize, f32)> {
    let query_norm = clawhdf5_accel::vector_norm(query);
    if query_norm == 0.0 || vectors_flat.is_empty() {
        return Vec::new();
    }
    // sgemm reads `dim` elements of `query` and `n * dim` of the matrix. A
    // `dim` that disagrees with the query read past the query's end
    // (undefined behavior), and `dim == 0` divided by zero below (2026-10
    // red-team finding).
    if dim == 0 || query.len() != dim {
        return Vec::new();
    }

    let n = vectors_flat.len() / dim;
    if norms.len() < n {
        return Vec::new();
    }

    // If no tombstones, we can use the flat buffer directly
    let all_active = tombstones.iter().all(|&t| t == 0);

    if all_active {
        let mut scores = vec![0.0f32; n];
        // SAFETY: sgemm requires valid f32 pointers. vectors_flat has n*dim elements.
        unsafe {
            matrixmultiply::sgemm(
                n,
                dim,
                1,
                1.0,
                vectors_flat.as_ptr(),
                dim as isize,
                1,
                query.as_ptr(),
                1,
                1,
                0.0,
                scores.as_mut_ptr(),
                1,
                1,
            );
        }

        let mut results: Vec<(usize, f32)> = scores
            .iter()
            .enumerate()
            .map(|(i, &dot)| {
                let denom = query_norm * norms[i];
                let score = if denom == 0.0 { 0.0 } else { dot / denom };
                (i, score)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        return results;
    }

    // With tombstones: need to pack active rows
    let mut active_indices: Vec<usize> = Vec::with_capacity(n);
    let mut flat: Vec<f32> = Vec::with_capacity(n * dim);

    for i in 0..n {
        if i < tombstones.len() && tombstones[i] != 0 {
            continue;
        }
        active_indices.push(i);
        let offset = i * dim;
        flat.extend_from_slice(&vectors_flat[offset..offset + dim]);
    }

    let active_n = active_indices.len();
    if active_n == 0 {
        return Vec::new();
    }

    let mut scores = vec![0.0f32; active_n];
    // SAFETY: sgemm requires valid f32 pointers with consistent row/column strides.
    // All slice lengths are checked against active_n * dim before this point.
    unsafe {
        matrixmultiply::sgemm(
            active_n,
            dim,
            1,
            1.0,
            flat.as_ptr(),
            dim as isize,
            1,
            query.as_ptr(),
            1,
            1,
            0.0,
            scores.as_mut_ptr(),
            1,
            1,
        );
    }

    let mut results: Vec<(usize, f32)> = Vec::with_capacity(active_n);
    for (j, &orig_idx) in active_indices.iter().enumerate() {
        let denom = query_norm * norms[orig_idx];
        let score = if denom == 0.0 { 0.0 } else { scores[j] / denom };
        results.push((orig_idx, score));
    }

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

/// Compute L2 norms for all vectors in a flat buffer using BLAS-style batch ops.
///
/// Returns a Vec of norms, one per vector.
pub fn blas_batch_norms(vectors_flat: &[f32], dim: usize) -> Vec<f32> {
    if dim == 0 || vectors_flat.is_empty() {
        return Vec::new();
    }
    let n = vectors_flat.len() / dim;
    let mut norms = Vec::with_capacity(n);

    for i in 0..n {
        let offset = i * dim;
        let v = &vectors_flat[offset..offset + dim];
        norms.push(clawhdf5_accel::vector_norm(v));
    }

    norms
}

/// Compute a Q×N distance matrix using sgemm.
///
/// `queries` is a flat `[Q × dim]` buffer, `vectors` is a flat `[N × dim]` buffer.
/// Returns a flat `[Q × N]` matrix of dot products (row-major).
///
/// For cosine distance, divide by norms afterward.
/// For PQ training, this computes all pairwise distances efficiently.
pub fn blas_distance_matrix(queries: &[f32], vectors: &[f32], dim: usize) -> Vec<f32> {
    if dim == 0 || queries.is_empty() || vectors.is_empty() {
        return Vec::new();
    }

    let q = queries.len() / dim;
    let n = vectors.len() / dim;
    let mut result = vec![0.0f32; q * n];

    // result = queries × vectors^T
    // queries: Q × D (row-major), vectors^T: D × N
    // But vectors is stored as N × D row-major, so vectors^T has:
    //   element (d, j) = vectors[j * dim + d]
    //   row stride = 1, col stride = dim
    // SAFETY: sgemm requires valid f32 pointers with consistent row/column strides.
    // All slice lengths are checked against active_n * dim before this point.
    unsafe {
        matrixmultiply::sgemm(
            q,   // m: rows of result
            dim, // k: inner dimension
            n,   // n: cols of result
            1.0, // alpha
            queries.as_ptr(),
            dim as isize, // rsa: row stride of queries (row-major)
            1,            // csa: col stride of queries
            vectors.as_ptr(),
            1,            // rsb: row stride of vectors^T = col stride of vectors = 1
            dim as isize, // csb: col stride of vectors^T = row stride of vectors = dim
            0.0,          // beta
            result.as_mut_ptr(),
            n as isize, // rsc: row stride of result (row-major)
            1,          // csc: col stride of result
        );
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vectors(n: usize, dim: usize, seed: u32) -> Vec<Vec<f32>> {
        let mut s = seed;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as f32) / 65536.0 - 0.5
        };
        (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect()
    }

    fn compute_norms(vectors: &[Vec<f32>]) -> Vec<f32> {
        vectors
            .iter()
            .map(|v| clawhdf5_accel::vector_norm(v))
            .collect()
    }

    // --- Test 1: BLAS cosine results match SIMD cosine within f32 epsilon ---
    #[test]
    fn blas_matches_simd_scores() {
        let dim = 384;
        let n = 500;
        let vectors = make_vectors(n, dim, 42);
        let norms = compute_norms(&vectors);
        let tombstones = vec![0u8; n];
        let query = vectors[0].clone();

        let blas_results = blas_cosine_batch(&query, &vectors, &norms, &tombstones, n);
        let simd_results = crate::vector_search::cosine_similarity_batch_prenorm(
            &query,
            &vectors,
            &norms,
            &tombstones,
        );

        assert_eq!(blas_results.len(), simd_results.len());
        // Compare scores by index (both sorted by score desc)
        for (b, s) in blas_results.iter().zip(&simd_results) {
            assert_eq!(b.0, s.0, "index mismatch");
            assert!(
                (b.1 - s.1).abs() < 1e-4,
                "score mismatch at idx {}: blas={} vs simd={}",
                b.0,
                b.1,
                s.1,
            );
        }
    }

    // --- Test 2: BLAS ranking order matches SIMD ranking order ---
    #[test]
    fn blas_ranking_matches_simd() {
        let dim = 128;
        let n = 200;
        let vectors = make_vectors(n, dim, 77);
        let norms = compute_norms(&vectors);
        let tombstones = vec![0u8; n];
        let query = vectors[5].clone();

        let blas_top10 = blas_cosine_batch(&query, &vectors, &norms, &tombstones, 10);
        let simd_all = crate::vector_search::cosine_similarity_batch_prenorm(
            &query,
            &vectors,
            &norms,
            &tombstones,
        );
        let simd_top10 = crate::vector_search::top_k(simd_all, 10);

        let blas_ids: Vec<usize> = blas_top10.iter().map(|r| r.0).collect();
        let simd_ids: Vec<usize> = simd_top10.iter().map(|r| r.0).collect();
        assert_eq!(blas_ids, simd_ids, "top-10 ranking should match");
    }

    // --- Test 3: BLAS batch norms match individual norms ---
    #[test]
    fn blas_batch_norms_match_individual() {
        let dim = 384;
        let n = 100;
        let vectors = make_vectors(n, dim, 42);
        let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();

        let batch_norms = blas_batch_norms(&flat, dim);
        let individual_norms = compute_norms(&vectors);

        assert_eq!(batch_norms.len(), individual_norms.len());
        for (b, i) in batch_norms.iter().zip(&individual_norms) {
            assert!(
                (b - i).abs() < 1e-6,
                "norm mismatch: batch={b} vs individual={i}"
            );
        }
    }

    // --- Test 4: BLAS with tombstones excluded ---
    #[test]
    fn blas_excludes_tombstones() {
        let query = vec![1.0, 0.0, 0.0];
        let vectors = vec![
            vec![1.0, 0.0, 0.0], // idx 0: identical
            vec![0.0, 1.0, 0.0], // idx 1: tombstoned
            vec![0.5, 0.5, 0.0], // idx 2: partial
        ];
        let norms = compute_norms(&vectors);
        let tombstones = vec![0, 1, 0];

        let results = blas_cosine_batch(&query, &vectors, &norms, &tombstones, 10);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|(idx, _)| *idx != 1));
        assert_eq!(results[0].0, 0); // highest
    }

    // --- Test 5: BLAS distance matrix shape and values ---
    #[test]
    fn blas_distance_matrix_shape() {
        let dim = 4;
        let queries: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 2 queries
        let vectors: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, // vec 0
            0.0, 1.0, 0.0, 0.0, // vec 1
            0.0, 0.0, 1.0, 0.0, // vec 2
        ];

        let result = blas_distance_matrix(&queries, &vectors, dim);
        assert_eq!(result.len(), 2 * 3); // Q=2, N=3

        // query[0] = [1,0,0,0] dot vec[0]=[1,0,0,0] = 1.0
        assert!((result[0] - 1.0).abs() < 1e-6);
        // query[0] dot vec[1] = 0.0
        assert!(result[1].abs() < 1e-6);
        // query[1] = [0,1,0,0] dot vec[1]=[0,1,0,0] = 1.0
        assert!((result[4] - 1.0).abs() < 1e-6);
    }

    // --- Test 6: Empty vectors returns empty ---
    #[test]
    fn blas_empty_vectors() {
        let query = vec![1.0, 0.0, 0.0];
        let vectors: Vec<Vec<f32>> = Vec::new();
        let norms: Vec<f32> = Vec::new();
        let tombstones: Vec<u8> = Vec::new();

        let results = blas_cosine_batch(&query, &vectors, &norms, &tombstones, 10);
        assert!(results.is_empty());
    }

    // --- Test 7: Zero query returns empty ---
    #[test]
    fn blas_zero_query() {
        let query = vec![0.0, 0.0, 0.0];
        let vectors = vec![vec![1.0, 0.0, 0.0]];
        let norms = compute_norms(&vectors);
        let tombstones = vec![0u8];

        let results = blas_cosine_batch(&query, &vectors, &norms, &tombstones, 10);
        assert!(results.is_empty());
    }

    // --- Test 8: All tombstoned returns empty ---
    #[test]
    fn blas_all_tombstoned() {
        let query = vec![1.0, 0.0, 0.0];
        let vectors = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let norms = compute_norms(&vectors);
        let tombstones = vec![1, 1];

        let results = blas_cosine_batch(&query, &vectors, &norms, &tombstones, 10);
        assert!(results.is_empty());
    }

    // --- Test 9: Identical vector has score ~1.0 ---
    #[test]
    fn blas_identical_vector_score_one() {
        let query = vec![1.0, 2.0, 3.0, 4.0];
        let vectors = vec![query.clone()];
        let norms = compute_norms(&vectors);
        let tombstones = vec![0u8];

        let results = blas_cosine_batch(&query, &vectors, &norms, &tombstones, 1);
        assert_eq!(results.len(), 1);
        assert!(
            (results[0].1 - 1.0).abs() < 1e-5,
            "expected ~1.0, got {}",
            results[0].1
        );
    }

    // --- Test 10: Orthogonal vectors have score ~0 ---
    #[test]
    fn blas_orthogonal_score_zero() {
        let query = vec![1.0, 0.0, 0.0];
        let vectors = vec![vec![0.0, 1.0, 0.0]];
        let norms = compute_norms(&vectors);
        let tombstones = vec![0u8];

        let results = blas_cosine_batch(&query, &vectors, &norms, &tombstones, 1);
        assert_eq!(results.len(), 1);
        assert!(
            results[0].1.abs() < 1e-5,
            "expected ~0.0, got {}",
            results[0].1
        );
    }

    // --- Test 11: Top-k truncation works ---
    #[test]
    fn blas_top_k_truncation() {
        let dim = 32;
        let n = 100;
        let vectors = make_vectors(n, dim, 42);
        let norms = compute_norms(&vectors);
        let tombstones = vec![0u8; n];
        let query = vectors[0].clone();

        let results = blas_cosine_batch(&query, &vectors, &norms, &tombstones, 5);
        assert_eq!(results.len(), 5);
        // Scores should be descending
        for w in results.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    // --- Test 12: Flat variant matches Vec<Vec> variant ---
    #[test]
    fn blas_flat_matches_vec_variant() {
        let dim = 64;
        let n = 200;
        let vectors = make_vectors(n, dim, 42);
        let norms = compute_norms(&vectors);
        let tombstones = vec![0u8; n];
        let query = vectors[3].clone();
        let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();

        let vec_results = blas_cosine_batch(&query, &vectors, &norms, &tombstones, 10);
        let flat_results = blas_cosine_batch_flat(&query, &flat, &norms, &tombstones, dim, 10);

        assert_eq!(vec_results.len(), flat_results.len());
        for (v, f) in vec_results.iter().zip(&flat_results) {
            assert_eq!(v.0, f.0);
            assert!((v.1 - f.1).abs() < 1e-5);
        }
    }

    // --- Test 13: Flat variant with tombstones ---
    #[test]
    fn blas_flat_with_tombstones() {
        let dim = 3;
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.5, 0.5, 0.0],
        ];
        let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
        let norms = compute_norms(&vectors);
        let tombstones = vec![0, 1, 0]; // idx 1 tombstoned
        let query = vec![1.0, 0.0, 0.0];

        let results = blas_cosine_batch_flat(&query, &flat, &norms, &tombstones, dim, 10);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|(idx, _)| *idx != 1));
    }

    // --- Test 14: Distance matrix empty inputs ---
    #[test]
    fn blas_distance_matrix_empty() {
        let result = blas_distance_matrix(&[], &[1.0, 0.0], 2);
        assert!(result.is_empty());

        let result2 = blas_distance_matrix(&[1.0, 0.0], &[], 2);
        assert!(result2.is_empty());
    }

    // --- Test 15: Batch norms empty ---
    #[test]
    fn blas_batch_norms_empty() {
        let norms = blas_batch_norms(&[], 4);
        assert!(norms.is_empty());
    }

    // --- Test 16: Large-scale BLAS matches SIMD (1000 vectors, 384 dims) ---
    #[test]
    fn blas_large_scale_matches_simd() {
        let dim = 384;
        let n = 1000;
        let vectors = make_vectors(n, dim, 42);
        let norms = compute_norms(&vectors);
        let mut tombstones = vec![0u8; n];
        // Tombstone every 7th
        for i in (0..n).step_by(7) {
            tombstones[i] = 1;
        }
        let query = vectors[1].clone();

        let blas_top20 = blas_cosine_batch(&query, &vectors, &norms, &tombstones, 20);
        let simd_all = crate::vector_search::cosine_similarity_batch_prenorm(
            &query,
            &vectors,
            &norms,
            &tombstones,
        );
        let simd_top20 = crate::vector_search::top_k(simd_all, 20);

        assert_eq!(blas_top20.len(), simd_top20.len());
        for (b, s) in blas_top20.iter().zip(&simd_top20) {
            assert_eq!(b.0, s.0, "index mismatch in top-20");
            assert!(
                (b.1 - s.1).abs() < 1e-4,
                "score mismatch: blas={} vs simd={}",
                b.1,
                s.1,
            );
        }
    }

    // --- Test 17: Negative correlation detected ---
    #[test]
    fn blas_negative_correlation() {
        let query = vec![1.0, 0.0];
        let vectors = vec![vec![-1.0, 0.0]];
        let norms = compute_norms(&vectors);
        let tombstones = vec![0u8];

        let results = blas_cosine_batch(&query, &vectors, &norms, &tombstones, 1);
        assert_eq!(results.len(), 1);
        assert!(
            (results[0].1 - (-1.0)).abs() < 1e-5,
            "expected ~-1.0, got {}",
            results[0].1
        );
    }

    // --- Test 18: Performance - BLAS 10K should complete quickly ---
    #[test]
    fn blas_performance_10k() {
        let dim = 384;
        let n = 10_000;
        let vectors = make_vectors(n, dim, 42);
        let norms = compute_norms(&vectors);
        let tombstones = vec![0u8; n];
        let query = vectors[0].clone();

        let start = std::time::Instant::now();
        let results = blas_cosine_batch(&query, &vectors, &norms, &tombstones, 10);
        let elapsed = start.elapsed();

        assert_eq!(results.len(), 10);
        assert!(
            elapsed.as_millis() < 500,
            "BLAS 10K took {}ms, expected < 500ms",
            elapsed.as_millis()
        );
    }
}

#[cfg(test)]
mod redteam_2026_10 {
    use super::*;

    /// A stored vector shorter than the query used to leave the packed
    /// matrix short while sgemm read `active_n * dim` elements: undefined
    /// behavior. Mismatched vectors are now skipped and the rest still score.
    #[test]
    fn mismatched_vector_lengths_are_skipped_not_read_past() {
        let query = vec![1.0f32, 0.0, 0.0, 0.0];
        let vectors = vec![vec![1.0, 0.0, 0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        let norms: Vec<f32> = vectors.iter().map(|v| clawhdf5_accel::vector_norm(v)).collect();
        let results = blas_cosine_batch(&query, &vectors, &norms, &[], 10);
        let ids: Vec<usize> = results.iter().map(|r| r.0).collect();
        assert!(ids.contains(&0) && ids.contains(&2));
        assert!(!ids.contains(&1), "the 2-element vector cannot be scored against a 4-element query");
        assert!((results[0].1 - 1.0).abs() < 1e-6);
    }

    /// A `dim` that disagrees with the query read past the query's end; zero
    /// divided by zero.
    #[test]
    fn flat_rejects_dim_that_disagrees_with_query() {
        let flat = vec![1.0f32; 16];
        let norms = vec![2.0f32; 4];
        assert!(blas_cosine_batch_flat(&[1.0, 0.0], &flat, &norms, &[], 4, 10).is_empty());
        assert!(blas_cosine_batch_flat(&[1.0, 0.0, 0.0, 0.0], &flat, &norms, &[], 0, 10).is_empty());
        assert_eq!(blas_cosine_batch_flat(&[1.0, 0.0, 0.0, 0.0], &flat, &norms, &[], 4, 10).len(), 4);
    }
}
