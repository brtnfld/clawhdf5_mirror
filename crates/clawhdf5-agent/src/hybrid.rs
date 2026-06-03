//! Hybrid search combining vector similarity and BM25 keyword scores.
//!
//! Normalizes both score sets to [0, 1] and computes a weighted merge.

use std::collections::HashMap;

use crate::bm25::BM25Index;
use crate::vector_search;

/// Perform hybrid search combining cosine vector similarity and BM25 keyword search.
///
/// Both score distributions are independently normalized to [0, 1] before
/// being combined with the specified weights. Uses pre-computed norms when
/// available for faster vector search.
///
/// # Arguments
///
/// * `query_embedding` - The query vector for cosine similarity.
/// * `query_text` - The query text for BM25 keyword search.
/// * `vectors` - All stored embedding vectors.
/// * `_chunks` - All stored text chunks (parallel to `vectors`).
/// * `tombstones` - Tombstone flags (non-zero = deleted).
/// * `bm25_index` - Pre-built BM25 index.
/// * `vector_weight` - Weight for vector similarity scores (default 0.7).
/// * `keyword_weight` - Weight for keyword search scores (default 0.3).
/// * `k` - Number of top results to return.
#[allow(clippy::too_many_arguments)]
pub fn hybrid_search(
    query_embedding: &[f32],
    query_text: &str,
    vectors: &[Vec<f32>],
    _chunks: &[String],
    tombstones: &[u8],
    bm25_index: &BM25Index,
    vector_weight: f32,
    keyword_weight: f32,
    k: usize,
) -> Vec<(usize, f32)> {
    // Get raw scores from both systems. Request all results so normalization
    // covers the full distribution.
    // Use parallel search when rayon feature is enabled and vector count > 10K.
    let vec_scores = {
        #[cfg(feature = "parallel")]
        {
            if vectors.len() > 10_000 {
                vector_search::parallel_cosine_batch(
                    query_embedding,
                    vectors,
                    tombstones,
                    vectors.len(),
                )
            } else {
                vector_search::cosine_similarity_batch(query_embedding, vectors, tombstones)
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            vector_search::cosine_similarity_batch(query_embedding, vectors, tombstones)
        }
    };
    let kw_scores = bm25_index.search(query_text, vectors.len());

    merge_vector_keyword(vec_scores, kw_scores, vector_weight, keyword_weight, k)
}

/// Merge pre-computed vector-similarity and keyword scores into a single ranking.
///
/// Both score sets are independently min-max normalized to [0, 1] and combined
/// with the given weights. This is the shared core of [`hybrid_search`]; it is
/// also used by the optional HNSW path, which supplies vector scores from an
/// approximate-nearest-neighbour index instead of a full linear scan.
pub fn merge_vector_keyword(
    vec_scores: Vec<(usize, f32)>,
    kw_scores: Vec<(usize, f32)>,
    vector_weight: f32,
    keyword_weight: f32,
    k: usize,
) -> Vec<(usize, f32)> {
    // Normalize each set to [0, 1].
    let vec_normalized = normalize_scores(&vec_scores);
    let kw_normalized = normalize_scores(&kw_scores);

    // Merge scores with weights.
    let mut merged: HashMap<usize, f32> = HashMap::new();

    for (idx, score) in &vec_normalized {
        *merged.entry(*idx).or_insert(0.0) += vector_weight * score;
    }
    for (idx, score) in &kw_normalized {
        *merged.entry(*idx).or_insert(0.0) += keyword_weight * score;
    }

    let mut results: Vec<(usize, f32)> = merged.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

/// Normalize a set of scores to the [0, 1] range using min-max normalization.
///
/// If all scores are identical, returns 0.0 for each entry.
fn normalize_scores(scores: &[(usize, f32)]) -> Vec<(usize, f32)> {
    if scores.is_empty() {
        return Vec::new();
    }

    let min = scores.iter().map(|(_, s)| *s).fold(f32::INFINITY, f32::min);
    let max = scores
        .iter()
        .map(|(_, s)| *s)
        .fold(f32::NEG_INFINITY, f32::max);

    let range = max - min;
    if range == 0.0 {
        return scores.iter().map(|(idx, _)| (*idx, 0.0)).collect();
    }

    scores
        .iter()
        .map(|(idx, s)| (*idx, (s - min) / range))
        .collect()
}

/// Perform hybrid search using Reciprocal Rank Fusion (RRF).
///
/// RRF combines rankings from multiple retrieval systems without requiring
/// score normalization. Each result is scored as:
///
///   `score = Σ 1 / (k + rank_i)`
///
/// where `k = 60` (standard constant that dampens the impact of high ranks)
/// and `rank_i` is the 1-based rank of the document in retrieval system `i`.
///
/// Documents only present in one system still receive a partial score.
///
/// # Arguments
///
/// * `query_embedding` - The query vector for cosine similarity.
/// * `query_text` - The query text for BM25 keyword search.
/// * `vectors` - All stored embedding vectors.
/// * `_chunks` - All stored text chunks (parallel to `vectors`).
/// * `tombstones` - Tombstone flags (non-zero = deleted).
/// * `bm25_index` - Pre-built BM25 index.
/// * `k` - Number of top results to return.
#[allow(clippy::too_many_arguments)]
pub fn rrf_hybrid_search(
    query_embedding: &[f32],
    query_text: &str,
    vectors: &[Vec<f32>],
    _chunks: &[String],
    tombstones: &[u8],
    bm25_index: &BM25Index,
    k: usize,
) -> Vec<(usize, f32)> {
    const RRF_K: f32 = 60.0;

    // Retrieve all results from both systems sorted descending by score.
    let mut vec_scores = {
        #[cfg(feature = "parallel")]
        {
            if vectors.len() > 10_000 {
                vector_search::parallel_cosine_batch(
                    query_embedding,
                    vectors,
                    tombstones,
                    vectors.len(),
                )
            } else {
                vector_search::cosine_similarity_batch(query_embedding, vectors, tombstones)
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            vector_search::cosine_similarity_batch(query_embedding, vectors, tombstones)
        }
    };
    let mut kw_scores = bm25_index.search(query_text, vectors.len());

    // Sort both lists descending so rank 1 = best.
    vec_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    kw_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Accumulate RRF scores.
    let mut rrf_scores: HashMap<usize, f32> = HashMap::new();

    for (rank, (idx, _score)) in vec_scores.iter().enumerate() {
        let rrf = 1.0 / (RRF_K + (rank + 1) as f32);
        *rrf_scores.entry(*idx).or_insert(0.0) += rrf;
    }
    for (rank, (idx, _score)) in kw_scores.iter().enumerate() {
        let rrf = 1.0 / (RRF_K + (rank + 1) as f32);
        *rrf_scores.entry(*idx).or_insert(0.0) += rrf;
    }

    let mut results: Vec<(usize, f32)> = rrf_scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<Vec<f32>>, Vec<String>, Vec<u8>, BM25Index) {
        // 4 documents with 3-dim embeddings
        let vectors = vec![
            vec![1.0, 0.0, 0.0], // doc 0: points in x direction
            vec![0.0, 1.0, 0.0], // doc 1: points in y direction
            vec![0.7, 0.7, 0.0], // doc 2: between x and y
            vec![0.0, 0.0, 1.0], // doc 3: points in z direction
        ];
        let chunks = vec![
            "rust programming language".to_string(),
            "python scripting language".to_string(),
            "rust and python comparison".to_string(),
            "javascript web development".to_string(),
        ];
        let tombstones = vec![0u8; 4];
        let bm25 = BM25Index::build(&chunks, &tombstones);
        (vectors, chunks, tombstones, bm25)
    }

    #[test]
    fn vector_only_search() {
        let (vectors, chunks, tombstones, bm25) = make_test_data();
        let query_emb = vec![1.0, 0.0, 0.0]; // points in x, should match doc 0

        let results = hybrid_search(
            &query_emb,
            "nonexistent_xyz",
            &vectors,
            &chunks,
            &tombstones,
            &bm25,
            1.0, // vector only
            0.0, // no keyword
            4,
        );

        assert!(!results.is_empty());
        assert_eq!(
            results[0].0, 0,
            "doc 0 should be top match for x-direction query"
        );
    }

    #[test]
    fn keyword_only_search() {
        let (vectors, chunks, tombstones, bm25) = make_test_data();
        // Use a zero vector so vector similarity contributes nothing meaningful
        let query_emb = vec![0.0, 0.0, 0.0];

        let results = hybrid_search(
            &query_emb,
            "rust programming",
            &vectors,
            &chunks,
            &tombstones,
            &bm25,
            0.0, // no vector
            1.0, // keyword only
            4,
        );

        assert!(!results.is_empty());
        // Doc 0 ("rust programming language") should rank highest for "rust programming"
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn balanced_merge_ranking() {
        let (vectors, chunks, tombstones, bm25) = make_test_data();
        // Query embedding close to doc 0, text query for "rust"
        let query_emb = vec![0.9, 0.1, 0.0];

        let results = hybrid_search(
            &query_emb,
            "rust",
            &vectors,
            &chunks,
            &tombstones,
            &bm25,
            0.7,
            0.3,
            4,
        );

        assert!(!results.is_empty());
        // Doc 0 should rank high (good vector match + contains "rust")
        // Doc 2 should also appear (contains "rust" + decent vector match)
        let top_ids: Vec<usize> = results.iter().map(|(idx, _)| *idx).collect();
        assert!(top_ids.contains(&0), "doc 0 should appear in results");
        assert!(top_ids.contains(&2), "doc 2 should appear in results");
    }

    #[test]
    fn empty_results_when_no_data() {
        let vectors: Vec<Vec<f32>> = Vec::new();
        let chunks: Vec<String> = Vec::new();
        let tombstones: Vec<u8> = Vec::new();
        let bm25 = BM25Index::build(&chunks, &tombstones);

        let results = hybrid_search(
            &[],
            "anything",
            &vectors,
            &chunks,
            &tombstones,
            &bm25,
            0.7,
            0.3,
            10,
        );

        assert!(results.is_empty());
    }

    #[test]
    fn normalize_scores_empty() {
        let result = normalize_scores(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn normalize_scores_single() {
        let result = normalize_scores(&[(0, 5.0)]);
        assert_eq!(result.len(), 1);
        // Single score normalizes to 0.0 (range is 0)
        assert_eq!(result[0].1, 0.0);
    }

    #[test]
    fn normalize_scores_range() {
        let scores = vec![(0, 2.0), (1, 4.0), (2, 6.0)];
        let result = normalize_scores(&scores);

        assert_eq!(result.len(), 3);
        assert!((result[0].1 - 0.0).abs() < 1e-6); // min -> 0
        assert!((result[1].1 - 0.5).abs() < 1e-6); // mid -> 0.5
        assert!((result[2].1 - 1.0).abs() < 1e-6); // max -> 1
    }

    #[test]
    fn hybrid_respects_k_limit() {
        let (vectors, chunks, tombstones, bm25) = make_test_data();
        let query_emb = vec![0.5, 0.5, 0.0];

        let results = hybrid_search(
            &query_emb,
            "language",
            &vectors,
            &chunks,
            &tombstones,
            &bm25,
            0.5,
            0.5,
            2,
        );

        assert!(results.len() <= 2);
    }

    // --- RRF tests ---

    #[test]
    fn rrf_vector_dominant_query() {
        let (vectors, chunks, tombstones, bm25) = make_test_data();
        let query_emb = vec![1.0, 0.0, 0.0]; // strong match on doc 0

        let results = rrf_hybrid_search(
            &query_emb,
            "nonexistent_xyz",
            &vectors,
            &chunks,
            &tombstones,
            &bm25,
            4,
        );

        assert!(!results.is_empty());
        assert_eq!(
            results[0].0, 0,
            "doc 0 should top RRF for x-direction query"
        );
    }

    #[test]
    fn rrf_keyword_dominant_query() {
        let (vectors, chunks, tombstones, bm25) = make_test_data();
        let query_emb = vec![0.0, 0.0, 0.0];

        let results = rrf_hybrid_search(
            &query_emb,
            "rust programming",
            &vectors,
            &chunks,
            &tombstones,
            &bm25,
            4,
        );

        assert!(!results.is_empty());
        assert_eq!(
            results[0].0, 0,
            "doc 0 should top RRF for rust programming query"
        );
    }

    #[test]
    fn rrf_respects_k_limit() {
        let (vectors, chunks, tombstones, bm25) = make_test_data();
        let query_emb = vec![0.5, 0.5, 0.0];

        let results = rrf_hybrid_search(
            &query_emb,
            "language",
            &vectors,
            &chunks,
            &tombstones,
            &bm25,
            2,
        );

        assert!(results.len() <= 2);
    }

    #[test]
    fn rrf_scores_are_positive() {
        let (vectors, chunks, tombstones, bm25) = make_test_data();
        let query_emb = vec![0.5, 0.5, 0.0];

        let results =
            rrf_hybrid_search(&query_emb, "rust", &vectors, &chunks, &tombstones, &bm25, 4);

        for (_, score) in &results {
            assert!(*score > 0.0, "RRF scores must be positive");
        }
    }

    #[test]
    fn rrf_empty_data() {
        let vectors: Vec<Vec<f32>> = Vec::new();
        let chunks: Vec<String> = Vec::new();
        let tombstones: Vec<u8> = Vec::new();
        let bm25 = BM25Index::build(&chunks, &tombstones);

        let results = rrf_hybrid_search(&[], "anything", &vectors, &chunks, &tombstones, &bm25, 10);

        assert!(results.is_empty());
    }

    #[test]
    fn rrf_scores_sorted_descending() {
        let (vectors, chunks, tombstones, bm25) = make_test_data();
        let query_emb = vec![0.7, 0.3, 0.0];

        let results = rrf_hybrid_search(
            &query_emb,
            "rust language",
            &vectors,
            &chunks,
            &tombstones,
            &bm25,
            4,
        );

        for window in results.windows(2) {
            assert!(
                window[0].1 >= window[1].1,
                "RRF results must be sorted descending"
            );
        }
    }
}
