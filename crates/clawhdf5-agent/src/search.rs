//! Search and agents_md methods for HDF5Memory.

use std::path::Path;

use crate::bm25;
use crate::hybrid;
use crate::{HDF5Memory, MemoryError, Result, SearchResult};

impl HDF5Memory {
    /// Vector + keyword scoring stage of [`HDF5Memory::hybrid_search`].
    ///
    /// Without the `hnsw` feature this is a full linear cosine scan (the exact
    /// previous behaviour, also used as the correctness oracle in tests). With
    /// `hnsw` enabled and an index available, the vector candidates come from an
    /// approximate-nearest-neighbour search over an over-fetched pool, then merge
    /// with BM25 via the shared [`hybrid::merge_vector_keyword`].
    #[cfg(feature = "hnsw")]
    fn vector_keyword_search(
        &mut self,
        query_embedding: &[f32],
        query_text: &str,
        bm25: &bm25::BM25Index,
        vector_weight: f32,
        keyword_weight: f32,
        k: usize,
    ) -> Vec<(usize, f32)> {
        self.ensure_hnsw_fresh();
        match self.hnsw.as_ref() {
            Some(index)
                if !index.is_empty() && index.dimension() == query_embedding.len() =>
            {
                // Over-fetch so the merge sees a useful vector pool; cosine
                // distance from the index converts back to similarity (1 - d).
                let pool = (k * 8).max(64);
                let vec_scores: Vec<(usize, f32)> = index
                    .search(query_embedding, pool, pool)
                    .into_iter()
                    .map(|(id, dist)| (id, 1.0 - dist))
                    .collect();
                let kw_scores = bm25.search(query_text, self.cache.len());
                hybrid::merge_vector_keyword(vec_scores, kw_scores, vector_weight, keyword_weight, k)
            }
            _ => hybrid::hybrid_search(
                query_embedding,
                query_text,
                &self.cache.embeddings,
                &self.cache.chunks,
                &self.cache.tombstones,
                bm25,
                vector_weight,
                keyword_weight,
                k,
            ),
        }
    }

    #[cfg(not(feature = "hnsw"))]
    fn vector_keyword_search(
        &mut self,
        query_embedding: &[f32],
        query_text: &str,
        bm25: &bm25::BM25Index,
        vector_weight: f32,
        keyword_weight: f32,
        k: usize,
    ) -> Vec<(usize, f32)> {
        hybrid::hybrid_search(
            query_embedding,
            query_text,
            &self.cache.embeddings,
            &self.cache.chunks,
            &self.cache.tombstones,
            bm25,
            vector_weight,
            keyword_weight,
            k,
        )
    }

    /// Perform hybrid search combining cosine vector similarity and BM25 keyword search.
    pub fn hybrid_search(
        &mut self,
        query_embedding: &[f32],
        query_text: &str,
        vector_weight: f32,
        keyword_weight: f32,
        k: usize,
    ) -> Vec<SearchResult> {
        let bm25 = bm25::BM25Index::build(&self.cache.chunks, &self.cache.tombstones);
        let scored = self.vector_keyword_search(
            query_embedding,
            query_text,
            &bm25,
            vector_weight,
            keyword_weight,
            k,
        );
        let mut results: Vec<SearchResult> = scored
            .into_iter()
            .map(|(idx, score)| {
                let w = self.cache.activation_weights[idx];
                SearchResult {
                    score: score * w.sqrt(),
                    chunk: self.cache.chunks[idx].clone(),
                    index: idx,
                    timestamp: self.cache.timestamps[idx],
                    source_channel: self.cache.source_channels[idx].clone(),
                    activation: w,
                }
            })
            .collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let hit_indices: Vec<usize> = results.iter().map(|r| r.index).collect();
        self.apply_hebbian_boost(&hit_indices);
        self.flush().ok();

        results
    }

    fn apply_hebbian_boost(&mut self, hit_indices: &[usize]) {
        for &idx in hit_indices {
            self.cache.activation_weights[idx] += self.config.hebbian_boost;
        }
    }

    /// Get the chunk text for a memory entry by index.
    pub fn get_chunk(&self, index: usize) -> Option<&str> {
        if index < self.cache.chunks.len() && self.cache.tombstones[index] == 0 {
            Some(&self.cache.chunks[index])
        } else {
            None
        }
    }

    /// Generate an AGENTS.md string from current memory state.
    pub fn generate_agents_md(&self) -> String {
        crate::agents_md::generate(&self.config, &self.cache, &self.sessions, &self.knowledge)
    }

    /// Write AGENTS.md to disk alongside the .h5 file.
    pub fn write_agents_md(&self) -> Result<()> {
        let md = self.generate_agents_md();
        let md_path = self.config.path.with_extension("agents.md");
        std::fs::write(&md_path, md).map_err(MemoryError::Io)
    }

    /// Read AGENTS.md from disk (if it exists).
    pub fn read_agents_md(path: &Path) -> Result<String> {
        let md_path = path.with_extension("agents.md");
        std::fs::read_to_string(&md_path).map_err(MemoryError::Io)
    }
}
