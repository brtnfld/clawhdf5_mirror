//! ZeroClaw agent memory HDF5 backend.
//!
//! Provides persistent memory storage for AI agents using HDF5 files.
//! All data is cached in-memory for fast access and flushed to disk
//! on mutations.

#[cfg(any(feature = "accelerate", feature = "openblas"))]
pub mod accelerate_search;
#[cfg(feature = "async")]
pub mod async_memory;
#[cfg(feature = "fast-math")]
pub mod blas_search;
pub mod bm25;
pub mod gpu_search;
pub mod hybrid;
pub mod ivf;
pub mod pq;
pub mod strategy;
pub mod vector_search;

pub mod agents_md;
pub mod anomaly;
pub mod cache;
pub mod confidence;
pub mod consolidation;
pub mod decision_gate;
pub mod entity_extract;
pub mod ephemeral;
pub mod knowledge;
pub mod memory_strategy;
pub mod multimodal;
pub mod openclaw;
pub mod provenance;
pub mod query_expand;
pub mod reranker;
pub mod schema;
pub mod search;
pub mod session;
pub mod storage;
pub mod temporal;
pub mod wal;

/// Cosine similarity with pre-computed norms using clawhdf5_accel primitives.
///
/// Avoids recomputing the query/vector norms on every comparison.
#[inline]
pub fn cosine_similarity_prenorm(
    query: &[f32],
    query_norm: f32,
    vec: &[f32],
    vec_norm: f32,
) -> f32 {
    let denom = query_norm * vec_norm;
    if denom == 0.0 {
        return 0.0;
    }
    clawhdf5_accel::dot_product(query, vec) / denom
}

use std::path::{Path, PathBuf};

use cache::MemoryCache;
#[cfg(feature = "hnsw")]
use clawhdf5_ann::{DistanceMetric, HnswIndex};
use ephemeral::{EphemeralConfig, EphemeralStore};

/// HNSW construction parameters used for the agent's vector index. Cosine is the
/// agent's similarity metric, so the index is built with cosine distance.
#[cfg(feature = "hnsw")]
const HNSW_M: usize = 16;
#[cfg(feature = "hnsw")]
const HNSW_EF_CONSTRUCTION: usize = 64;
// EphemeralEntry and EphemeralStats are part of the crate public API via
// the `ephemeral` module; they are not needed directly in lib.rs internals.
#[allow(unused_imports)]
pub use ephemeral::{EphemeralEntry, EphemeralStats};
use knowledge::KnowledgeCache;
use memory_strategy::{Exchange, MemoryStrategy, StrategyOutput};
use session::SessionCache;

// --- Error type ---

#[derive(Debug)]
pub enum MemoryError {
    Io(std::io::Error),
    Hdf5(String),
    Schema(String),
    NotFound(String),
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryError::Io(e) => write!(f, "I/O error: {e}"),
            MemoryError::Hdf5(e) => write!(f, "HDF5 error: {e}"),
            MemoryError::Schema(e) => write!(f, "schema error: {e}"),
            MemoryError::NotFound(e) => write!(f, "not found: {e}"),
        }
    }
}

impl std::error::Error for MemoryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MemoryError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for MemoryError {
    fn from(e: std::io::Error) -> Self {
        MemoryError::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, MemoryError>;

// --- Config and data types ---

#[derive(Debug, Clone)]
pub struct MemoryConfig {
    pub path: PathBuf,
    pub agent_id: String,
    pub embedder: String,
    pub embedding_dim: usize,
    pub chunk_size: usize,
    pub overlap: usize,
    pub float16: bool,
    pub compression: bool,
    pub compression_level: u32,
    pub compact_threshold: f32,
    pub hebbian_boost: f32,
    pub decay_factor: f32,
    pub created_at: String,
    pub wal_enabled: bool,
    pub wal_max_entries: usize,
}

impl MemoryConfig {
    pub fn new(path: PathBuf, agent_id: &str, embedding_dim: usize) -> Self {
        let created_at = now_iso8601();
        Self {
            path,
            agent_id: agent_id.to_string(),
            embedder: "openai:text-embedding-3-small".to_owned(),
            embedding_dim,
            chunk_size: 512,
            overlap: 50,
            float16: false,
            compression: false,
            compression_level: 0,
            compact_threshold: 0.3,
            hebbian_boost: 0.15,
            decay_factor: 0.98,
            created_at,
            wal_enabled: true,
            wal_max_entries: 500,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryEntry {
    pub chunk: String,
    pub embedding: Vec<f32>,
    pub source_channel: String,
    pub timestamp: f64,
    pub session_id: String,
    pub tags: String,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub score: f32,
    pub chunk: String,
    pub index: usize,
    pub timestamp: f64,
    pub source_channel: String,
    pub activation: f32,
}

// --- Trait ---

pub trait AgentMemory {
    fn save(&mut self, entry: MemoryEntry) -> Result<usize>;
    fn save_batch(&mut self, entries: Vec<MemoryEntry>) -> Result<Vec<usize>>;
    fn delete(&mut self, id: usize) -> Result<()>;
    fn compact(&mut self) -> Result<usize>;
    fn count(&self) -> usize;
    fn count_active(&self) -> usize;
    fn snapshot(&self, dest: &Path) -> Result<PathBuf>;
    fn add_session(
        &mut self,
        id: &str,
        start: usize,
        end: usize,
        channel: &str,
        summary: &str,
    ) -> Result<()>;
    fn get_session_summary(&self, session_id: &str) -> Result<Option<String>>;
}

// --- HDF5Memory ---

pub struct HDF5Memory {
    pub(crate) config: MemoryConfig,
    pub cache: MemoryCache,
    pub(crate) sessions: SessionCache,
    pub(crate) knowledge: KnowledgeCache,
    wal: Option<wal::WalFile>,
    strategy: Option<Box<dyn MemoryStrategy>>,
    pub ephemeral: Option<EphemeralStore>,
    /// Optional HNSW index accelerating the vector stage of `hybrid_search`.
    /// `None` when the store isn't indexable (no/zero-dim/mixed-dim embeddings);
    /// rebuilt from the cache whenever it drifts out of sync (see
    /// [`HDF5Memory::ensure_hnsw_fresh`]).
    #[cfg(feature = "hnsw")]
    hnsw: Option<HnswIndex>,
    /// Set when an in-place update/compaction may have invalidated `hnsw`,
    /// forcing a rebuild before the next search.
    #[cfg(feature = "hnsw")]
    hnsw_dirty: bool,
    /// Cache length the current `hnsw` value reflects. A mismatch with the live
    /// cache length triggers a rebuild — this both picks up unhooked cache
    /// pushes and avoids re-attempting to build an unindexable store every
    /// search.
    #[cfg(feature = "hnsw")]
    hnsw_synced_len: usize,
}

impl std::fmt::Debug for HDF5Memory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HDF5Memory({:?})", self.config.agent_id)
    }
}

impl HDF5Memory {
    /// Create a new HDF5 memory file with the given configuration.
    pub fn create(config: MemoryConfig) -> Result<Self> {
        let cache = MemoryCache::new(config.embedding_dim);
        let sessions = SessionCache::new();
        let knowledge = KnowledgeCache::new();

        // Write initial empty file
        storage::write_to_disk(&config.path, &config, &cache, &sessions, &knowledge)?;

        let wal = if config.wal_enabled {
            let wal_path = config.path.with_extension("h5.wal");
            Some(wal::WalFile::open(&wal_path)?)
        } else {
            None
        };

        Ok(Self {
            config,
            cache,
            sessions,
            knowledge,
            wal,
            strategy: None,
            ephemeral: None,
            #[cfg(feature = "hnsw")]
            hnsw: None,
            #[cfg(feature = "hnsw")]
            hnsw_dirty: false,
            #[cfg(feature = "hnsw")]
            hnsw_synced_len: 0,
        })
    }

    /// Open an existing HDF5 memory file.
    pub fn open(path: &Path) -> Result<Self> {
        let (config, mut cache, sessions, knowledge) = storage::read_from_disk(path)?;

        // Replay WAL if present
        let wal_path = path.with_extension("h5.wal");
        let wal = if wal_path.exists() {
            let entries = wal::WalFile::read_entries(&wal_path)?;
            wal::replay_into_cache(&entries, &mut cache);
            Some(wal::WalFile::open(&wal_path)?)
        } else if config.wal_enabled {
            Some(wal::WalFile::open(&wal_path)?)
        } else {
            None
        };

        Ok(Self {
            config,
            cache,
            sessions,
            knowledge,
            wal,
            strategy: None,
            ephemeral: None,
            // Existing data is loaded from disk + WAL replay; mark the index
            // dirty so it is (re)built from the cache on the first search.
            #[cfg(feature = "hnsw")]
            hnsw: None,
            #[cfg(feature = "hnsw")]
            hnsw_dirty: true,
            #[cfg(feature = "hnsw")]
            hnsw_synced_len: 0,
        })
    }

    /// Flush current state to disk and truncate the WAL.
    ///
    /// Every code path that persists the full cache to the .h5 file must
    /// also clear the WAL, otherwise `open()` will replay stale entries
    /// on top of the already-persisted data, duplicating them.
    fn flush(&mut self) -> Result<()> {
        storage::write_to_disk(
            &self.config.path,
            &self.config,
            &self.cache,
            &self.sessions,
            &self.knowledge,
        )?;
        if let Some(ref mut w) = self.wal {
            w.truncate()?;
        }
        Ok(())
    }

    // ---- HNSW index maintenance --------------------------------------------
    //
    // The index mirrors the cache: HNSW node id == cache index, kept aligned by
    // appending to both in lock-step and mirroring deletes. The incremental
    // hooks below are an optimization for the hot path; correctness is
    // guaranteed by `ensure_hnsw_fresh`, which rebuilds from the cache whenever
    // the index length drifts from the cache length (covering any mutation path
    // that doesn't call a hook, e.g. consolidation pushes).

    /// Build an HNSW index over the entire cache, re-applying tombstones as
    /// soft-deletions so node ids stay aligned with cache indices.
    ///
    /// Returns `None` for stores that aren't usefully indexable: no embeddings,
    /// a zero embedding dimension, or embeddings of mixed dimension (in which
    /// case `hybrid_search` keeps using the linear scan).
    #[cfg(feature = "hnsw")]
    fn build_hnsw_from_cache(&self) -> Option<HnswIndex> {
        let dim = self.cache.embedding_dim;
        if dim == 0 || self.cache.embeddings.is_empty() {
            return None;
        }
        if self.cache.embeddings.iter().any(|e| e.len() != dim) {
            return None;
        }
        let mut index = HnswIndex::build_with_metric(
            &self.cache.embeddings,
            HNSW_M,
            HNSW_EF_CONSTRUCTION,
            DistanceMetric::Cosine,
        );
        for (i, &t) in self.cache.tombstones.iter().enumerate() {
            if t != 0 {
                index.mark_deleted(i);
            }
        }
        Some(index)
    }

    /// Ensure the HNSW index reflects the current cache. Rebuilds when marked
    /// dirty or when the cache length no longer matches what the index reflects.
    #[cfg(feature = "hnsw")]
    fn ensure_hnsw_fresh(&mut self) {
        let n = self.cache.embeddings.len();
        if self.hnsw_dirty || self.hnsw_synced_len != n {
            self.hnsw = self.build_hnsw_from_cache();
            self.hnsw_synced_len = n;
            self.hnsw_dirty = false;
        }
    }

    /// Incrementally index the embedding just pushed at `idx`. Falls back to a
    /// rebuild (via the dirty flag) for the first vector, dimension mismatches,
    /// or id drift.
    #[cfg(feature = "hnsw")]
    fn hnsw_on_insert(&mut self, idx: usize) {
        if self.hnsw_dirty {
            return; // a rebuild is already pending; it will pick this up
        }
        let emb_len = self.cache.embeddings[idx].len();
        match self.hnsw.as_mut() {
            Some(index) if emb_len == index.dimension() => {
                let id = index.insert(self.cache.embeddings[idx].clone());
                if id == idx {
                    self.hnsw_synced_len = self.cache.embeddings.len();
                } else {
                    self.hnsw_dirty = true;
                }
            }
            // Dimension mismatch, first-ever vector, or no index yet: defer to a
            // rebuild, which decides indexability uniformly.
            _ => self.hnsw_dirty = true,
        }
    }

    #[cfg(not(feature = "hnsw"))]
    #[inline]
    fn hnsw_on_insert(&mut self, _idx: usize) {}

    /// Mirror a cache deletion into the index.
    #[cfg(feature = "hnsw")]
    fn hnsw_on_delete(&mut self, id: usize) {
        if let Some(index) = self.hnsw.as_mut() {
            index.mark_deleted(id);
        }
    }

    #[cfg(not(feature = "hnsw"))]
    #[inline]
    fn hnsw_on_delete(&mut self, _id: usize) {}

    /// Mark the index for rebuild after a mutation that may have changed
    /// existing embeddings or renumbered ids (in-place update, compaction).
    #[cfg(feature = "hnsw")]
    #[inline]
    fn hnsw_mark_dirty(&mut self) {
        self.hnsw_dirty = true;
    }

    #[cfg(not(feature = "hnsw"))]
    #[inline]
    fn hnsw_mark_dirty(&mut self) {}

    /// Get a reference to the config.
    pub fn config(&self) -> &MemoryConfig {
        &self.config
    }

    /// Get a reference to the knowledge cache.
    pub fn knowledge(&self) -> &KnowledgeCache {
        &self.knowledge
    }

    /// Get a mutable reference to the knowledge cache.
    pub fn knowledge_mut(&mut self) -> &mut KnowledgeCache {
        &mut self.knowledge
    }

    /// Add an entity to the knowledge graph and flush.
    pub fn add_entity(&mut self, name: &str, entity_type: &str, embedding_idx: i64) -> Result<u64> {
        let id = self.knowledge.add_entity(name, entity_type, embedding_idx);
        self.flush()?;
        Ok(id)
    }

    /// Add an alias for a knowledge graph entity and flush.
    pub fn add_entity_alias(&mut self, alias: &str, entity_id: i64) -> Result<()> {
        self.knowledge.add_alias(alias, entity_id);
        self.flush()
    }

    /// Add a relation to the knowledge graph and flush.
    pub fn add_relation(&mut self, src: u64, tgt: u64, relation: &str, weight: f32) -> Result<()> {
        self.knowledge.add_relation(src, tgt, relation, weight);
        self.flush()?;
        Ok(())
    }

    /// Extract entities from a text chunk and add them to the knowledge graph.
    ///
    /// Runs `EntityExtractor::extract()` on `text`, then calls
    /// `knowledge_cache.resolve_or_create()` for each extracted entity to find
    /// or create the corresponding node.  Returns the list of
    /// `(entity_id, extracted_entity)` pairs.
    pub fn extract_and_store_entities(
        &mut self,
        text: &str,
        config: Option<entity_extract::ExtractorConfig>,
    ) -> Vec<(u64, entity_extract::ExtractedEntity)> {
        let cfg = config.unwrap_or_default();
        let extractor = entity_extract::EntityExtractor::new(cfg);
        let entities = extractor.extract(text);
        let mut result = Vec::with_capacity(entities.len());
        for entity in entities {
            let type_str = format!("{:?}", entity.entity_type).to_lowercase();
            let (id, _created) = self
                .knowledge
                .resolve_or_create(&entity.text, &type_str, -1, 1);
            result.push((id, entity));
        }
        // Best-effort flush; ignore errors here so the method remains infallible.
        let _ = self.flush();
        result
    }
}

impl HDF5Memory {
    /// Upsert: if an active entry with the same tags (key) exists, update it in-place.
    /// Otherwise append a new entry. Use this for key-based memory stores where
    /// the same key should not create duplicates.
    pub fn save_or_update(&mut self, entry: MemoryEntry) -> Result<usize> {
        if let Some(existing_idx) = self.cache.find_by_tags(&entry.tags) {
            if let Some(ref mut w) = self.wal {
                let wal_entry = wal::WalEntry {
                    entry_type: wal::WalEntryType::Save,
                    timestamp: entry.timestamp,
                    chunk: entry.chunk.clone(),
                    embedding: entry.embedding.clone(),
                    source_channel: entry.source_channel.clone(),
                    session_id: entry.session_id.clone(),
                    tags: entry.tags.clone(),
                    tombstone_index: None,
                };
                w.append_save(&wal_entry)?;
            }
            self.cache.update(
                existing_idx,
                entry.chunk,
                entry.embedding,
                entry.source_channel,
                entry.timestamp,
                entry.session_id,
            );
            // In-place embedding change: the index node is stale, force rebuild.
            self.hnsw_mark_dirty();
            let needs_flush = self
                .wal
                .as_ref()
                .is_none_or(|w| w.pending_count() as usize > self.config.wal_max_entries);
            if needs_flush {
                self.flush()?;
                if let Some(ref mut w) = self.wal {
                    w.truncate()?;
                }
            }
            return Ok(existing_idx);
        }
        // No existing entry — fall through to regular save
        AgentMemory::save(self, entry)
    }
}

impl AgentMemory for HDF5Memory {
    fn save(&mut self, entry: MemoryEntry) -> Result<usize> {
        if let Some(ref mut w) = self.wal {
            let wal_entry = wal::WalEntry {
                entry_type: wal::WalEntryType::Save,
                timestamp: entry.timestamp,
                chunk: entry.chunk.clone(),
                embedding: entry.embedding.clone(),
                source_channel: entry.source_channel.clone(),
                session_id: entry.session_id.clone(),
                tags: entry.tags.clone(),
                tombstone_index: None,
            };
            w.append_save(&wal_entry)?;
        }
        let idx = self.cache.push(
            entry.chunk,
            entry.embedding,
            entry.source_channel,
            entry.timestamp,
            entry.session_id,
            entry.tags,
        );
        self.hnsw_on_insert(idx);
        let needs_flush = self
            .wal
            .as_ref()
            .is_none_or(|w| w.pending_count() as usize > self.config.wal_max_entries);
        if needs_flush {
            self.flush()?;
            if let Some(ref mut w) = self.wal {
                w.truncate()?;
            }
        }
        Ok(idx)
    }

    fn save_batch(&mut self, entries: Vec<MemoryEntry>) -> Result<Vec<usize>> {
        let mut indices = Vec::with_capacity(entries.len());
        for entry in entries {
            let idx = self.cache.push(
                entry.chunk,
                entry.embedding,
                entry.source_channel,
                entry.timestamp,
                entry.session_id,
                entry.tags,
            );
            indices.push(idx);
        }
        // Batch inserts rebuild the index once rather than node-by-node.
        self.hnsw_mark_dirty();
        self.flush()?;
        Ok(indices)
    }

    fn delete(&mut self, id: usize) -> Result<()> {
        if !self.cache.mark_deleted(id) {
            return Err(MemoryError::NotFound(format!(
                "entry {id} not found or already deleted"
            )));
        }
        self.hnsw_on_delete(id);
        self.flush()?;

        // Auto-compact if threshold exceeded
        if self.config.compact_threshold > 0.0
            && self.cache.tombstone_fraction() > self.config.compact_threshold
        {
            self.compact()?;
        }

        Ok(())
    }

    fn compact(&mut self) -> Result<usize> {
        let (removed, _index_map) = self.cache.compact();
        if removed > 0 {
            // Compaction renumbers cache indices; rebuild the index to match.
            self.hnsw_mark_dirty();
            self.flush()?;
        }
        Ok(removed)
    }

    fn count(&self) -> usize {
        self.cache.len()
    }

    fn count_active(&self) -> usize {
        self.cache.count_active()
    }

    fn snapshot(&self, dest: &Path) -> Result<PathBuf> {
        storage::snapshot_file(&self.config.path, dest)
    }

    fn add_session(
        &mut self,
        id: &str,
        start: usize,
        end: usize,
        channel: &str,
        summary: &str,
    ) -> Result<()> {
        self.sessions.add(id, start, end, channel, summary);
        self.flush()?;
        Ok(())
    }

    fn get_session_summary(&self, session_id: &str) -> Result<Option<String>> {
        Ok(self.sessions.find_summary(session_id).map(String::from))
    }
}

fn now_iso8601() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let minutes = (time_secs % 3600) / 60;
    let seconds = time_secs % 60;

    let mut y = 1970i64;
    let mut remaining_days = (secs / 86400) as i64;
    loop {
        let days_in_year = if is_leap(y) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        y += 1;
    }
    let month_days = if is_leap(y) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut m = 1u32;
    for &md in &month_days {
        if remaining_days < md {
            break;
        }
        remaining_days -= md;
        m += 1;
    }
    let day = remaining_days + 1;

    format!("{y:04}-{m:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_config(dir: &TempDir) -> MemoryConfig {
        let mut c = MemoryConfig::new(dir.path().join("test.h5"), "agent-test", 4);
        c.wal_enabled = false;
        c
    }

    fn make_entry(chunk: &str, embedding: &[f32]) -> MemoryEntry {
        MemoryEntry {
            chunk: chunk.to_string(),
            embedding: embedding.to_vec(),
            source_channel: "test".to_owned(),
            timestamp: 1000000.0,
            session_id: "session-1".to_owned(),
            tags: "tag1,tag2".to_owned(),
        }
    }

    #[test]
    fn create_new_file() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let mem = HDF5Memory::create(config).unwrap();
        assert_eq!(mem.count(), 0);
        assert_eq!(mem.count_active(), 0);
        assert!(dir.path().join("test.h5").exists());
    }

    #[test]
    fn save_single_entry() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let mut mem = HDF5Memory::create(config).unwrap();

        let idx = mem
            .save(make_entry("hello world", &[1.0, 2.0, 3.0, 4.0]))
            .unwrap();
        assert_eq!(idx, 0);
        assert_eq!(mem.count(), 1);
        assert_eq!(mem.count_active(), 1);
    }

    #[test]
    fn save_batch() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let mut mem = HDF5Memory::create(config).unwrap();

        let entries = vec![
            make_entry("chunk 1", &[1.0, 0.0, 0.0, 0.0]),
            make_entry("chunk 2", &[0.0, 1.0, 0.0, 0.0]),
            make_entry("chunk 3", &[0.0, 0.0, 1.0, 0.0]),
        ];
        let indices = mem.save_batch(entries).unwrap();
        assert_eq!(indices, vec![0, 1, 2]);
        assert_eq!(mem.count(), 3);
    }

    #[test]
    fn delete_entry() {
        let dir = TempDir::new().unwrap();
        let mut config = make_config(&dir);
        config.compact_threshold = 0.0;
        let mut mem = HDF5Memory::create(config).unwrap();

        mem.save(make_entry("chunk 1", &[1.0, 0.0, 0.0, 0.0]))
            .unwrap();
        mem.save(make_entry("chunk 2", &[0.0, 1.0, 0.0, 0.0]))
            .unwrap();

        mem.delete(0).unwrap();
        assert_eq!(mem.count(), 2);
        assert_eq!(mem.count_active(), 1);
    }

    #[test]
    fn compact_removes_tombstoned() {
        let dir = TempDir::new().unwrap();
        let mut config = make_config(&dir);
        config.compact_threshold = 0.0;
        let mut mem = HDF5Memory::create(config).unwrap();

        mem.save(make_entry("chunk 1", &[1.0, 0.0, 0.0, 0.0]))
            .unwrap();
        mem.save(make_entry("chunk 2", &[0.0, 1.0, 0.0, 0.0]))
            .unwrap();
        mem.save(make_entry("chunk 3", &[0.0, 0.0, 1.0, 0.0]))
            .unwrap();

        mem.delete(0).unwrap();
        mem.delete(2).unwrap();

        let removed = mem.compact().unwrap();
        assert_eq!(removed, 2);
        assert_eq!(mem.count(), 1);
        assert_eq!(mem.count_active(), 1);
    }

    #[test]
    fn snapshot_creates_copy() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let mut mem = HDF5Memory::create(config).unwrap();

        mem.save(make_entry("snapshot test", &[1.0, 2.0, 3.0, 4.0]))
            .unwrap();

        let snap_dir = TempDir::new().unwrap();
        let snap_path = mem.snapshot(snap_dir.path()).unwrap();
        assert!(snap_path.exists());

        let snap_mem = HDF5Memory::open(&snap_path).unwrap();
        assert_eq!(snap_mem.count(), 1);
    }

    #[test]
    fn session_tracking() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let mut mem = HDF5Memory::create(config).unwrap();

        mem.add_session("sess-1", 0, 5, "whatsapp", "discussed AI topics")
            .unwrap();
        mem.add_session("sess-2", 6, 10, "slack", "code review session")
            .unwrap();

        let summary = mem.get_session_summary("sess-1").unwrap();
        assert_eq!(summary.as_deref(), Some("discussed AI topics"));

        let summary2 = mem.get_session_summary("sess-2").unwrap();
        assert_eq!(summary2.as_deref(), Some("code review session"));

        let missing = mem.get_session_summary("sess-999").unwrap();
        assert!(missing.is_none());
    }

    #[test]
    fn knowledge_add_entity() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let mut mem = HDF5Memory::create(config).unwrap();

        let id1 = mem.add_entity("Rust", "language", -1).unwrap();
        let id2 = mem.add_entity("HDF5", "format", -1).unwrap();

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);

        let entity = mem.knowledge().get_entity(0).unwrap();
        assert_eq!(entity.name, "Rust");
        assert_eq!(entity.entity_type, "language");
    }

    #[test]
    fn knowledge_add_relation() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let mut mem = HDF5Memory::create(config).unwrap();

        let rust_id = mem.add_entity("Rust", "language", -1).unwrap();
        let hdf5_id = mem.add_entity("HDF5", "format", -1).unwrap();
        mem.add_relation(rust_id, hdf5_id, "uses", 1.0).unwrap();

        let rels = mem.knowledge().get_relations_from(rust_id);
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].relation, "uses");
        assert_eq!(rels[0].tgt, hdf5_id);
    }

    #[test]
    fn open_existing() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let path = config.path.clone();

        {
            let mut mem = HDF5Memory::create(config).unwrap();
            mem.save(make_entry("persisted chunk", &[1.0, 2.0, 3.0, 4.0]))
                .unwrap();
        }

        let mem = HDF5Memory::open(&path).unwrap();
        assert_eq!(mem.count(), 1);
        assert_eq!(mem.config().agent_id, "agent-test");
        assert_eq!(mem.config().embedding_dim, 4);
    }

    #[test]
    fn schema_version_mismatch() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bad.h5");

        let mut builder = clawhdf5::FileBuilder::new();
        let mut meta = builder.create_group("meta");
        meta.set_attr("schema_version", clawhdf5::AttrValue::String("99.0".into()));
        meta.set_attr("created_at", clawhdf5::AttrValue::String("now".into()));
        meta.set_attr("agent_id", clawhdf5::AttrValue::String("test".into()));
        meta.set_attr("embedder", clawhdf5::AttrValue::String("test".into()));
        meta.set_attr("embedding_dim", clawhdf5::AttrValue::I64(4));
        meta.set_attr("chunk_size", clawhdf5::AttrValue::I64(512));
        meta.set_attr("overlap", clawhdf5::AttrValue::I64(50));
        meta.create_dataset("_marker").with_u8_data(&[1]);
        let finished = meta.finish();
        builder.add_group(finished);
        builder.write(&path).unwrap();

        let err = HDF5Memory::open(&path).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("schema version mismatch"), "got: {msg}");
    }

    #[test]
    fn round_trip() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let path = config.path.clone();

        {
            let mut mem = HDF5Memory::create(config).unwrap();
            mem.save(make_entry("round trip data", &[0.1, 0.2, 0.3, 0.4]))
                .unwrap();
            mem.add_session("sess-rt", 0, 0, "api", "round trip session")
                .unwrap();
            mem.add_entity("TestEntity", "test", 0).unwrap();
        }

        let mem = HDF5Memory::open(&path).unwrap();
        assert_eq!(mem.count(), 1);
        let summary = mem.get_session_summary("sess-rt").unwrap();
        assert_eq!(summary.as_deref(), Some("round trip session"));
        let entity = mem.knowledge().get_entity(0).unwrap();
        assert_eq!(entity.name, "TestEntity");
    }

    #[test]
    fn delete_nonexistent() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let mut mem = HDF5Memory::create(config).unwrap();

        let err = mem.delete(999).unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn double_delete() {
        let dir = TempDir::new().unwrap();
        let mut config = make_config(&dir);
        config.compact_threshold = 0.0;
        let mut mem = HDF5Memory::create(config).unwrap();

        mem.save(make_entry("double del", &[1.0, 0.0, 0.0, 0.0]))
            .unwrap();
        mem.delete(0).unwrap();
        let err = mem.delete(0).unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn compact_no_tombstones() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let mut mem = HDF5Memory::create(config).unwrap();

        mem.save(make_entry("no compact", &[1.0, 0.0, 0.0, 0.0]))
            .unwrap();
        let removed = mem.compact().unwrap();
        assert_eq!(removed, 0);
        assert_eq!(mem.count(), 1);
    }

    #[test]
    fn empty_file_operations() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let path = config.path.clone();
        let mem = HDF5Memory::create(config).unwrap();
        assert_eq!(mem.count(), 0);
        assert_eq!(mem.count_active(), 0);

        let mem2 = HDF5Memory::open(&path).unwrap();
        assert_eq!(mem2.count(), 0);
    }

    #[test]
    fn multiple_sessions() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let path = config.path.clone();

        {
            let mut mem = HDF5Memory::create(config).unwrap();
            for i in 0..5 {
                mem.add_session(
                    &format!("sess-{i}"),
                    i * 10,
                    (i + 1) * 10,
                    "api",
                    &format!("session {i} summary"),
                )
                .unwrap();
            }
        }

        let mem = HDF5Memory::open(&path).unwrap();
        for i in 0..5 {
            let summary = mem
                .get_session_summary(&format!("sess-{i}"))
                .unwrap()
                .unwrap();
            assert_eq!(summary, format!("session {i} summary"));
        }
    }

    #[test]
    fn knowledge_graph_persistence() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let path = config.path.clone();

        {
            let mut mem = HDF5Memory::create(config).unwrap();
            let id1 = mem.add_entity("Alice", "person", -1).unwrap();
            let id2 = mem.add_entity("Bob", "person", -1).unwrap();
            mem.add_relation(id1, id2, "knows", 0.9).unwrap();
        }

        let mem = HDF5Memory::open(&path).unwrap();
        assert_eq!(mem.knowledge().entities.len(), 2);
        assert_eq!(mem.knowledge().relations.len(), 1);
        assert_eq!(mem.knowledge().get_entity(0).unwrap().name, "Alice");
        assert_eq!(mem.knowledge().get_entity(1).unwrap().name, "Bob");

        let rels = mem.knowledge().get_relations_from(0);
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].relation, "knows");
    }

    #[test]
    fn different_channels() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let path = config.path.clone();

        {
            let mut mem = HDF5Memory::create(config).unwrap();
            let e1 = MemoryEntry {
                chunk: "whatsapp msg".into(),
                embedding: vec![1.0, 0.0, 0.0, 0.0],
                source_channel: "whatsapp".into(),
                timestamp: 100.0,
                session_id: "s1".into(),
                tags: "chat".into(),
            };
            let e2 = MemoryEntry {
                chunk: "slack msg".into(),
                embedding: vec![0.0, 1.0, 0.0, 0.0],
                source_channel: "slack".into(),
                timestamp: 200.0,
                session_id: "s2".into(),
                tags: "work".into(),
            };
            mem.save_batch(vec![e1, e2]).unwrap();
        }

        let mem = HDF5Memory::open(&path).unwrap();
        assert_eq!(mem.count(), 2);
    }

    #[test]
    fn compact_then_reopen() {
        let dir = TempDir::new().unwrap();
        let mut config = make_config(&dir);
        config.compact_threshold = 0.0;
        let path = config.path.clone();

        {
            let mut mem = HDF5Memory::create(config).unwrap();
            mem.save(make_entry("keep", &[1.0, 0.0, 0.0, 0.0])).unwrap();
            mem.save(make_entry("delete me", &[0.0, 1.0, 0.0, 0.0]))
                .unwrap();
            mem.save(make_entry("also keep", &[0.0, 0.0, 1.0, 0.0]))
                .unwrap();

            mem.delete(1).unwrap();
            mem.compact().unwrap();
        }

        let mem = HDF5Memory::open(&path).unwrap();
        assert_eq!(mem.count(), 2);
        assert_eq!(mem.count_active(), 2);
    }

    #[test]
    fn config_preserved() {
        let dir = TempDir::new().unwrap();
        let mut config = make_config(&dir);
        config.embedder = "custom-embedder".into();
        config.chunk_size = 1024;
        config.overlap = 100;
        let path = config.path.clone();

        HDF5Memory::create(config).unwrap();

        let mem = HDF5Memory::open(&path).unwrap();
        assert_eq!(mem.config().embedder, "custom-embedder");
        assert_eq!(mem.config().chunk_size, 1024);
        assert_eq!(mem.config().overlap, 100);
    }

    #[test]
    fn large_batch() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let path = config.path.clone();

        {
            let mut mem = HDF5Memory::create(config).unwrap();
            let entries: Vec<MemoryEntry> = (0..100)
                .map(|i| MemoryEntry {
                    chunk: format!("chunk number {i} with some content"),
                    embedding: vec![i as f32, 0.0, 0.0, 0.0],
                    source_channel: "api".into(),
                    timestamp: i as f64 * 1000.0,
                    session_id: format!("batch-sess-{}", i / 10),
                    tags: format!("batch,item-{i}"),
                })
                .collect();
            mem.save_batch(entries).unwrap();
        }

        let mem = HDF5Memory::open(&path).unwrap();
        assert_eq!(mem.count(), 100);
        assert_eq!(mem.count_active(), 100);
    }

    #[test]
    fn auto_compact() {
        let dir = TempDir::new().unwrap();
        let mut config = make_config(&dir);
        config.compact_threshold = 0.4;
        let mut mem = HDF5Memory::create(config).unwrap();

        mem.save(make_entry("a", &[1.0, 0.0, 0.0, 0.0])).unwrap();
        mem.save(make_entry("b", &[0.0, 1.0, 0.0, 0.0])).unwrap();
        mem.save(make_entry("c", &[0.0, 0.0, 1.0, 0.0])).unwrap();

        mem.delete(0).unwrap();
        assert_eq!(mem.count(), 3);

        mem.delete(1).unwrap();
        assert_eq!(mem.count(), 1);
        assert_eq!(mem.count_active(), 1);
    }

    #[test]
    fn snapshot_to_file() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let mut mem = HDF5Memory::create(config).unwrap();
        mem.save(make_entry("snap", &[1.0, 2.0, 3.0, 4.0])).unwrap();

        let snap_path = dir.path().join("my_snapshot.h5");
        let result = mem.snapshot(&snap_path).unwrap();
        assert_eq!(result, snap_path);
        assert!(snap_path.exists());
    }

    #[test]
    fn entity_id_continuity() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let path = config.path.clone();

        {
            let mut mem = HDF5Memory::create(config).unwrap();
            mem.add_entity("First", "test", -1).unwrap();
            mem.add_entity("Second", "test", -1).unwrap();
        }

        let mut mem = HDF5Memory::open(&path).unwrap();
        let id3 = mem.add_entity("Third", "test", -1).unwrap();
        assert_eq!(id3, 2);
    }

    #[test]
    fn multiple_relations() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let mut mem = HDF5Memory::create(config).unwrap();

        let a = mem.add_entity("A", "node", -1).unwrap();
        let b = mem.add_entity("B", "node", -1).unwrap();
        let c = mem.add_entity("C", "node", -1).unwrap();

        mem.add_relation(a, b, "connects", 1.0).unwrap();
        mem.add_relation(a, c, "connects", 0.5).unwrap();
        mem.add_relation(b, c, "depends_on", 0.8).unwrap();

        assert_eq!(mem.knowledge().get_relations_from(a).len(), 2);
        assert_eq!(mem.knowledge().get_relations_from(b).len(), 1);
        assert_eq!(mem.knowledge().get_relations_to(c).len(), 2);
    }

    #[test]
    fn empty_strings() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let path = config.path.clone();

        {
            let mut mem = HDF5Memory::create(config).unwrap();
            let entry = MemoryEntry {
                chunk: "content".into(),
                embedding: vec![1.0, 0.0, 0.0, 0.0],
                source_channel: "".into(),
                timestamp: 0.0,
                session_id: "".into(),
                tags: "".into(),
            };
            mem.save(entry).unwrap();
        }

        let mem = HDF5Memory::open(&path).unwrap();
        assert_eq!(mem.count(), 1);
    }

    // ---------------------------------------------------------------
    // Hebbian activation & decay tests
    // ---------------------------------------------------------------

    #[test]
    fn test_hebbian_activation_boost() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let mut mem = HDF5Memory::create(config).unwrap();

        // Save 10 entries; entry 0 has embedding [1,0,0,0]
        for i in 0..10 {
            let emb = if i == 0 {
                vec![1.0, 0.0, 0.0, 0.0]
            } else {
                // orthogonal-ish embeddings
                vec![0.0, (i as f32).sin(), (i as f32).cos(), 0.0]
            };
            mem.save(make_entry(&format!("chunk {i}"), &emb)).unwrap();
        }

        // Search 5 times for a query that matches entry 0 best
        let query = vec![1.0, 0.0, 0.0, 0.0];
        for _ in 0..5 {
            let results = mem.hybrid_search(&query, "chunk", 1.0, 0.0, 3);
            assert!(!results.is_empty());
        }

        // Entry 0 should have a higher activation weight than all others
        let w0 = mem.cache.activation_weights[0];
        for i in 1..10 {
            assert!(
                w0 > mem.cache.activation_weights[i],
                "entry 0 weight ({w0}) should be > entry {i} weight ({})",
                mem.cache.activation_weights[i]
            );
        }
    }

    #[test]
    fn test_hebbian_decay() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let mut mem = HDF5Memory::create(config).unwrap();

        for i in 0..5 {
            mem.save(make_entry(
                &format!("decay {i}"),
                &[i as f32, 1.0, 0.0, 0.0],
            ))
            .unwrap();
        }

        // Call tick_session 100 times with no searches
        for _ in 0..100 {
            mem.tick_session().unwrap();
        }

        // All weights should approach 0 (< 0.2)
        for (i, &w) in mem.cache.activation_weights.iter().enumerate() {
            assert!(
                w < 0.2,
                "weight[{i}] = {w}, expected < 0.2 after 100 decay ticks"
            );
        }
    }

    #[test]
    fn test_hebbian_no_effect_at_default() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let mut mem = HDF5Memory::create(config).unwrap();

        mem.save(make_entry("alpha", &[1.0, 0.0, 0.0, 0.0]))
            .unwrap();
        mem.save(make_entry("beta", &[0.0, 1.0, 0.0, 0.0])).unwrap();
        mem.save(make_entry("gamma", &[0.5, 0.5, 0.0, 0.0]))
            .unwrap();

        // All weights should be 1.0 (default)
        for &w in &mem.cache.activation_weights {
            assert!(
                (w - 1.0).abs() < 1e-6,
                "default weight should be 1.0, got {w}"
            );
        }

        // Search: since sqrt(1.0) == 1.0, scores should be pure cosine
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = mem.hybrid_search(&query, "", 1.0, 0.0, 3);
        // Entry 0 should be best (perfect match)
        assert_eq!(results[0].index, 0);
        assert!((results[0].activation - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hebbian_persistence() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let path = config.path.clone();

        {
            let mut mem = HDF5Memory::create(config).unwrap();
            mem.save(make_entry("persist me", &[1.0, 0.0, 0.0, 0.0]))
                .unwrap();

            // Boost via search
            let query = vec![1.0, 0.0, 0.0, 0.0];
            mem.hybrid_search(&query, "", 1.0, 0.0, 1);
            let boosted_weight = mem.cache.activation_weights[0];
            assert!(
                boosted_weight > 1.0,
                "weight should be boosted after search"
            );
        }

        // Reopen and check weight persisted
        let mem = HDF5Memory::open(&path).unwrap();
        assert!(
            mem.cache.activation_weights[0] > 1.0,
            "persisted weight should be > 1.0, got {}",
            mem.cache.activation_weights[0]
        );
    }

    #[test]
    fn test_hebbian_compact_preserves_weights() {
        let dir = TempDir::new().unwrap();
        let mut config = make_config(&dir);
        config.compact_threshold = 0.0;
        let mut mem = HDF5Memory::create(config).unwrap();

        // Save 5 entries
        for i in 0..5 {
            mem.save(make_entry(
                &format!("compact {i}"),
                &[i as f32, 1.0, 0.0, 0.0],
            ))
            .unwrap();
        }

        // Manually set distinct weights
        mem.cache.activation_weights = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Delete entries 1 and 3
        mem.delete(1).unwrap();
        mem.delete(3).unwrap();
        mem.compact().unwrap();

        // Remaining: indices 0, 2, 4 -> weights 1.0, 3.0, 5.0
        assert_eq!(mem.cache.activation_weights.len(), 3);
        assert!((mem.cache.activation_weights[0] - 1.0).abs() < 1e-6);
        assert!((mem.cache.activation_weights[1] - 3.0).abs() < 1e-6);
        assert!((mem.cache.activation_weights[2] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_activation_in_search_result() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let mut mem = HDF5Memory::create(config).unwrap();

        mem.save(make_entry("search me", &[1.0, 0.0, 0.0, 0.0]))
            .unwrap();
        mem.save(make_entry("also me", &[0.0, 1.0, 0.0, 0.0]))
            .unwrap();

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = mem.hybrid_search(&query, "", 1.0, 0.0, 2);

        // Every result should have a populated activation field
        for r in &results {
            assert!(
                r.activation > 0.0,
                "activation should be > 0, got {}",
                r.activation
            );
        }
    }

    #[test]
    fn test_add_entity_alias_on_memory() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let mut mem = HDF5Memory::create(config).unwrap();

        let id = mem.add_entity("Henry", "person", -1).unwrap();
        mem.add_entity_alias("my son", id as i64).unwrap();

        let aliases = mem.knowledge().get_aliases(id as i64);
        assert_eq!(aliases.len(), 1);
        assert_eq!(aliases[0], "my son");
    }

    #[test]
    fn tombstone_fraction() {
        let dir = TempDir::new().unwrap();
        let mut config = make_config(&dir);
        config.compact_threshold = 0.0;
        let mut mem = HDF5Memory::create(config).unwrap();

        assert_eq!(mem.cache.tombstone_fraction(), 0.0);

        mem.save(make_entry("a", &[1.0, 0.0, 0.0, 0.0])).unwrap();
        mem.save(make_entry("b", &[0.0, 1.0, 0.0, 0.0])).unwrap();
        mem.save(make_entry("c", &[0.0, 0.0, 1.0, 0.0])).unwrap();
        mem.save(make_entry("d", &[0.0, 0.0, 0.0, 1.0])).unwrap();

        mem.delete(0).unwrap();
        assert!((mem.cache.tombstone_fraction() - 0.25).abs() < 0.01);

        mem.delete(1).unwrap();
        assert!((mem.cache.tombstone_fraction() - 0.50).abs() < 0.01);
    }
}

impl HDF5Memory {
    pub fn set_strategy(&mut self, s: Box<dyn MemoryStrategy>) {
        self.strategy = Some(s);
    }
    pub fn record(&mut self, exchange: Exchange) -> Result<StrategyOutput> {
        let strat = self.strategy.as_ref().ok_or_else(|| {
            MemoryError::Schema(
                "strategy not initialized: call set_strategy() before record()".to_owned(),
            )
        })?;
        let view = memory_strategy::CacheStoreView::new(&self.cache, &self.knowledge);
        let output = strat.evaluate(&exchange, &view);
        for e in &output.entries {
            self.cache.push(
                e.chunk.clone(),
                e.embedding.clone(),
                e.source_channel.clone(),
                e.timestamp,
                e.session_id.clone(),
                e.tags.clone(),
            );
        }
        for eu in &output.entity_updates {
            let id = self.knowledge.add_entity(&eu.name, &eu.entity_type, -1);
            for a in &eu.aliases {
                self.knowledge.add_alias(a, id as i64);
            }
        }
        if !output.entries.is_empty() || !output.entity_updates.is_empty() {
            self.flush()?;
        }
        Ok(output)
    }
}

impl HDF5Memory {
    pub fn tick_session(&mut self) -> Result<()> {
        let d = self.config.decay_factor;
        for w in self.cache.activation_weights.iter_mut() {
            *w *= d;
        }
        self.flush()?;
        if let Some(ref mut w) = self.wal {
            w.truncate()?;
        }
        Ok(())
    }

    /// Number of pending WAL entries (0 if WAL disabled).
    pub fn wal_pending_count(&self) -> usize {
        self.wal.as_ref().map_or(0, |w| w.pending_count() as usize)
    }

    /// Explicit WAL merge: flush .h5, truncate WAL.
    pub fn flush_wal(&mut self) -> Result<()> {
        self.flush()?;
        if let Some(ref mut w) = self.wal {
            w.truncate()?;
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Ephemeral tier integration
// ─────────────────────────────────────────────────────────────────────────────

impl HDF5Memory {
    /// Enable the ephemeral working memory tier with the given configuration.
    pub fn enable_ephemeral(&mut self, config: EphemeralConfig) {
        self.ephemeral = Some(EphemeralStore::new(config));
    }

    /// Return a shared reference to the ephemeral store, if enabled.
    pub fn ephemeral(&self) -> Option<&EphemeralStore> {
        self.ephemeral.as_ref()
    }

    /// Return a mutable reference to the ephemeral store, if enabled.
    pub fn ephemeral_mut(&mut self) -> Option<&mut EphemeralStore> {
        self.ephemeral.as_mut()
    }

    /// Promote frequently-accessed ephemeral entries into the persistent cache.
    ///
    /// Every entry whose `access_count >= min_access_count` is removed from the
    /// ephemeral store and written to the HDF5 cache, then the file is flushed.
    /// Returns the number of entries promoted.
    pub fn promote_ephemeral(&mut self, min_access_count: u32) -> Result<usize> {
        let candidates = match &self.ephemeral {
            None => return Ok(0),
            Some(s) => s.promotion_candidates(min_access_count),
        };

        if candidates.is_empty() {
            return Ok(0);
        }

        let dim = self.config.embedding_dim;
        let mut promoted = 0;

        for key in candidates {
            let entry = match self
                .ephemeral
                .as_mut()
                .and_then(|s| s.take_for_promotion(&key))
            {
                Some(e) => e,
                None => continue,
            };

            let chunk = entry
                .text
                .clone()
                .unwrap_or_else(|| String::from_utf8_lossy(&entry.value).into_owned());
            let embedding = entry.embedding.clone().unwrap_or_else(|| vec![0.0f32; dim]);

            self.cache.push(
                chunk,
                embedding,
                format!("ephemeral::{key}"),
                entry.created_at,
                String::new(),
                entry.tags.join(","),
            );
            promoted += 1;
        }

        if promoted > 0 {
            self.flush()?;
        }
        Ok(promoted)
    }

    /// Search both the persistent HDF5 tier and the ephemeral tier, returning
    /// the top `k` results sorted by score descending.
    ///
    /// Ephemeral results are boosted by a factor of 1.2 to surface recent
    /// in-context information above older persisted data.
    pub fn unified_search(
        &mut self,
        query_embedding: &[f32],
        query_text: &str,
        k: usize,
    ) -> Vec<SearchResult> {
        // Persistent tier.
        let persistent = self.hybrid_search(query_embedding, query_text, 0.7, 0.3, k);
        const EPHEMERAL_BOOST: f32 = 1.2;
        let mut results = persistent;

        if self.ephemeral.is_none() {
            return results;
        }

        let eph = self.ephemeral.as_mut().unwrap();

        // Collect (key, score) pairs from ephemeral — borrow ends before we
        // access entries again below.
        let eph_hits: Vec<(String, f32)> = if !query_embedding.is_empty() {
            eph.search_embedding(query_embedding, k)
        } else if !query_text.is_empty() {
            eph.search_text(query_text, k)
        } else {
            Vec::new()
        };

        for (key, score) in &eph_hits {
            if let Some(entry) = eph.get_entry(key) {
                let chunk = entry
                    .text
                    .clone()
                    .unwrap_or_else(|| String::from_utf8_lossy(&entry.value).into_owned());
                results.push(SearchResult {
                    score: score * EPHEMERAL_BOOST,
                    chunk,
                    index: usize::MAX,
                    timestamp: entry.created_at,
                    source_channel: format!("ephemeral::{key}"),
                    activation: 1.0,
                });
            }
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);
        results
    }
}
