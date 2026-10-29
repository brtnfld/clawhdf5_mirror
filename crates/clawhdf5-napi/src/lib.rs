//! Node.js native addon for clawhdf5-agent.
//!
//! Built with [napi-rs](https://napi.rs).  Exposes [`ClawhdfMemory`], a Node.js
//! class that wraps [`clawhdf5_agent::openclaw::ClawhdfBackend`] and the
//! hippocampal consolidation engine.
//!
//! # Build
//! ```
//! npm install -g @napi-rs/cli
//! cd packages/clawhdf5-node
//! napi build --platform --release
//! ```

#![deny(clippy::all)]
#![allow(clippy::too_many_arguments)]

use napi::bindgen_prelude::*;
use napi_derive::napi;

use clawhdf5_agent::openclaw::ClawhdfBackend;

// ─────────────────────────────────────────────────────────────────────────────
// Plain JS object types (returned by value from napi methods)
// ─────────────────────────────────────────────────────────────────────────────

/// A single result returned from a memory search.
#[napi(object)]
pub struct MemorySearchResult {
    /// Text content of the matching memory record.
    pub text: String,
    /// Relevance score (higher = more relevant). Uses f64 for JS number
    /// compatibility.
    pub score: f64,
    /// Source file / section path the record originated from.
    pub path: String,
    /// Optional `[start, end]` line range within the source file.
    pub line_range: Option<Vec<u32>>,
    /// Unix-epoch timestamp of the record, if available.
    pub timestamp: Option<f64>,
    /// Human-readable source description (channel / file type).
    pub source: String,
}

/// Aggregate statistics about the memory store.
#[napi(object)]
pub struct BackendStats {
    /// Number of active (non-deleted) records.
    pub total_records: u32,
    /// Number of records that have a non-empty embedding vector.
    pub total_embeddings: u32,
    /// On-disk size of the .h5 file in bytes.  Reported as f64 because JS
    /// does not have a native u64 (BigInt support would require an extra dep).
    pub file_size_bytes: f64,
    /// List of modalities present (e.g. `["text"]`).
    pub modalities: Vec<String>,
    /// Unix-epoch seconds of the most recently stored record, if any.
    pub last_updated: Option<f64>,
}

/// Statistics for the ephemeral (in-memory only) working memory tier.
#[napi(object)]
pub struct EphemeralStatsJs {
    /// Number of entries currently stored.
    pub total_entries: u32,
    /// Total bytes used by stored values.
    pub total_bytes: f64,
    /// Number of entries removed by TTL expiry in the last cleanup.
    pub expired_count: u32,
    /// Number of entries evicted for capacity since the store was created.
    pub evicted_count: u32,
    /// Age in seconds of the oldest entry.
    pub oldest_entry_age_secs: f64,
    /// Total successful get calls.
    pub hit_count: f64,
    /// Total failed get calls (absent or expired).
    pub miss_count: f64,
}

/// Per-tier counts and running totals from one consolidation cycle.
#[napi(object)]
pub struct ConsolidationStats {
    /// Number of records in the Working tier after consolidation.
    pub working_count: u32,
    /// Number of records in the Episodic tier after consolidation.
    pub episodic_count: u32,
    /// Number of records in the Semantic tier after consolidation.
    pub semantic_count: u32,
    /// Cumulative records evicted (capacity overflow, decay).
    pub total_evictions: f64,
    /// Cumulative records promoted (Working→Episodic, Episodic→Semantic).
    pub total_promotions: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// ClawhdfMemory class
// ─────────────────────────────────────────────────────────────────────────────

/// Node.js class wrapping the clawhdf5 HDF5-backed memory backend.
///
/// ```ts
/// import { ClawhdfMemory } from '@redclaw/clawhdf5';
///
/// const mem = ClawhdfMemory.openOrCreate('./agent.brain', 768);
/// mem.write('memory/user.md', '# Goals\n\nLearn Rust.');
/// const results = mem.search('Rust', new Float32Array(768), 5);
/// ```
/// Input accepted by [] and [].
///
/// Mirrors [] with JS-compatible types.
/// Addresses issue #10: Node.js callers can now write embedding-aware records
/// directly without shelling out to the CLI binary.
#[napi(object)]
pub struct MemoryEntryInput {
    /// The text chunk to store.
    pub chunk: String,
    /// Pre-computed embedding vector.  Length must match the store's
    /// , or pass  / empty array to store without a
    /// vector (text-only recall).
    pub embedding: Option<Vec<f64>>,
    /// Source identifier, e.g. , , .
    pub source_channel: String,
    /// Unix-epoch seconds.  Pass  to use the current wall-clock time.
    pub timestamp: Option<f64>,
    /// Session identifier (e.g. a UUID or a human-readable label).
    pub session_id: String,
    /// Comma-separated tags, e.g. .
    pub tags: String,
}

#[napi]
pub struct ClawhdfMemory {
    inner: ClawhdfBackend,
}

// SAFETY: Node.js executes JavaScript on a single thread.  The napi-rs runtime
// serialises all JS↔native calls through the event loop, so there is no
// concurrent access to `inner` from multiple threads.
unsafe impl Send for ClawhdfMemory {}

// Every exported method carries `catch_unwind`. napi-rs v2 wraps a call in
// `std::panic::catch_unwind` only when asked (napi-derive's `catch_unwind`
// flag defaults to off), and a panic that reaches the N-API callback aborts
// the Node process. Two such panics were reachable before the 2026-10
// red-team round (a wrong-dimension save followed by any search, and a
// wrong-dimension query); both are fixed at their source, and this turns the
// next one into a JavaScript exception instead of a crash.
#[napi]
impl ClawhdfMemory {
    // ── Lifecycle ────────────────────────────────────────────────────────────

    /// Create a new HDF5 memory store at `path`.
    ///
    /// `embeddingDim` must match the external embedder you plan to use
    /// (e.g. `384` for `all-MiniLM-L6-v2`, `768` for `nomic-embed-text`,
    /// `1536` for OpenAI `text-embedding-3-small`).
    #[napi(factory, catch_unwind)]
    pub fn create(path: String, embedding_dim: u32) -> napi::Result<ClawhdfMemory> {
        let backend = ClawhdfBackend::create(std::path::Path::new(&path), embedding_dim as usize)
            .map_err(napi::Error::from_reason)?;
        Ok(Self { inner: backend })
    }

    /// Open an existing HDF5 memory store.  Replays the WAL if present.
    #[napi(factory, catch_unwind)]
    pub fn open(path: String) -> napi::Result<ClawhdfMemory> {
        let backend =
            ClawhdfBackend::open(std::path::Path::new(&path)).map_err(napi::Error::from_reason)?;
        Ok(Self { inner: backend })
    }

    /// Open an existing HDF5 memory store, failing closed when it carries no
    /// content Merkle root or its WAL sidecar is unauthenticated. Use this
    /// for stores that must always be integrity-protected; `open` stays
    /// fail-open for backward compatibility (2026-10 red-team finding: the
    /// Node binding had no way to request the guarantee).
    #[napi(factory, catch_unwind)]
    pub fn open_strict(path: String) -> napi::Result<ClawhdfMemory> {
        let backend = ClawhdfBackend::open_strict(std::path::Path::new(&path))
            .map_err(napi::Error::from_reason)?;
        Ok(Self { inner: backend })
    }

    /// Open the store at `path` if it exists; otherwise create it.
    #[napi(factory, catch_unwind)]
    pub fn open_or_create(path: String, embedding_dim: u32) -> napi::Result<ClawhdfMemory> {
        let backend =
            ClawhdfBackend::open_or_create(std::path::Path::new(&path), embedding_dim as usize)
                .map_err(napi::Error::from_reason)?;
        Ok(Self { inner: backend })
    }

    // ── MemoryBackend methods ─────────────────────────────────────────────────

    /// Hybrid vector + BM25 search.
    ///
    /// `queryEmbedding` must have the same dimension the store was created with.
    /// Pass a zero-length `Float32Array` to skip vector similarity and rely on
    /// BM25 text matching only.
    #[napi(catch_unwind)]
    pub fn search(
        &mut self,
        query_text: String,
        query_embedding: Float32Array,
        k: u32,
    ) -> Vec<MemorySearchResult> {
        use clawhdf5_agent::openclaw::MemoryBackend;
        let raw = self
            .inner
            .search(&query_text, query_embedding.as_ref(), k as usize);
        raw.into_iter()
            .map(|r| MemorySearchResult {
                text: r.text,
                score: r.score as f64,
                path: r.path,
                line_range: r.line_range.map(|(s, e)| vec![s as u32, e as u32]),
                timestamp: r.timestamp,
                source: r.source,
            })
            .collect()
    }

    /// Retrieve content stored at `path`.
    ///
    /// `fromLine` and `numLines` apply a line-range filter (0-indexed).
    /// Returns `null` when no records exist for that path.
    #[napi(catch_unwind)]
    pub fn get(
        &self,
        path: String,
        from_line: Option<u32>,
        num_lines: Option<u32>,
    ) -> Option<String> {
        use clawhdf5_agent::openclaw::MemoryBackend;
        self.inner.get(
            &path,
            from_line.map(|n| n as usize),
            num_lines.map(|n| n as usize),
        )
    }

    /// Store raw `content` at `path`, overwriting any previous data for that path.
    #[napi(catch_unwind)]
    pub fn write(&mut self, path: String, content: String) -> napi::Result<()> {
        use clawhdf5_agent::openclaw::MemoryBackend;
        self.inner
            .write(&path, &content)
            .map_err(napi::Error::from_reason)
    }

    /// Parse `content` as Markdown, split into sections, and ingest each section.
    ///
    /// Returns the number of sections ingested.
    #[napi(catch_unwind)]
    pub fn ingest_markdown(&mut self, path: String, content: String) -> napi::Result<u32> {
        use clawhdf5_agent::openclaw::MemoryBackend;
        self.inner
            .ingest_markdown(&path, &content)
            .map(|n| n as u32)
            .map_err(napi::Error::from_reason)
    }

    /// Reconstruct stored sections for `path` back into a Markdown string.
    #[napi(catch_unwind)]
    pub fn export_markdown(&self, path: String) -> napi::Result<String> {
        use clawhdf5_agent::openclaw::MemoryBackend;
        self.inner
            .export_markdown(&path)
            .map_err(napi::Error::from_reason)
    }

    /// Return aggregate statistics about the memory store.
    #[napi(catch_unwind)]
    pub fn stats(&self) -> BackendStats {
        use clawhdf5_agent::openclaw::MemoryBackend;
        let s = self.inner.stats();
        BackendStats {
            total_records: s.total_records as u32,
            total_embeddings: s.total_embeddings as u32,
            file_size_bytes: s.file_size_bytes as f64,
            modalities: s.modalities,
            last_updated: s.last_updated,
        }
    }

    // ── Consolidation hooks (7.6) ─────────────────────────────────────────────

    /// Remove tombstoned entries and return the number of records compacted.
    #[napi(catch_unwind)]
    pub fn compact(&mut self) -> napi::Result<u32> {
        self.inner
            .run_compaction()
            .map(|(_, n, _)| n as u32)
            .map_err(napi::Error::from_reason)
    }

    /// Apply one Hebbian decay tick to all activation weights and flush to disk.
    #[napi(catch_unwind)]
    pub fn tick_session(&mut self) -> napi::Result<()> {
        self.inner.tick_session().map_err(napi::Error::from_reason)
    }

    /// Force a WAL merge: flush the .h5 file and truncate the WAL log.
    #[napi(catch_unwind)]
    pub fn flush_wal(&mut self) -> napi::Result<()> {
        self.inner.flush_wal().map_err(napi::Error::from_reason)
    }

    /// Run a full compaction cycle (decay + compact + WAL flush) and return
    /// stats.  Call this at the end of an agent session for best performance.
    #[napi(catch_unwind)]
    pub fn run_consolidation(&mut self, now_secs: f64) -> napi::Result<ConsolidationStats> {
        let s = self
            .inner
            .run_consolidation(now_secs)
            .map_err(napi::Error::from_reason)?;
        Ok(ConsolidationStats {
            working_count: s.working_count as u32,
            episodic_count: s.episodic_count as u32,
            semantic_count: s.semantic_count as u32,
            total_evictions: s.total_evictions as f64,
            total_promotions: s.total_promotions as f64,
        })
    }

    // ── Metadata ─────────────────────────────────────────────────────────────

    /// Number of pending WAL entries waiting to be merged (0 if WAL disabled).
    #[napi(catch_unwind)]
    pub fn wal_pending_count(&self) -> u32 {
        self.inner.wal_pending_count() as u32
    }

    // ── Ephemeral tier ────────────────────────────────────────────────────────

    /// Enable the ephemeral (in-memory only) working memory tier.
    ///
    /// `maxEntries` caps capacity before eviction (default 10 000).
    /// `defaultTtlSecs` sets the TTL applied when callers do not supply one
    /// (default 3600 s = 1 hour).
    #[napi(catch_unwind)]
    pub fn enable_ephemeral(&mut self, max_entries: Option<u32>, default_ttl_secs: Option<f64>) {
        use clawhdf5_agent::ephemeral::EphemeralConfig;
        let config = EphemeralConfig {
            max_entries: max_entries.unwrap_or(10_000) as usize,
            default_ttl_secs: default_ttl_secs.unwrap_or(3600.0),
            track_access: true,
        };
        self.inner.enable_ephemeral(config);
    }

    /// Store a text value in ephemeral memory (never persisted to disk).
    ///
    /// `ttlSecs` overrides the store-level default TTL; `null` uses the
    /// default configured when `enableEphemeral` was called.
    #[napi(catch_unwind)]
    pub fn ephemeral_set(&mut self, key: String, value: String, ttl_secs: Option<f64>) {
        let _ = self.inner.ephemeral_set(&key, &value, ttl_secs);
    }

    /// Retrieve a value from ephemeral memory.
    ///
    /// Returns `null` if the tier is disabled, the key is absent, or the
    /// entry has expired.
    #[napi(catch_unwind)]
    pub fn ephemeral_get(&mut self, key: String) -> Option<String> {
        self.inner.ephemeral_get(&key)
    }

    /// Delete a key from ephemeral memory.
    ///
    /// Returns `true` if the key existed and was removed.
    #[napi(catch_unwind)]
    pub fn ephemeral_delete(&mut self, key: String) -> bool {
        self.inner.ephemeral_delete(&key)
    }

    /// Return statistics for the ephemeral tier, or `null` if not enabled.
    #[napi(catch_unwind)]
    pub fn ephemeral_stats(&self) -> Option<EphemeralStatsJs> {
        self.inner.ephemeral_stats().map(|s| EphemeralStatsJs {
            total_entries: s.total_entries as u32,
            total_bytes: s.total_bytes as f64,
            expired_count: s.expired_count as u32,
            evicted_count: s.evicted_count as u32,
            oldest_entry_age_secs: s.oldest_entry_age_secs,
            hit_count: s.hit_count as f64,
            miss_count: s.miss_count as f64,
        })
    }

    /// Promote frequently-accessed ephemeral entries to persistent HDF5 storage.
    ///
    /// `minAccessCount` is the minimum number of times an entry must have been
    /// retrieved before it is eligible for promotion (default 3).
    /// Returns the number of entries promoted.
    #[napi(catch_unwind)]
    pub fn promote_ephemeral(&mut self, min_access_count: Option<u32>) -> napi::Result<u32> {
        self.inner
            .promote_ephemeral(min_access_count.unwrap_or(3))
            .map(|n| n as u32)
            .map_err(napi::Error::from_reason)
    }

    // ── Raw MemoryEntry write ─────────────────────────────────────────────────

    /// Store a single [] record directly in the HDF5 backend.
    ///
    /// Unlike , this method lets callers supply pre-computed
    /// embeddings and fine-grained metadata.  Addresses issue #10.
    ///
    /// Returns the record index assigned by the store.
    #[napi(catch_unwind)]
    pub fn save(&mut self, entry: MemoryEntryInput) -> napi::Result<u32> {
        use clawhdf5_agent::MemoryEntry;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        let me = MemoryEntry {
            chunk: entry.chunk,
            embedding: entry
                .embedding
                .unwrap_or_default()
                .into_iter()
                .map(|x| x as f32)
                .collect(),
            source_channel: entry.source_channel,
            timestamp: entry.timestamp.unwrap_or(now),
            session_id: entry.session_id,
            tags: entry.tags,
        };
        self.inner
            .save_entry(me)
            .map(|n| n as u32)
            .map_err(napi::Error::from_reason)
    }

    /// Store a batch of [] records in a single call.
    ///
    /// Returns one record index per entry in the same order as the input.
    #[napi(catch_unwind)]
    pub fn save_batch(&mut self, entries: Vec<MemoryEntryInput>) -> napi::Result<Vec<u32>> {
        use clawhdf5_agent::MemoryEntry;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        let mes: Vec<MemoryEntry> = entries
            .into_iter()
            .map(|e| MemoryEntry {
                chunk: e.chunk,
                embedding: e
                    .embedding
                    .unwrap_or_default()
                    .into_iter()
                    .map(|x| x as f32)
                    .collect(),
                source_channel: e.source_channel,
                timestamp: e.timestamp.unwrap_or(now),
                session_id: e.session_id,
                tags: e.tags,
            })
            .collect();
        self.inner
            .save_batch_entries(mes)
            .map(|v| v.into_iter().map(|n| n as u32).collect())
            .map_err(napi::Error::from_reason)
    }
}
