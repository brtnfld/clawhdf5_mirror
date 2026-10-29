//! OpenClaw Integration Layer.
//!
//! Bridge between OpenClaw agent gateway (Markdown + sqlite-vec) and the
//! clawhdf5 HDF5-backed memory backend.  Provides:
//!
//! - [`MemoryBackend`] — the trait OpenClaw implements against.
//! - [`ClawhdfBackend`] — concrete HDF5-backed implementation.
//! - [`MarkdownParser`] — splits Markdown into [`MarkdownSection`] records.
//! - [`MarkdownExporter`] — renders sections back to Markdown text.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{
    AgentMemory, HDF5Memory, MemoryConfig, MemoryEntry,
    confidence::{ConfidenceConfig, ScoredResult, reject_low_confidence},
    reranker::{ReRankConfig, RerankInput, rerank},
};

// ─────────────────────────────────────────────────────────────────────────────
// MemorySearchResult
// ─────────────────────────────────────────────────────────────────────────────

/// A single result returned from a memory search.
#[derive(Debug, Clone)]
pub struct MemorySearchResult {
    /// Text content of the matching memory record.
    pub text: String,
    /// Relevance score (higher = more relevant).
    pub score: f32,
    /// Source file / section path the record originated from.
    pub path: String,
    /// Optional line range `(start, end)` within the source file.
    pub line_range: Option<(usize, usize)>,
    /// Unix-epoch timestamp of the record, if available.
    pub timestamp: Option<f64>,
    /// Human-readable source description (channel / file type).
    pub source: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// BackendStats
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics reported by a [`MemoryBackend`].
#[derive(Debug, Clone)]
pub struct BackendStats {
    /// Total number of active (non-tombstoned) memory records.
    pub total_records: usize,
    /// Number of records that carry a non-empty embedding vector.
    pub total_embeddings: usize,
    /// Size of the backing store file in bytes (0 if unknown).
    pub file_size_bytes: u64,
    /// Distinct modalities present in the store (e.g. `["text"]`).
    pub modalities: Vec<String>,
    /// Unix-epoch timestamp of the most recently written record, if any.
    pub last_updated: Option<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// MemoryBackend trait
// ─────────────────────────────────────────────────────────────────────────────

/// Interface that OpenClaw uses to interact with a memory backend.
///
/// Implementors provide persistent storage, full-text + vector search,
/// Markdown ingestion / export, and statistics.
pub trait MemoryBackend {
    /// Search memory using combined text + embedding retrieval.
    ///
    /// Results are returned sorted by relevance descending.
    fn search(
        &mut self,
        query_text: &str,
        query_embedding: &[f32],
        k: usize,
    ) -> Vec<MemorySearchResult>;

    /// Retrieve raw text stored at `path`.
    ///
    /// * `from_line` — 0-based start line (inclusive).  `None` = beginning.
    /// * `num_lines` — maximum lines to return.  `None` = all.
    ///
    /// Returns `None` if no record exists for `path`.
    fn get(&self, path: &str, from_line: Option<usize>, num_lines: Option<usize>)
    -> Option<String>;

    /// Write raw text content at `path`, replacing any existing record.
    fn write(&mut self, path: &str, content: &str) -> Result<(), String>;

    /// Parse `content` as Markdown, split into sections, and ingest each
    /// section as a separate memory record tagged with `path`.
    ///
    /// Returns the number of sections ingested.
    fn ingest_markdown(&mut self, path: &str, content: &str) -> Result<usize, String>;

    /// Export all memory records associated with `path` as Markdown text.
    fn export_markdown(&self, path: &str) -> Result<String, String>;

    /// Return aggregate statistics for the backend.
    fn stats(&self) -> BackendStats;
}

// ─────────────────────────────────────────────────────────────────────────────
// MarkdownSection
// ─────────────────────────────────────────────────────────────────────────────

/// A logical section parsed from a Markdown document.
#[derive(Debug, Clone, PartialEq)]
pub struct MarkdownSection {
    /// Heading text without the leading `#` characters, if present.
    pub heading: Option<String>,
    /// Body text of the section (lines after the heading).
    pub content: String,
    /// 0-based index of the first line in the original document.
    pub line_start: usize,
    /// 0-based index of the last line in the original document (inclusive).
    pub line_end: usize,
    /// ATX heading level (1 = `#`, 2 = `##`, …, 0 = preamble with no heading).
    pub level: u8,
}

// ─────────────────────────────────────────────────────────────────────────────
// MarkdownParser
// ─────────────────────────────────────────────────────────────────────────────

/// Splits Markdown documents into [`MarkdownSection`] records.
pub struct MarkdownParser;

impl MarkdownParser {
    /// Parse a generic memory Markdown file.
    ///
    /// Splits on ATX headings (`#`, `##`, …).  Text before the first heading
    /// becomes a preamble section (`level = 0`, `heading = None`).  Empty
    /// preamble sections (no content) are discarded.
    pub fn parse_memory_md(content: &str) -> Vec<MarkdownSection> {
        Self::parse_sections(content, 1)
    }

    /// Parse a daily log Markdown file.
    ///
    /// Only `##`-level (and deeper) headings act as section boundaries.
    /// The `date` parameter is informational — callers should embed it in
    /// tags when ingesting the returned sections.
    pub fn parse_daily_log(content: &str, _date: &str) -> Vec<MarkdownSection> {
        Self::parse_sections(content, 2)
    }

    // ── private ──────────────────────────────────────────────────────────────

    /// Detect an ATX heading on `line`.
    ///
    /// Returns `Some((level, title))` where `level` is in 1..=6.
    fn heading_level(line: &str) -> Option<(u8, &str)> {
        if !line.starts_with('#') {
            return None;
        }
        let trimmed = line.trim_start_matches('#');
        let hashes = line.len() - trimmed.len();
        if hashes > 6 {
            return None;
        }
        // Must be followed by a space (or end-of-line for empty headings).
        if !trimmed.is_empty() && !trimmed.starts_with(' ') {
            return None;
        }
        Some((hashes as u8, trimmed.trim()))
    }

    /// Generic section splitter.
    ///
    /// `min_level` is the minimum ATX heading level that starts a new section.
    fn parse_sections(content: &str, min_level: u8) -> Vec<MarkdownSection> {
        let lines: Vec<&str> = content.lines().collect();
        let mut sections: Vec<MarkdownSection> = Vec::new();

        let mut current_heading: Option<String> = None;
        let mut current_level: u8 = 0;
        let mut current_start: usize = 0;
        let mut body_lines: Vec<&str> = Vec::new();

        for (i, &line) in lines.iter().enumerate() {
            if let Some((lvl, title)) = Self::heading_level(line)
                && lvl >= min_level
            {
                // Flush previous section.
                Self::flush_section(
                    current_heading.take(),
                    current_level,
                    current_start,
                    i.saturating_sub(1),
                    &body_lines,
                    &mut sections,
                );
                body_lines.clear();
                current_heading = Some(title.to_string());
                current_level = lvl;
                current_start = i;
                continue;
            }
            body_lines.push(line);
        }

        // Flush trailing section.
        let last_line = lines.len().saturating_sub(1);
        Self::flush_section(
            current_heading.take(),
            current_level,
            current_start,
            last_line,
            &body_lines,
            &mut sections,
        );

        sections
    }

    fn flush_section(
        heading: Option<String>,
        level: u8,
        start: usize,
        end: usize,
        body: &[&str],
        out: &mut Vec<MarkdownSection>,
    ) {
        // Build body by trimming trailing blank lines.
        let content_raw = body.join("\n");
        let content = content_raw.trim_end_matches('\n').to_string();

        // Discard empty preamble sections.
        if heading.is_none() && content.trim().is_empty() {
            return;
        }

        out.push(MarkdownSection {
            heading,
            content,
            line_start: start,
            line_end: end,
            level,
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MarkdownExporter
// ─────────────────────────────────────────────────────────────────────────────

/// Renders [`MarkdownSection`] slices back to Markdown text.
pub struct MarkdownExporter;

impl MarkdownExporter {
    /// Render `sections` as a Markdown string.
    ///
    /// Headings are reproduced using ATX syntax (`# …`); preamble sections
    /// (level 0, no heading) are emitted without a heading line.
    pub fn export_sections(sections: &[MarkdownSection]) -> String {
        let mut out = String::new();
        for (i, sec) in sections.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            if let Some(h) = &sec.heading {
                let hashes = "#".repeat(sec.level.max(1) as usize);
                out.push_str(&format!("{hashes} {h}\n"));
            }
            if !sec.content.is_empty() {
                out.push_str(&sec.content);
                out.push('\n');
            }
        }
        out
    }

    /// Render sections with optional metadata annotations.
    ///
    /// * `include_timestamps` — inject `<!-- timestamp: 0.0 -->` after each
    ///   heading (placeholder; sections do not carry timestamp data).
    /// * `include_sources` — inject `<!-- lines: start-end -->` after each
    ///   heading.
    pub fn export_with_metadata(
        sections: &[MarkdownSection],
        include_timestamps: bool,
        include_sources: bool,
    ) -> String {
        let mut out = String::new();
        for (i, sec) in sections.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            if let Some(h) = &sec.heading {
                let hashes = "#".repeat(sec.level.max(1) as usize);
                out.push_str(&format!("{hashes} {h}\n"));
            }
            if include_timestamps {
                out.push_str("<!-- timestamp: 0.0 -->\n");
            }
            if include_sources {
                out.push_str(&format!(
                    "<!-- lines: {}-{} -->\n",
                    sec.line_start, sec.line_end
                ));
            }
            if !sec.content.is_empty() {
                out.push_str(&sec.content);
                out.push('\n');
            }
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ClawhdfBackend
// ─────────────────────────────────────────────────────────────────────────────

/// HDF5-backed implementation of [`MemoryBackend`].
///
/// # Path mapping
///
/// OpenClaw addresses memories by file path (e.g. `"memory/user.md"`).
/// Internally every [`MemoryEntry`] stores the originating path as its
/// `source_channel`.  Section sub-paths are stored as
/// `"<path>::<heading>"`.
pub struct ClawhdfBackend {
    hdf5_path: PathBuf,
    memory: HDF5Memory,
    /// md_path → source_channel prefix (usually the path itself).
    path_map: HashMap<String, String>,
    rerank_config: ReRankConfig,
    confidence_config: ConfidenceConfig,
}

impl ClawhdfBackend {
    /// Create a new HDF5-backed memory store at `path`.
    ///
    /// `embedding_dim` must match the external embedder (e.g. `384` for
    /// `all-MiniLM-L6-v2`, `1536` for `text-embedding-3-small`).
    pub fn create(path: &Path, embedding_dim: usize) -> Result<Self, String> {
        let config = MemoryConfig::new(path.to_path_buf(), "openclaw", embedding_dim);
        let memory = HDF5Memory::create(config).map_err(|e| e.to_string())?;
        Ok(Self {
            hdf5_path: path.to_path_buf(),
            memory,
            path_map: HashMap::new(),
            rerank_config: ReRankConfig::default(),
            confidence_config: ConfidenceConfig::default(),
        })
    }

    /// Open an existing HDF5 memory store.
    ///
    /// Fail-open on a missing integrity root, like [`HDF5Memory::open`]; use
    /// [`open_strict`](Self::open_strict) for a store that must always be
    /// integrity-protected.
    pub fn open(path: &Path) -> Result<Self, String> {
        let memory = HDF5Memory::open(path).map_err(|e| e.to_string())?;
        Ok(Self::from_memory(path, memory))
    }

    /// Open an existing HDF5 memory store, failing closed when the file has
    /// no content Merkle root or its WAL sidecar is unauthenticated (see
    /// [`HDF5Memory::open_strict`]).
    ///
    /// This backend is the integration point other agents (ZeroClaw) use, and
    /// it previously had no strict constructor at all, so an integrator could
    /// not obtain the stripped-root and WAL-downgrade protection the rest of
    /// the crate offers (2026-10 red-team finding).
    pub fn open_strict(path: &Path) -> Result<Self, String> {
        let memory = HDF5Memory::open_strict(path).map_err(|e| e.to_string())?;
        Ok(Self::from_memory(path, memory))
    }

    fn from_memory(path: &Path, memory: HDF5Memory) -> Self {
        // Rebuild path_map from cached source channels.
        let mut path_map: HashMap<String, String> = HashMap::new();
        for ch in &memory.cache.source_channels {
            // Strip section suffix so the key is just the file path.
            let key = ch.split("::").next().unwrap_or(ch.as_str()).to_string();
            path_map.entry(key).or_insert_with(|| ch.clone());
        }

        Self {
            hdf5_path: path.to_path_buf(),
            memory,
            path_map,
            rerank_config: ReRankConfig::default(),
            confidence_config: ConfidenceConfig::default(),
        }
    }

    /// Open the store if it exists, otherwise create it.
    pub fn open_or_create(path: &Path, embedding_dim: usize) -> Result<Self, String> {
        if path.exists() {
            Self::open(path)
        } else {
            Self::create(path, embedding_dim)
        }
    }

    /// Override the default re-ranking configuration.
    pub fn with_rerank_config(mut self, config: ReRankConfig) -> Self {
        self.rerank_config = config;
        self
    }

    /// Override the default confidence-rejection configuration.
    pub fn with_confidence_config(mut self, config: ConfidenceConfig) -> Self {
        self.confidence_config = config;
        self
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    fn now_secs() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
    }

    /// Return all active entry indices whose `source_channel` starts with
    /// `path_prefix`.
    fn indices_for_path(&self, path_prefix: &str) -> Vec<usize> {
        self.memory
            .cache
            .source_channels
            .iter()
            .enumerate()
            .filter_map(|(i, ch)| {
                if self.memory.cache.tombstones[i] == 0
                    && (ch == path_prefix || ch.starts_with(&format!("{path_prefix}::")))
                {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    // ── Compaction & Consolidation hooks (7.6) ────────────────────────────

    /// Run a compaction cycle — called by OpenClaw during session compaction.
    ///
    /// Sequence:
    /// 1. `tick_session()` — apply Hebbian decay to all activation weights.
    /// 2. `compact()` — remove tombstoned entries, returning the count removed.
    /// 3. `flush_wal()` — merge any pending WAL entries to the .h5 file.
    ///
    /// Returns `(decayed, compacted, wal_flushed)`.
    pub fn run_compaction(&mut self) -> Result<(bool, usize, bool), String> {
        self.memory.tick_session().map_err(|e| e.to_string())?;
        let compacted = AgentMemory::compact(&mut self.memory).map_err(|e| e.to_string())?;
        self.memory.flush_wal().map_err(|e| e.to_string())?;
        Ok((true, compacted, true))
    }

    /// Run the hippocampal consolidation engine over the currently stored records.
    ///
    /// Snapshots active cache entries into a [`crate::consolidation::ConsolidationEngine`]
    /// and runs one consolidation cycle at time `now_secs` (Unix epoch seconds).
    /// Returns per-tier counts and eviction/promotion totals.
    pub fn run_consolidation(
        &mut self,
        now_secs: f64,
    ) -> Result<crate::consolidation::ConsolidationStats, String> {
        use crate::consolidation::{
            ConsolidationConfig, ConsolidationEngine, ConsolidationStats, MemoryRecord,
            MemorySource, MemoryTier,
        };

        let cache = &self.memory.cache;
        let mut engine = ConsolidationEngine {
            config: ConsolidationConfig::default(),
            records: Vec::with_capacity(cache.chunks.len()),
            next_id: 0,
            stats: ConsolidationStats::default(),
        };

        for i in 0..cache.chunks.len() {
            if cache.tombstones[i] != 0 {
                continue;
            }
            let record = MemoryRecord {
                id: i as u64,
                chunk: cache.chunks[i].clone(),
                embedding: cache.embeddings[i].clone(),
                tier: MemoryTier::Working,
                importance: cache.activation_weights[i],
                access_count: 0,
                last_accessed: now_secs,
                created_at: cache.timestamps[i],
                source: MemorySource::System,
            };
            engine.records.push(record);
            engine.next_id = engine.next_id.max(i as u64 + 1);
        }

        engine.consolidate(now_secs);
        Ok(engine.get_stats())
    }

    /// Apply one Hebbian decay tick to all activation weights and flush.
    ///
    /// Delegates to [`crate::HDF5Memory::tick_session`].
    pub fn tick_session(&mut self) -> Result<(), String> {
        self.memory.tick_session().map_err(|e| e.to_string())
    }

    /// Force a WAL merge: flush the .h5 file and truncate the WAL log.
    ///
    /// Delegates to [`crate::HDF5Memory::flush_wal`].
    pub fn flush_wal(&mut self) -> Result<(), String> {
        self.memory.flush_wal().map_err(|e| e.to_string())
    }

    /// Number of pending WAL entries (0 if WAL is disabled).
    pub fn wal_pending_count(&self) -> usize {
        self.memory.wal_pending_count()
    }

    /// Store a single [] directly in the HDF5 backend, bypassing
    /// Markdown parsing.
    ///
    ///  may be empty — if so the entry is stored without a vector
    /// and will only participate in text-based recall.  Returns the record
    /// index assigned by the store.
    ///
    /// Addresses the Node.js integration request in issue #10: callers can
    /// now supply pre-computed embeddings without going through the Markdown parser.
    pub fn save_entry(&mut self, entry: MemoryEntry) -> Result<usize, String> {
        AgentMemory::save(&mut self.memory, entry).map_err(|e| e.to_string())
    }

    /// Store a batch of [] records atomically.
    ///
    /// Returns one record index per input entry in the same order.
    pub fn save_batch_entries(&mut self, entries: Vec<MemoryEntry>) -> Result<Vec<usize>, String> {
        AgentMemory::save_batch(&mut self.memory, entries).map_err(|e| e.to_string())
    }
}

impl MemoryBackend for ClawhdfBackend {
    /// Search using hybrid vector + BM25 retrieval, then re-rank and
    /// confidence-filter.
    fn search(
        &mut self,
        query_text: &str,
        query_embedding: &[f32],
        k: usize,
    ) -> Vec<MemorySearchResult> {
        // 1. Hybrid retrieval (RRF-blended vector + BM25).
        let candidates = k.saturating_mul(3).max(10);
        let raw = self
            .memory
            .hybrid_search(query_embedding, query_text, 0.7, 0.3, candidates);

        if raw.is_empty() {
            return Vec::new();
        }

        let now = Self::now_secs();

        // 2. Re-rank using temporal recency, source authority, Hebbian weight.
        let rerank_inputs: Vec<RerankInput> = raw
            .iter()
            .map(|r| RerankInput {
                index: r.index,
                timestamp: r.timestamp,
                source_channel: r.source_channel.clone(),
                raw_activation: r.activation,
            })
            .collect();

        let reranked = rerank(&rerank_inputs, &self.rerank_config, now);

        // 3. Confidence rejection.
        let scored: Vec<ScoredResult> = reranked
            .iter()
            .map(|r| ScoredResult {
                index: r.index,
                score: r.combined_score,
            })
            .collect();

        let confident = reject_low_confidence(&scored, &self.confidence_config);

        // 4. Map back to MemorySearchResult; preserve raw text via index lookup.
        let raw_by_idx: HashMap<usize, &crate::SearchResult> =
            raw.iter().map(|r| (r.index, r)).collect();

        confident
            .into_iter()
            .take(k)
            .filter_map(|sr| {
                let r = raw_by_idx.get(&sr.index)?;
                let path = r.source_channel.clone();
                Some(MemorySearchResult {
                    text: r.chunk.clone(),
                    score: sr.score,
                    path: path.clone(),
                    line_range: None,
                    timestamp: Some(r.timestamp),
                    source: path,
                })
            })
            .collect()
    }

    fn get(
        &self,
        path: &str,
        from_line: Option<usize>,
        num_lines: Option<usize>,
    ) -> Option<String> {
        let indices = self.indices_for_path(path);
        if indices.is_empty() {
            return None;
        }

        // Join all chunks belonging to this path in insertion order.
        let combined: String = indices
            .iter()
            .map(|&i| self.memory.cache.chunks[i].as_str())
            .collect::<Vec<_>>()
            .join("\n");

        // Apply optional line-range filter.
        let lines: Vec<&str> = combined.lines().collect();
        let start = from_line.unwrap_or(0).min(lines.len());
        let slice = &lines[start..];
        let slice = if let Some(n) = num_lines {
            &slice[..n.min(slice.len())]
        } else {
            slice
        };

        if slice.is_empty() {
            None
        } else {
            Some(slice.join("\n"))
        }
    }

    fn write(&mut self, path: &str, content: &str) -> Result<(), String> {
        let now = Self::now_secs();
        let entry = MemoryEntry {
            chunk: content.to_string(),
            embedding: Vec::new(), // embedding supplied externally
            source_channel: path.to_string(),
            timestamp: now,
            session_id: "openclaw".to_string(),
            tags: "openclaw,write".to_string(),
        };
        self.memory.save(entry).map_err(|e| e.to_string())?;
        self.path_map
            .entry(path.to_string())
            .or_insert_with(|| path.to_string());
        Ok(())
    }

    fn ingest_markdown(&mut self, path: &str, content: &str) -> Result<usize, String> {
        let sections = MarkdownParser::parse_memory_md(content);
        let count = sections.len();
        if count == 0 {
            return Ok(0);
        }

        let now = Self::now_secs();

        for (i, sec) in sections.iter().enumerate() {
            let source_channel = match &sec.heading {
                Some(h) => format!("{path}::{h}"),
                None => path.to_string(),
            };

            let entry = MemoryEntry {
                chunk: sec.content.clone(),
                embedding: Vec::new(),
                source_channel: source_channel.clone(),
                timestamp: now,
                session_id: "openclaw".to_string(),
                tags: format!("openclaw,markdown,section-{i}"),
            };

            self.memory.save(entry).map_err(|e| e.to_string())?;
        }

        self.path_map
            .entry(path.to_string())
            .or_insert_with(|| path.to_string());
        Ok(count)
    }

    fn export_markdown(&self, path: &str) -> Result<String, String> {
        let indices = self.indices_for_path(path);
        if indices.is_empty() {
            return Err(format!("no records found for path: {path}"));
        }

        // Rebuild MarkdownSection values from the stored entries.
        let sections: Vec<MarkdownSection> = indices
            .iter()
            .map(|&i| {
                let ch = &self.memory.cache.source_channels[i];
                let (heading, level) = if let Some(sep_pos) = ch.find("::") {
                    let h = &ch[sep_pos + 2..];
                    if h.is_empty() {
                        (None, 0u8)
                    } else {
                        (Some(h.to_string()), 2u8)
                    }
                } else {
                    (None, 0u8)
                };
                MarkdownSection {
                    heading,
                    content: self.memory.cache.chunks[i].clone(),
                    line_start: 0,
                    line_end: 0,
                    level,
                }
            })
            .collect();

        Ok(MarkdownExporter::export_sections(&sections))
    }

    fn stats(&self) -> BackendStats {
        let cache = &self.memory.cache;

        let total_records = cache.count_active();

        let total_embeddings = cache
            .embeddings
            .iter()
            .enumerate()
            // A text-only memory is stored as the zero vector (the file format
            // has no "no embedding" state), so "has an embedding" means
            // "has a non-zero vector" -- which also stays correct after reload.
            .filter(|(i, emb)| cache.tombstones[*i] == 0 && emb.iter().any(|&x| x != 0.0))
            .count();

        let file_size_bytes = std::fs::metadata(&self.hdf5_path)
            .map(|m| m.len())
            .unwrap_or(0);

        let modalities = if total_records > 0 {
            vec!["text".to_string()]
        } else {
            Vec::new()
        };

        let last_updated = cache
            .timestamps
            .iter()
            .enumerate()
            .filter(|(i, _)| cache.tombstones[*i] == 0)
            .map(|(_, &ts)| ts)
            .fold(None::<f64>, |acc, ts| Some(acc.map_or(ts, |a| a.max(ts))));

        BackendStats {
            total_records,
            total_embeddings,
            file_size_bytes,
            modalities,
            last_updated,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ── MarkdownParser ────────────────────────────────────────────────────────

    #[test]
    fn parse_empty_content() {
        let sections = MarkdownParser::parse_memory_md("");
        assert!(
            sections.is_empty(),
            "empty input should produce no sections"
        );
    }

    #[test]
    fn parse_whitespace_only() {
        let sections = MarkdownParser::parse_memory_md("   \n\n  ");
        assert!(
            sections.is_empty(),
            "whitespace-only input should be discarded"
        );
    }

    #[test]
    fn parse_preamble_only() {
        let content = "Some text without any headings.\nAnother line.";
        let sections = MarkdownParser::parse_memory_md(content);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].heading, None);
        assert_eq!(sections[0].level, 0);
        assert!(sections[0].content.contains("Some text"));
    }

    #[test]
    fn parse_single_h1() {
        let content = "# My Title\n\nSome body text.";
        let sections = MarkdownParser::parse_memory_md(content);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].heading.as_deref(), Some("My Title"));
        assert_eq!(sections[0].level, 1);
        assert!(sections[0].content.contains("Some body text"));
    }

    #[test]
    fn parse_multiple_sections() {
        let content = "# First\n\nBody one.\n\n## Sub\n\nBody two.\n\n# Second\n\nBody three.";
        let sections = MarkdownParser::parse_memory_md(content);
        assert_eq!(sections.len(), 3);
        assert_eq!(sections[0].heading.as_deref(), Some("First"));
        assert_eq!(sections[0].level, 1);
        assert_eq!(sections[1].heading.as_deref(), Some("Sub"));
        assert_eq!(sections[1].level, 2);
        assert_eq!(sections[2].heading.as_deref(), Some("Second"));
        assert_eq!(sections[2].level, 1);
    }

    #[test]
    fn parse_section_line_numbers() {
        let content = "# Title\n\nLine 2.\nLine 3.\n\n## Sub\n\nLine 7.";
        let sections = MarkdownParser::parse_memory_md(content);
        assert_eq!(sections.len(), 2);
        // First section starts at line 0.
        assert_eq!(sections[0].line_start, 0);
        // Second section starts at line 5 (the "## Sub" line).
        assert_eq!(sections[1].line_start, 5);
    }

    #[test]
    fn parse_preamble_then_heading() {
        let content = "Preamble text.\n\n# Section One\n\nContent.";
        let sections = MarkdownParser::parse_memory_md(content);
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].heading, None);
        assert_eq!(sections[0].level, 0);
        assert!(sections[0].content.contains("Preamble"));
        assert_eq!(sections[1].heading.as_deref(), Some("Section One"));
    }

    #[test]
    fn parse_heading_levels_preserved() {
        let content = "# H1\n\n## H2\n\n### H3\n\n#### H4";
        let sections = MarkdownParser::parse_memory_md(content);
        assert_eq!(sections.len(), 4);
        assert_eq!(sections[0].level, 1);
        assert_eq!(sections[1].level, 2);
        assert_eq!(sections[2].level, 3);
        assert_eq!(sections[3].level, 4);
    }

    #[test]
    fn parse_daily_log_ignores_h1() {
        // H1 should NOT start a new section for daily logs.
        let content =
            "# Daily 2024-01-01\n\nIntro.\n\n## Morning\n\nTask 1.\n\n## Evening\n\nReview.";
        let sections = MarkdownParser::parse_daily_log(content, "2024-01-01");
        // H1 is treated as plain text; H2s split into sections.
        assert_eq!(sections.len(), 3, "expected preamble + 2 H2 sections");
        let headings: Vec<Option<&str>> = sections.iter().map(|s| s.heading.as_deref()).collect();
        assert!(headings.contains(&Some("Morning")));
        assert!(headings.contains(&Some("Evening")));
    }

    #[test]
    fn parse_daily_log_date_ignored() {
        // The date parameter must not panic.
        let content = "## Entry\n\nSome text.";
        let sections = MarkdownParser::parse_daily_log(content, "2024-03-19");
        assert_eq!(sections.len(), 1);
    }

    #[test]
    fn parse_hash_in_content_not_treated_as_heading() {
        // A line like "foo # not a heading" should not be parsed as a heading.
        let content = "# Real Heading\n\nThis has a # in the middle.";
        let sections = MarkdownParser::parse_memory_md(content);
        assert_eq!(sections.len(), 1);
        assert!(sections[0].content.contains("# in the middle"));
    }

    #[test]
    fn parse_empty_heading() {
        // "# " (with trailing space, no title) should still create a section.
        let content = "# \n\nSome content.";
        let sections = MarkdownParser::parse_memory_md(content);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].heading.as_deref(), Some(""));
    }

    // ── MarkdownExporter ─────────────────────────────────────────────────────

    fn make_sections() -> Vec<MarkdownSection> {
        vec![
            MarkdownSection {
                heading: Some("Section A".to_string()),
                content: "Content A.".to_string(),
                line_start: 0,
                line_end: 2,
                level: 2,
            },
            MarkdownSection {
                heading: Some("Section B".to_string()),
                content: "Content B.".to_string(),
                line_start: 3,
                line_end: 5,
                level: 2,
            },
        ]
    }

    #[test]
    fn export_sections_basic() {
        let sections = make_sections();
        let out = MarkdownExporter::export_sections(&sections);
        assert!(out.contains("## Section A"), "heading A missing");
        assert!(out.contains("Content A."), "content A missing");
        assert!(out.contains("## Section B"), "heading B missing");
        assert!(out.contains("Content B."), "content B missing");
    }

    #[test]
    fn export_sections_empty() {
        let out = MarkdownExporter::export_sections(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn export_sections_preamble_no_heading_line() {
        let sections = vec![MarkdownSection {
            heading: None,
            content: "Preamble text.".to_string(),
            line_start: 0,
            line_end: 1,
            level: 0,
        }];
        let out = MarkdownExporter::export_sections(&sections);
        assert!(
            !out.contains('#'),
            "preamble should not emit a heading line"
        );
        assert!(out.contains("Preamble text."));
    }

    #[test]
    fn export_sections_level_respected() {
        let sections = vec![MarkdownSection {
            heading: Some("Deep".to_string()),
            content: "x".to_string(),
            line_start: 0,
            line_end: 0,
            level: 3,
        }];
        let out = MarkdownExporter::export_sections(&sections);
        assert!(out.starts_with("### Deep"));
    }

    #[test]
    fn export_with_metadata_timestamps() {
        let sections = make_sections();
        let out = MarkdownExporter::export_with_metadata(&sections, true, false);
        assert!(out.contains("<!-- timestamp: 0.0 -->"));
        assert!(!out.contains("<!-- lines:"));
    }

    #[test]
    fn export_with_metadata_sources() {
        let sections = make_sections();
        let out = MarkdownExporter::export_with_metadata(&sections, false, true);
        assert!(out.contains("<!-- lines:"));
        assert!(!out.contains("<!-- timestamp:"));
    }

    #[test]
    fn export_with_metadata_both() {
        let sections = make_sections();
        let out = MarkdownExporter::export_with_metadata(&sections, true, true);
        assert!(out.contains("<!-- timestamp: 0.0 -->"));
        assert!(out.contains("<!-- lines:"));
    }

    #[test]
    fn export_roundtrip_via_parser() {
        let original = "# Alpha\n\nAlpha body.\n\n# Beta\n\nBeta body.";
        let sections = MarkdownParser::parse_memory_md(original);
        let exported = MarkdownExporter::export_sections(&sections);
        // Re-parse the exported text.
        let re_sections = MarkdownParser::parse_memory_md(&exported);
        assert_eq!(sections.len(), re_sections.len());
        for (a, b) in sections.iter().zip(re_sections.iter()) {
            assert_eq!(a.heading, b.heading);
            assert_eq!(a.level, b.level);
            // Content must be equivalent after trim.
            assert_eq!(a.content.trim(), b.content.trim());
        }
    }

    // ── ClawhdfBackend ────────────────────────────────────────────────────────

    fn make_backend(dir: &TempDir) -> ClawhdfBackend {
        let path = dir.path().join("openclaw_test.h5");
        ClawhdfBackend::create(&path, 4).expect("create backend")
    }

    #[test]
    fn backend_create_and_stats_empty() {
        let dir = TempDir::new().unwrap();
        let backend = make_backend(&dir);
        let stats = backend.stats();
        assert_eq!(stats.total_records, 0);
        assert_eq!(stats.total_embeddings, 0);
        assert!(stats.modalities.is_empty());
        assert!(stats.last_updated.is_none());
    }

    #[test]
    fn backend_write_and_get() {
        let dir = TempDir::new().unwrap();
        let mut backend = make_backend(&dir);

        backend.write("memory/notes.md", "Hello, memory!").unwrap();

        let result = backend.get("memory/notes.md", None, None);
        assert!(result.is_some());
        assert!(result.unwrap().contains("Hello, memory!"));
    }

    #[test]
    fn backend_get_nonexistent_returns_none() {
        let dir = TempDir::new().unwrap();
        let backend = make_backend(&dir);
        let result = backend.get("nonexistent/path.md", None, None);
        assert!(result.is_none());
    }

    #[test]
    fn backend_get_with_line_range() {
        let dir = TempDir::new().unwrap();
        let mut backend = make_backend(&dir);

        let content = "line0\nline1\nline2\nline3\nline4";
        backend.write("test.md", content).unwrap();

        // from_line=1, num_lines=2 → "line1\nline2"
        let result = backend.get("test.md", Some(1), Some(2)).unwrap();
        let lines: Vec<&str> = result.lines().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "line1");
        assert_eq!(lines[1], "line2");
    }

    #[test]
    fn backend_get_from_line_beyond_end() {
        let dir = TempDir::new().unwrap();
        let mut backend = make_backend(&dir);

        backend.write("test.md", "only one line").unwrap();

        let result = backend.get("test.md", Some(100), None);
        assert!(result.is_none());
    }

    #[test]
    fn backend_write_stats_updated() {
        let dir = TempDir::new().unwrap();
        let mut backend = make_backend(&dir);

        backend.write("a.md", "content A").unwrap();
        backend.write("b.md", "content B").unwrap();

        let stats = backend.stats();
        assert_eq!(stats.total_records, 2);
        assert_eq!(stats.total_embeddings, 0, "no embeddings provided");
        assert!(stats.modalities.contains(&"text".to_string()));
        assert!(stats.last_updated.is_some());
        assert!(stats.file_size_bytes > 0);
    }

    #[test]
    fn backend_ingest_markdown_basic() {
        let dir = TempDir::new().unwrap();
        let mut backend = make_backend(&dir);

        let md = "# Goals\n\nLearn Rust.\n\n# Projects\n\nBuild clawhdf5.";
        let count = backend.ingest_markdown("memory/user.md", md).unwrap();
        assert_eq!(count, 2, "two H1 sections");

        let stats = backend.stats();
        assert_eq!(stats.total_records, 2);
    }

    #[test]
    fn backend_ingest_markdown_empty() {
        let dir = TempDir::new().unwrap();
        let mut backend = make_backend(&dir);

        let count = backend.ingest_markdown("empty.md", "").unwrap();
        assert_eq!(count, 0);
        assert_eq!(backend.stats().total_records, 0);
    }

    #[test]
    fn backend_ingest_markdown_single_section_no_heading() {
        let dir = TempDir::new().unwrap();
        let mut backend = make_backend(&dir);

        let count = backend
            .ingest_markdown("notes.md", "Just a paragraph.\nNo heading here.")
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn backend_export_markdown_roundtrip() {
        let dir = TempDir::new().unwrap();
        let mut backend = make_backend(&dir);

        let md = "## Section One\n\nContent one.\n\n## Section Two\n\nContent two.";
        backend.ingest_markdown("docs/test.md", md).unwrap();

        let exported = backend.export_markdown("docs/test.md").unwrap();
        assert!(
            exported.contains("Section One"),
            "section one missing in export"
        );
        assert!(exported.contains("Content one."), "content one missing");
        assert!(
            exported.contains("Section Two"),
            "section two missing in export"
        );
        assert!(exported.contains("Content two."), "content two missing");
    }

    #[test]
    fn backend_export_markdown_nonexistent() {
        let dir = TempDir::new().unwrap();
        let backend = make_backend(&dir);

        let result = backend.export_markdown("missing.md");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no records found"));
    }

    #[test]
    fn backend_get_after_ingest() {
        let dir = TempDir::new().unwrap();
        let mut backend = make_backend(&dir);

        let md = "## Alpha\n\nAlpha content.\n\n## Beta\n\nBeta content.";
        backend.ingest_markdown("ctx.md", md).unwrap();

        // get by exact section path
        let alpha = backend.get("ctx.md::Alpha", None, None);
        assert!(alpha.is_some());
        assert!(alpha.unwrap().contains("Alpha content"));

        // get by base path returns both sections combined
        let all = backend.get("ctx.md", None, None);
        assert!(all.is_some());
    }

    #[test]
    fn backend_open_or_create_creates() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("new.h5");

        assert!(!path.exists());
        let backend = ClawhdfBackend::open_or_create(&path, 4).unwrap();
        assert_eq!(backend.stats().total_records, 0);
        assert!(path.exists());
    }

    #[test]
    fn backend_open_existing() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("persist.h5");

        {
            let mut b = ClawhdfBackend::create(&path, 4).unwrap();
            b.write("note.md", "persisted content").unwrap();
        }

        let b2 = ClawhdfBackend::open(&path).unwrap();
        let result = b2.get("note.md", None, None);
        assert!(result.is_some());
        assert!(result.unwrap().contains("persisted content"));
    }

    #[test]
    fn backend_search_empty_store() {
        let dir = TempDir::new().unwrap();
        let mut backend = make_backend(&dir);

        let results = backend.search("anything", &[1.0, 0.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn backend_search_returns_results() {
        let dir = TempDir::new().unwrap();
        let mut backend = ClawhdfBackend::create(dir.path().join("s.h5").as_path(), 4)
            .unwrap()
            .with_confidence_config(ConfidenceConfig {
                min_score: 0.0,
                min_gap: f32::INFINITY,
                max_results: 10,
            });

        backend
            .write("docs/a.md", "rust programming language")
            .unwrap();
        backend
            .write("docs/b.md", "python scripting tools")
            .unwrap();
        backend.write("docs/c.md", "rust async runtime").unwrap();

        // Query by text keyword only (empty embedding → zero vector similarity)
        let results = backend.search("rust", &[0.0; 4], 5);
        // With confidence threshold 0.0 and infinite gap, we expect some results.
        // Exact count depends on BM25 + reranker, but at least 1 should surface.
        assert!(
            !results.is_empty(),
            "expected at least one search result for 'rust'"
        );
    }

    #[test]
    fn backend_search_result_fields_populated() {
        let dir = TempDir::new().unwrap();
        let mut backend = ClawhdfBackend::create(dir.path().join("f.h5").as_path(), 4)
            .unwrap()
            .with_confidence_config(ConfidenceConfig {
                min_score: 0.0,
                min_gap: f32::INFINITY,
                max_results: 10,
            });

        backend.write("mem/x.md", "hello world greeting").unwrap();

        let results = backend.search("hello", &[0.0; 4], 3);
        if let Some(r) = results.first() {
            assert!(!r.text.is_empty());
            assert!(!r.path.is_empty());
            assert!(!r.source.is_empty());
            assert!(r.timestamp.is_some());
        }
    }

    #[test]
    fn backend_stats_file_size() {
        let dir = TempDir::new().unwrap();
        let mut backend = make_backend(&dir);

        backend.write("a.md", "some content here").unwrap();

        let stats = backend.stats();
        assert!(
            stats.file_size_bytes > 0,
            "file size should be > 0 after write"
        );
    }

    #[test]
    fn backend_multiple_paths_isolated() {
        let dir = TempDir::new().unwrap();
        let mut backend = make_backend(&dir);

        backend.write("user.md", "user specific info").unwrap();
        backend
            .write("project.md", "project specific info")
            .unwrap();

        let user = backend.get("user.md", None, None).unwrap();
        let project = backend.get("project.md", None, None).unwrap();

        assert!(user.contains("user specific"), "wrong content for user.md");
        assert!(
            project.contains("project specific"),
            "wrong content for project.md"
        );
        assert!(!user.contains("project"), "user.md leaked project content");
    }

    #[test]
    fn backend_ingest_multiple_files() {
        let dir = TempDir::new().unwrap();
        let mut backend = make_backend(&dir);

        backend
            .ingest_markdown("file1.md", "## A\n\nContent A.")
            .unwrap();
        backend
            .ingest_markdown("file2.md", "## B\n\nContent B.\n\n## C\n\nContent C.")
            .unwrap();

        assert_eq!(backend.stats().total_records, 3);

        let md2 = backend.export_markdown("file2.md").unwrap();
        assert!(md2.contains("Content B."));
        assert!(md2.contains("Content C."));
    }

    #[test]
    fn backend_open_or_create_opens_existing() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("exist.h5");

        {
            let mut b = ClawhdfBackend::create(&path, 4).unwrap();
            b.write("k.md", "known content").unwrap();
        }

        let b2 = ClawhdfBackend::open_or_create(&path, 4).unwrap();
        assert_eq!(b2.stats().total_records, 1);
    }

    #[test]
    fn markdown_parser_no_space_after_hash_not_heading() {
        // "#nospace" should NOT be treated as a heading.
        let content = "#nospace\n\nRegular paragraph.";
        let sections = MarkdownParser::parse_memory_md(content);
        assert_eq!(sections.len(), 1, "should be a single preamble section");
        assert_eq!(sections[0].heading, None);
        assert!(sections[0].content.contains("#nospace"));
    }

    #[test]
    fn markdown_exporter_level_zero_uses_level_one() {
        // level=0 sections should clamp to # when exported with a heading.
        let sections = vec![MarkdownSection {
            heading: Some("Title".to_string()),
            content: "body".to_string(),
            line_start: 0,
            line_end: 1,
            level: 0,
        }];
        let out = MarkdownExporter::export_sections(&sections);
        // level.max(1) == 1 → "#"
        assert!(out.starts_with("# Title"));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Ephemeral tier methods on ClawhdfBackend
// ─────────────────────────────────────────────────────────────────────────────

impl ClawhdfBackend {
    /// Enable the ephemeral (in-memory only) working memory tier.
    pub fn enable_ephemeral(&mut self, config: crate::ephemeral::EphemeralConfig) {
        self.memory.enable_ephemeral(config);
    }

    /// Store a text value in ephemeral memory.
    ///
    /// Returns an error string if the ephemeral tier has not been enabled.
    pub fn ephemeral_set(
        &mut self,
        key: &str,
        value: &str,
        ttl_secs: Option<f64>,
    ) -> Result<(), String> {
        match self.memory.ephemeral_mut() {
            Some(s) => {
                s.set_text(key, value, ttl_secs);
                Ok(())
            }
            None => Err("ephemeral tier not enabled".to_string()),
        }
    }

    /// Retrieve a text value from ephemeral memory.
    ///
    /// Returns `None` if the tier is disabled, the key is absent, or the
    /// entry has expired.
    pub fn ephemeral_get(&mut self, key: &str) -> Option<String> {
        self.memory
            .ephemeral_mut()?
            .get_text(key)
            .map(|s| s.to_string())
    }

    /// Delete a key from ephemeral memory.
    ///
    /// Returns `true` if the key existed and was removed.
    pub fn ephemeral_delete(&mut self, key: &str) -> bool {
        self.memory.ephemeral_mut().is_some_and(|s| s.delete(key))
    }

    /// Return a snapshot of ephemeral tier statistics, or `None` if the tier
    /// is not enabled.
    pub fn ephemeral_stats(&self) -> Option<crate::ephemeral::EphemeralStats> {
        self.memory.ephemeral().map(|s| s.stats())
    }

    /// Promote frequently-accessed ephemeral entries to persistent HDF5 storage.
    ///
    /// Entries with `access_count >= min_access_count` are moved from the
    /// ephemeral store into the persistent cache.  Returns the count promoted.
    pub fn promote_ephemeral(&mut self, min_access_count: u32) -> Result<usize, String> {
        self.memory
            .promote_ephemeral(min_access_count)
            .map_err(|e| e.to_string())
    }
}
