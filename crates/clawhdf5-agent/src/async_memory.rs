//! Async wrapper for [`HDF5Memory`] with background flush support.
//!
//! Provides non-blocking access to the synchronous HDF5Memory core by
//! offloading all I/O and CPU-bound work to `spawn_blocking`. Includes
//! a background flush task that:
//!
//! - **Batches saves** through an mpsc channel (avoids per-entry disk writes)
//! - **Auto-flushes** on a configurable interval (default 5s)
//! - **Threshold-flushes** when pending WAL entries exceed a limit
//! - **Tracks dirty state** to skip no-op flushes
//!
//! # Architecture
//!
//! ```text
//! ┌───────────────┐                     ┌──────────────┐
//! │ AsyncHDF5     │  spawn_blocking     │  HDF5Memory  │
//! │   Memory      │ ─────────────────── │  (sync core) │
//! └───────┬───────┘                     └──────────────┘
//!         │
//!         │ mpsc channel (saves + commands)
//!         ▼
//! ┌───────────────┐
//! │  Background   │  interval timer + threshold check
//! │  Writer Task  │  batch save → WAL → periodic .h5 merge
//! └───────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use clawhdf5_agent::async_memory::{AsyncHDF5Memory, AsyncConfig};
//!
//! let config = AsyncConfig {
//!     flush_interval: Duration::from_secs(10),
//!     flush_threshold: 100,
//! };
//! let mem = AsyncHDF5Memory::open_with(path, config).await?;
//! mem.save(entry).await?;           // buffered → background writer
//! mem.save_batch(entries).await?;    // also buffered
//! let results = mem.hybrid_search(emb, "query".into(), 0.7, 0.3, 5).await;
//! mem.shutdown().await?;             // final flush + stop
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::task::spawn_blocking;

use crate::memory_strategy::{Exchange, StrategyOutput};
use crate::{
    AgentMemory, HDF5Memory, MemoryConfig, MemoryEntry, MemoryError, Result, SearchResult,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the async background writer.
#[derive(Debug, Clone)]
pub struct AsyncConfig {
    /// How often the background task auto-flushes WAL → .h5.
    /// Set to `Duration::ZERO` to disable periodic flush (threshold-only).
    pub flush_interval: Duration,

    /// Flush WAL → .h5 when pending WAL entries reach this count.
    /// Set to `0` to disable threshold-based flush (interval-only).
    pub flush_threshold: usize,

    /// Channel capacity for buffered save commands.
    /// Higher = more batching, more memory. Default 256.
    pub channel_capacity: usize,
}

impl Default for AsyncConfig {
    fn default() -> Self {
        Self {
            flush_interval: Duration::from_secs(5),
            flush_threshold: 200,
            channel_capacity: 256,
        }
    }
}

// ---------------------------------------------------------------------------
// Background writer commands
// ---------------------------------------------------------------------------

enum WriteCmd {
    /// Buffer one or more entries for saving.
    Save {
        entries: Vec<MemoryEntry>,
        reply: oneshot::Sender<Result<Vec<usize>>>,
    },
    /// Flush WAL → .h5 now.
    FlushNow(oneshot::Sender<Result<()>>),
    /// Tick session (decay + flush).
    TickSession(oneshot::Sender<Result<()>>),
    /// Shut down the background task.
    Shutdown(oneshot::Sender<()>),
}

// ---------------------------------------------------------------------------
// AsyncHDF5Memory
// ---------------------------------------------------------------------------

/// Async wrapper around [`HDF5Memory`].
///
/// Saves are buffered through a channel and batched by the background
/// writer task. Reads/searches use `spawn_blocking` directly (they need
/// the latest state, so they acquire the lock and run immediately).
pub struct AsyncHDF5Memory {
    inner: Arc<Mutex<HDF5Memory>>,
    write_tx: mpsc::Sender<WriteCmd>,
}

impl AsyncHDF5Memory {
    // -- Construction -------------------------------------------------------

    /// Create a new HDF5 memory file with default async config.
    pub async fn create(config: MemoryConfig) -> Result<Self> {
        Self::create_with(config, AsyncConfig::default()).await
    }

    /// Create a new HDF5 memory file with custom async config.
    pub async fn create_with(config: MemoryConfig, async_config: AsyncConfig) -> Result<Self> {
        let mem = spawn_blocking(move || HDF5Memory::create(config))
            .await
            .map_err(join_err)??;
        Ok(Self::wrap_with(mem, async_config))
    }

    /// Open an existing HDF5 memory file with default async config.
    pub async fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with(path, AsyncConfig::default()).await
    }

    /// Open an existing HDF5 memory file with custom async config.
    pub async fn open_with(path: impl AsRef<Path>, async_config: AsyncConfig) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mem = spawn_blocking(move || HDF5Memory::open(&path))
            .await
            .map_err(join_err)??;
        Ok(Self::wrap_with(mem, async_config))
    }

    /// Open an existing HDF5 memory file, failing closed when it carries no
    /// content Merkle root or its WAL sidecar is unauthenticated (see
    /// [`HDF5Memory::open_strict`]). Added alongside the CLI's `--strict` so
    /// every public way to open a store can request the guarantee (2026-10
    /// red-team finding).
    pub async fn open_strict(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_strict_with(path, AsyncConfig::default()).await
    }

    /// [`open_strict`](Self::open_strict) with a custom async config.
    pub async fn open_strict_with(
        path: impl AsRef<Path>,
        async_config: AsyncConfig,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mem = spawn_blocking(move || HDF5Memory::open_strict(&path))
            .await
            .map_err(join_err)??;
        Ok(Self::wrap_with(mem, async_config))
    }

    /// Wrap a sync `HDF5Memory` with default async config.
    pub fn wrap(mem: HDF5Memory) -> Self {
        Self::wrap_with(mem, AsyncConfig::default())
    }

    /// Wrap a sync `HDF5Memory` with custom async config.
    pub fn wrap_with(mem: HDF5Memory, async_config: AsyncConfig) -> Self {
        let inner = Arc::new(Mutex::new(mem));
        let (write_tx, write_rx) = mpsc::channel(async_config.channel_capacity);
        let bg_inner = Arc::clone(&inner);
        tokio::spawn(background_writer(bg_inner, write_rx, async_config));
        Self { inner, write_tx }
    }

    // -- Buffered save ops --------------------------------------------------

    /// Save a single memory entry. Buffered through the background writer.
    ///
    /// Returns the entry index once the background writer has applied it
    /// to the in-memory cache (does NOT wait for .h5 flush).
    pub async fn save(&self, entry: MemoryEntry) -> Result<usize> {
        let (tx, rx) = oneshot::channel();
        self.write_tx
            .send(WriteCmd::Save {
                entries: vec![entry],
                reply: tx,
            })
            .await
            .map_err(|_| channel_gone())?;
        let indices = rx.await.map_err(|_| channel_gone())??;
        Ok(indices[0])
    }

    /// Save a batch of entries. Buffered through the background writer.
    pub async fn save_batch(&self, entries: Vec<MemoryEntry>) -> Result<Vec<usize>> {
        let (tx, rx) = oneshot::channel();
        self.write_tx
            .send(WriteCmd::Save { entries, reply: tx })
            .await
            .map_err(|_| channel_gone())?;
        rx.await.map_err(|_| channel_gone())?
    }

    // -- Direct mutating ops (not buffered — need immediate consistency) ----

    /// Delete a memory entry by index.
    pub async fn delete(&self, id: usize) -> Result<()> {
        let inner = Arc::clone(&self.inner);
        spawn_blocking(move || {
            let mut mem = inner.blocking_lock();
            mem.delete(id)
        })
        .await
        .map_err(join_err)?
    }

    /// Compact tombstoned entries.
    pub async fn compact(&self) -> Result<usize> {
        let inner = Arc::clone(&self.inner);
        spawn_blocking(move || {
            let mut mem = inner.blocking_lock();
            mem.compact()
        })
        .await
        .map_err(join_err)?
    }

    /// Record an exchange using the configured memory strategy.
    pub async fn record(&self, exchange: Exchange) -> Result<StrategyOutput> {
        let inner = Arc::clone(&self.inner);
        spawn_blocking(move || {
            let mut mem = inner.blocking_lock();
            mem.record(exchange)
        })
        .await
        .map_err(join_err)?
    }

    // -- Search ops ---------------------------------------------------------

    /// Hybrid vector + BM25 search.
    pub async fn hybrid_search(
        &self,
        query_embedding: Vec<f32>,
        query_text: String,
        vector_weight: f32,
        keyword_weight: f32,
        k: usize,
    ) -> Vec<SearchResult> {
        let inner = Arc::clone(&self.inner);
        spawn_blocking(move || {
            let mut mem = inner.blocking_lock();
            mem.hybrid_search(
                &query_embedding,
                &query_text,
                vector_weight,
                keyword_weight,
                k,
            )
        })
        .await
        .unwrap_or_default()
    }

    // -- Read ops -----------------------------------------------------------

    /// Number of entries (including tombstoned).
    pub async fn count(&self) -> usize {
        self.inner.lock().await.count()
    }

    /// Number of active (non-tombstoned) entries.
    pub async fn count_active(&self) -> usize {
        self.inner.lock().await.count_active()
    }

    /// Get a chunk by index.
    pub async fn get_chunk(&self, index: usize) -> Option<String> {
        self.inner.lock().await.get_chunk(index).map(String::from)
    }

    /// Number of pending WAL entries.
    pub async fn wal_pending_count(&self) -> usize {
        self.inner.lock().await.wal_pending_count()
    }

    /// Get a clone of the config.
    pub async fn config(&self) -> MemoryConfig {
        self.inner.lock().await.config().clone()
    }

    /// Generate AGENTS.md content.
    pub async fn generate_agents_md(&self) -> String {
        let inner = Arc::clone(&self.inner);
        spawn_blocking(move || {
            let mem = inner.blocking_lock();
            mem.generate_agents_md()
        })
        .await
        .unwrap_or_default()
    }

    // -- Session ops --------------------------------------------------------

    /// Add a session record.
    pub async fn add_session(
        &self,
        id: String,
        start: usize,
        end: usize,
        channel: String,
        summary: String,
    ) -> Result<()> {
        let inner = Arc::clone(&self.inner);
        spawn_blocking(move || {
            let mut mem = inner.blocking_lock();
            mem.add_session(&id, start, end, &channel, &summary)
        })
        .await
        .map_err(join_err)?
    }

    /// Get a session summary by ID.
    pub async fn get_session_summary(&self, session_id: String) -> Result<Option<String>> {
        let inner = Arc::clone(&self.inner);
        spawn_blocking(move || {
            let mem = inner.blocking_lock();
            mem.get_session_summary(&session_id)
        })
        .await
        .map_err(join_err)?
    }

    // -- Knowledge graph ops ------------------------------------------------

    /// Add an entity to the knowledge graph.
    pub async fn add_entity(
        &self,
        name: String,
        entity_type: String,
        embedding_idx: i64,
    ) -> Result<u64> {
        let inner = Arc::clone(&self.inner);
        spawn_blocking(move || {
            let mut mem = inner.blocking_lock();
            mem.add_entity(&name, &entity_type, embedding_idx)
        })
        .await
        .map_err(join_err)?
    }

    /// Add an entity alias.
    pub async fn add_entity_alias(&self, alias: String, entity_id: i64) -> Result<()> {
        let inner = Arc::clone(&self.inner);
        spawn_blocking(move || {
            let mut mem = inner.blocking_lock();
            mem.add_entity_alias(&alias, entity_id)
        })
        .await
        .map_err(join_err)?
    }

    /// Add a relation to the knowledge graph.
    pub async fn add_relation(
        &self,
        src: u64,
        tgt: u64,
        relation: String,
        weight: f32,
    ) -> Result<()> {
        let inner = Arc::clone(&self.inner);
        spawn_blocking(move || {
            let mut mem = inner.blocking_lock();
            mem.add_relation(src, tgt, &relation, weight)
        })
        .await
        .map_err(join_err)?
    }

    // -- Snapshot -----------------------------------------------------------

    /// Snapshot the memory file to a destination path.
    pub async fn snapshot(&self, dest: PathBuf) -> Result<PathBuf> {
        let inner = Arc::clone(&self.inner);
        spawn_blocking(move || {
            let mem = inner.blocking_lock();
            mem.snapshot(&dest)
        })
        .await
        .map_err(join_err)?
    }

    // -- Flush / lifecycle --------------------------------------------------

    /// Request an immediate WAL → .h5 flush.
    pub async fn flush(&self) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.write_tx
            .send(WriteCmd::FlushNow(tx))
            .await
            .map_err(|_| channel_gone())?;
        rx.await.map_err(|_| channel_gone())?
    }

    /// Tick the session (decay activations + flush).
    pub async fn tick_session(&self) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.write_tx
            .send(WriteCmd::TickSession(tx))
            .await
            .map_err(|_| channel_gone())?;
        rx.await.map_err(|_| channel_gone())?
    }

    /// Gracefully shut down the background writer.
    ///
    /// Performs a final flush before stopping. Call before drop to
    /// ensure all buffered data is persisted.
    pub async fn shutdown(&self) -> Result<()> {
        self.flush().await?;
        let (tx, rx) = oneshot::channel();
        let _ = self.write_tx.send(WriteCmd::Shutdown(tx)).await;
        let _ = rx.await;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Background writer task
// ---------------------------------------------------------------------------

/// The background writer loop. Handles:
/// 1. Batched saves from the channel
/// 2. Periodic auto-flush on a timer
/// 3. Threshold-based flush when WAL grows too large
async fn background_writer(
    inner: Arc<Mutex<HDF5Memory>>,
    mut rx: mpsc::Receiver<WriteCmd>,
    config: AsyncConfig,
) {
    let use_interval = config.flush_interval > Duration::ZERO;
    let use_threshold = config.flush_threshold > 0;

    // Dirty flag: true when cache has unsaved changes that haven't been
    // flushed to .h5 yet. Saves always go through WAL first, so data
    // is durable — this just tracks whether we need a full .h5 rewrite.
    let mut dirty = false;

    let mut interval = tokio::time::interval(if use_interval {
        config.flush_interval
    } else {
        // If disabled, set a very long interval so it never fires
        Duration::from_secs(86400)
    });
    // Don't fire immediately on creation
    interval.tick().await;

    loop {
        tokio::select! {
            // --- Channel commands ---
            cmd = rx.recv() => {
                match cmd {
                    Some(WriteCmd::Save { entries, reply }) => {
                        let mem = Arc::clone(&inner);
                        let result = spawn_blocking(move || {
                            let mut m = mem.blocking_lock();
                            let mut indices = Vec::with_capacity(entries.len());
                            for entry in entries {
                                // Push to cache + WAL only (no .h5 rewrite).
                                // We use the existing save() which handles
                                // WAL append + auto-merge at wal_max_entries.
                                match m.save(entry) {
                                    Ok(idx) => indices.push(idx),
                                    Err(e) => return Err(e),
                                }
                            }
                            Ok(indices)
                        })
                        .await
                        .unwrap_or_else(|e| Err(MemoryError::Io(std::io::Error::other(
                            e,
                        ))));
                        dirty = result.is_ok();
                        let _ = reply.send(result);

                        // Check threshold
                        if use_threshold && dirty {
                            let mem = Arc::clone(&inner);
                            let threshold = config.flush_threshold;
                            let pending = spawn_blocking(move || {
                                let m = mem.blocking_lock();
                                m.wal_pending_count()
                            }).await.unwrap_or(0);

                            if pending >= threshold {
                                let mem = Arc::clone(&inner);
                                let _ = spawn_blocking(move || {
                                    let mut m = mem.blocking_lock();
                                    m.flush_wal()
                                }).await;
                                dirty = false;
                            }
                        }
                    }

                    Some(WriteCmd::FlushNow(reply)) => {
                        if dirty {
                            let mem = Arc::clone(&inner);
                            let result = spawn_blocking(move || {
                                let mut m = mem.blocking_lock();
                                m.flush_wal()
                            })
                            .await
                            .unwrap_or_else(|e| Err(MemoryError::Io(std::io::Error::other(
                                e,
                            ))));
                            if result.is_ok() { dirty = false; }
                            let _ = reply.send(result);
                        } else {
                            let _ = reply.send(Ok(()));
                        }
                    }

                    Some(WriteCmd::TickSession(reply)) => {
                        let mem = Arc::clone(&inner);
                        let result = spawn_blocking(move || {
                            let mut m = mem.blocking_lock();
                            m.tick_session()
                        })
                        .await
                        .unwrap_or_else(|e| Err(MemoryError::Io(std::io::Error::other(
                            e,
                        ))));
                        if result.is_ok() { dirty = false; }
                        let _ = reply.send(result);
                    }

                    Some(WriteCmd::Shutdown(reply)) => {
                        let _ = reply.send(());
                        break;
                    }

                    None => break, // channel closed
                }
            }

            // --- Periodic auto-flush ---
            _ = interval.tick(), if use_interval && dirty => {
                let mem = Arc::clone(&inner);
                let ok = spawn_blocking(move || {
                    let mut m = mem.blocking_lock();
                    m.flush_wal()
                }).await;
                if ok.is_ok() { dirty = false; }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn join_err(e: tokio::task::JoinError) -> MemoryError {
    MemoryError::Io(std::io::Error::other(e))
}

fn channel_gone() -> MemoryError {
    MemoryError::Io(std::io::Error::new(
        std::io::ErrorKind::BrokenPipe,
        "background writer task gone",
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(dir: &tempfile::TempDir) -> MemoryConfig {
        let mut c = MemoryConfig::new(dir.path().join("async_test.h5"), "async-agent", 4);
        c.wal_enabled = true;
        c
    }

    fn fast_async_config() -> AsyncConfig {
        AsyncConfig {
            flush_interval: Duration::from_millis(100),
            flush_threshold: 50,
            channel_capacity: 64,
        }
    }

    fn make_entry(chunk: &str, embedding: &[f32]) -> MemoryEntry {
        MemoryEntry {
            chunk: chunk.to_string(),
            embedding: embedding.to_vec(),
            source_channel: "test".to_string(),
            timestamp: 1000000.0,
            session_id: "session-1".to_string(),
            tags: "".to_string(),
        }
    }

    #[tokio::test]
    async fn create_and_count() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = make_config(&dir);
        let mem = AsyncHDF5Memory::create_with(config, fast_async_config())
            .await
            .unwrap();
        assert_eq!(mem.count().await, 0);
        assert_eq!(mem.count_active().await, 0);
        mem.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn save_and_search() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = make_config(&dir);
        let mem = AsyncHDF5Memory::create_with(config, fast_async_config())
            .await
            .unwrap();

        let idx = mem
            .save(make_entry("hello async world", &[1.0, 0.0, 0.0, 0.0]))
            .await
            .unwrap();
        assert_eq!(idx, 0);
        assert_eq!(mem.count().await, 1);

        let results = mem
            .hybrid_search(vec![1.0, 0.0, 0.0, 0.0], String::new(), 1.0, 0.0, 5)
            .await;
        assert!(!results.is_empty());
        assert_eq!(results[0].index, 0);

        mem.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn save_batch_async() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = make_config(&dir);
        let mem = AsyncHDF5Memory::create_with(config, fast_async_config())
            .await
            .unwrap();

        let entries = vec![
            make_entry("a", &[1.0, 0.0, 0.0, 0.0]),
            make_entry("b", &[0.0, 1.0, 0.0, 0.0]),
            make_entry("c", &[0.0, 0.0, 1.0, 0.0]),
        ];
        let indices = mem.save_batch(entries).await.unwrap();
        assert_eq!(indices, vec![0, 1, 2]);
        assert_eq!(mem.count().await, 3);

        mem.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn delete_and_compact() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut config = make_config(&dir);
        config.compact_threshold = 0.0;
        let mem = AsyncHDF5Memory::create_with(config, fast_async_config())
            .await
            .unwrap();

        mem.save(make_entry("a", &[1.0, 0.0, 0.0, 0.0]))
            .await
            .unwrap();
        mem.save(make_entry("b", &[0.0, 1.0, 0.0, 0.0]))
            .await
            .unwrap();
        mem.save(make_entry("c", &[0.0, 0.0, 1.0, 0.0]))
            .await
            .unwrap();

        mem.delete(1).await.unwrap();
        assert_eq!(mem.count_active().await, 2);

        let removed = mem.compact().await.unwrap();
        assert_eq!(removed, 1);
        assert_eq!(mem.count().await, 2);

        mem.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn flush_and_reopen() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = make_config(&dir);
        let path = config.path.clone();

        {
            let mem = AsyncHDF5Memory::create_with(config, fast_async_config())
                .await
                .unwrap();
            mem.save(make_entry("persist me", &[1.0, 0.0, 0.0, 0.0]))
                .await
                .unwrap();
            mem.shutdown().await.unwrap();
        }

        let mem = AsyncHDF5Memory::open_with(&path, fast_async_config())
            .await
            .unwrap();
        assert_eq!(mem.count().await, 1);
        let chunk = mem.get_chunk(0).await;
        assert_eq!(chunk.as_deref(), Some("persist me"));
        mem.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn session_tracking_async() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = make_config(&dir);
        let mem = AsyncHDF5Memory::create_with(config, fast_async_config())
            .await
            .unwrap();

        mem.add_session(
            "s1".into(),
            0,
            5,
            "discord".into(),
            "talked about rust".into(),
        )
        .await
        .unwrap();

        let summary = mem.get_session_summary("s1".into()).await.unwrap();
        assert_eq!(summary.as_deref(), Some("talked about rust"));

        mem.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn knowledge_graph_async() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = make_config(&dir);
        let mem = AsyncHDF5Memory::create_with(config, fast_async_config())
            .await
            .unwrap();

        let id1 = mem
            .add_entity("Rust".into(), "language".into(), -1)
            .await
            .unwrap();
        let id2 = mem
            .add_entity("HDF5".into(), "format".into(), -1)
            .await
            .unwrap();
        mem.add_relation(id1, id2, "uses".into(), 1.0)
            .await
            .unwrap();
        mem.add_entity_alias("rustlang".into(), id1 as i64)
            .await
            .unwrap();

        mem.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn tick_session_async() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = make_config(&dir);
        let mem = AsyncHDF5Memory::create_with(config, fast_async_config())
            .await
            .unwrap();

        mem.save(make_entry("decay test", &[1.0, 0.0, 0.0, 0.0]))
            .await
            .unwrap();

        mem.tick_session().await.unwrap();
        mem.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn snapshot_async() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = make_config(&dir);
        let mem = AsyncHDF5Memory::create_with(config, fast_async_config())
            .await
            .unwrap();

        mem.save(make_entry("snap", &[1.0, 0.0, 0.0, 0.0]))
            .await
            .unwrap();

        let snap_dest = dir.path().join("snapshot.h5");
        let snap_path = mem.snapshot(snap_dest).await.unwrap();
        assert!(snap_path.exists());

        mem.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn concurrent_saves() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = make_config(&dir);
        let mem = Arc::new(
            AsyncHDF5Memory::create_with(config, fast_async_config())
                .await
                .unwrap(),
        );

        let mut handles = Vec::new();
        for i in 0..20 {
            let m = Arc::clone(&mem);
            handles.push(tokio::spawn(async move {
                m.save(make_entry(
                    &format!("concurrent-{i}"),
                    &[i as f32, 0.0, 0.0, 0.0],
                ))
                .await
                .unwrap()
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        assert_eq!(mem.count().await, 20);
        mem.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn periodic_auto_flush() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = make_config(&dir);
        let path = config.path.clone();

        let async_config = AsyncConfig {
            flush_interval: Duration::from_millis(50),
            flush_threshold: 0, // disable threshold
            channel_capacity: 64,
        };
        let mem = AsyncHDF5Memory::create_with(config, async_config)
            .await
            .unwrap();

        mem.save(make_entry("auto-flush", &[1.0, 0.0, 0.0, 0.0]))
            .await
            .unwrap();

        // Wait for the periodic flush to fire
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Verify data is on disk by reopening without explicit flush
        drop(mem);
        let mem2 = AsyncHDF5Memory::open(&path).await.unwrap();
        assert_eq!(mem2.count().await, 1);
        mem2.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn threshold_flush() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut config = make_config(&dir);
        config.wal_max_entries = 1000; // high so sync auto-merge doesn't trigger
        let path = config.path.clone();

        let async_config = AsyncConfig {
            flush_interval: Duration::ZERO, // disable periodic
            flush_threshold: 5,
            channel_capacity: 64,
        };
        let mem = AsyncHDF5Memory::create_with(config, async_config)
            .await
            .unwrap();

        // Save enough entries to cross the threshold
        for i in 0..6 {
            mem.save(make_entry(
                &format!("thresh-{i}"),
                &[i as f32, 0.0, 0.0, 0.0],
            ))
            .await
            .unwrap();
        }

        // Give background task a moment to process the threshold flush
        tokio::time::sleep(Duration::from_millis(50)).await;

        assert_eq!(mem.count().await, 6);
        mem.shutdown().await.unwrap();

        // Verify persistence
        let mem2 = AsyncHDF5Memory::open(&path).await.unwrap();
        assert_eq!(mem2.count().await, 6);
        mem2.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn dirty_flag_skips_noop_flush() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = make_config(&dir);
        let mem = AsyncHDF5Memory::create_with(config, fast_async_config())
            .await
            .unwrap();

        // Flush with nothing dirty — should be instant no-op
        mem.flush().await.unwrap();
        mem.flush().await.unwrap();
        mem.flush().await.unwrap();

        // Save something, flush, then flush again (second should be no-op)
        mem.save(make_entry("dirty", &[1.0, 0.0, 0.0, 0.0]))
            .await
            .unwrap();
        mem.flush().await.unwrap();
        mem.flush().await.unwrap(); // no-op

        mem.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn default_config_works() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = make_config(&dir);
        // Use default AsyncConfig (no _with variant)
        let mem = AsyncHDF5Memory::create(config).await.unwrap();
        mem.save(make_entry("default", &[1.0, 0.0, 0.0, 0.0]))
            .await
            .unwrap();
        assert_eq!(mem.count().await, 1);
        mem.shutdown().await.unwrap();
    }
}
