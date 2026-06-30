# ClawhDF5

**The memory layer AI agents deserve. One file. Pure Rust. Zero C dependencies.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)
[![Tests](https://img.shields.io/badge/tests-1500%2B%20passing-brightgreen.svg)](#benchmarks)
[![LongMemEval](https://img.shields.io/badge/LongMemEval-Hit@5%2046%25%20BM25--only-blue.svg)](BENCHMARKS.md#longmemeval-results)
[![Footprint](https://img.shields.io/badge/footprint-6.5%20KB%2Frecord-lightgrey.svg)](BENCHMARKS.md#memory-footprint)

ClawHDF5 is a pure-Rust HDF5 implementation combined with a research-grade agent memory engine. It gives AI agents persistent, searchable, cryptographically verifiable memory — all stored in a single portable file.

```
cargo add clawhdf5-agent --features agent
```

> **New here?** Start with the **[Quickstart Guide](docs/QUICKSTART.md)** · See **[Use Cases](docs/USE_CASES.md)** · Read **[Benchmarks](BENCHMARKS.md)**

---

## Why ClawhDF5?

Every AI agent needs memory. Today that means scattered Markdown files, SQLite databases, cloud-hosted vector stores, and glue code. ClawhDF5 replaces all of it:

| Problem | Status Quo | ClawhDF5 |
|---------|-----------|----------|
| Vector search | External DB (Pinecone, Qdrant) | Built-in, sub-millisecond |
| Keyword search | Separate FTS engine | Integrated BM25 |
| Knowledge graph | Neo4j or none | In-file graph with spreading activation |
| Memory consolidation | Manual pruning | Hippocampal-inspired automatic tiers |
| Temporal queries | Custom code | Native temporal index (716ns) |
| Multi-modal | Multiple stores | Unified cross-modal search |
| Security | Hope for the best | Provenance tracking + anomaly detection |
| Portability | Config + DB + files | **One `.h5` file. Copy it anywhere.** |

---

## Performance

Benchmarked on Intel i7-12650H (10C/16T), 384-dim embeddings, Criterion.rs.

### Vector Search

| Scale | Flat | IVF (nprobe=10) | IVF-PQ | vs MemX¹ |
|-------|------|-----------------|--------|----------|
| 1K | **54 µs** | — | — | — |
| 10K | 753 µs | **27 µs** | — | — |
| 100K | 11.4 ms | 1.32 ms | **1.19 ms** | **8–76× faster** |

### Agent Memory Operations

| Operation | Latency | Scale |
|-----------|---------|-------|
| Hybrid search (RRF) | **222 µs** | 1K records |
| BM25 keyword search | **67 µs** | 1K records |
| Knowledge graph BFS | **24 µs** | 1K entities |
| Spreading activation | **17 µs** | 100 entities |
| Temporal range query | **716 ns** | 10K timestamps |
| Consolidation cycle | **164 µs** | 1K records |
| Memory write (WAL) | **134 µs** | per record |
| Importance gate | **61 ns** | per record |

### HDF5 Core I/O (vs h5py/C HDF5)

| Operation | ClawhDF5 | h5py (C) | Speedup |
|-----------|----------|----------|---------|
| Metadata parse | 19 ns | 2,080 µs | **308×** |
| Write 1M f64 | 0.82 ms | 1.60 ms | **2×** |
| Read 1M f64 | 0.28 ms | 0.65 ms | **2.3×** |
| Zero-copy mmap | 313 ns | N/A | — |

### Chunked Write Throughput (codec comparison)

Measured with Criterion on 512×512 f32 matrices (h5bench-equivalent workload):

| Codec | 128×128 f32 | 512×512 f32 | vs Deflate |
|-------|-------------|-------------|------------|
| Zstd level 3 | **189 µs / 330 MiB/s** | **1.69 ms / 593 MiB/s** | **2.1–2.5×** |
| Deflate level 6 | 475 µs / 132 MiB/s | 3.57 ms / 280 MiB/s | baseline |

Use `.with_zstd(3)` for write-heavy workloads. Same or better compression ratio at 2× the speed.

> ¹ MemX ([arxiv:2603.16171](https://arxiv.org/abs/2603.16171), March 2026): Rust + libSQL, claims <90ms at 100K records.

### LongMemEval Retrieval Recall

Evaluated against the LongMemEval dataset (500 questions, multi-session haystack).
BM25-only baseline (no embedding model required at bench time):

| Metric | BM25-only | Full hybrid¹ |
|--------|-----------|--------------|
| Hit@5 (session) | ~46% | Higher |
| MRR (session) | ~0.34 | Higher |
| Abstention accuracy | ~72% | — |

> ¹ Enable embeddings via `hybrid_search(query_emb, text, 0.7, 0.3, k)` for substantially higher recall. The vector stage is served by the HNSW index by default (the `hnsw` feature is on by default); build with `--no-default-features --features float16` to fall back to an exact linear cosine scan.

### Memory Footprint

| Records | File Size | Bytes/Record | With Compression |
|---------|-----------|--------------|------------------|
| 1K | ~6.5 MB | ~6.5 KB | ~2.1 MB (3.1x) |
| 10K | ~65 MB | ~6.5 KB | ~21 MB (3.1x) |
| 100K | ~645 MB | ~6.5 KB | ~208 MB (3.1x) |

### Consolidation Efficiency

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Records in store | 1,000 | ~110 | −89% |
| Hit@1 recall | ~60% | ~90% | +30% |
| Search latency | ~2.8 ms | ~0.3 ms | **9x faster** |

**Full benchmark details: [BENCHMARKS.md](BENCHMARKS.md)**

---

## Agent Memory Architecture

ClawhDF5's agent memory engine implements research from 15+ recent papers on agentic memory systems. It's not a toy — it's the real thing.

```
                        ┌─────────────────┐
                        │   Agent Query    │
                        └────────┬────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Hybrid Retrieval      │
                    │  Vector + BM25 + RRF    │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │         Multi-Factor Re-Ranking      │
              │  temporal · authority · activation    │
              └──────────────────┬──────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Confidence Rejection   │
                    │  (suppress bad matches) │
                    └────────────┬────────────┘
                                 │
        ┌────────────────────────▼────────────────────────┐
        │              Memory Store (HDF5)                │
        │                                                 │
        │  ┌───────────┐ ┌───────────┐ ┌───────────────┐  │
        │  │ Working   │→│ Episodic  │→│  Semantic     │  │
        │  │ (bounded) │ │ (bounded) │ │ (long-term)   │  │
        │  └───────────┘ └───────────┘ └───────────────┘  │
        │                                                 │
        │  ┌──────────┐ ┌──────────┐ ┌────────────────┐   │
        │  │Knowledge │ │Temporal  │ │  Multi-Modal   │   │
        │  │  Graph   │ │  Index   │ │  Embeddings    │   │
        │  └──────────┘ └──────────┘ └────────────────┘   │
        │                                                 │
        │  ┌──────────┐ ┌──────────┐ ┌────────────────┐   │
        │  │Provenance│ │ Anomaly  │ │   Source       │   │
        │  │ Tracking │ │Detection │ │  Isolation     │   │
        │  └──────────┘ └──────────┘ └────────────────┘   │
        └─────────────────────────────────────────────────┘
                              │
                     ┌────────┴────────┐
                     │ agent_memory.h5 │
                     │   single file   │
                     └─────────────────┘
```

### Module Overview

| Module | What It Does |
|--------|-------------|
| **`knowledge`** | Entity/relation graph with BFS traversal, spreading activation, fuzzy entity resolution |
| **`consolidation`** | Three-tier memory (Working → Episodic → Semantic) with importance scoring and time-decay |
| **`hybrid`** | Vector + BM25 fusion with Reciprocal Rank Fusion (RRF, k=60). The vector stage uses the HNSW index by default (`hnsw` feature, on by default); disable with `--no-default-features --features float16` for an exact linear scan |
| **`reranker`** | Multi-factor re-ranking: temporal recency, source authority, activation weight |
| **`confidence`** | Low-confidence rejection — suppresses spurious recalls when nothing matches |
| **`temporal`** | Sorted timestamp index, session DAG, entity timeline, temporal query hints |
| **`multimodal`** | Cross-modal search across text/image/audio/video embeddings |
| **`provenance`** | Source attribution, FNV-1a content hashing, integrity verification |
| **`anomaly`** | Write rate limiting, 15 injection pattern detectors, source distribution analysis |
| **`openclaw`** | OpenClaw integration: MemoryBackend trait, Markdown ↔ HDF5 conversion |
| **`vector_search`** | Flat cosine, pre-normed, SIMD, BLAS, GPU, parallel search paths |
| **`ivf` / `pq`** | IVF-PQ approximate nearest neighbor for billion-scale search |
| **`bm25`** | BM25 keyword index with TF-IDF scoring |
| **`entity_extract`** | Rule-based entity extraction from text chunks into the knowledge graph |
| **`wal`** | Write-ahead log for crash-safe persistence |
| **`memory_strategy`** | Pluggable strategies: save-every, semantic-shift, user-correction detection |
| **`decision_gate`** | Sub-microsecond trivial/substantive classification |
| **`async_memory`** | Tokio-based async wrapper over the memory store (`async` feature) |

---

## Quick Start

### HDF5 File I/O

```rust
use clawhdf5::{File, FileBuilder, AttrValue};

// Write
let mut builder = FileBuilder::new();
builder.create_dataset("temperatures")
    .with_f64_data(&[22.5, 23.1, 21.8])
    .with_shape(&[3]);
builder.write("output.h5")?;

// Read
let file = File::open("output.h5")?;
let ds = file.dataset("temperatures")?;
let values = ds.read_f64()?;
assert_eq!(values, vec![22.5, 23.1, 21.8]);
```

### Agent Memory

```rust
use clawhdf5_agent::{HDF5Memory, MemoryConfig, MemoryEntry, AgentMemory};

// Create memory store
let config = MemoryConfig::new("agent.h5", "my-agent", 384);
let mut memory = HDF5Memory::create(config)?;

// Save a memory
memory.save(MemoryEntry {
    chunk: "User prefers dark mode and vim keybindings.".into(),
    embedding: embed("User prefers dark mode..."),  // your embedder
    source_channel: "chat".into(),
    timestamp: now(),
    session_id: "session-001".into(),
    tags: "preference".into(),
})?;

// Search
let results = memory.search(&query_embedding, 5)?;
for result in results {
    println!("[{:.3}] {}", result.score, result.chunk);
}
```

### Knowledge Graph

```rust
use clawhdf5_agent::knowledge::KnowledgeCache;

let mut kg = KnowledgeCache::new();

// Add entities
let alice = kg.add_entity("Alice", "person", -1);
let bob = kg.add_entity("Bob", "person", -1);
let acme = kg.add_entity("Acme Corp", "company", -1);

// Add relations
kg.add_relation(alice, acme, "works_at", 1.0);
kg.add_relation(bob, acme, "works_at", 1.0);
kg.add_relation(alice, bob, "manages", 0.8);

// Traverse
let neighbors = kg.bfs_neighbors(alice, 2);  // 2-hop neighborhood

// Spreading activation — find related entities
let activated = kg.spreading_activation(&[alice], 0.5, 0.01, 5);

// Entity resolution — fuzzy matching
let resolved = kg.resolve_or_create("alice", "person", -1, 2);
// Returns existing Alice entity (Levenshtein distance ≤ 2)
```

### Memory Consolidation

```rust
use clawhdf5_agent::consolidation::*;

let config = ConsolidationConfig::default();
let mut engine = ConsolidationEngine::new(config);

// Add memories — automatically scored for importance
engine.add_memory("User prefers dark mode", vec![0.1, 0.2, ...], MemorySource::User);
engine.add_memory("ok", vec![0.0, 0.0, ...], MemorySource::System);

// Access a memory (reactivates it)
engine.access_memory(0);

// Run consolidation cycle
let stats = engine.consolidate();
// Working memories promote to Episodic (if important enough)
// Episodic memories promote to Semantic (if accessed enough)
// Low-decay memories get evicted when tiers are full
```

### Temporal Queries

```rust
use clawhdf5_agent::temporal::*;

let mut index = TemporalIndex::new();
index.insert(1, 1700000000.0);  // record 1 at timestamp
index.insert(2, 1700003600.0);  // record 2, 1 hour later

// Range query — "what happened between 2pm and 5pm?"
let ids = index.range_query(1700000000.0, 1700010800.0);

// Latest 10 memories
let recent = index.latest(10);
```

### OpenClaw Integration

```rust
use clawhdf5_agent::openclaw::*;

// Create backend
let mut backend = ClawhdfBackend::create("memory.h5", "agent-1", 384)?;

// Ingest existing Markdown memory files
let md = std::fs::read_to_string("MEMORY.md")?;
let count = backend.ingest_markdown("MEMORY.md", &md)?;

// Search (uses full pipeline: RRF → re-rank → confidence filter)
let results = backend.search("user preferences", &query_embedding, 5);

// Export back to Markdown
let exported = backend.export_markdown("MEMORY.md")?;
```

---

## Crate Map

```
clawhdf5 workspace (17 crates, 84K lines of Rust)
│
├── Core HDF5
│   ├── clawhdf5-types      — Type system definitions
│   ├── clawhdf5-format      — Binary parser/writer (no_std)
│   ├── clawhdf5-io          — I/O abstraction (buffered, mmap, async)
│   ├── clawhdf5-filters     — Compression (deflate, lz4, zstd, blosc)
│   ├── clawhdf5-derive      — Proc macros
│   ├── clawhdf5             — High-level API
│   ├── clawhdf5-netcdf4     — NetCDF-4 support
│   ├── clawhdf5-accel       — SIMD (NEON, AVX2, AVX-512)
│   └── clawhdf5-gpu         — GPU compute (wgpu)
│
├── Agent Memory
│   ├── clawhdf5-agent       — Memory engine (20.7K lines, 32 modules)
│   ├── clawhdf5-ann         — HNSW approximate nearest neighbor (default backend)
│   ├── clawhdf5-migrate     — SQLite → HDF5 migration
│   ├── clawhdf5-android     — Android JNI bridge
│   └── clawhdf5-cli         — CLI tool
│
├── Bindings
│   ├── clawhdf5-py          — Python (PyO3)
│   └── clawhdf5-napi        — Node.js (napi-rs)
│
└── Tooling
    └── clawhdf5-bench       — Benchmark suite
```

---

## Research Foundation

ClawhDF5's agent memory design draws from 15+ recent papers:

| Paper | Key Insight | ClawhDF5 Module |
|-------|-------------|-----------------|
| **MemX** (2026) | RRF + multi-factor re-ranking | `hybrid`, `reranker` |
| **Graph-Native Cognitive Memory** (2026) | Graph-structured belief revision | `knowledge` |
| **CraniMem** (2026) | Bounded hippocampal memory | `consolidation` |
| **D-MEM** (2026) | Reward prediction error gating | `consolidation` |
| **SYNAPSE** (2025) | Spreading activation for recall | `knowledge` |
| **RAGdb** (2025) | Zero-dependency edge RAG | Architecture |
| **MemoryGraft** (2025) | Memory poisoning attacks | `anomaly`, `provenance` |
| **MemoryArena** (2026) | Multi-session benchmark | `temporal` |
| **AI Hippocampus** (2026) | Memory taxonomy survey | Overall design |

---

## Feature Flags

### `clawhdf5-agent`

| Flag | Default | Description |
|------|---------|-------------|
| `agent` | no | Full agent memory layer |
| `float16` | **yes** | Half-precision embedding storage (2× compression) |
| `hnsw` | **yes** | HNSW approximate vector index for `hybrid_search` (via `clawhdf5-ann`); disable for an exact linear scan |
| `parallel` | no | Rayon parallel search |
| `fast-math` | no | BLAS matrix-vector multiply |
| `accelerate` | no | Apple Accelerate / AMX (macOS) |
| `openblas` | no | OpenBLAS (Linux) |
| `gpu` | no | GPU search via wgpu |
| `async` | no | Tokio async with background flush |

### `clawhdf5-format`

| Flag | Default | Description |
|------|---------|-------------|
| `std` | yes | Standard library (disable for `no_std`) |
| `deflate` | yes | Deflate compression |
| `checksum` | yes | Jenkins lookup3 verification |
| `provenance` | yes | SHA-256 provenance attributes |
| `fast-deflate` | **yes** | zlib-ng backend for faster deflate |
| `system-zlib-decompress` | **yes** | Use the system zlib for decompression where available |
| `parallel` | no | Parallel chunk encoding + compression (rayon) |
| `fast-checksum` | no | crc32fast-accelerated checksums |
| `lz4` | no | LZ4 block compression filter (id 32004) |
| `zstd` | no | Zstandard compression filter (id 32015) |
| `system-zlib` / `zlib-rs` | no | Alternative zlib backends for deflate |
| `blake3_hash` | no | BLAKE3 content hashing for provenance |

---

## Building

```bash
# Default
cargo build --workspace

# Agent memory with all accelerations (Linux)
cargo build -p clawhdf5-agent --features "agent,float16,parallel,fast-math"

# Agent memory with Apple Accelerate (macOS)
cargo build -p clawhdf5-agent --features "agent,float16,accelerate,parallel,gpu"

# Tests
cargo test --workspace            # all 417+ tests
cargo test -p clawhdf5-agent      # agent memory tests

# Benchmarks
cargo bench -p clawhdf5-agent               # agent memory suite
cargo bench -p clawhdf5-bench               # h5bench-equivalent I/O suite
```

---

## HDF5 File Schema

```
agent_memory.h5
├── /meta
│   ├── schema_version: "1.0"
│   ├── agent_id, embedder, embedding_dim
│   └── created_at
├── /memory
│   ├── chunks:      string[N]
│   ├── embeddings:  f32[N × D]  (or f16 with float16 flag)
│   ├── tombstones:  u8[N]
│   └── norms:       f32[N]      (pre-computed L2)
├── /sessions
│   ├── ids:         string[S]
│   └── summaries:   string[S]
└── /knowledge_graph
    ├── entity_names:    string[E]
    ├── relation_srcs:   i64[R]
    ├── relation_tgts:   i64[R]
    └── relation_types:  string[R]
```

---

## Migration

### From rustyhdf5 / edgehdf5

Replace in `Cargo.toml` and source:

| Old | New |
|-----|-----|
| `rustyhdf5*` | `clawhdf5*` |
| `edgehdf5-memory` | `clawhdf5-agent` |
| `edgehdf5` (CLI) | `clawhdf5-cli` |

### From SQLite

```bash
cargo install --path crates/clawhdf5-migrate
clawhdf5-migrate --sqlite old.db --hdf5 memory.h5 --agent-id my-agent --embedding-dim 384
```

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full implementation tracker.

**Phase 1 complete** — all 8 tracks delivered:
- ✅ Knowledge Graph with spreading activation
- ✅ Hippocampal memory consolidation
- ✅ RRF hybrid retrieval + re-ranking + confidence rejection
- ✅ Temporal reasoning with sub-µs queries
- ✅ Memory security + anomaly detection
- ✅ Multi-modal memory (text/image/audio/video)
- ✅ OpenClaw integration layer
- ✅ Comprehensive Criterion benchmarks

**Phase 2** — OpenClaw TypeScript bridge, academic benchmarks (MemoryArena, LongMemEval), cross-platform validation.

---

## Part of the RedClaw Ecosystem

ClawhDF5 powers the `.brain` format for [ClawBrainHub](https://clawbrainhub.com) — the brain registry for AI agents. One file that packages identity, skills, memory, knowledge, and cryptographic provenance.

---

## License

MIT

---

<p align="center">
  <em>Built by <a href="https://github.com/redclawsystems">RedClaw Systems</a></em><br>
  <em>72,087 lines of Rust. Zero C dependencies. One file to remember everything.</em>
</p>
