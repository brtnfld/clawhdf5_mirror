# Changelog

## v2.1.0 (2026-06-03)

### New Features
- `clawhdf5-agent`: HNSW now backs the vector stage of `hybrid_search`. The
  `hnsw` feature is **on by default**, so semantic search uses the approximate
  `clawhdf5-ann` index instead of a linear cosine scan. The index mirrors the
  memory cache (node id == cache index) and self-heals — it rebuilds whenever it
  drifts from the cache length, so no mutation path can desync it. Non-indexable
  stores (no/zero-dim/mixed embeddings) and dimension-mismatched queries fall
  back to the exact linear scan. Disable with
  `--no-default-features --features float16` for exact search.
- `clawhdf5-ann`: HNSW is now a live, mutable index — added `insert`,
  `mark_deleted` (soft-delete bitset; deleted nodes are traversed for
  connectivity but never returned), `compact` (drops deleted vectors and
  renumbers survivors), and `new` (empty index). Serialization gains a format
  version tag (`HNSW_FORMAT_VERSION` = 2) and persists the deleted bitset;
  pre-existing v1 files still load.
- `clawhdf5-agent`: `hybrid::merge_vector_keyword` exposes the shared
  normalize-and-fuse step used by both the linear and HNSW vector paths.
- Expose `max_dimensions()` API on Dataset, MmapDataset, and LazyDataset
- NetCDF-4 unlimited dimension detection now works correctly
- Python bindings (`clawhdf5-py`) build and link on macOS with system Python

### Bug Fixes
- `clawhdf5-py`: upgrade PyO3 and numpy `0.23` → `0.28` so the bindings build on
  Python 3.14 (PyO3 0.23 capped at 3.13 and hard-failed `cargo build
  --workspace`). Updated for the removed `PyObject` alias (`Py<PyAny>`) and the
  `Python::allow_threads` → `Python::detach` rename.
- Fix GPU L2 distance test (squared vs actual L2 mismatch in test helper)
- Mark Android JNI functions as `unsafe` for Rust 2024 edition compliance
- Add `# Safety` documentation to all public unsafe extern functions
- Fix all clippy warnings: needless_range_loop, manual_strip, ptr_arg, etc.
- Rename `RelationType::from_str` to `from_label` to avoid trait confusion
- Isolate h5py interop tests with `#[ignore]` when h5py unavailable

### Code Quality
- Full rustfmt pass across workspace (61 files)
- Refine inner unsafe blocks for Rust 2024 edition style
- Zero clippy warnings, zero clippy errors across entire workspace
- 1,546 tests passing, 0 failures

## v2.0.0 (2026-03-19)

- Unified rustyhdf5 (11 crates) and edgehdf5 (4 crates) into a single workspace
- All crates renamed to clawhdf5-* prefix
- Version bumped to 2.0.0 across all crates
- Git dependencies replaced with in-workspace path dependencies
- Added `agent` feature flag to clawhdf5-agent

