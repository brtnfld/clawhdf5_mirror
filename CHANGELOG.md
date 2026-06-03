# Changelog

## Unreleased

### New Features
- `clawhdf5-format`: decode the **scale-offset filter** (id 6) — both the
  integer variant (`H5Z_SO_INT`) and the floating-point **D-scale** variant
  (`H5Z_SO_FLOAT_DSCALE`). Handles signed/unsigned int sizes, f32/f64, negative
  minima, decimal scale factors and fill values; reverse-engineered against
  HDF5 2.0 and validated end-to-end. The float E-scale variant remains
  unsupported.
- `clawhdf5-format`: decode the **N-Bit filter** (id 5) — atomic, **compound**
  and **array** layouts (the full recursive type tree, nestable to any depth),
  previously unsupported. Signed and unsigned reduced-precision integers and
  float members all read end-to-end, validated against HDF5 2.0.

### Bug Fixes
- `clawhdf5-format`: read **array-typed datatypes** (e.g. an array-typed
  compound member) via `read_as_i32/i64/u64/f32/f64` — previously a
  `TypeMismatch`. The array is read as a flat sequence of its base elements
  (recursing for nested arrays), applying base-type precision rules.
- `clawhdf5-format`: **sign-extend reduced-precision fixed-point integers** on
  read. A signed integer whose datatype precision is smaller than its storage
  size is stored zero-filled, so e.g. a 16-bit-precision `-1` previously read as
  `65535`. The integer read paths now extract the precision field and
  sign-extend (full-width types are unchanged). Completes signed N-Bit reads and
  also fixes un-filtered reduced-precision integer datasets.
- `clawhdf5-format`: read datasets written by modern HDF5 (1.14+/2.0, i.e.
  `libver=latest`). Compound (class 6) and array (class 10) datatype **version 5**
  messages and data layout **version 5** messages were rejected as invalid; they
  reuse the v3/v4 binary structure, so they are now accepted. This unblocks
  reading compound types and — critically — every chunked/compressed dataset
  written by HDF5 2.0. Found by running the h5py interop tests against
  h5py 3.16 / HDF5 2.0.

### Performance
- `clawhdf5-format`: chunked writes now compress all chunks up front via
  `compress_all_chunks`, running across rayon threads under the `parallel`
  feature when there are more than 4 filtered chunks. On-disk layout is
  unchanged. Speeds up compressed embedding writes in `clawhdf5-agent` (which
  enables `parallel`).

### Documentation
- Fix stale package names across all 13 per-crate READMEs (`rustyhdf5-*` /
  `edgehdf5-*` → `clawhdf5-*`, usage versions → 2.1.0).
- Correct README workspace/test/crate stats and the CLAUDE.md CLI subcommand
  list; document the `hnsw` and format compression/checksum feature flags and
  the `entity_extract` / `async_memory` modules.

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

