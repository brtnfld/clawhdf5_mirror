# Changelog

## Unreleased

### Robustness
- `clawhdf5-format`: harden the readers added this cycle against malformed /
  hostile input — they parse untrusted bytes and must return errors, never
  panic, OOM, or recurse without bound. Fixed concrete vectors found by audit
  and locked in with adversarial tests:
  - **Paged Fixed Array**: `1 << max_nelmts_bits` shift overflow (a `u8` ≥ 64);
    element/page offset multiplications now checked; element count bounded by
    file size.
  - **H5S selection decoder**: `ALL`/`NONE` no longer claim 16 bytes they don't
    have; hyperslab `rank` capped at 32 (`H5S_MAX_RANK`) to stop a giant
    allocation; `iter_linear` coordinate/stride/product arithmetic is checked.
  - **VDS mapping parser**: no pre-allocation from the untrusted `nused`; all
    selection slicing is bounds-checked.
  - **scale-offset / N-Bit filters**: `1 << minbits` overflow at `minbits == 64`;
    N-Bit `bit_offset + precision` overflow; N-Bit type-tree recursion depth
    capped (no stack overflow from a crafted nested tree); element counts
    bounded by the chunk's expected decompressed size so a bogus count can't
    drive a huge allocation.
  - **Virtual Dataset assembly**: a virtual dataset whose source is itself
    virtual (a cycle) now errors instead of recursing into a stack overflow.

### New Features
- `clawhdf5-agent`: **compress fixed-length string datasets** (memory text
  chunks, session summaries, ids, tags, entity/relation names, …). These were
  always stored uncompressed with a "chunked compound not yet supported" note
  that was simply stale — chunked writes work for fixed-size string/compound
  datatypes like any other. `write_string_dataset` now chunks + deflates a
  string dataset once its payload reaches 4 KiB, so large, highly-redundant
  NullPad content shrinks substantially while tiny metadata stays contiguous
  (no chunk-overhead bloat).
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

### New Features
- `clawhdf5` / `clawhdf5-format`: read **external-file Virtual Datasets (VDS)**.
  The format layer gains `read_raw_data_full_with_resolver` and a
  `VdsSourceResolver` callback (`Fn(&str) -> Option<Vec<u8>>`) that maps a
  stored source file name to its bytes, so the pure-byte reader can pull in
  external sources without a filesystem of its own. The `clawhdf5` `File` API
  wires a default resolver that reads sibling source files relative to the
  opened file's directory, so `File::open(...).dataset(...).read_*()` now
  transparently assembles cross-file VDS. A source file the resolver cannot
  supply leaves its region at the fill value (matching HDF5); an external
  source with no resolver at all is a clean error. In-memory files
  (`File::from_bytes`) have no directory, so only same-file VDS resolves there.
- `clawhdf5-format`: assemble **same-file Virtual Datasets (VDS)** of any rank.
  Previously a virtual layout returned `UnsupportedVersion`. The reader now
  decodes the global-heap mapping block (reverse-engineered against HDF5 2.0:
  `version · nused · [source-file · source-dataset · source-selection ·
  virtual-selection]* · checksum`, including the block-version-1 same-file
  marker), decodes the `H5S` source/virtual dataspace **selections** (ALL,
  NONE, and version-3 regular hyperslabs), reads each same-file source dataset,
  and scatters its selected elements into the virtual buffer in row-major order
  (so multi-dimensional block mappings land correctly); unmapped regions are
  left at the zero fill value. External-file sources return a clean unsupported
  error. The previous `parse_vds_mappings` used a guessed layout that did not
  match real files and is replaced.

### Tests
- `clawhdf5-format`: regression test for **scale-offset float E-scale**
  datasets. The HDF5 library does not implement E-scale encoding — when asked
  for it (`cd_values[0] = 1`) it stores the chunk raw and sets the chunk filter
  mask to skip the filter — so these files read back verbatim purely by
  honoring the per-chunk filter mask. The test locks in that behavior against a
  fixture produced via the HDF5 low-level API; no E-scale decoder is needed.

### Bug Fixes
- `clawhdf5-format`: scope the per-file **chunk cache by dataset**. The shared
  `ChunkCache` built its chunk index once and reused it for every chunked
  dataset in the file, keyed only by chunk coordinate with no dataset
  discrimination. With a single chunked dataset per file this was latent; once a
  file holds two chunked datasets of different rank (e.g. a 1-D compressed
  string array and the 2-D embeddings matrix), the first dataset's index was
  reused for the second, panicking with an out-of-bounds chunk coordinate. The
  cache now rebinds (dropping its index, chunk-index map, layout, and
  decompressed slots) whenever the dataset being read changes, while still
  caching repeated/sequential access to the same dataset.
- `clawhdf5-format`: read **paged Fixed Array** chunk indexes. A filtered,
  fixed-dimension dataset with more than one data-block page (>1024 chunks by
  default) previously failed with "paged Fixed Array data blocks not yet
  supported". The reader now walks the page-init bitmap (MSB-first), skips
  uninitialized pages, and resolves each page's fixed full-size slot (including
  the short final page). Reverse-engineered and validated end-to-end against an
  HDF5 2.0 file.
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

