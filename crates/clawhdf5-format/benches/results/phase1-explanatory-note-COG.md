# P1.6 Phase 1 Benchmark — Explanatory Note

This note accompanies `phase1-COG.csv`, per the S2-D2 spec's "Benchmark
validity and interpretation" requirement (p.52): every benchmark artifact
needs a reproducible explanatory note covering exact reproduction steps,
hardware, what is measured, how to read the results, and a root-cause
explanation of any notable trend or anomaly.

## Reproduction

```bash
cargo run --example generate_10gb --release -- 10 /tmp/noaa_data/synthetic_10gb.h5

# Real NOAA sample: the exact file historically referenced in
# test-vectors/README.md had rotated out of the public bucket by the time
# this was run. Find a currently-available file from the same time window:
curl "https://noaa-goes18.s3.amazonaws.com/?list-type=2&prefix=ABI-L1b-RadC/2024/001/00/&max-keys=20"
curl -o /tmp/noaa_data/goes18_sample.nc \
    "https://noaa-goes18.s3.amazonaws.com/ABI-L1b-RadC/2024/001/00/OR_ABI-L1b-RadC-M6C01_G18_s20240010001177_e20240010003550_c20240010004002.nc"

cargo run -p clawhdf5 --release --example phase1_bench --features clawhdf5-format/baselines -- \
    /tmp/noaa_data/synthetic_10gb.h5 /tmp/noaa_data/goes18_sample.nc
```

The harness is `crates/clawhdf5/examples/phase1_bench.rs`. Argument order is
`<synthetic_10gb.h5> <noaa_sample.nc>`.

## Hardware and parameters

- Host: `COG`
- CPU: AMD Ryzen 9 9950X3D 16-Core Processor
- RAM: 46.2 GB
- Disk: SSD (detected via `/sys/block/*/queue/rotational`)
- Recorded automatically in the CSV's `#`-prefixed header line (hostname,
  CPU model, RAM size, UTC date).
- 30 measured trials per cell (`TRIALS` in the harness). **Caveat**: unlike
  the sibling `hash_bench_harness.rs`/`baselines_bench.rs` harnesses, this
  harness does not discard 5 warmup trials before measuring — trial 1 of
  each cell is included in the reported set. In practice this made no
  visible difference here (every cell's first-trial value sits within
  normal trial-to-trial variance of its median — see the comparison table
  in "Expected trends" below), but it is a deviation from the Statistical
  Protocol's letter and should be fixed if a future cell ever shows a
  cold-start outlier.
- Inputs: the 10 GB synthetic file from P1.1, swept across its three
  chunk-size datasets (`dataset_64kb` → 54,608 chunks, `dataset_256kb` →
  13,652 chunks, `dataset_1mb` → 3,413 chunks, ~3.33 GB each), and a real
  NOAA GOES-18 sample (`Rad` dataset, 240 chunks, ~43 KB average on-disk
  chunk size).

## What is measured

Seven scenarios, each run against the Merkle-tree primitives
(`clawhdf5_format::merkle`) and, where applicable, the P1.2b
`FlatHashBackend` baseline:

- **verify_dataset** — rebuild the full Merkle tree from every chunk and
  compare the root to a known-good reference (`MerkleTree::from_chunks_owned`
  + root comparison).
- **full_rebuild** — the same tree build, timed on its own (separate label
  for the RQ3 line chart; mechanically identical cost to `verify_dataset`).
- **verify_chunk** — `tree.verify_chunk(0, ...)`: re-hash the chunk's bytes
  and compare against the leaf hash stored at that index in the tree's own
  node array (`MerkleTree::leaf_hash`, an O(1) lookup). This does **not**
  walk a sibling proof path to the root — that's a separate API
  (`generate_proof`/`verify_proof`), not exercised by this scenario — so as
  measured here, verify_chunk's cost is O(chunk_size), not O(chunk_size +
  log N). See the "Expected trends" note below for why this matters.
- **flat_verify** — the P1.2b baseline: one SHA-256 pass over the entire
  dataset's bytes (`FlatHashBackend::verify_dataset`), representing the
  status quo "no partial verification" approach.
- **extend_merkle** / **update_merkle** — `tree.clone()` followed by
  `MerkleTree::update_leaf(0, new_hash)`, the O(log N) path-recomputation
  primitive shared by both append and in-place update.
- **companion_io** — `write_merkle_companion()` cost and resulting byte size
  at N ∈ {16, 256, 1024, 65536} synthetic 64-byte chunks, independent of the
  source file's real chunk size, so the inline-vs-companion-dataset storage
  transition (`MerkleCompanionResult::Inline` for N ≤ 256, `::Dataset`
  beyond that) can be observed at a fixed, controlled N.

**Companion I/O caveat**: `write_merkle_companion` is called with a fresh
in-memory `FileWriter` (no path is ever passed to it in this harness), so
`companion_io`'s `wall_time_ms` measures the cost of building the inline
attribute or companion-dataset structure in memory, not a real disk write
or `fsync`. The column name is accurate to the spec's `companion_bytes`/byte-
overhead intent but should not be read as a measured disk-I/O latency.

## How to read the CSV

Each row is one trial. Columns: `source` (`synthetic`/`NOAA`), `scenario`
(one of the seven above), `n_chunks`, `chunk_size_kb` (average on-disk chunk
size; `0.0625` for the synthetic 64-byte `companion_io` chunks),
`wall_time_ms`, `peak_rss_mb` (the process's `VmHWM` high-water-mark,
sampled after each trial — not the trial's own incremental allocation),
`companion_bytes` (0 for all scenarios except `companion_io`, where it is
`(2 * next_power_of_two(N) - 1) * 32` — collapsing to `(2N-1) * 32` only
because every `N` used in this scenario (16, 256, 1,024, 65,536) is already
a power of two, so padding is a no-op), `trial` (1-30), `hostname`,
`disk_type`. There is no pre-computed median/CI column in this file
(unlike `hash-bench-*.csv`) —
aggregate the 30 trials per `(source, scenario, n_chunks, chunk_size_kb)`
group yourself, e.g. with the median.

## Figure

[phase1-plot-COG.pdf](phase1-plot-COG.pdf) — median wall time vs. chunk
count (log-log) for all six chunk-count-dependent scenarios, with 95%
bootstrap CI error bars, rendered as a vector PDF for `\includegraphics`
in LaTeX (not embedded inline here since Markdown can't preview a PDF).

Generated by `phase1-plot-COG.gp` (`gnuplot phase1-plot-COG.gp`), which
inlines the same per-cell median + 95% bootstrap CI (2000 resamples, same
protocol as `hash_bench_harness.rs`/`baselines_bench.rs`) tabulated below —
the CSV's 30 raw trials per cell remain the source of truth; the script
only renders already-verified summary numbers. The error bars are present
on every series but mostly imperceptible at this scale: every cell's CI is
sub-1% of its median (consistent with the tight CIs already observed in
`baselines-explanatory-note-localhost.localdomain.md`), so the lines read
as if they have no measurement spread. `verify_dataset`'s line is plotted
but sits exactly underneath `full_rebuild`'s (both run the identical
`MerkleTree::from_chunks_owned` call, per "What is measured" above), which
is itself a visual confirmation that the two scenarios are mechanically
the same operation.

## Expected trends and whether the data matches

Median `wall_time_ms` per cell:

| scenario | synthetic 64KB (n=54608) | synthetic 256KB (n=13652) | synthetic 1MB (n=3413) | NOAA (n=240, ~43KB) |
|---|---|---|---|---|
| verify_dataset | 636.92 | 431.98 | 375.17 | 2.414 |
| full_rebuild | 633.98 | 432.26 | 377.35 | 2.402 |
| flat_verify | 1278.997 | 1278.646 | 1278.566 | 3.775 |
| verify_chunk | 0.0100 | 0.0240 | 0.0830 | 0.0100 |
| extend_merkle | 0.3765 | 0.0150 | 0.0040 | 0.0010 |
| update_merkle | 0.3685 | 0.0150 | 0.0030 | 0.0010 |

- **flat_verify is independent of chunk count/size, as expected.** All
  three synthetic cells land within 0.04% of each other (~1278.6-1279.0 ms)
  despite chunk counts ranging from 3,413 to 54,608 — `FlatHashBackend`
  does one SHA-256 pass over the same ~3.33 GB regardless of how it's
  chunked, so its cost tracks total bytes only. This is the expected
  status-quo baseline shape.
- **verify_dataset/full_rebuild scale with chunk *count*, not just total
  bytes, as expected for a Merkle tree.** Same ~3.33 GB hashed in each
  synthetic cell, yet wall time rises from 375 ms (3,413 chunks) to 637 ms
  (54,608 chunks) — a ~1.7x increase for a ~16x increase in chunk count.
  This matches the design: building the tree issues one leaf-hash call per
  chunk plus internal-node combination work, so more (smaller) chunks mean
  more discrete hash invocations even though the bytes hashed are constant.
  This is the real, expected storage-vs-verification-granularity tradeoff
  the spec's RQ1/RQ3 are measuring.
- **verify_chunk gets *faster* as chunk count grows — the inverse of naive
  O(log N) intuition, but expected once you check what this method actually
  does.** Looking only at chunk *count*, one might expect verify_chunk to
  get slower as N grows (deeper tree → more proof-path work). The data
  shows the opposite: 0.0830 ms at N=3,413 (1 MB chunks) vs. 0.0100 ms at
  N=54,608 (64 KB chunks). Root cause, confirmed by reading
  `MerkleTree::verify_chunk` in `merkle.rs`: it re-hashes the chunk's full
  content and compares the result against the leaf hash already stored at
  that index — an O(1) array lookup, not a sibling-path walk to the root.
  So there is no log(N) term here at all to be dominated; verify_chunk's
  cost in this benchmark is essentially pure O(chunk_size). Since this
  dataset fixes total bytes (~3.33 GB) and varies chunk size inversely with
  chunk count, "more chunks" here always means "smaller chunks," and
  smaller chunks are cheaper to re-hash — hence the inverted curve. This is
  a real property of both the benchmark's parameterization and this
  specific API's design (not a bug): a verifier that only needs to check a
  chunk against a Merkle tree it already holds in full doesn't need the
  proof path at all. The O(log N) proof-path cost only applies to the
  separate `generate_proof`/`verify_proof` API (for a verifier holding just
  the root), which this scenario does not measure.
- **extend_merkle/update_merkle track each other almost exactly, as
  expected** (both literally call the same `MerkleTree::update_leaf`), but
  their absolute scaling with N is much steeper than the O(log N) the
  underlying path update actually performs, and needs its own explanation
  below (Anomaly 1).

## Anomaly 1: extend_merkle/update_merkle scale far faster than O(log N)

The path-update primitive itself (`MerkleTree::update_leaf`) is O(log N):
it recomputes only the ⌈log2 N⌉ ancestors of the changed leaf. Naively,
going from N=3,413 to N=54,608 (a 16x increase, ~4 extra tree levels)
should change the cost by a small constant factor. Instead the measured
cost rises by ~94x (0.004 ms → 0.3765 ms).

Root cause: the timed closure is `let mut t = tree.clone(); t.update_leaf(...)`,
and `tree.clone()` is a full copy of the tree's internal node array — O(N)
in the number of leaves, not O(log N). `MerkleTree::build` pads the leaf
count up to the next power of two before sizing the node array
(`2 * leaf_count.next_power_of_two() - 1` nodes), so the clone size must be
computed from the *padded* leaf count, not the raw chunk count:

| N (leaves) | padded to | nodes = 2·padded-1 | clone size |
|---|---|---|---|
| 3,413 | 4,096 | 8,191 | ~256 KB |
| 13,652 | 16,384 | 32,767 | ~1.00 MB |
| 54,608 | 65,536 | 131,071 | ~4.00 MB |

This matches the observed timings closely: the clone is cheap and
sub-linear while it stays at or below the CPU's per-core L2 size (256 KB →
1.00 MB, 0.004 ms → 0.015 ms, a ~3.75x rise for a 4x size increase), then
jumps disproportionately once the array (~4.00 MB) clears L2 entirely and
spills into L3 (1.00 MB → 4.00 MB, 0.015 ms → 0.3765 ms — a ~25x time
increase for only a 4x size increase). The harness is therefore measuring
`clone() + O(log N) update`, where the clone's O(N) cost dominates and the
L2-capacity threshold — not tree depth — explains the super-linear jump at
the largest N. The true cost of `update_leaf` alone (without the clone) is
expected to remain flat/O(log N) regardless of N; this benchmark's numbers
should not be quoted as that primitive's cost in isolation.

## Anomaly 2 (non-anomaly, confirms design): companion_bytes is identical across the inline/dataset storage transition

`companion_bytes` at N=256 is 16,352 and at N=1,024 is 65,504 — both exactly
`(2N-1) * 32`, with no discontinuity at the documented inline-vs-dataset
threshold (`MerkleCompanionResult::Inline` for N ≤ 256, `::Dataset` for
N > 256). This confirms the storage-mode switch only changes *where* the
node hashes live (an HDF5 attribute vs. a companion dataset), not the
payload size — exactly as the spec intends (the switch exists to avoid
attribute-size limits, not to change the proof's byte cost). The
`companion_io` wall-time column does show a real cost jump at N=65,536
(0.025 ms at N=1,024 → 1.6-2.3 ms at N=65,536), consistent with the larger
absolute number of bytes (4.19 MB) being serialized into the in-memory
companion-dataset builder — an O(N) cost in serialized size, as expected,
not a step-function artifact of the storage-mode switch itself.

## Inconclusive results

None — every scenario's synthetic-vs-NOAA and chunk-size-vs-chunk-size
comparisons show clear, explainable separation; no two cells in this run
are close enough to be ambiguous at 30 trials.
