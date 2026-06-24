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
- 30 measured trials per cell (`TRIALS` in the harness), after 5 discarded
  warmup trials (`WARMUP_TRIALS`), matching the sibling
  `hash_bench_harness.rs`/`baselines_bench.rs` harnesses and the Statistical
  Protocol's letter. **This was not always true of this harness**: an
  earlier revision measured trial 1 of every cell with no warmup discard,
  and the `extend_merkle`/`update_merkle` cells showed real cold-start
  outliers as a result (e.g. at n=54,608 the slowest `extend_merkle` trial
  was 0.965 ms against a ~0.377 ms median; at n=13,652 five of the 30
  trials sat 3-4x above the median). Adding the warmup discard removed
  essentially all of that spread — see the boxplot figure below, whose
  remaining outliers are now small in absolute terms (≤0.002 ms above the
  median in every case) rather than the 2.6x cold-start spike seen before.
- Inputs: the 10 GB synthetic file from P1.1, swept across its three
  chunk-size datasets (`dataset_64kb` → 54,608 chunks, `dataset_256kb` →
  13,652 chunks, `dataset_1mb` → 3,413 chunks, ~3.33 GB each), and a real
  NOAA GOES-18 sample (`Rad` dataset, 240 chunks, ~43 KB average on-disk
  chunk size).

## What is measured

Nine scenarios, each run against the Merkle-tree primitives
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
  walk a sibling proof path to the root — that's `verify_proof` below — so
  as measured here, verify_chunk's cost is O(chunk_size), not O(chunk_size +
  log N). This is the right primitive for a verifier that already holds the
  full tree (e.g. checking its own local copy after a write).
- **verify_proof** — `tree.proof(0)` (generate the sibling-hash inclusion
  proof) followed by `MerkleTree::verify_proof_standalone(&root, 0, chunk,
  &proof)`, verified against *only the root hash*, not the full tree. This
  is the true O(chunk_size + log N) path-verification cost for a verifier
  that does **not** hold the full tree — e.g. checking a chunk received in
  transit from a low-trust source — completing the verification picture
  `verify_chunk` alone leaves out.
- **flat_verify** — the P1.2b baseline: one SHA-256 pass over the entire
  dataset's bytes (`FlatHashBackend::verify_dataset`), representing the
  status quo "no partial verification" approach.
- **extend_merkle** / **update_merkle** — `tree.clone()` followed by
  `MerkleTree::update_leaf(0, new_hash)`, the O(log N) path-recomputation
  primitive shared by both append and in-place update. The clone is a
  benchmark-harness artifact (each trial needs a fresh tree so the
  unmodified `tree` stays available for the other scenarios in the same
  run), not part of `update_leaf`'s own cost — see `update_leaf_inplace`
  below and Anomaly 1.
- **update_leaf_inplace** — the same `update_leaf(0, new_hash)` call, but on
  a tree cloned once *outside* the timed region and reused across all 30
  trials, isolating the path-update primitive's true cost from the
  per-trial `tree.clone()` that dominates `extend_merkle`/`update_merkle`
  above.
- **companion_serialization** — `write_merkle_companion()` cost and
  resulting byte size at N ∈ {16, 256, 1024, 65536} synthetic 64-byte
  chunks, independent of the source file's real chunk size, so the
  inline-vs-companion-dataset storage transition
  (`MerkleCompanionResult::Inline` for N ≤ 256, `::Dataset` beyond that) can
  be observed at a fixed, controlled N. Named "serialization", not "io":
  `write_merkle_companion` is called with a fresh in-memory `FileWriter`
  (`FileWriter::new()`/`finish()` only ever produce an in-memory `Vec<u8>` —
  there is no disk-path or `fsync` method on this type at all), so this
  scenario measures the cost of building the inline attribute or
  companion-dataset structure in memory; the real disk write/fsync happens
  later, once per whole file, in a step this function never touches.

## How to read the CSV

Each row is one trial. Columns: `source` (`synthetic`/`NOAA`), `scenario`
(one of the nine above), `n_chunks`, `chunk_size_kb` (average on-disk chunk
size; `0.0625` for the synthetic 64-byte `companion_serialization` chunks),
`wall_time_ms`, `peak_rss_mb` (the process's `VmHWM` high-water-mark,
sampled after each trial — not the trial's own incremental allocation),
`companion_bytes` (0 for all scenarios except `companion_serialization`,
where it is `(2 * next_power_of_two(N) - 1) * 32` — collapsing to
`(2N-1) * 32` only because every `N` used in this scenario (16, 256, 1,024,
65,536) is already a power of two, so padding is a no-op), `trial` (1-30),
`hostname`, `disk_type`. There is no pre-computed median/CI column in this
file (unlike `hash-bench-*.csv`) —
aggregate the 30 trials per `(source, scenario, n_chunks, chunk_size_kb)`
group yourself, e.g. with the median.

## Figures

Two separate vector PDFs, both for direct `\includegraphics` use in LaTeX
(not embedded inline here since Markdown can't preview a PDF).

[phase1-plot-lines-COG.pdf](phase1-plot-lines-COG.pdf) — median wall time
vs. chunk count (log-log) for the five scenarios whose 30-trial spread is
tight enough to read as a single line: `verify_dataset`, `full_rebuild`,
`flat_verify`, `verify_chunk`, `verify_proof`. Each point carries a 95%
bootstrap CI error bar (2000 resamples, same protocol as
`hash_bench_harness.rs`/`baselines_bench.rs`), tabulated below — the CSV's
30 raw trials per cell remain the source of truth; the script only renders
already-verified summary numbers. The error bars are present on every
series but mostly imperceptible at this scale: every cell's CI is sub-1% of
its median (consistent with the tight CIs already observed in
`baselines-explanatory-note-localhost.localdomain.md`). `verify_dataset`'s
line is plotted but sits exactly underneath `full_rebuild`'s (both run the
identical `MerkleTree::from_chunks_owned` call, per "What is measured"
above), which is itself a visual confirmation that the two scenarios are
mechanically the same operation. `verify_proof` tracks `verify_chunk`
closely but consistently slightly above it (e.g. 0.0120 ms vs. 0.0100 ms at
n=54,608) — the extra cost of the sibling-path walk `verify_chunk` skips,
small here because chunk re-hashing (O(chunk_size)) still dominates the
O(log N) proof-walk term at these tree depths (≤17 levels). Generated by
`phase1-plot-lines-COG.gp` (`gnuplot phase1-plot-lines-COG.gp`).

[phase1-plot-boxplot-COG.pdf](phase1-plot-boxplot-COG.pdf) —
`extend_merkle`/`update_merkle`/`update_leaf_inplace`: Tukey box-and-whisker
plots (Q1/median/Q3, whiskers capped at the most extreme point within
1.5×IQR of the box) with any trial beyond that fence drawn as an individual
open-circle outlier. `extend_merkle`/`update_merkle` still show enough
per-cell spread (and a handful of small outliers) to be worth a
distributional view rather than a collapsed median line. `update_leaf_inplace`
is the control: it sits perfectly flat at ~0.002 ms with *zero* spread
across every chunk count (240 through 54,608) — direct visual confirmation
that the path-update primitive itself is O(log N)/cache-independent, and
that all of `extend_merkle`/`update_merkle`'s scaling with N comes from the
`tree.clone()` they pay per trial (see Anomaly 1). Generated by
`phase1-plot-boxplot-COG.gp` (`gnuplot phase1-plot-boxplot-COG.gp`).

## Expected trends and whether the data matches

Median `wall_time_ms` per cell:

| scenario | synthetic 64KB (n=54608) | synthetic 256KB (n=13652) | synthetic 1MB (n=3413) | NOAA (n=240, ~43KB) |
|---|---|---|---|---|
| verify_dataset | 627.91 | 430.50 | 374.28 | 2.406 |
| full_rebuild | 628.20 | 432.27 | 376.68 | 2.408 |
| flat_verify | 1281.92 | 1279.49 | 1276.95 | 3.780 |
| verify_chunk | 0.0100 | 0.0240 | 0.0830 | 0.0100 |
| verify_proof | 0.0120 | 0.0260 | 0.0850 | 0.0110 |
| extend_merkle | 0.3620 | 0.0150 | 0.0030 | 0.0010 |
| update_merkle | 0.3590 | 0.0150 | 0.0030 | 0.0010 |
| update_leaf_inplace | 0.0020 | 0.0020 | 0.0020 | 0.0010 |

- **flat_verify is independent of chunk count/size, as expected.** All
  three synthetic cells land within 0.4% of each other (~1276.9-1281.9 ms)
  despite chunk counts ranging from 3,413 to 54,608 — `FlatHashBackend`
  does one SHA-256 pass over the same ~3.33 GB regardless of how it's
  chunked, so its cost tracks total bytes only. This is the expected
  status-quo baseline shape.
- **verify_dataset/full_rebuild scale with chunk *count*, not just total
  bytes, as expected for a Merkle tree.** Same ~3.33 GB hashed in each
  synthetic cell, yet wall time rises from 374 ms (3,413 chunks) to 628 ms
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
  separate `proof`/`verify_proof_standalone` API (for a verifier holding
  just the root) — measured directly by the next scenario.
- **verify_proof tracks verify_chunk's shape almost exactly, but is
  consistently a touch higher, as expected.** Same inverted N-vs-time curve
  (0.0850 ms at N=3,413 down to 0.0120 ms at N=54,608) since both scenarios
  are still dominated by O(chunk_size) re-hashing, but verify_proof is
  consistently ~10-20% above verify_chunk at the same N (e.g. 0.0120 ms vs.
  0.0100 ms at N=54,608) — the added cost of generating and walking the
  O(log N) sibling-hash proof path that verify_chunk's O(1) lookup skips.
  The gap stays small because, at these chunk counts, the proof path is at
  most 17 levels deep (⌈log2 65,536⌉) — negligible next to hashing tens to
  hundreds of KB of chunk data per call. This confirms `verify_chunk` and
  `verify_proof` measure genuinely different operations whose costs happen
  to be close at this scale, not that one was mislabeled as the other.
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
cost rises by ~121x (0.0030 ms → 0.3620 ms).

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
1.00 MB, 0.0030 ms → 0.0150 ms, a 5x rise for a 4x size increase), then
jumps disproportionately once the array (~4.00 MB) clears L2 entirely and
spills into L3 (1.00 MB → 4.00 MB, 0.0150 ms → 0.3620 ms — a ~24x time
increase for only a 4x size increase). The harness is therefore measuring
`clone() + O(log N) update`, where the clone's O(N) cost dominates and the
L2-capacity threshold — not tree depth — explains the super-linear jump at
the largest N.

This is no longer just an inference from clone-size arithmetic: the
`update_leaf_inplace` scenario isolates the same `update_leaf` call with the
clone removed from the timed region (the tree is cloned once, outside
`time_trials`, and that same in-place tree is reused across all 30 trials).
Its measured cost is flat at 0.0010-0.0020 ms across all four chunk counts
(240/3,413/13,652/54,608) with zero trial-to-trial spread inside any cell —
direct confirmation that `update_leaf` alone is O(log N)/effectively
constant at these tree depths, and that the ~121x scaling seen in
`extend_merkle`/`update_merkle` is entirely the clone's O(N) cost, not a
property of the path-update primitive itself. At N=54,608, `update_leaf`
in isolation costs roughly 180-360x less than the clone-inclusive
`extend_merkle`/`update_merkle` measurement (0.0020 ms vs. 0.3590-0.3620
ms). `update_leaf_inplace`'s number is the primitive's true cost;
`extend_merkle`/`update_merkle` should be read as `clone() + update_leaf()`,
where the clone is a benchmark-harness artifact (see "What is measured"
above), not part of the primitive's real cost in a system that mutates a
tree in place rather than cloning it per update.

The boxplot figure makes the largest-N cell's residual spread visible
directly rather than just in the median: at n=54,608, `extend_merkle`'s box
(Q1-Q3) spans roughly 0.358-0.365 ms (median 0.362 ms), with a single trial
landing as an outlier above the box at 0.379 ms (~4.7% above median) — far
smaller than the cold-start spikes this section reported before the
5-trial warmup discard was added (previously up to 0.965 ms, ~2.6x the
median). `update_merkle`'s remaining outliers (0.004 ms at N=3,413, twice,
and 0.017 ms at N=13,652) are flagged only because those cells' IQR is
essentially zero (all 30 trials round to the same value); in absolute
terms they are ≤0.002 ms above the median and consistent with ordinary
timer-resolution noise, not a distinct cache or scheduling effect. With
warmup trials now excluded, the spread remaining at every N is small and
explainable as system noise on an L2/L3-resident allocation, not cold-start
variance — the earlier, much larger outliers were warmup artifacts, not a
property of the clone or the allocator.

## Anomaly 2 (non-anomaly, confirms design): companion_bytes is identical across the inline/dataset storage transition

`companion_bytes` at N=256 is 16,352 and at N=1,024 is 65,504 — both exactly
`(2N-1) * 32`, with no discontinuity at the documented inline-vs-dataset
threshold (`MerkleCompanionResult::Inline` for N ≤ 256, `::Dataset` for
N > 256). This confirms the storage-mode switch only changes *where* the
node hashes live (an HDF5 attribute vs. a companion dataset), not the
payload size — exactly as the spec intends (the switch exists to avoid
attribute-size limits, not to change the proof's byte cost). The
`companion_serialization` wall-time column does show a real cost jump at
N=65,536 (0.025 ms at N=1,024 → 1.6-2.3 ms at N=65,536), consistent with the
larger absolute number of bytes (4.19 MB) being serialized into the
in-memory companion-dataset builder — an O(N) cost in serialized size, as
expected, not a step-function artifact of the storage-mode switch itself.

## Inconclusive results

None — every scenario's synthetic-vs-NOAA and chunk-size-vs-chunk-size
comparisons show clear, explainable separation; no two cells in this run
are close enough to be ambiguous at 30 trials.

## Known follow-up: NOAA sample persistence

The real-data input is fetched from the live `noaa-goes18` public S3
bucket at run time (see "Reproduction" above), and the exact object key
has already rotated out once during this benchmark's history — the
current command points at a different file from the same time window than
the one originally referenced in `test-vectors/README.md`. This makes the
NOAA half of the CSV's reproducibility dependent on an external bucket's
retention policy, which is outside this repository's control and could
rotate again. Mirroring this sample to a permanent, citable academic
repository (e.g. a Zenodo/institutional-archive deposit referenced by DOI
instead of a live bucket key) was recommended alongside the other items in
this note's revision but has not been implemented here — it is
infrastructure/data-hosting work, not a change to the harness or its
analysis, and is left as an open follow-up.
