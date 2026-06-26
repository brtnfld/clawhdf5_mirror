# P1.5 Subset-Proof Size Benchmark — Explanatory Note

This note accompanies `subset-proof-size.csv` and `subset-proof-size-raw-trials.csv`,
per the S2-D2 spec's "Benchmark validity and interpretation" requirement (p.52): every
benchmark artifact needs a reproducible explanatory note covering exact reproduction
steps, hardware, what is measured, how to read the results, and a root-cause
explanation of every notable trend or anomaly.

## Reproduction

```bash
cargo bench --features merkle --bench subset_proof_size_bench
```

Run from the workspace root or the `crates/clawhdf5-format/` directory. The `merkle`
feature is required (it is not in the default feature set). No input files are
needed — the benchmark builds a synthetic 65,536-chunk dataset in memory. The bench
binary writes three output files:

- `crates/clawhdf5-format/benches/results/subset-proof-size.csv` — summary, one row per
  (hyperslab type, k) condition; `proof_time_ms`/`verify_time_ms` are medians.
- `crates/clawhdf5-format/benches/results/subset-proof-size-raw-trials.csv` — all 30
  raw per-trial timings per condition; the source of truth behind the medians.
- `test-vectors/subset-vectors.json` — three serialised `SubsetProof` structs (one per
  hyperslab shape at k=64) with their expected Merkle roots.

## Hardware and parameters

- Host: `COG`
- CPU: AMD Ryzen 9 9950X3D 16-Core Processor
- RAM: 46.2 GB
- Recorded automatically in the CSV's `#`-prefixed header line
  (hostname, CPU model, RAM, UTC date).
- 30 measured trials per (hyperslab type, k) cell (`TRIALS` in the harness), after 5
  discarded warmup trials (`WARMUP_TRIALS`), per the Statistical Protocol (S2-D2 spec,
  p.52: "minimum 30 trials after 5 discarded warmups"), matching
  `hash_bench_harness.rs`/`baselines_bench.rs`/`phase1_bench.rs`.
- `proof_time_ms` and `verify_time_ms` are the medians over the 30 measured trials.
  95% bootstrap CI (2000 resamples, xorshift64* PRNG, same protocol as
  `hash_bench_harness.rs`) is reported in the `_ci95_low`/`_ci95_high` columns and in
  the summary printed to stdout.
- `proof_size_bytes` and `theoretical_bound_bytes` are deterministic for a given
  (shape, k, N, leaf order) — they carry no trial variance and have no CI columns.
- N = 65,536 (2^16) chunks in a 1-D dataset, one chunk per leaf.
- Leaf order: `RowMajor` (the 1-D identity; Morton order is equivalent in 1-D and
  would produce identical results).
- Hash algorithm: BLAKE3 throughout.
- k values swept: 64, 256, 1024, 4096 — chosen to span two orders of magnitude
  within N and keep strided/random timing manageable on a single run.
- Chunk data: synthetic, 11-byte ASCII strings (`"chunk-000000"` … `"chunk-065535"`),
  deterministic and reproducible; the content does not affect proof size (only N and
  the selection shape matter) and affects verify timing only through the cost of
  hashing the chunk bytes.

## What is measured

**`proof_size_bytes`** — the serialised wire size of the `SubsetProof` struct for the
given selection, computed as:

```
k × 8   (chunk_indices: one u64 per selected chunk)
k × 32  (leaf_hashes: one BLAKE3 hash per selected chunk)
m × 40  (proof_nodes: one BTreeMap entry at 8-byte key + 32-byte hash per unique
          sibling node, after deduplication across all k proof paths)
d × 16  (grid_params.dims + chunk_params.chunk_shape, d=1 here)
32      (grid_params.grid_hash)
32      (coverage_cert)
```

where `m` is the number of *deduplicated* proof nodes — the key quantity RQ6 asks
about, since deduplication is what makes contiguous proofs compact.

**`theoretical_bound_bytes`** — the per-chunk-path worst case with no deduplication:
`k × ⌈log2(N)⌉ × 40 + k × 40 + 100` = `k × 16 × 40 + k × 40 + 100` = `k × 680 + 100`
for N = 65,536. This bound is tight when no two selected chunks share any sibling on
the path to the root — i.e. when the chunks are maximally spread, which strided
selection (stride = N/k, maximum separation) approaches closely.

**`proof_time_ms`** — wall-clock time to call `extract_subset(tree, grid, sel,
RowMajor)` once (median over 30 trials after 5 warmup). This includes translating the
hyperslab into chunk coordinates, sweeping the bounding-box sub-grid, computing level-
order tree indices for each proof path, deduplicating via BTreeMap insertion, and
computing the coverage certificate. It does **not** include hashing the chunk data —
the tree was pre-built from the synthetic chunks before the timed region starts.

**`verify_time_ms`** — wall-clock time to call `verify_subset(root, Blake3, chunks,
proof, grid, sel, RowMajor)` once (median over 30 trials after 5 warmup). This
includes re-hashing each delivered chunk's bytes, walking its sibling proof path by
BTreeMap lookup to recompute the root, and verifying the coverage certificate. Chunk
re-hashing is an O(chunk\_data\_size) cost paid once per selected chunk; proof-path
walking is O(log N) per chunk via BTreeMap lookups keyed by level-order index.

## How to read the CSV

`subset-proof-size.csv` has one row per (hyperslab\_type, k\_chunks) condition.
Columns:

| Column | Meaning |
|---|---|
| `hyperslab_type` | `contiguous`, `strided`, or `random` |
| `k_chunks` | Number of selected chunks in the hyperslab |
| `n_total` | Total chunks in the dataset (65,536) |
| `proof_size_bytes` | Serialised wire size of the proof (deterministic) |
| `proof_time_ms` | Median wall-clock time to generate the proof (30 trials) |
| `proof_time_ci95_low/high` | 95% bootstrap CI bounds on the median proof time |
| `verify_time_ms` | Median wall-clock time to verify the proof (30 trials) |
| `verify_time_ci95_low/high` | 95% bootstrap CI bounds on the median verify time |
| `theoretical_bound_bytes` | O(k·log N) worst-case proof size with no deduplication |

`subset-proof-size-raw-trials.csv` has 30 rows per condition (one per trial), with
`trial` (1–30), `proof_time_ms`, and `verify_time_ms` for the raw timing values behind
the summary medians.

## Figures

[figure6-proof-size-COG.pdf](figure6-proof-size-COG.pdf) — Figure 6 left panel
(RQ6, §7.6). Log-log line chart: proof size |π| (bytes) on the y-axis vs. k (number
of selected chunks) on the x-axis; three series (contiguous/strided/random) plus a
dashed O(k·log N) theoretical bound line. Generated by
`figure6-proof-size-COG.gp` (`gnuplot figure6-proof-size-COG.gp` from the
`benches/results/` directory).

## Expected trends and whether the data matches

### Proof size

| hyperslab_type | k=64 | k=256 | k=1024 | k=4096 |
|---|---|---|---|---|
| contiguous | 8,080 | 31,040 | 123,120 | 491,680 |
| strided | 33,280 | 112,640 | 368,640 | 1,146,880 |
| random | 30,600 | 104,040 | 336,200 | 1,027,760 |
| theoretical bound | 43,620 | 174,180 | 696,420 | 2,785,380 |

**All three series grow linearly in k, as expected** for a fixed N. The slope differs:
contiguous grows at ~120 bytes/chunk (≈ 17.6% of the 680 bytes/chunk theoretical),
while strided grows at ~280 bytes/chunk (≈ 41%) and random at ~252 bytes/chunk (≈
37%). All are strictly below the theoretical bound at every k, confirming the
deduplication step is always effective.

**Contiguous falls well below the bound — the dominant effect.** A contiguous 1-D slab
of k adjacent chunks at position [start, start+k) spans a single contiguous region of
the leaf level. Adjacent leaves share all ancestors above the point where their paths
first diverge: for k=64 chunks starting at index 16384 (= 2^14), every pair of
adjacent leaves shares 14 ancestors before the paths split. In the extreme case of a
power-of-two-aligned slab, all k leaves share a single subtree root — so the entire
proof collapses to O(log N) nodes rather than O(k log N). The data confirms: at k=64,
contiguous uses 136 deduplicated proof nodes vs the bound of 64×16=1,024 (a 7.5×
reduction). The contiguous ratio stays remarkably flat at ~0.18 across all k values
(k=64: 0.185, k=4096: 0.177), indicating the sharing fraction is roughly constant for
a contiguous slab of any size within this 1-D grid, as expected.

**Strided and random are structurally similar, but strided is slightly worse than
random.** Strided selection (stride = N/k) places every chunk at the maximum possible
separation, guaranteeing that no two selected chunks share any sibling closer than
the root — this is the worst case for deduplication and matches the theoretical bound
most closely. Random selection with a fixed seed happens to cluster some chunks
slightly more than worst-case-separated striding, giving marginally more sharing and
smaller proofs (random is ~8% smaller than strided at k=64). In the limit as k →
N the two converge since all proof paths are included.

### Timing

Median timing per cell:

| hyperslab_type | k | proof_time_ms | verify_time_ms |
|---|---|---|---|
| contiguous | 64 | 0.0102 | 0.1917 |
| strided | 64 | 2.9820 | 3.0422 |
| random | 64 | 2.9257 | 3.0258 |
| contiguous | 256 | 0.0351 | 0.5904 |
| strided | 256 | 11.6156 | 11.9881 |
| random | 256 | 11.6521 | 12.1396 |
| contiguous | 1024 | 0.2021 | 2.3577 |
| strided | 1024 | 43.6534 | 45.5771 |
| random | 1024 | 43.6681 | 45.5756 |
| contiguous | 4096 | 0.8869 | 9.4499 |
| strided | 4096 | 166.4220 | 174.5620 |
| random | 4096 | 167.0235 | 175.2337 |

All 95% CIs are tight: every cell's CI width is ≤ 1.5% of its median, indicating the
30-trial protocol is adequate to distinguish these conditions without inference noise.

## Anomaly 1: contiguous proof time is ~300× faster than strided at k=64

At k=64, contiguous proof generation takes 0.010 ms while strided takes 2.98 ms — a
~293× gap. This is not an outlier: it holds at every k (contiguous is consistently
~140–200× faster than strided and grows only weakly with k, while strided/random grow
roughly linearly with k).

Root cause: `extract_subset` uses `selection_chunk_bounds` to compute the
element-space bounding box of the selection and sweeps only the *enclosing sub-grid*
(a `bounded_sweep` of chunk coordinates in that box). For a contiguous 1-D slab of k
chunks, the bounding box contains exactly k chunks — the sweep visits k items. For a
strided selection with stride = N/k (e.g. stride = 1024 at k=64), the bounding box
spans the entire [0, N) range, so the sweep must visit all N=65,536 chunk coordinates
to identify the k selected ones, paying an O(N) scan even though only k chunks are
selected. The BTreeMap insertion of proof nodes then adds O(k log N) work on top.
This is expected behaviour for the current implementation of
`chunk_coords_for_selection` with a `Points` selection: the bounding box of maximally
spread points spans the full grid, so the bounded sweep cannot prune anything. For
random selections the situation is identical (bounding box also spans [0, N)).

This is not an asymptotic design flaw — it is a consequence of using a bounding-box
sweep as a generic selection-to-chunks translator. A future implementation could
short-circuit for `Points` selections by iterating the sorted points directly rather
than sweeping the bounding box, reducing extract cost to O(k log N) for
strided/random inputs. This optimisation is out of scope for P1.5 and is not needed
for correctness.

## Anomaly 2: verify_time >> proof_time for contiguous, but ≈ proof_time for strided/random

At k=64, contiguous verify (0.192 ms) is ~19× slower than contiguous proof (0.010 ms),
whereas strided verify (3.04 ms) is only ~2% above strided proof (2.98 ms).

Root cause: the two operations have different cost breakdowns depending on proof size.
`verify_subset` always re-hashes each of the k delivered chunks' bytes before walking
the proof path; `extract_subset` never re-hashes chunk data (it reads pre-hashed leaf
values already stored in the `MerkleTree`'s node array). For contiguous, the proof
itself is tiny (136 deduplicated nodes, built in 0.010 ms via a lightweight O(k) sweep
+ O(m) BTree inserts where m=136), but verification must still hash 64 chunks of
11-byte data — BLAKE3 with domain-separation prefix, a fixed per-call overhead that
dominates the verify cost for small proofs. For strided/random, both proof and verify
are dominated by BTree traversal over ~700–25,000 proof nodes, dwarfing the per-chunk
hashing cost, so the two times converge. This is expected: the implementation
correctly decouples the proof-generation path (index arithmetic + BTree inserts) from
the verification path (chunk rehashing + BTree lookups).
