# P1.2 Hash Algorithm Benchmark — Explanatory Note

This note accompanies `hash-bench-localhost.localdomain.csv` and
`hash-bench-raw-trials-localhost.localdomain.csv`, per the S2-D2 spec's
"Benchmark validity and interpretation" requirement (p.52): every
benchmark artifact needs a reproducible explanatory note covering exact
reproduction steps, hardware, what is measured, how to read the results,
and a root-cause explanation of any notable trend or anomaly.

## Reproduction

```bash
cargo run -p clawhdf5-format --release --example hash_bench_harness --features merkle
```

The harness is `crates/clawhdf5-format/examples/hash_bench_harness.rs`. It
is a separate, spec-conformant alternative to the pre-existing
`benches/hash_bench.rs` (run via `./benches/run_hash_bench.sh`): that
Criterion microbenchmark is still useful for ad hoc performance work, but
its parsed CSV (`parse_hash_bench.py`) reports `throughput_mibs` (base-1024)
under Criterion's own adaptive-sampling statistics, not the spec's literal
`throughput_mbs` column with an explicit 30-trial/5-warmup/bootstrap-CI
protocol — this harness exists to produce the actual required artifact.

## Hardware and parameters

- Host: `localhost.localdomain`
- CPU: AMD Ryzen 9 9950X3D 16-Core Processor (has `sha_ni`, `avx2`,
  `avx512` per `/proc/cpuinfo` flags — relevant to Anomaly 1 below)
- RAM: 46.2 GB
- Recorded automatically in the CSV's `#`-prefixed header line (hostname,
  CPU model, RAM size, UTC date), same convention as the P1.2b baseline CSV.
- 30 measured trials per (algorithm, chunk size) cell, after 5 discarded
  warmup iterations (`WARMUP_TRIALS`/`TRIALS` in the harness).
- Each trial times one `hash_chunk()` call (includes the `0x00` leaf
  domain-separation prefix, per §5.5) over a deterministic, non-trivially
  compressible in-memory buffer of the given size (same data-generation
  pattern as `hash_bench.rs::make_chunk`) — no disk I/O is measured, only
  hashing throughput.
- Chunk sizes: 64 KB, 256 KB, 1 MB, the HDF5-typical range cited in RQ4.
- `throughput_mbs` is decimal MB/s (1 MB = 1,000,000 bytes), not MiB/s, to
  match the spec's column name literally.

## What is measured

`HashAlg::{Sha256, Blake3, K12}` via `clawhdf5_format::merkle::hash_chunk`,
the same leaf-hashing helper the `MerkleTree` itself uses (P1.2 step 3) —
so these throughput numbers characterize the actual function on the actual
critical path, not a standalone copy of the algorithm.

## How to read the plot

- `plot_hash_throughput.png`: one main log-scale point+errorbar plot
  comparing all three algorithms across chunk sizes, plus three small
  zoomed-in inset panels below it (one per algorithm, each individually
  y-scaled) — the main plot's 95% CIs are too narrow to see at that shared
  scale, so each inset re-plots its algorithm's own CI on a tight axis
  range where the whiskers are actually visible.
- `plot_hash_throughput_spread.png`: a separate plot, one panel per
  algorithm, showing the **raw spread of the 30 measured trials** rather
  than the bootstrap CI on the median: a thin whisker for min/max and a
  thick band for the IQR (25th-75th percentile), with the median marked
  as a dot. This is a different statistic than the CI plot above — see
  "How to read the CSV" below for the distinction — and is generally much
  wider, since it reflects trial-to-trial variability directly rather than
  uncertainty in the median estimate.

## How to read the CSV

`hash-bench-localhost.localdomain.csv` (the spec-mandated artifact): each
row is one (algorithm, chunk size) cell, `throughput_mbs` is the **median
of the 30 measured trials**, with `ci95_low`/`ci95_high` from a **95%
bootstrap confidence interval** (2000 resamples) — never a bare mean, per
the statistical protocol. This CI describes how much the *median* would
plausibly vary if the 30-trial experiment were rerun, not how spread out
the raw trials themselves were.

`hash-bench-raw-trials-localhost.localdomain.csv` (supplementary, not a
spec-mandated column set): one row per individual trial — `alg,
chunk_size_kb, trial, throughput_mbs` — the same 30 raw measurements per
cell that feed into the summary CSV's median/CI, kept around so the raw
spread (min/max, IQR) can also be reported, e.g. in
`plot_hash_throughput_spread.png`.

## Expected trends and whether the data matches

- **BLAKE3 throughput increases with chunk size** (6.7 GB/s at 64 KB → 10.6
  GB/s at 256 KB → 12.5 GB/s at 1 MB). Matches expectation: BLAKE3 processes
  input in parallel 1 KB-leaf lanes internally (SIMD-width dependent), so
  larger inputs better amortize per-call setup cost and expose more
  parallelism — consistent with RQ4's framing of BLAKE3 as the
  parallelism-oriented design.
- **SHA-256 and K12 throughput are flat across chunk sizes** (SHA-256:
  ~2.73–2.74 GB/s; K12: ~1.59–1.60 GB/s). Matches expectation for both:
  neither this `sha2` usage nor `k12`'s sponge construction exploits
  cross-chunk-size parallelism the way BLAKE3's tree does, so per-byte cost
  is roughly constant once warmed up.

## Anomaly: SHA-256 is faster than K12, despite K12 being the
"very fast" candidate

At every chunk size, SHA-256 (~2.7 GB/s) outperforms K12 (~1.6 GB/s) by
~1.7x — the opposite of §5.3's framing of K12 as a faster SHA-3-family
alternative to SHA-256 for throughput-bound deployments.

Root cause: this is specifically about *this* CPU's hardware support, not
the algorithms in the abstract. `/proc/cpuinfo` on this host has the
`sha_ni` flag — the `sha2` crate (used by `HashAlg::Sha256`) auto-detects
SHA extensions at runtime (`cpufeatures` dependency, pulled in during the
build) and uses the hardware SHA-256 instruction sequence when available,
which is dramatically faster than a software round-function implementation.
`k12` (Keccak-p1600-based) has no equivalent widely-deployed hardware
instruction on this consumer Zen 5 part — AVX-512 is present, but there is
no dedicated Keccak/SHA-3 instruction extension comparable to `sha_ni`, so
K12 runs the software permutation. The published "K12 is faster than
SHA-256" comparisons (e.g., the spec's §5.3 framing) implicitly assume
software-only SHA-256; on hardware with SHA extensions, that ordering
inverts. This is a real, reproducible property of this specific CPU, not a
measurement artifact — on a CPU without `sha_ni` (e.g., an older or
low-power x86 part, or most current ARM cores without the ARMv8.2
SHA3/SHA512 crypto extension), SHA-256 would be expected to fall back to
software and K12 would likely come out ahead, matching §5.3's framing.

**Confirmed experimentally.** The `sha2` crate exposes a `force-soft`
feature that disables the hardware-accelerated path regardless of detected
CPU support. Rebuilding and rerunning with it forced on:

```bash
cargo run -p clawhdf5-format --release --example hash_bench_harness \
    --features "merkle,sha2/force-soft"
```

drops SHA-256 from ~2.70–2.72 GB/s to **~666–676 MB/s** at every chunk
size — now *below* K12's ~1.56–1.61 GB/s, exactly the ordering §5.3
predicts. This isolates the effect to the `sha_ni` hardware path (the only
thing `force-soft` changes) rather than some other confound, confirming
the root-cause explanation above rather than merely asserting it. (This
forced-software run is not the artifact's committed numbers — the
committed CSV correctly reflects the project's default build, which uses
hardware acceleration when available.)

## Observation: CI width (as % of median) grows with chunk size for SHA-256 and K12

Within the committed 64/256/1024 KB cells (most recent run), SHA-256's CI
width is ~0.04% → 0.04% → 0.30% of the median, and K12's is ~0.31% →
0.30% → 0.25%. The exact figures are noisy run-to-run (rerunning the
harness for the spread plot above shifted them somewhat from an earlier
run, where SHA-256 went 0.04% → 0.15% → 0.71%), but the broader pattern —
SHA-256 and K12 having visibly wider CIs at 1024 KB than at 64 KB in most
runs, all still comfortably sub-1% — motivated checking whether this is a
real, chunk-size-driven effect rather than noise.

To see the actual shape of this trend (not just three points) and check
for a sharp cache-capacity-wall effect, `CHUNK_SIZES_KB` was temporarily
widened to a 16 KB–8 MB sweep and rerun (output discarded, not part of
the committed artifact — the CSV was restored via `git checkout` and the
source edit reverted afterward). The widened sweep shows a gradual,
roughly monotonic increase in CI width starting well below the L2 size
(this CPU's L2 is 1 MiB/core per `lscpu`), continuing smoothly through
1 MB and out to 8 MB, rather than a step change exactly at the 1 MiB
boundary:

```
sha256:  16KB→0.02%  64KB→0.04%  256KB→0.15%  1024KB→0.71%  4096KB→0.93%  8192KB→0.88%
k12:     16KB→0.19%  64KB→0.21%* 256KB→0.22%  1024KB→1.10%  4096KB→1.09%  8192KB→7.35%†
```

(`*` excludes one anomalous run where k12@64KB's median itself dropped to
~1054 MB/s, a one-off perturbation not seen elsewhere in the sweep; `†` is
a clear outlier even relative to the otherwise-smooth trend.)

The smooth, gradual growth — rather than a sharp jump at the cache-size
boundary — points to measurement methodology rather than a cache-capacity
wall: each trial times a single `hash_chunk()` call end-to-end via
`Instant::now()`, so the wall-clock window being measured itself grows
with chunk size (roughly 24 µs at 64 KB vs. 360–400 µs at 1 MB vs. several
ms at 8 MB). A longer timed window has proportionally more exposure to
OS-level jitter — scheduler timeslicing, timer interrupts, P-state/turbo
frequency transitions, background cache eviction from other processes —
landing inside the window during at least one of the 30 trials. That
produces occasional high-latency outlier trials, which widen the
bootstrap CI (sensitive to tail samples) much more than they move the
median (robust to a single outlier). This is consistent with BLAKE3
showing the same widening trend despite its different (parallel,
tree-based) internal structure — the effect tracks wall-clock window
length, not an algorithm-specific property.

Practical takeaway: this does not affect the committed artifact's
validity — all three committed chunk sizes (64/256/1024 KB) sit at the
low end of this effect and are comfortably sub-1% CI width, consistent
with the tight-CI claim below. It does mean that extending this benchmark
to much larger chunk sizes in the future should expect proportionally
wider confidence intervals, and that such an extension may benefit from
more trials or from timing a fixed number of repeated calls per trial
instead of a single call.

## Inconclusive results

None — every measured cell's 95% bootstrap CI is tight relative to the
median (sub-1% width in all cases), so no algorithm/chunk-size pairs are
statistically indistinguishable at this sample size.
