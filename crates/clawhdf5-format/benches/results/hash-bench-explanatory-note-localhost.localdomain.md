# P1.2 Hash Algorithm Benchmark — Explanatory Note

This note accompanies `hash-bench-localhost.localdomain.csv`, per the S2-D2
spec's "Benchmark validity and interpretation" requirement (p.52): every
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

`plot_hash_throughput.png`: grouped bars per algorithm at each chunk size,
with 95% CI error bars (too narrow to see at this scale — see the CSV for
exact bounds).

## How to read the CSV

Each row is one (algorithm, chunk size) cell: `throughput_mbs` is the
**median of the 30 measured trials**, with `ci95_low`/`ci95_high` from a
**95% bootstrap confidence interval** (2000 resamples) — never a bare mean,
per the statistical protocol.

## Expected trends and whether the data matches

- **BLAKE3 throughput increases with chunk size** (6.6 GB/s at 64 KB → 10.5
  GB/s at 256 KB → 12.2 GB/s at 1 MB). Matches expectation: BLAKE3 processes
  input in parallel 1 KB-leaf lanes internally (SIMD-width dependent), so
  larger inputs better amortize per-call setup cost and expose more
  parallelism — consistent with RQ4's framing of BLAKE3 as the
  parallelism-oriented design.
- **SHA-256 and K12 throughput are flat across chunk sizes** (SHA-256:
  ~2.70–2.72 GB/s; K12: ~1.57–1.58 GB/s). Matches expectation for both:
  neither this `sha2` usage nor `k12`'s sponge construction exploits
  cross-chunk-size parallelism the way BLAKE3's tree does, so per-byte cost
  is roughly constant once warmed up.

## Anomaly: SHA-256 is faster than K12, despite K12 being the
"very fast" candidate

At every chunk size, SHA-256 (~2.7 GB/s) outperforms K12 (~1.58 GB/s) by
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

## Inconclusive results

None — every measured cell's 95% bootstrap CI is tight relative to the
median (sub-0.5% width in all cases), so no algorithm/chunk-size pairs are
statistically indistinguishable at this sample size.
