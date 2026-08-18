# P2.8c — Contiguous Verification Grid: Measured Cost Model

**Status:** Partial. The tile-shape cost model and whole-dataset audit
throughput are measured on two storage classes. The Bao and `h5repack`
baselines are **not** measured — see [Not measured](#not-measured). Nothing
below is extrapolated to the missing conditions.

P2.8a and P2.8b established that contiguous support *works*. This document
reports what it *costs*, and which leaf construction should ship as the
default. A negative result here is a real outcome, and there is one: the
design's whole-dataset audit claim does not hold on fast storage.

---

## Recommendation

**Ship the DAOS-derived shape (`1×250×1000` at a 1 MiB target) as the
default.** The evidence is that it is the only shape that is never
catastrophic, not that it wins everywhere:

- It wins **decisively** where absolute cost is large: on HDD it reads a
  compact 100³ sub-cube in 1.40 s against 2.52 s for `16×16×1000`, and a
  plane in 9.6 ms against 451 ms — a 47× gap.
- It **loses** only for the tiny 10³ sub-cube, 20.8 ms vs 4.7 ms on HDD.
  That is a 16 ms absolute difference on a selection that reads 4 KB of
  useful data; no workload notices it.
- On NVMe the three non-cubic shapes are within 13% of each other, so the
  choice costs nothing there — the default is chosen by its HDD behavior
  because that is the only class where it matters.

The cubic shape should not be offered as an option. It is worst or
near-worst on every selection and every device, and its 256 B runs put it
squarely over the run-length cliff (below).

**On the Phase-3 items (P3.2):** the measured data supports building the
general index-space tile option *only* for workloads dominated by small
sub-cubes, which is the one case where a narrower tile wins — and there the
absolute saving is milliseconds. On this evidence P3.2's tile work is **not
justified** by sub-cube performance alone. The other P3.2 items
(detached-sidecar mode, encryption, partial-leaf writes) are motivated by
capability, not by these numbers, and are unaffected.

---

## What was measured

A 1000³ float32 contiguous dataset (4.0 GB), on two storage classes, for
four leaf shapes × three selections. Reported: bytes actually transferred at
the block layer, discontiguous I/O count, read latency, proof size, and
verify latency.

| | NVMe | HDD |
|---|---|---|
| Device | Samsung SSD 9100 PRO 4TB | WDC WD6002FZWX |
| Path | `/home/brtnfld` (`nvme0n1p4`) | `/home/brtnfld_2025/brtnfld` (`sda1`) |
| Filesystem | ext4 | ext4 |
| Measured sequential read | 12,805 MB/s | 174 MB/s |
| `read_ahead_kb` | 1024 | 1024 |
| logical / physical block | 512 / 512 | 512 / 4096 |
| Scheduler | none | mq-deadline |

**Both classes are ext4 on the same host**, so the device is the only
variable between them — no filesystem confound.

---

## Finding 1 — I/O counts reproduce the analytical table exactly

Every one of the 12 cells matched the published `runs`, `ℓ`, and I/O counts,
and the derived leaf sizes (0.95 / 0.98 / 0.98 / 1.00 MiB) matched to two
decimals. The one deviation is cubic/plane: 1,024,000 measured vs 1,048,576
published, because the published figure ignores truncation of the 16th tile
on a 1000-element axis (1000 = 15·64 + 40). The geometry is validated
independently by `--selftest`, which refuses to run the benchmark if the run
enumerator disagrees with the table.

## Finding 2 — the run-length cliff is real, but the model uses the wrong quantum

Over-read relative to the model's `runs · ⌈ℓ/B⌉ · B` prediction, by run
length (mean across selections; NVMe and HDD agree to within 0.05×):

| run length | measured / model |
|---|---|
| 1,000,000 B | 1.05–1.09× |
| 256,000 B | 1.14× |
| 64,000 B | 1.40× |
| 256 B | **8.5×** |

The cliff appears exactly where the design predicts. **But the model bills
the tax against `B` = the 512 B logical block, and the real quantum is the
1 MiB readahead window.** A 256 B run is not charged one sector; it is
charged against a readahead window ~2000× larger. This is why the model
under-predicts by 8.5× at the bottom of the range and is accurate at the
top, where runs already exceed the window.

Consequence for the model: replace `B` with `max(B, readahead)` when
predicting for a device, and record the readahead setting alongside any
quoted figure. The readahead configuration for both devices is in the table
above, as the plan requires.

## Finding 3 — the published ordering breaks for the compact sub-cube

| selection | measured ordering (bytes) | published | |
|---|---|---|---|
| `cube10` | `16×16×1000` < `4×64×1000` < `1×250×1000` < cubic | same | matches |
| `cube100` | **`1×250×1000`** < `4×64×1000` < `16×16×1000` < cubic | `16×16×1000` first | **differs** |
| `plane` | `1×250×1000` < `4×64×1000` < `16×16×1000` < cubic | same | matches |

The table predicts `16×16×1000` wins `cube100` at 49.0 MiB. Measured, it
costs **101.3 MiB** — 2.07× the prediction — and loses to the DAOS shape's
95.8 MiB. Root cause is Finding 2: its 64 KB runs sit far enough below the
1 MiB readahead window to be taxed, while the DAOS shape's 1 MB runs are
not. The analytical table's `cube100` row should be corrected.

Cubic/plane is over-predicted in the other direction: 4096.0 MiB published
vs **245.9 MiB** measured (0.06×). The published figure is consistent with
`1,048,576 I/Os × 4 KiB`, i.e. charging every run a separate block fetch;
in reality those runs are spatially clustered inside a 4 MB region and
readahead coalesces them. A selection cannot transfer 4 GiB reading a
region that small.

## Finding 4 — the device penalty grows as runs shrink (the γ/β effect)

Median read latency, 30 trials, caches evicted per trial, `cube100`:

| leaf shape | run len | NVMe | HDD | HDD/NVMe |
|---|---|---|---|---|
| `1×250×1000` (DAOS) | 1 MB | 40.5 ms | **1396 ms** | 34× |
| `4×64×1000` | 256 KB | 45.9 ms | 2074 ms | 45× |
| `16×16×1000` | 64 KB | 42.2 ms | **2523 ms** | 60× |
| `64×64×64` cubic | 256 B | 79.7 ms | 3735 ms | 47× |

This is the central result. **On NVMe the first three shapes are within 13%
of each other — leaf shape is nearly irrelevant. On HDD they spread 1.8×,
ordered by run length**, and the HDD/NVMe penalty climbs monotonically
across the first three (34× → 45× → 60×) as runs shrink. The design predicts
the ranking becomes device-sensitive as γ/β changes; it does.

**Anomaly, explained.** Cubic/plane shows an 11× HDD penalty — *lower* than
every other cell, breaking the monotonic trend. Root cause: on NVMe it takes
283.9 ms to issue 1,024,000 reads ≈ 277 ns per read, which is `pread`
syscall overhead, not device time. That cell is **syscall-bound on NVMe and
I/O-bound on HDD**, so its inflated NVMe baseline compresses the ratio. The
binding resource differs by device for this cell alone; it is not evidence
that the cubic shape degrades gracefully.

## Finding 5 — the "tree is nearly free" claim holds only when I/O-bound

Whole-dataset audit, single-pass streaming Merkle build vs a flat BLAKE3
hash over the same 4.0 GB, median of 5 trials, caches evicted:

| storage | flat hash | grid tree | overhead |
|---|---|---|---|
| HDD | 22,774 ms (176 MB/s) | 22,759 ms (176 MB/s) | **−0.1%** |
| NVMe | 619 ms (6,459 MB/s) | 3,352 ms (1,193 MB/s) | **+441%** |

**On HDD the design's claim is confirmed exactly**: both run at the measured
sequential-read bandwidth (176 vs 174 MB/s), so the tree is free — the
device is the bottleneck and hashing hides entirely behind it.

**On NVMe the claim fails.** The device supplies 12.8 GB/s; the flat hash
sustains 6.5 GB/s and the streaming tree build only 1.2 GB/s, so the tree
costs 5.4× rather than being free. Root cause: the streaming build issues
~1,000,000 `Hasher::update` calls of 4000 B each (one per row per leaf)
against ~4000 live hashers, so it runs below BLAKE3's efficient batch size
and thrashes on hasher state, while the flat hash feeds 8 MiB at a time and
gets full SIMD throughput. The build is CPU-bound on NVMe and I/O-bound on
HDD.

This is a **negative result against the design's stated claim**, and it is
actionable rather than fatal: the build is single-threaded here, and leaf
hashing is embarrassingly parallel (`clawhdf5-format` already has a
`parallel` feature using rayon). Parallelising leaf hashing across the 32
available cores, and batching larger spans per `update`, are the obvious
fixes. Neither was attempted here — the number reported is what the current
implementation does.

---

## Not measured

Stated rather than estimated, so no reader mistakes a gap for a result:

- **Parallel/network filesystem (Lustre).** Not available on this host. The
  design predicts the ranking inverts again as γ/β shifts; that prediction
  is untested. Every ordering claim above is scoped to NVMe and HDD.
- **`h5repack` baseline, and therefore the conversion crossover.** No
  libhdf5 is installed on this host, so the reference `h5repack` binary
  could not be run. A clawhdf5-native repack would measure a different
  program and was not substituted silently. Consequently
  `contiguous-baselines.csv`, `setup_*` columns, and the per-storage-class
  `crossover_verifications` figure are **not produced**.
- **Bao / BLAKE3 byte-range baseline.** Not implemented. The design's
  specific predictions (Bao wins ~25× on bytes for the 10³ sub-cube, loses
  ~10× on I/O count and ~18× on proof size, advantage collapsing to ~2.4×
  at 100³) remain **unconfirmed and uncorrected**.
- **`seq_write_mbps`** is absent from the CSV: nothing in the measured set
  writes, and it exists to parameterise the `h5repack` comparison that was
  not run.

## Protocol deviations

- **CPU governor is `powersave`, not `performance`** — changing it needs
  root, unavailable here. This inflates variance on CPU-bound quantities
  (`verify_ms`, and Finding 5's NVMe numbers) but not on the block-layer
  byte and I/O counters. Finding 5's NVMe overhead should be treated as an
  upper bound on the gap for that reason.
- **Cache eviction uses `posix_fadvise(POSIX_FADV_DONTNEED)`**, not
  `/proc/sys/vm/drop_caches`, for the same reason. It is strictly more
  targeted — it evicts only this file's clean pages — and was validated on
  this host: a warm re-read reports 0 bytes through `/proc/self/io`, and
  the same read after eviction reports the full file size.
- Cores were not pinned with `taskset`. The I/O-bound measurements are
  insensitive to this; Finding 5's NVMe figure is the one most affected.

## Reproducing

```bash
# validate the grid geometry against the published table first
cargo bench --features merkle,blake3 --bench contiguous_tileshape_bench -- --selftest

# tile-shape matrix (generates a 4 GB dataset in --dir on first run)
cargo bench --features merkle,blake3 --bench contiguous_tileshape_bench -- \
    --dir <path-on-device> --storage-class <nvme|hdd> --trials 30 --warmups 5

# whole-dataset audit throughput
cargo bench --features merkle,blake3 --bench contiguous_tileshape_bench -- \
    --audit --dir <path-on-device> --storage-class <nvme|hdd> --trials 5

# measured-vs-published comparison
crates/clawhdf5-format/benches/analyze_contiguous_tileshape.py \
    crates/clawhdf5-format/benches/results/contiguous-tileshape.csv
```

`--calibrate` runs one trial per condition and reports per-cell timing, so a
full run can be budgeted before it is started. Each measurement evicts the
data file's page cache immediately before the trial it times.

**What the numbers mean.** `bytes_transferred` is the delta in
`/proc/self/io`'s `read_bytes` across the gather — actual block-layer reads,
excluding page-cache hits, not a computed figure. `io_ops` is the count of
`pread` calls issued, i.e. the discontiguous I/O count. `read_ms` times the
gather alone; `verify_ms` times `verify_subset` over the gathered buffers.
`model_bytes_predicted` is `runs · ⌈ℓ/B⌉ · B` with `B` = 512 B, so
measured-minus-model is one subtraction — and Finding 2 is what that
subtraction shows.

## Artifacts

- `crates/clawhdf5-format/benches/results/contiguous-tileshape.csv` — 24 rows
  (4 shapes × 3 selections × 2 storage classes), median plus 95% bootstrap CI
  over 30 trials.
- `crates/clawhdf5-format/benches/results/contiguous-audit-throughput.csv` —
  raw per-trial audit timings, 5 trials × 2 approaches × 2 storage classes.
- `crates/clawhdf5-format/benches/contiguous_tileshape_bench.rs` — the harness.
- `crates/clawhdf5-format/benches/analyze_contiguous_tileshape.py` — the
  measured-vs-published comparison.
