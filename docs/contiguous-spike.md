# P2.8c — Contiguous Verification Grid: Measured Cost Model

**Status:** Partial. The tile-shape cost model, leaf-target sweep,
whole-dataset audit throughput, and the Bao baseline are measured on two
storage classes. The `h5repack` baseline is **not** measured — see
[Not measured](#not-measured). Nothing below is extrapolated to the missing
conditions.

P2.8a and P2.8b established that contiguous support *works*. This document
reports what it *costs*, and which leaf construction should ship as the
default. A negative result here is a real outcome, and there is one: the
design's whole-dataset audit claim did not hold on fast storage, and two of
the three Bao predictions are refuted. The audit defect has since been fixed
(Finding 5): the streaming build is now 7.87× faster and within 1.36× of the
pure-read ceiling.

---

## Recommendation

**Ship the DAOS-derived shape (`1×250×1000` at a 1 MiB target) as the
default.** The evidence is that it is the only shape that is never
catastrophic, not that it wins everywhere:

- It wins **decisively** where absolute cost is large: on HDD it reads a
  compact 100³ sub-cube in 1.40 s against 2.52 s for `16×16×1000`. On the
  plane selection it reads 4.78 MiB against 62.79 MiB, a ~13× advantage once
  normalised for the drive-cache artefact described below.
- It **loses** only for the tiny 10³ sub-cube, 20.8 ms vs 4.7 ms on HDD.
  That is a 16 ms absolute difference on a selection that reads 4 KB of
  useful data; no workload notices it.
- On NVMe the three non-cubic shapes are within 13% of each other, so the
  choice costs nothing there — the default is chosen by its HDD behavior
  because that is the only class where it matters.
- The 1 MiB **target** is also a true optimum, not an inherited constant: a
  sweep from 256 KiB to 4 MiB puts the `cube100` byte minimum exactly at
  1 MiB (Finding 6). The transferable rule is to match the device's readahead
  window, which is 1 MB on this host.

The streaming builder (`contiguous_tree_streaming`) should be the API used
for whole-dataset audits: it is bounded in memory, parallel, and within
1.36× of the pure-read ceiling, where the original interleaved build was
10.7× off it. `contiguous_tree_and_grid` remains correct but needs the whole
dataset as one slice, so it does not scale to the sizes this work targets.

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

`cube100` is used for this table deliberately: it is the only selection whose
HDD footprint (96–302 MiB) exceeds the drive's 128 MB onboard cache, so these
latencies are genuinely platter-bound. See
[Drive-cache limitation](#drive-cache-limitation) — the `cube10` and `plane`
HDD latencies are *not* trustworthy in absolute terms, and an earlier draft of
this document overstated the plane result because of it.

**Anomaly, explained.** Cubic/plane shows an 11× HDD penalty — *lower* than
every other cell, breaking the monotonic trend. Root cause: on NVMe it takes
283.9 ms to issue 1,024,000 reads ≈ 277 ns per read, which is `pread`
syscall overhead, not device time. That cell is **syscall-bound on NVMe and
I/O-bound on HDD**, so its inflated NVMe baseline compresses the ratio. The
binding resource differs by device for this cell alone; it is not evidence
that the cubic shape degrades gracefully.

## Finding 5 — the audit claim held only when I/O-bound; the build has since been fixed

Whole-dataset audit, single-pass streaming Merkle build vs a flat BLAKE3 hash
over the same 4.0 GB, caches evicted per trial.

**As originally measured** (single-threaded build, interleaved per-leaf hashers):

| storage | flat hash | grid tree | overhead |
|---|---|---|---|
| HDD | 22,644 ms (177 MB/s) | 22,639 ms (177 MB/s) | **−0.0%** |
| NVMe | 626 ms (6,384 MB/s) | 3,478 ms (1,150 MB/s) | **+456%** |

On HDD the design's claim was confirmed exactly: both ran at the measured
sequential-read bandwidth, so the tree was free — the device was the
bottleneck and hashing hid entirely behind it. On NVMe it failed badly. Root
cause: the build issued ~10^6 `Hasher::update` calls of 4000 B each across
~4000 live hashers, below BLAKE3's efficient batch size and thrashing hasher
state, while the flat hash fed 8 MiB at a time.

**This has now been fixed** (P2.9). The structural observation is that a
verification grid's leaves are each a *single contiguous byte run in
increasing index order*, so the interleaved hashers were never necessary:
`subset_proof::contiguous_tree_streaming` reads bounded batches sequentially,
splits them into per-leaf slices by arithmetic, and hashes those slices in
parallel — with a scoped producer thread overlapping the next batch's read
against the current batch's hashing.

| stage | median | MB/s | vs pure-read floor |
|---|---|---|---|
| interleaved (original) | 3,478 ms | 1,150 | 10.7× |
| parallel, serialised reads | 503 ms | 7,952 | 1.55× |
| parallel + double-buffered reads | **442 ms** | **9,057** | **1.36×** |
| *`read_only` control (no hashing)* | *325 ms* | *12,312* | *1.00×* |

**Cumulative 7.87×**, of which parallel leaf hashing contributed 7.20× and
overlapping the reads a further 1.08×. The build now runs within 1.36× of the
hard ceiling for this device and access pattern.

Two honesty notes on this table:

- **The flat-hash row is not a like-for-like baseline.** `blake3`'s `rayon`
  feature is not enabled, so the flat hash is single-threaded (626 ms) while
  the fixed build uses 32 cores. The build now *appears* to beat it by 1.4×;
  that comparison should not be quoted as a win, because a parallelised flat
  hash would improve too. The sound claims are the before/after speedup (same
  workload, same machine, only the builder changed) and the distance to the
  `read_only` control.
- **The residual 117 ms is not explained.** A prediction that overlapping
  reads would reach the 325 ms floor was wrong: only ~74 ms of hash time hid
  behind I/O and 117 ms stayed exposed. One hypothesis was *ruled out* by the
  `read_only` control — the device genuinely delivers 12.3 GB/s on a
  sustained 4 GB read, so the ceiling is real and not a bad denominator. The
  remaining candidates, **not** distinguished by any measurement here, are
  memory-bandwidth contention between the reader's page-cache copy and 32
  concurrent hashing threads, and imperfect pipeline fill across the ~62
  batches. Recorded rather than guessed, so a later reader can resume it.

On HDD the fix changes nothing, exactly as the original explanation predicted:
all three approaches land at 177 MB/s because the platters bind. That is the
confirmation that "free relative to the flat hash" was a statement about the
*device*, not about the construction.

## Finding 6 — 1 MiB is the right target, and the reason generalises

The default target was never itself tested, so it was swept from 256 KiB to
4 MiB (the DAOS rule re-derives a different grid at each), 30 trials,
measured bytes:

| target | derived grid | run len | `cube10` | `cube100` | `plane` | proof (c100) |
|---|---|---|---|---|---|---|
| 256 KiB | `1×63×1000` | 252 KB | 2.44 MiB | 173.84 MiB | 5.53 MiB | 48,352 B |
| 512 KiB | `1×125×1000` | 500 KB | 4.81 MiB | 148.07 MiB | 4.53 MiB | 32,352 B |
| **1 MiB** | `1×250×1000` | 1000 KB | 9.57 MiB | **95.75 MiB** | 4.78 MiB | 20,352 B |
| 2 MiB | `1×500×1000` | 2000 KB | 29.60 MiB | 296.84 MiB | 5.53 MiB | 16,352 B |
| 4 MiB | `1×1000×1000` | 4000 KB | 39.53 MiB | 382.53 MiB | 5.53 MiB | 12,352 B |

**Going above 1 MiB never helps and usually hurts.** `cube100` costs 3.1×
more bytes at 2 MiB and 4.0× more at 4 MiB; `cube10` degrades monotonically;
`plane` is flat. Read amplification (whole leaves read per useful byte) rises
without bound as leaves outgrow the selection — analytically 25× at 1 MiB,
100× at 4 MiB, 119× at 64 MiB for `cube100`.

**`cube100` has a true minimum exactly at 1 MiB**, and the mechanism is
Finding 2. At a 1 MiB target the run length (1,000,000 B) essentially equals
this host's readahead window (1,048,576 B), so measured bytes are 100.4 MB
against 100 MB wanted — a 1.0× tax. At 256 KiB the amplification model
predicts a *better* result (18.9× vs 25×), but the 252 KB runs fall under the
readahead window and are taxed 2.4×, so it measures 1.8× **worse**. The
readahead tax more than cancels the amplification advantage.

Smaller targets do win for the tiny `cube10` selection (2.44 MiB at 256 KiB
vs 9.57 MiB at 1 MiB), and larger targets do shrink proofs monotonically
(48 KB → 12 KB for `cube100`). Neither outweighs the `cube100` byte minimum.

**The generalisable rule is better than the constant:** set the leaf target
to the device's readahead window (`/sys/block/<dev>/queue/read_ahead_kb`),
which is 1 MB here and is a tunable, not a law. "Use 1 MiB" is the right
default *on this class of host*; "match the readahead window" is the rule
that transfers.

## Finding 7 — the Bao comparison: right about I/O, wrong about bytes and proofs

Bao is BLAKE3's own Merkle tree exposed for verified streaming: fixed 1 KiB
leaves over a flat byte stream. It is the strongest existing alternative to
the verification grid — same problem, opposite granularity choice — so it is
the fair baseline. Measured with the reference `bao` 0.13.1 crate in outboard
mode (a 249 MB sidecar for the 4 GB dataset), against the DAOS grid at a
1 MiB target, on NVMe:

| | selection | grid | Bao | outcome | predicted |
|---|---|---|---|---|---|
| bytes | `cube10` | 10,039,296 | 2,584,576 | Bao wins **3.9×** | Bao wins 25× |
| | `cube100` | 100,405,248 | 105,144,320 | Bao **loses** 1.05× | Bao wins 2.4× |
| | `plane` | 5,013,504 | 5,799,936 | Bao loses 1.16× | — |
| I/O ops | `cube10` | 10 | 100 | Bao loses **10×** | Bao loses 10× |
| | `cube100` | 100 | 10,000 | Bao loses 100× | — |
| | `plane` | 4 | 1 | Bao wins 4× | — |
| proof (naive) | `cube10` | 2,352 | 141,856 | Bao loses **60×** | Bao loses 18× |
| | `cube100` | 20,352 | 14,390,000 | Bao loses 707× | — |
| | `plane` | 912 | 1,664,256 | Bao loses 1825× | — |
| proof (dedup) | `cube10` | 2,352 | 16,416 | Bao loses 7× | — |
| | `cube100` | 20,352 | 1,357,472 | Bao loses 67× | — |
| | `plane` | 912 | 125,856 | Bao loses 138× | — |

**The I/O-count prediction is exactly right** — 10× for the 10³ sub-cube, to
the digit. The other two are not.

*Measurement note.* `cube10` and `plane` naive proof sizes are a full census
(100 and 1000 slice extractions). `cube100`'s is an **estimate**: 2,000 of
its 10,000 ranges were extracted and scaled by the sampled mean. The ranges
are structurally identical — same length, same tree depth, differing only in
position — so the estimate is faithful, but it is an estimate and is marked
as one rather than presented as a count. Deduplicated proof sizes are exact
for all three: they are computed as the distinct witness set over every
covered chunk, with no sampling.

**The byte prediction is wrong, and Finding 2 explains why.** Bao was
predicted to win 25× on bytes at `cube10`; it wins 3.9×. At `cube100` it was
predicted to win 2.4×; it *loses* 1.05×. Root cause: the prediction assumes
fetching a 1 KiB chunk costs 1 KiB. It does not. Bao touched 103 chunks for
`cube10` — 105,472 B of ideal traffic — and actually transferred 2,584,576 B,
a **24.5× over-read**, because every scattered 1 KiB chunk is billed against
the same 1 MiB readahead window that Finding 2 identified. The finer the
granularity, the worse that tax bites, so Bao suffers from it more than any
grid shape does. The two findings are the same mechanism.

**The proof-size prediction is wrong in the harsh direction**: 60× rather
than 18× at `cube10` naive, and 707× at `cube100`. The design's claim that
proof size "grows to exceed the delivered data" is **confirmed and then
some** — at `cube100` the naive proof is 14.39 MB against 4.00 MB of
delivered data, i.e. **3.6× the payload**. Deduplication helps a great deal
(707× → 67×) but does not close the gap.

**The decisive result is the `plane` control.** That selection is a single
contiguous 4 MB byte range, so Bao's flat-stream model costs it nothing —
it needs just 1 I/O against the grid's 4, its best showing anywhere. Yet its
proof is still **138× larger than the grid's even fully deduplicated**
(125,856 B vs 912 B). Since dimensional scatter is entirely absent from this
case, **Bao's proof-size disadvantage is intrinsic to the 1 KiB leaf size,
not an artefact of hyperslab geometry**. The 4 GB dataset is 4,194,304 Bao
leaves against ~4,000 grid leaves — a 1000× leaf count, hence a 22-deep tree
against 12 — and every covered chunk drags its own witness path. That is the
structural argument for array-aware leaves, and it is not visible in the
analytical treatment, which only ever compares scattered selections.

**Bao's wall-clock advantage is an NVMe artefact, and it inverts on HDD.**
Median read latency, grid (DAOS @ 1 MiB) vs Bao:

| selection | grid NVMe | Bao NVMe | grid HDD | Bao HDD | HDD winner |
|---|---|---|---|---|---|
| `cube10` | 2.45 ms | **1.81 ms** | **20.5 ms** | 65.2 ms | grid by 3.2× |
| `cube100` | 39.1 ms | **26.7 ms** | 1492 ms | **1222 ms** | Bao by 1.2× |
| `plane` | **0.80 ms** | 1.02 ms | **9.1 ms** | 32.0 ms | grid by 3.5× |

On NVMe Bao wins two of three; on HDD it loses two of three, and its one
remaining win narrows to 1.2×. This is Finding 4's γ/β effect applied to the
baseline comparison: Bao's 1 KiB chunks become scattered seeks on rotational
media, and 100 scattered reads cost more than 10 contiguous ones however
few bytes each carries. (The `cube10` and `plane` HDD rows sit in the
drive-cache regime — see below — which if anything *flatters* the grid there;
but Bao's `cube10` at 2.58 MB in 65.2 ms is 40 MB/s, far under sequential, so
it is genuinely seek-bound and the direction is robust.)

**Setup cost.** Building the outboard took **6.1 s on NVMe and 134.6 s on
HDD** — a 22× spread, reported separately from steady-state verification and
never summed into it. The 249 MB sidecar is 6.2% of the dataset, against the
grid's ~128 KB of companion nodes for 4,000 leaves.

So the honest summary is narrower than either side's prior: if verification
is local, on flash, and proofs never travel, Bao is competitive and
occasionally faster. The verification grid's case rests on proof size, I/O
count, and rotational media — which is precisely the archival-storage regime
that motivates contiguous support in the first place.

---

## Drive-cache limitation

`posix_fadvise(POSIX_FADV_DONTNEED)` evicts the kernel page cache but has no
reach into the **drive's own 128 MB DRAM cache** (WD Black 6TB). Any HDD
condition whose footprint fits in that cache is therefore served at SATA-link
speed rather than platter speed. Implied throughput makes this unmistakable:

| condition | bytes read | implied MB/s | verdict |
|---|---|---|---|
| `cube10`, all targets ≤ 2 MiB | 2–30 MiB | 416–513 | drive cache |
| `plane`, all targets | 4–6 MiB | 317–528 | drive cache |
| `cube100`, all targets | 96–387 MiB | 67–172 | **platter-bound, trustworthy** |

The drive's measured sequential read is 174 MB/s, so any figure materially
above that cannot have come from the platters.

**What this invalidates.** An earlier draft of this document claimed the DAOS
shape beats `16×16×1000` on the plane selection by **47×** (9.6 ms vs 451 ms)
on HDD. That compared a cache-served number against a platter-served one and
is **overstated**. Normalising both to the measured 174 MB/s sequential rate
via their (trustworthy) byte counts — 4.78 MiB vs 62.79 MiB — gives a real
advantage of about **13×**. The conclusion is unchanged; the magnitude was
inflated 3.6×, and the corrected figure is the one to quote.

**What survives untouched:** all byte and I/O counts (Findings 1–3, 6) are
read from `/proc/self/io` and are not timings; the `cube100` latencies
(Finding 4) exceed the cache; and the audit throughput (Finding 5) streams
4.0 GB, far beyond it.

Fixing this properly needs either a device larger-footprint than its cache
for every condition, or `hdparm -W0`/O_DIRECT — neither available without
root here.

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
- **`seq_write_mbps`** is absent from the CSV: nothing in the measured set
  writes, and it exists to parameterise the `h5repack` comparison that was
  not run.

## Protocol deviations

- **CPU governor is `powersave`, not `performance`** — changing it needs
  root, unavailable here. This inflates variance on CPU-bound quantities
  (`verify_ms`, and Finding 5's NVMe numbers) but not on the block-layer
  byte and I/O counters. Finding 5's NVMe figures should be read with that
  caveat, including its `read_only` control, which is affected equally and so
  does not bias the ratio between them.
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
  raw per-trial audit timings: NVMe 7 trials × 4 approaches (including the
  `read_only` ceiling control), HDD 5 trials × 3.
- `crates/clawhdf5-format/benches/results/contiguous-target-sweep.csv` —
  leaf-target sweep, 5 targets × 3 selections × 2 storage classes.
- `crates/clawhdf5-format/benches/results/contiguous-baselines.csv` — the Bao
  baseline, 3 selections × 2 storage classes, with naive and deduplicated
  proof sizes.
- `crates/clawhdf5-format/benches/contiguous_bao_baseline.rs` — the Bao harness.
- `crates/clawhdf5-format/benches/contiguous_tileshape_bench.rs` — the harness.
- `crates/clawhdf5-format/benches/analyze_contiguous_tileshape.py` — the
  measured-vs-published comparison.
