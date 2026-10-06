# Four-Way Integrity Benchmark — Explanatory Note

This note accompanies `fourway.csv`, `fourway-ed25519-nofast.csv` and the
LaTeX tables in `fourway.tex`, per the S2-D2 spec's "Benchmark validity and
interpretation" requirement (p.52).

## Why this benchmark exists

P1.6 (`phase1-explanatory-note-COG.md`) compared the Merkle tree only against
flat SHA-256, and P1.2b (`baselines-explanatory-note-localhost.localdomain.md`)
measured flat SHA-256, per-chunk HMAC and per-chunk Ed25519 but never the
Merkle tree. The two were separate runs, and P1.6's Merkle tree used BLAKE3
while every baseline uses SHA-256, so a cross-note comparison conflated tree
structure with hash choice. This harness runs all backends in one process on
the same chunk sets with one definition of each operation, and runs the
Merkle tree with both SHA-256 (hash-matched to the baselines) and BLAKE3.

## Reproduction

Requirements: Rust >= 1.85 (edition 2024), Python 3 (stdlib only), ~11 GB free
disk, >= 16 GB RAM (the loader reads the whole 10.7 GB file, then copies one
3.33 GB dataset out of it). No libhdf5 is needed.

```bash
mkdir -p /tmp/noaa_data
cargo run -p clawhdf5 --release --example generate_10gb -- 10 /tmp/noaa_data/synthetic_10gb.h5
curl -o /tmp/noaa_data/goes18_sample.nc \
  "https://noaa-goes18.s3.amazonaws.com/ABI-L1b-RadC/2024/001/00/OR_ABI-L1b-RadC-M6C01_G18_s20240010001177_e20240010003550_c20240010004002.nc"

# Main run: all five backends, Ed25519 with precomputed tables
cargo run -p clawhdf5 --release --example fourway_bench --features bench-ed25519-fast -- \
  /tmp/noaa_data/synthetic_10gb.h5 /tmp/noaa_data/goes18_sample.nc

# Ed25519 without precomputed tables (the library's default build)
cargo run -p clawhdf5 --release --example fourway_bench -- \
  /tmp/noaa_data/synthetic_10gb.h5 /tmp/noaa_data/goes18_sample.nc \
  --backends ed25519 --tag ed25519-nofast

cd crates/clawhdf5-format/benches/results
python3 fourway_table.py fourway.csv fourway-ed25519-nofast.csv > fourway.tex
```

Output overwrites `fourway.csv`; pass `--tag <name>` to write
`fourway-<name>.csv` instead. Drop the second CSV argument to the table
script to omit the "no `fast`" row. Wall-clock: ~45 min for the main run,
~25 min for the Ed25519-only run on the reference CPU below.

The NOAA object is fetched from a live bucket and may rotate out (see the
P1.6 note's "Known follow-up"); if so, list a current `M6C01` file with
`curl "https://noaa-goes18.s3.amazonaws.com/?list-type=2&prefix=ABI-L1b-RadC/2024/001/00/&max-keys=20"`.

## Hardware and parameters

- AMD Ryzen 9 7900X3D (12-core), recorded in each CSV's `#` header line.
  This is a different machine from the Ryzen 9 9950X3D used for P1.6, so
  absolute times are not comparable to `phase1-COG.csv`; every comparison
  in this note is within this run.
- Single-threaded for every backend. 30 measured trials after 5 discarded
  warm-ups per cell. Medians with 95% bootstrap CIs (2000 resamples).
- Inputs: the P1.1 synthetic file's `dataset_64kb` (54,608 chunks),
  `dataset_256kb` (13,652) and `dataset_1mb` (3,413), 3.33 GiB each, and the
  NOAA GOES-18 `Rad` dataset (240 DEFLATE-compressed chunks, 43.0 KiB average
  on disk). Raw on-disk chunk bytes are hashed, as in P1.6/P1.2b.
- Data is memory-resident: no disk I/O is timed.

## What is measured

Backends: `merkle_sha256`, `merkle_blake3` (`clawhdf5_format::merkle`), and
the unmodified P1.2b `FlatHashBackend`, `PerChunkMacBackend` (HMAC-SHA-256)
and `PerChunkSigBackend` (Ed25519).

- **commit** — build all integrity metadata from every chunk.
- **verify_dataset** — check every chunk against the committed metadata.
- **verify_chunk** — check chunk N/2 holding only the trusted commitment:
  Merkle root + inclusion proof (`verify_proof_standalone`, proof generated
  outside the timed region), HMAC key + tag, or Ed25519 public key +
  signature. Flat SHA-256 cannot, and is recorded as unsupported.
- **update** — overwrite chunk N/2 with same-size data and bring metadata up
  to date, in place (unlike P1.6's `update_merkle`, no per-trial
  `tree.clone()`). Flat SHA-256 rehashes everything.
- **append** — add one chunk. Merkle writes the leaf into the first padding
  slot and recomputes its path; every N here is below its next power of two.
- **metadata_bytes / chunk_proof_bytes / public_verification** — derived,
  not timed. Merkle metadata is the full node array that
  `write_merkle_companion` persists.

The Merkle leaf hash measured is `HashAlg::hash_leaf` (`0x00 || data`,
streamed into the hasher). See "Not measured" for the bound-leaf preimage.

## Results

Full tables: `fourway.tex` (Table 1: all metrics at 256 KiB; Table 2: every
operation across all four datasets).

1. **Tree overhead is negligible.** Merkle-SHA-256, flat SHA-256 and HMAC
   all take 1.39–1.43 s per 3.33 GiB (~2.4 GB/s) at every chunk size. The
   N−1 internal-node hashes cost ~2% at 64 KiB and are within noise above.
   P1.6's "Merkle 2–3x faster than flat" was BLAKE3 vs SHA-256, not the tree.
2. **Merkle-SHA-256 matches HMAC on every operation** (within ~2% bulk,
   ~8% single-chunk at 64 KiB, <1% at 256 KiB and above), while being
   publicly verifiable. Single-chunk verify/update/append all equal one
   chunk hash (26.6 µs at 64 KiB, 98 µs at 256 KiB, ~391 µs at 1 MiB); the
   12–16-level path walk is not measurable.
3. **Merkle-SHA-256 beats per-chunk Ed25519** — the only other publicly
   verifiable per-chunk scheme — by 5.6–6.4x on commit, 2.8–4.1x on
   verify_dataset, and 3–6x on single-chunk operations.
4. **BLAKE3 is a separate 1.8–3.3x gain**, growing with chunk size
   (BLAKE3's intra-input SIMD parallelism improves on longer inputs). Report
   it as a hash-choice result, not a tree result.
5. **Storage is the Merkle tree's cost.** The full companion is the largest
   metadata (4.00 MiB at 64 KiB vs 3.33 MiB Ed25519, 1.67 MiB HMAC): it
   stores internal nodes, and power-of-two padding wastes 20% at
   N=54,608. Single-chunk proofs are 457–521 B vs 64 B for Ed25519.
6. **Projection to §4.3's 10^7 x 1 MiB, single core:** Merkle-BLAKE3 ~0.46,
   Merkle-SHA-256 ~1.2, HMAC ~1.1, flat ~1.1, Ed25519 ~6.5 CPU-hours.

## Anomaly 1: Ed25519 signing is ~2x verification, and `fast` barely matters

Signing (update/append) costs ~2x verify_chunk at every chunk size, with or
without `ed25519-dalek`'s `fast` feature (precomputed base-point tables).
`fast` changed Ed25519 timings by 0–9%, within cross-run noise (256 KiB
commit was 8.28 s with `fast`, 8.14 s without).

Root cause: Ed25519 signing runs SHA-512 over the message twice (nonce
derivation and challenge), verification once. A linear fit of the measured
medians against chunk size gives verify ~16 µs + 1.18 µs/KiB and sign
~5 µs + 2.33 µs/KiB — the per-KiB slope doubles exactly, and the fixed
curve-arithmetic term (what `fast` accelerates) is 5–16 µs against
75–2400 µs of hashing. At these chunk sizes Ed25519 is SHA-512-bound, and
SHA-512 has no SHA-NI acceleration on this CPU while SHA-256 does.

**This corrects P1.2b's Anomaly 1**, which attributed signing being slower
than verifying to `fast` being disabled, and its expectation that enabling
`fast` would bring the 1 MiB-chunk 10^7 projection below ~3 CPU-hours.

## Anomaly 2: wide CIs in four cells

Every cell's 95% CI half-width is within 1.9% of the median except:

- NOAA flat_sha256 verify_dataset (±9.5%) and update (±7.4%): a few slow
  outlier trials on a ~4 ms operation, consistent with scheduler noise.
- 1 MiB append for merkle_sha256 and hmac_sha256 (~±7%): bimodal at
  ~390 µs vs ~500 µs. A 1 MiB chunk equals this CPU's per-core L2, so the
  modes are consistent with the input being L2-resident or not. Both
  backends are affected identically, so the comparison holds.

## Not measured / known limitations

- **Bound-leaf preimage.** `hash_chunk_bound` (introduced with the
  "leaf preimage is a property of the dataset" change) copies the whole chunk
  into a freshly allocated `Vec` before hashing, unlike `hash_leaf`, which
  streams. Datasets using the bound format pay an extra allocation + memcpy
  per leaf hash that these numbers do not include. Streaming the fields into
  the hasher would remove it.
- **Parallelism.** All backends are single-threaded. Merkle, HMAC and
  Ed25519 parallelize per chunk (`from_chunks_parallel` exists); flat SHA-256
  cannot, so a multi-core run would widen the gap to flat.
- **Flat append is pessimistic.** Retaining the SHA-256 midstate would make
  flat append O(chunk); the P1.2b definition rehashes everything.
- **Merkle append across a power of two** (an O(N) internal-node rebuild over
  32-byte digests, no data rehash) is not exercised, and `update_leaf` does
  not advance `leaf_count`, so there is no first-class append API yet.
- **Merkle + one root signature** (the P2.4 hybrid design) is not a row
  here; it would add one ~100 µs signature per commit to the Merkle rows.
- ML-DSA-65 per-chunk signatures are not measured.

## Inconclusive results

None. Apart from the cells in Anomaly 2, CIs are under ±1.9%, and the
backend orderings are well separated in every cell; Merkle-SHA-256 vs HMAC is
the closest pair and is reported as equivalent rather than ranked.
