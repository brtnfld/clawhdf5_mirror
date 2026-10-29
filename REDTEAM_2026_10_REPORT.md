# Red-Team Engagement 2026-10 — ClawHDF5

**Repo:** clawhdf5 · **Window:** 2026-10-08 → 2026-10-29
**First pass:** branch `main`, 2026-10-08 → 10-10, 4 fix commits, `5b84998..6c89a91`
**Follow-up pass:** branch `review-round3`, 2026-10-28 → 10-29, 10 fix commits, `be7bd2d..b2cc968`

A second adversarial pass over the Merkle/signing/AEAD stack and the
pure-Rust HDF5 parser, three months after the P2.4 engagement. Scope was
the S2-D2 "schema" (the on-disk layout the design document, the C
integration spec, and `clawhdf5-format`'s `MerkleAttr`/`canonical_payload`
all describe) *and* every public verifier it feeds. The P2.4 report
(`P2.4_REDTEAM_REPORT.md`) is the baseline this engagement started from;
nothing listed there is re-reported here, except one P2.4 refutation that
this engagement found to be wrong (see "Corrections").

The engagement ran in two passes. The first pass, closed on 2026-10-10,
covered only part of the planned lanes. Asked whether the white-box review
was complete, the honest answer was no, so a follow-up pass worked through
the remaining lanes one at a time. "Coverage" below lists what each pass
reviewed and what neither pass reached.

| | First pass | Follow-up | Total |
|---|---|---|---|
| Findings fixed | 16 | 24 | 40 |
| Critical | 1 | 0 | 1 |
| High | 7 | 11 | 18 |
| Medium | 7 | 10 | 17 |
| Low | 1 | 3 | 4 |
| Documented, not fixed | — | 3 | 3 |
| Harness detection | 29/33 | 30/33 | 0 false positives throughout |

The harness moved from 29/33 to 30/33 between the passes. That change came
from `9120fb7` and `6a0e7d6` (subset verification for `LeafFormat::Bound`
datasets, and T4b routed through the real verifiers), which landed outside
this engagement. The follow-up pass then tightened four harness rows so
that a verifier that rejects everything can no longer score a detection
(`b2cc968`). The count did not change.

## Executive summary

**First pass.** Four classes of real defect showed up, each confirmed by a
reproduction against the actual code before any fix:

- The hybrid-signed payload left out two attribute fields that *change
  what the root means*.
- The agent-crate loader panicked on crafted on-disk state before any
  integrity check could run.
- The agent WAL's strict mode did not reach the sidecar.
- The untrusted-HDF5 parser had the July engagement's bug class in files
  the July engagement did not touch.

The signing finding is the sharpest. The canonical signed payload R was
exactly what the design equation specified — `root || companion_hash ||
version || timestamp || alg_id`, 81 bytes — but the attribute had grown
two more fields since: `leaf_format` (which preimage the tree was built
with) and `grid_hash` (the chunk-grid binding). An attacker with file
write access and no signing key could flip a `Bound` dataset to `Content`.
They could then re-frame every stored chunk as the bound preimage suffix
`be32(8) || le64(k) || be32(len) || data || le64(v)`. Every verifier then
accepted bytes the signer never committed to, while the signature still
verified. Payload v1 (114 bytes) brings every field a verifier reads out
of the attribute under the signature. Signatures over the v0 layout are
now rejected at parse time rather than verified against a payload they
never bound.

**Follow-up pass.** The most serious follow-up findings were in the
configurations the first pass never exercised end to end:

- **Encrypted.** A writer that followed the documented reopen protocol
  after a clean commit restarted every committed chunk's counter at 1. Its
  next write reused the nonce, a two-time pad with no attacker needed.
- **Signed.** A stored signature could not be verified from the file at
  all: `merkle_sig` carried no timestamp, so every "verify from the file"
  path, the secure demo's included, was reusing values it still held from
  writing. Deleting the newest provenance-journal record also rolled
  recovery back past a state the verifier had already accepted.
- **Untrusted parse.** A 261 KB deflate chunk decoded to 256 MiB, and the
  P2.4 engagement had recorded this attack as refuted. In addition,
  `read_as_slice` let safe code reinterpret file bytes as `&'static str`.
- **Integration surface (OpenClaw, Android, Node).** Text-only OpenClaw
  stores, the mode ZeroClaw uses, could not be reopened after their first
  flush. They raised a false tamper alarm, a defect present since the P2.4
  integrity wiring in July. The Android bridge shipped with integrity
  protection compiled out, and a panic in either native binding aborted
  the host process.

Every fix has a regression test. Each test is the crafted-bytes or
crafted-sequence reproduction, checked against the pre-fix tree where the
pre-fix tree could run it.

## Method

Three guard-rails from the P2.4 engagement stayed in force:

- **Re-verify every "confirmed" finding against the actual code before
  fixing it.** Leads were reproduced with a throwaway probe first. Probes
  for allocation-exhaustion cases ran under `ulimit -v 4194304` so an
  abort was observable rather than swapping. Leads that did not reproduce
  are listed in "Investigated and refuted."
- **Reproduce the attack as a test *first*, then fix.** Where possible the
  regression was run against the stashed pre-fix tree to confirm it fails
  there.
- **Fix as you go.** One commit per lane, each with the attack scenario,
  the root cause, what was checked and found sound, and the verification
  run in the message. The commit log is the audit trail; the entries below
  summarize it.

Lanes were run one at a time. Parallel sweeps exhausted the model's
session limit three times at the start of the engagement before anything
was written to disk. A serial pass finished each lane before the next
began.

The audit is manual white-box review with targeted reproductions. No
coverage-guided fuzzing was run in either pass.

## Coverage

| Area | First pass | Follow-up |
|---|---|---|
| Signed payload (`canonical_payload`, `verify_signed_root`) | ✓ | — |
| Signature storage (`merkle_sig`), Ed25519/ML-DSA backends, feature flags | — | ✓ |
| Provenance journal and restore (`merkle_recovery`) | — | ✓ |
| Subset proofs, SFC orders, contiguous layout | ✓ | — |
| Merkle core (domain separation, `MerkleAttr::unpack`, companion vs. inline) | partial | ✓ (no findings) |
| Parser: datatype, B-tree v2, chunked read | ✓ | — |
| Parser: filter pipeline / decompression, data_read, dataspace | — | ✓ |
| Parser: fixed and extensible array indexes, object header, global heap / VL, fractal heap | — | ✓ |
| `unsafe` blocks in `clawhdf5-format` and the `clawhdf5` facade | — | ✓ |
| AEAD filter: nonce derivation, version WAL, key separation | — | ✓ |
| Agent: loader, WAL, restore | ✓ | — |
| Agent: openclaw, async_memory, provenance, knowledge, consolidation, agents_md | — | ✓ |
| Agent: vector search, BLAS paths; `clawhdf5-accel` SIMD length guards | — | ✓ |
| `clawhdf5-io`: S3 range-GET proofs, HSDS client | — | ✓ |
| FFI: `clawhdf5-android`, `clawhdf5-napi`, `clawhdf5-py` | — | ✓ |
| Attack harness: per-row trace to a real verifier, control arm | — | ✓ |

**Not reviewed by either pass:**

- `clawhdf5-ann`. Its public `HnswIndex::load_from_hdf5` parses bytes,
  but nothing in the workspace calls it: the agent rebuilds the index from
  its cache. It is the largest unreviewed parser of untrusted input.
- `clawhdf5-gpu`, `clawhdf5-netcdf4`, `clawhdf5-migrate`,
  `clawhdf5-derive`, and `clawhdf5-bench`. Their test suites were run as
  downstream checks only.
- `clawhdf5-cli` beyond the `--strict` flag.

## Findings — first pass (2026-10-08 → 10-10)

### Hybrid-signed payload omitted `leaf_format` and `grid_hash`
**Critical · Fixed** · `clawhdf5-sign/src/lib.rs` (`canonical_payload`,
`HybridSignature`), `clawhdf5-format/src/merkle.rs` (`SignedRootVerifier`,
`verify_signed_root`)

The canonical signed payload R was exactly the design equation —
`root || companion_hash || version || timestamp || alg_id`, 81 bytes —
but the `merkle_root` attribute had grown two more fields that give the
root its meaning:

- `leaf_format`, added in commit `87f6b08` on 2026-08-27. It is protected
  only by the attribute's unkeyed integrity hash, which a storage-level
  attacker recomputes with no key material at all.
- `grid_hash`, added in P1.5. Nothing protects it, and its preimage is
  public.

Two attacks followed:

- **Leaf-format downgrade with re-framing.** Flip a `Bound` dataset to
  `Content`, recomputing the unkeyed integrity hash with the public
  constructor. Then rewrite each stored chunk as the bound preimage suffix
  `be32(8) || le64(k) || be32(len) || data || le64(v)`. Under `Content`,
  `H(0x00 || framed)` equals the original bound leaf, so the signed root
  is untouched: `verify_signed_root`, `verify_dataset`, and `verify_chunk`
  all accept bytes the signer never committed to. The attack strips the
  index and version binding that the `Bound` format exists to provide. It
  needs neither a hash collision nor a signature forgery.
- **Grid rewrite.** Rewrite `dims`/`chunk_shape` in the unsigned object
  header and recompute `grid_hash`. The root is unchanged and the
  signature passes. `verify_subset` trusts the attribute's `grid_hash`,
  exactly as the P2.3 security review intended. It therefore maps the
  same requested hyperslab to a different leaf and delivers the wrong
  chunk under a valid proof.

Both attacks were reproduced against the real Ed25519+ML-DSA-65 signer
before the fix. The first ran end to end in `verify_dataset`, the second
in `verify_subset`.

**Fix:**

- Payload v1 is 114 bytes: it adds `leaf_format(1)` and `grid_hash(32)`
  after `alg_id`.
- `SignedRootVerifier` now receives the whole `MerkleAttr`, so any field
  added later is at least available to the signer.
- `HybridSignature::VERSION` was bumped from 0 to 1, and `from_bytes`
  checks it. A stored v0 signature is rejected at parse time rather than
  verified against a payload it never bound.
- The spec table (`docs/c-hdf5-integration-spec.md`) and the
  hybrid-signature test vector were updated to match.
- Three new regressions pin both attacks and the v0 rejection.

### Agent loader panicked on crafted parallel-dataset lengths
**High · Fixed** · `clawhdf5-agent/src/schema.rs`,
`clawhdf5-agent/src/integrity.rs`

The content Merkle root's leaf for entry `i` is built from
`X.get(i).unwrap_or(default)`, so it cannot tell a missing trailing
element from a present default one. As a result, a `/memory/tombstones`
dataset shortened from `[n]` to `[1]`:

- hashed to the honest root and loaded under `open_strict`;
- undercounted active entries;
- made `delete()` report `NotFound`;
- panicked `get_chunk()`, and the `compact()` that `delete()` triggers
  once the tombstone fraction crosses its threshold.

Other datasets were worse: a short `/sessions/summaries`, or short entity
or relation arrays, indexed out of bounds before `verify_content` could
even run.

**Fix:** The loader rejects any per-entry dataset whose length disagrees
with its driver count, with `MemoryError::Schema`. This covers `/memory`,
`/sessions`, entities, relations, aliases, and the flat embeddings array
against `n * embedding_dim`. `integrity.rs`'s module doc records length
agreement as the loader's invariant, not the root's.

### Untrusted `/meta` attributes reached arithmetic before verification
**High · Fixed** · `clawhdf5-agent/src/schema.rs`

`embedding_dim` is not covered by the root, because it says how to parse
what the root covers. It was read as `extract_i64_attr(..) as usize`. A
value of 0 reached `slice::chunks(0)` and panicked, and a negative value
became a huge `usize`. Entity ids are stored as `u64 as i64`, so an id of
2^63 or more read back negative and `max + 1` overflowed. In debug builds
that panicked; in release it wrapped `next_id` to 0, so new entities
collided with existing ones.

**Fix:** Counts are parsed by a new `extract_count_attr` with a minimum (1
for `embedding_dim`). Ids pass through `non_negative` and `checked_add`. A
`/meta` value out of range is a `MemoryError::Schema`.

### WAL allocations driven by u32 length prefix
**High · Fixed** · `clawhdf5-agent/src/wal.rs`

`vec![0u8; len]` and `Vec::<f32>::with_capacity(count)` were sized
straight from a u32 prefix in the sidecar. A 4-byte edit to the WAL made
every `open`, `open_strict`, and `WalFile::open` attempt allocate 4 GiB
(string) or 16 GiB (embedding) before reading a single byte of payload.
Under `ulimit -v 2G` the Rust allocator aborted with SIGABRT. Without a
limit the allocation "succeeded" lazily, and the entry was then dropped as
if truncated.

**Fix:** Every prefix is bounded by the bytes actually remaining in the
file before any allocation. A larger prefix is handled exactly like a
truncation, which is the module's existing crash-safe path.

### `open_strict` accepted a version-1 (unchained) WAL sidecar
**High · Fixed** · `clawhdf5-agent/src/lib.rs`, `wal.rs`

The strict flag only reached `storage::read_from_disk_strict`. The WAL
branch was identical for strict and non-strict opens and called the
version-tolerant `WalFile::read_entries`. An attacker could rewrite the
sidecar's header version byte from `2` to `1` and drop the chain trailers,
stripping the v2 chain with less effort than recomputing it. Forged `Save`
and `Tombstone` entries then replayed under `open_strict`, and the process
kept appending unchained v1 records.

**Fix:** New `WalFile::open_strict` and `read_entries_strict` reject a v1
sidecar with `MemoryError::Integrity`, and `HDF5Memory::open_strict` uses
them. The CLI gains a global `--strict` flag (`CLAWHDF5_STRICT`) so an
operator can opt in. Previously every subcommand used the fail-open
`open()`, with no alternative.

### Chain-coverage gaps in the v2 WAL
**Medium · Fixed** · `clawhdf5-agent/src/wal.rs`

Two record types escaped the v2 chain:

- `ActivationUpdate` (`0x03`) records hit `continue` before the trailer
  read, so unchained 9-byte blobs could be spliced anywhere in a chained
  file.
- A present but unrecognized type byte silently ended replay and dropped
  every later entry, contradicting the module's own truncation-vs-
  corruption rule.

Both are now chain failures in v2.

### `restore_to_version` staged the journal's `snapshot_ref` before any check
**Medium · Fixed** · `clawhdf5-agent/src/storage.rs`

The signed payload binds root, version, and timestamp, but not the
snapshot path. A record naming a FIFO, a device, or a multi-GB file was
therefore copied in full next to the live file before any gate ran, and a
failed copy left the partial temp file behind. Fixed: only regular files
are staged, and the copy-failure path cleans up.

### `Datatype::parse` recursed with no depth guard
**High · Fixed** · `clawhdf5-format/src/datatype.rs`

Seven recursive call sites had no depth cap. They span Compound members,
Enum base, VL base, and Array base, in both the v1/v2 and v3–5 encodings.
A ~24 KiB datatype made of 16,000 nested VL types aborted the process
(SIGABRT) on any public `open` or `shape` call. Array and Compound chains
gave similar reach at about 5–13 bytes per level. The reproduction killed
the test runner process; after the fix the same input returns
`FormatError::NestingDepthExceeded`.

**Fix:** The recursion now goes through private `parse_with_depth` seams,
with a new `MAX_NEST = 32`. Real HDF5 datatypes nest 2–3 levels, so this
leaves a 10× safety margin, and it matches the sibling parsers' depth
caps.

### B-tree v2 recursion bounded only by the file's `depth` header
**High · Fixed** · `clawhdf5-format/src/btree_v2.rs`

`collect_internal_records` recursion was bounded only by the file's own
`u16 depth` field. A crafted header with `depth = 65535`, plus a BTIN node
that pointed its child at itself, cascaded 64K recursive calls and
overflowed the stack. This is the same class as the July engagement's
`btree_v1` and `chunked_read` fix. The sibling parsers already had
`MAX_BTREE_DEPTH` and `MAX_CHUNK_BTREE_DEPTH` of 64; this parser did not.
New `MAX_BTREE_V2_DEPTH = 64`.

### `chunked_read` rank underflow on empty `chunk_dimensions`
**Medium · Fixed** · `clawhdf5-format/src/chunked_read.rs` (4 sites)

With a zero-length `chunk_dimensions`, `rank = ndims - 1` wrapped to
`usize::MAX`. The following `chunk_dimensions[..rank]` slice then panicked
in release builds, and the subtraction itself panicked in debug builds.
Every site now rejects this before the subtraction.

### Five unchecked `c_addr + size` sites in `chunked_read`
**Medium · Fixed** · `clawhdf5-format/src/chunked_read.rs` (5 sites)

This is the same bug class as the July engagement's `offset + N` fix in
the same file. A crafted B-tree key, or fixed/extensible-array entry, with
an address near `usize::MAX` wrapped past the bound check and panicked the
following slice. All five sites now use `checked_add`, and failure returns
`FormatError::Overflow`.

### `generate_implicit_chunks` had five arithmetic hazards
**Medium · Fixed** · `clawhdf5-format/src/chunked_read.rs`

- A zero chunk dimension became a division by zero in `div_ceil`.
- The chunk-byte-size product overflowed `u64`.
- The `total_chunks` product overflowed `u64`.
- `Vec::with_capacity(total_chunks)` allocated terabyte-scale buffers from
  the attacker-controlled count.
- `base_address + linear_idx * chunk_byte_size` wrapped past `u64::MAX`.

All five are now checked, and the function returns `Result`. The four
production callers and the one `data_read` caller were updated.

### Space-filling-curve bit-budget overflow on skewed grids
**High · Fixed** · `clawhdf5-format/src/subset_proof.rs`

`morton_index` gives each axis `floor(64/rank)` bits and drops the rest.
`hilbert_index` needs `hilbert_bits * rank` bits, and its doc claimed the
padded-leaf-count check guaranteed that. It does not: the check caps the
total chunk count at 2^39, not the per-axis extent. Dims `[2^20,1,1,1]`
pass it yet need 80 Hilbert bits, and `[2^33,1]` passes it yet needs 33
Morton bits per axis against a budget of 32.

Past the budget, two distinct chunk coordinates map onto one leaf. The
verifier's expected chunk set for a hyperslab covering both is then one
index short, and a proof that delivers one chunk for two logical regions
passes as complete, violating R5b. This is not attacker-triggerable once
`grid_hash` is signed. But a writer who chose such a curve for such a grid
would get a tree that verifies against itself and certifies the wrong
coverage.

**Fix:** New `MerkleError::LeafOrderOverflow` and
`check_order_bit_budget`, called in `compute_expected_chunk_indices`
before any coordinate is mapped. Row-major order is unaffected.

### Saturating selection and chunk-space bounds
**Medium · Fixed** · `clawhdf5-format/src/subset_proof.rs`

`selection_chunk_bounds` wrapped on extreme operands at three points:

- `n_per_dim * chunk_shape`, with a `chunk_shape` near 2^64;
- the hyperslab end `start + (count-1)*stride + block`;
- the point set's `max + 1`.

A wrap turned "inside the grid" into "no chunks", and an empty expected
set is exactly what an omission attack needs the verifier to derive. All
three now saturate.

### `contiguous_layout` returned mutually inconsistent untrusted fields
**Medium · Fixed** · `clawhdf5-format/src/subset_proof.rs`

The layout message's `size`, the dataspace `dims`, and the datatype size
are independent fields of an untrusted object header. The function's
contract tells callers to slice `file[data_addr..data_addr + nbytes]` and
hand the buffer, with `dims` and `elem_size`, to the grid builders. Those
builders tile it assuming `len == product(dims) * elem_size`. A `size`
that disagreed with the dataspace, or an address range wrapping past
`u64::MAX`, was therefore a crafted-file panic in the caller or the
builder (T7) rather than a verification failure. Both cases are now
rejected as `MalformedHeader` before any data byte is touched.

### Spec document stale after `87f6b08`
**Low · Fixed** · `docs/c-hdf5-integration-spec.md`

Section 1.2's attribute table still showed the 129-byte,
pre-`leaf_format` layout after commit `87f6b08` grew the attribute to 130
bytes. Section 1.5's payload table omitted `leaf_format` and `grid_hash`
after this engagement's first commit made the hybrid signature cover them.
Both tables, plus §7's "Leaf hash formula" row, were updated to match the
code.

## Findings — follow-up pass (2026-10-28 → 10-29)

### A reopened encrypting writer reused a committed chunk's nonce
**High · Fixed** · `2715466` · `clawhdf5-filters/src/version_wal.rs`,
`chacha20_filter.rs`

`EncryptedChunkWriter::recover` is documented as the step to run "after
reopening an existing WAL, before any new write". It seeded the version
store only from *pending* WAL records, and `VersionWal::compact` deleted
committed records outright. Following the protocol after a clean commit
therefore restarted every committed chunk's counter at 1. The next write
to that chunk re-derived the same `(DEK, chunk_idx, version)` nonce.

This was reproduced with and without compaction:

1. Session 1 wrote chunk 0 at version 1 and committed.
2. Session 2 (new writer, then `recover`, nothing pending) wrote chunk 0
   at version 1 again.
3. The two ciphertexts XORed to the XOR of the two plaintexts, a two-time
   pad.

No attacker was involved.

**Fix:**

- New `VersionWal::high_water_marks` returns the highest version journaled
  per chunk over every record, pending and committed.
- `recover` seeds the version store from it.
- `compact` keeps each chunk's highest committed record.

The WAL is now bounded by the number of chunks ever written rather than
by the number of writes.

### A stored signature could not be verified from the file
**High · Fixed** · `be7bd2d` · `clawhdf5-sign/src/lib.rs`

`merkle_sig` held the bare 3,374-byte signature. Nothing in the file
carried the timestamp that `canonical_payload` binds, and without a
`merkle_version` attribute nothing carried the version either. Every
caller that "verified the file" was in fact reusing the version and
timestamp it still held from writing it. That includes the secure demo's
Stage 3, labeled "reopen from scratch & verify (out of the file, not held
state)". A second process holding only the file and the trusted keys had
no way to check the signature.

**Fix:**

- `merkle_sig` is now the 3,391-byte layout already proposed in C spec
  §1.5: `alg_id(1) || version(8,BE) || timestamp(8,BE) ||
  HybridSignature(3374)`.
- `SignedRootMeta` describes that prefix, and new `verify_sig_attr`
  rebuilds the v1 payload from `merkle_root` plus the prefix.
- An `alg_id` that disagrees with `merkle_root`'s is
  `SignError::AlgorithmMismatch`.
- A bare pre-v1 attribute is rejected.

### Journal truncation rolled recovery back past a verified state
**High · Fixed** · `be7bd2d` · `clawhdf5-format/src/merkle_recovery.rs`

The provenance journal is an unsigned attribute. Deleting its newest
record leaves an older, validly signed record as "last known good", and
`RestoreTarget::LastKnownGood` restores it without complaint. That is
threat T4 routed through the recovery path. Corrupting the newest record's
signature has the same effect.

**Fix:** New `RestoreTarget::LastKnownGoodAtLeast(floor)` refuses with
`RestoreError::JournalTruncated` when the best verifying record is older
than a version the verifier has already observed. The floor is the
`VersionObservationStore` high-water mark. `LastKnownGood` is kept and
documented as stateless, and the secure demo now restores with the floor
it observed.

### Decompression bombs (corrects a P2.4 refutation)
**High · Fixed** · `37a2cb4` · `clawhdf5-format/src/filters.rs`

`decompress_chunk` is handed the chunk's uncompressed size, but no decoder
used it:

- deflate and zstd ran to the end of the stream;
- LZ4 pre-allocated whatever u32 its in-band header claimed;
- the chunked readers clipped the result to the output buffer only
  afterward.

A 261 KB deflate chunk declared as 1 KiB decoded to 256 MiB in a direct
reproduction, and zstd's run-length blocks allow far higher ratios.

**Fix:**

- Each stage is bounded by the chunk size plus 64 bytes of slack, which
  allows for a checksum filter placed ahead of the compressor.
- deflate and zstd read through `Read::take(limit + 1)`.
- LZ4's header is checked before decoding.
- A per-stage catch-all rejects any decoder that cannot be bounded up
  front.

### `read_as_slice` was unsound
**High · Fixed** · `37a2cb4` · `clawhdf5/src/reader.rs`

`read_as_slice` was a safe `pub fn` bounded only by `T: Copy + 'static`.
It could therefore reinterpret file bytes as `bool`, `char`, an enum, or
`&'static str`. That is undefined behavior, with attacker-chosen pointers
in the reference case, reachable from safe code through a crafted file.

**Fix:** `T` is now bounded by a sealed `ZeroCopyElement` trait,
implemented for the ten integer and float primitives. `compile_fail`
doctests pin the `bool` and `&'static str` cases.

### Extensible-array direct blocks aborted the process
**High · Fixed** · `5d0dc6f` · `clawhdf5-format/src/extensible_array.rs`

`super_blk_min_nelmts` is a u8 from the file, and level `l` contributes
2^l direct blocks. The reader pushed one entry per block, guarded only by
a 64-level limit. A value of 40 in a 4 KiB index block aborted with
"memory allocation of 4294967296 bytes failed". Every direct block's
address occupies `offset_size` bytes of the index block, so a count larger
than the block can hold is now rejected before any allocation.

### Object-header continuation DAG (exponential parse)
**High · Fixed** · `5d0dc6f` · `clawhdf5-format/src/object_header.rs`

The v1 depth limit of 32 did not bound work, because one block may hold
several continuation messages. Blocks that each point twice at the next
level form a DAG whose path count doubles per level. A 1,232-byte file
with 22 levels parsed 8,388,607 messages, and thirty levels exhaust
memory. v2 capped continuations at 256 but still re-parsed a shared block
up to 256 times.

**Fix:** Both versions now track the continuation offsets visited per
header and reject a revisit. Real files chain continuations without
sharing, so each block parses once.

### Text-only stores could not be reopened (false tamper alarm)
**High · Fixed** · `e170ef9` · `clawhdf5-agent/src/lib.rs`, `wal.rs`

The OpenClaw backend, the integration point ZeroClaw uses, saves memories
with empty embeddings. The file format has no "no embedding" state, so an
empty row reaches disk as zeros. The content root was computed over the
empty row on write and over the zeros on read. After the first flush,
every open therefore failed with `CompanionTampered`. `hybrid_search`
flushes, so an OpenClaw write, one search, and a reopen were enough to
trigger it. The defect was present since the P2.4 integrity wiring in
July.

**Fix:** An empty embedding is normalized to the store's zero vector at
insert and at WAL replay.

### Wrong-dimension embeddings poisoned the store
**High · Fixed** · `e170ef9` · `clawhdf5-agent/src/lib.rs`,
`vector_search.rs`

`save`, `save_or_update`, `save_batch`, and `record` cached a vector of
any length. Every later search then panicked in `clawhdf5-accel`'s length
assertion. Through the Android C ABI or the Node binding, a panic aborts
the host process. On flush the store also became unopenable.

**Fix:**

- Each insert normalizes or rejects the vector with `MemoryError::Schema`
  before anything is journaled.
- `save_batch` is all-or-nothing.
- WAL replay applies the same rule.
- The cosine batch functions skip any vector whose length differs from
  the query.

### Android bridge shipped without integrity protection
**High · Fixed** · `3570a73` · `clawhdf5-android/Cargo.toml`

The bridge depended on `clawhdf5-agent` with `default-features = false`,
which also turned off `integrity`. Android stores carried no content
Merkle root and were never verified on open. `integrity` is now enabled,
and a new `edgehdf5_open_strict` export provides a fail-closed open.

### Panics crossed the Android C ABI
**High · Fixed** · `3570a73` · `clawhdf5-android/src/lib.rs`

None of the 14 exports caught unwinding. Since Rust 1.81, a panic that
reaches an `extern "C"` boundary aborts the process, so a panic inside the
agent took the host app down. Two such panics were reachable through this
ABI, and both are fixed at the source in `e170ef9`. Every export body now
runs inside `ffi_guard`, which turns a panic into that function's existing
error value.

### `mldsa` was not a default feature
**Medium · Fixed** · `be7bd2d` · `clawhdf5-sign/Cargo.toml`

`verify_sig`, `sign_root`, and the `merkle_sig` verify path exist only
with both `ed25519` and `mldsa` enabled. The default feature set was
`ed25519` alone, so a default dependency silently had no strict-AND
verifier to call.

### Output sizes from untrusted extents
**Medium · Fixed** · `37a2cb4` · `chunked_read.rs`, `data_read.rs`,
`dataspace.rs`

`Dataspace::num_elements` used a plain `product()`, which panics in debug
builds and wraps in release. The four chunked readers multiplied with
plain `*` and allocated with `vec![0u8; n]`, which aborts the process when
the allocation fails. A new `checked_read_sizes` validates the total, the
strides, and the chunk size, capped at HDF5's 4 GiB chunk limit.
`try_zeroed_vec` reports an impossible allocation as an error.

### Sixteen unchecked block-address sums in fixed/extensible arrays
**Medium · Fixed** · `5d0dc6f` · `fixed_array.rs`, `extensible_array.rs`

This is the July `checked_add` bug class, in two parsers that fix did not
reach. All sixteen sites now use `checked_add`. The chunk-byte-size
products and `pages_start` are checked, and the rank is validated before
it is used as an index.

### Variable-length reads were N × S in the global heap
**Medium · Fixed** · `d8dc912` · `clawhdf5-format/src/vl_data.rs`

Every element re-parsed its whole global-heap collection. At about 40 µs
per reference into a 1 MiB collection, a 100 MB file meant hours of CPU
time, and honest files paid the same quadratic cost. Each collection is
now parsed once per read and indexed by id: 4,096 references take 0.5 ms,
down from 166 ms.

### OpenClaw and async opens had no strict constructor
**Medium · Fixed** · `51fa9c0` · `clawhdf5-agent/src/openclaw.rs`,
`async_memory.rs`

ZeroClaw's integration point could not request the protection that
`open_strict` provides against a stripped root or a downgraded WAL. Added
`ClawhdfBackend::open_strict` and `AsyncHDF5Memory::open_strict[_with]`.
With the CLI's `--strict` flag, every public way to open a store can now
request fail-closed behavior.

### Provenance integrity used FNV-1a 64
**Medium · Fixed** · `51fa9c0` · `clawhdf5-agent/src/provenance.rs`

A birthday search finds colliding chunk texts in about 2^32 evaluations.
`content_hash` is now SHA-256. It is still unkeyed, so it detects modified
text but not an attacker who can also rewrite the stored record.

### HSDS request URLs were not encoded
**Medium · Fixed** · `59190aa` · `clawhdf5-io/src/hsds.rs`

A domain containing `&` or `#` injected query parameters, and an object id
containing `/` or `?` rewrote the request path. A new `endpoint()` builder
percent-encodes each path segment and the `host` value.

### Null pointers reached `slice::from_raw_parts` (Android)
**Medium · Fixed** · `3570a73` · `clawhdf5-android/src/lib.rs`

Passing a null pointer to `slice::from_raw_parts` is undefined behavior
even at length 0. A null embedding with length 0 now means "no vector",
and a null pointer with a non-zero length is rejected. The search export
also rejects null result arrays.

### Node binding aborted Node on a panic
**Medium · Fixed** · `3570a73` · `clawhdf5-napi/src/lib.rs`

napi-rs v2 catches unwinding only when a method is annotated for it. All
19 methods and 3 factories now carry `catch_unwind`, and a new
`openStrict` factory provides a fail-closed open.

### BLAS cosine paths read out of bounds (`fast-math`)
**Medium · Fixed** · `e170ef9` · `clawhdf5-agent/src/blas_search.rs`

The safe function `blas_cosine_batch` packed vectors of any length and had
`sgemm` read `active_n * dim` elements. A stored vector shorter than the
query left the matrix short, and `sgemm` read uninitialized memory, even
though a SAFETY comment claimed the lengths were checked. Both BLAS paths
now admit only exact-length rows and check the matrix length before the
unsafe call. The finding is rated Medium because `fast-math` is an opt-in
feature and the wrong-dimension fix above removes the usual way to reach
it. The SIMD kernels were checked and assert equal lengths.

### Ed25519 lenient verification and weak keys
**Low · Fixed** · `be7bd2d` · `clawhdf5-sign/src/lib.rs`

Verification now uses `verify_strict`, and `VerifyingKey::from_bytes`
rejects small-order keys with `SignError::WeakKey`. The verifier's keys
are trusted out of band, so this is hardening rather than a reachable
bypass.

### Fractal-heap ID decode shift
**Low · Fixed** · `37a2cb4` · `clawhdf5-format/src/fractal_heap.rs`

A `u64` was shifted by the file's `max_heap_size`, and a value of 64 or
more panicked in debug builds. It now uses `checked_shr`.

### Harness rows that a reject-everything verifier could pass
**Low · Fixed** · `b2cc968` · `clawhdf5-attack-harness/src/attacks.rs`

Every row reported detected=yes was traced to the verifier it names. Two
rows could have scored a detection against a verifier that rejects every
input:

- **T5.** Its honest-signature check was a `debug_assert!`, which compiles
  out of the release build the harness runs as.
- **T6a.** It tested only `unpack(zeros).is_err()`, and no control
  covered `MerkleAttr::unpack`.

Both now require the honest input to be accepted, the pattern T4b already
used. T3 and T4a matched specific error variants but had no honest arm, so
they gained one too.

The other rows are genuine:

- T1a, T1c, T2a, T2b, and T2c build the committed state from honest bytes,
  verify the tampered bytes with production `verify_chunk` or
  `verify_dataset`, and assert the exact victim index. Their honest sides
  are covered by controls C2 and C3.
- T3's `select_restore_record` uses a mock signature verifier. This is
  disclosed in the row and does not affect the restore logic under test.

The result stays at 30/33, with 0/29 false positives.

## Corrections to earlier reports

- **P2.4 "Investigated and refuted" — decompression bomb.** P2.4 recorded
  "allocation is bounded elsewhere before this point". It was not: see the
  decompression-bomb finding above. Both copies of `P2.4_REDTEAM_REPORT.md`
  now strike the entry and point here.
- **This report's first-pass open items.** Three items listed as open on
  2026-10-10 have since closed:
  - Subset proofs for `LeafFormat::Bound`, fixed in `9120fb7`, outside
    this engagement.
  - Object-header v2 continuation amplification, fixed in `5d0dc6f`.
  - The fractal-heap and local-heap debug panic sites. `37a2cb4` fixed the
    fractal-heap shift. No non-test `panic!` or `debug_assert!` remains in
    `local_heap.rs` or `fractal_heap.rs`.
- **The first-pass "Harness detection 29/33" row** was accurate on
  2026-10-10. The current figure is 30/33, for the reason given in the
  summary table.
- **The first-pass severity tally** read "High 4" and the `e892c03` ledger
  row read "High × 3 + Medium × 4". Neither matched the labels on the
  findings themselves, which give 7 High and 7 Medium overall and 4 High
  plus 2 Medium for `e892c03`. The tables now follow the labels.

## Investigated and refuted

A sweep that lists only confirmed findings reads as a rubber stamp. These
leads survived to the verification step and did not survive it.

**First pass:**

- `hash_chunk_bound`'s `u32` length saturation in release builds. The
  fixed-width `v_k` suffix after the data makes the preimage unambiguous
  regardless of the length field, so a saturated length never produces a
  collision.
- `verify_subset_mesh` fold from a forged fill-in. The verifier folds from
  its own recomputed leaves, and the proof's fill-in only resolves members
  absent from the delivered set.
- `LeafOrder`, `MeshLevel`, or `ProofConstruction` taken from the proof.
  All three are caller parameters and are never read from `SubsetProof` or
  `MeshProof`.
- `MeshLevel::new` upper bound driving `leaf_range` overflow. The level is
  bounded by the tree height before any range arithmetic.
- Mesh `proof_nodes`/`fill_in` unbounded sweep. Both are only looked up by
  key, never iterated.
- Agent-WAL `read_u8_dataset` sign confusion. The honest writer only ever
  writes 0 or 1, and the loaded `u8` is both what is hashed and what
  consumers use, so the verifier and the consumer cannot diverge.
- Agent-WAL `tombstone_index` beyond cache length. `cache.mark_deleted` is
  bounds-checked and returns `false` cleanly.
- Agent `restore_to_version` non-atomic swap. The swap is a rename after
  fsync, with cleanup on every failure path.

**Follow-up pass:**

- Interop between the RustCrypto and PQClean ML-DSA-65 backends. They
  interoperate in both directions, now pinned by
  `tests/mldsa_backend_interop.rs`.
- A public key read from the file being verified. No production path does
  this.
- Bypassing strict-AND through feature selection. `verify_sig` does not
  exist unless both features are enabled.
- Self-referencing v1 continuation. The depth limit catches it.
- Fixed- and extensible-array page-size shifts. They are already guarded.
- Every `unsafe` block in `data_read`, the chunk cache, the mmap reader,
  the facade's typed readers, and the libaec szip FFI. All are sound.
- `GlobalHeapCollection::parse`. Every read is length-checked and every
  loop advances.
- Merkle-core leftovers: domain separation (also locked in by harness row
  T1e), `MerkleAttr::unpack` bounds, `resolve_response`, chunk-count
  handling, and inline versus companion storage. All are sound.
- AEAD key separation, intra-session version reservation, crash recovery
  of pending writes, fail-closed handling of a corrupt WAL slot, no
  plaintext on tag failure, and `mock_dek` reachability. All are sound.
- Agent `knowledge`, `consolidation`, and `agents_md`. They do no file I/O
  of their own and rely on loader-enforced lengths.
- `clawhdf5-accel` SIMD kernels. Every backend asserts equal lengths, so a
  mismatch was a panic, never an out-of-bounds read.
- S3 proof arithmetic and HSDS JSON recursion. Both are bounded.
- PyO3 binding panics. PyO3 converts them to `PanicException`, and the
  binding contains no `unsafe`.

## Open items (documented, not fixed)

- **S3 `fetch_chunk_proof` transfers O(N) bytes, not O(log N).** This is
  a research-validity issue, not a security bug: every node is still
  verified. The function issues one range-GET that spans from a child of
  the root to the leaf. For a 2^20-leaf proof that needs 672 bytes, it
  fetches 33.5 MB to 67.1 MB. No cloud results have been committed, so
  nothing reported is affected. The two ways to reach O(log N) bytes are
  one request per level or a path-contiguous companion layout. Choosing
  between them is a measurement-protocol decision for the P2.5 plan.
- **The AEAD version WAL is CRC-protected, not authenticated.** An
  attacker who can rewrite it can roll back a chunk's high-water mark and
  make the next honest write reuse a nonce. `recover`'s documentation now
  says so and points to the design's random-nonce mode.
- **HSDS response bodies are read without a size limit.** A malicious
  server can exhaust client memory.
- **Signing and encryption remain unreachable from `clawhdf5-agent`
  production paths.** This is unchanged since P2.4. The design document
  scopes production wiring to the C library.
- **The default `open()` is fail-open** on a missing `_merkle_root`.
  Every public entry point now has a strict variant.
- **Per-chunk version counters are kept in memory and in the WAL only.**
  This is the spec's own §7 "Known Gaps" item 3.
- **WAL `flush()` does not call `fsync()`.** This is a pre-existing
  `TODO(P3)`.
- **Pre-existing build and lint debt.** None of it is security-relevant,
  and none was introduced here:
  - The `no_std` build of `clawhdf5-format` fails on two `f64::powi`
    calls.
  - clippy fails on the `clawhdf5` test targets (`lib.rs:470`,
    `tests/chunk_modify_test.rs`) and on the format examples and benches.

## Commit ledger

| Commit | Pass | Scope | Description |
|---|---|---|---|
| `5b84998` | 1 | sign | Bind `leaf_format` and `grid_hash` into the hybrid-signed payload (Critical) |
| `e892c03` | 1 | agent | Loader panics, WAL allocation / strip-to-v1 gaps, restore staging (High × 4, Medium × 2) |
| `7b2bca6` | 1 | subset | SFC bit-budget overflow, saturating selection bounds, contiguous layout (High, Medium × 2) |
| `6c89a91` | 1 | parse | Datatype recursion, btree_v2 depth, implicit chunks, chunked_read (High × 2, Medium × 3) |
| `0db1771` | 1 | docs | First-pass report |
| `be7bd2d` | 2 | sign | Self-describing `merkle_sig`, floored restore, strict Ed25519, `mldsa` default (High × 2, Medium, Low) |
| `37a2cb4` | 2 | read | Decompression bounds, fallible read allocation, sealed `read_as_slice`, heap shift (High × 2, Medium, Low) |
| `5d0dc6f` | 2 | parse | Array-index checked math, extensible-array block count, continuation DAG (High × 2, Medium) |
| `d8dc912` | 2 | vl | Parse each global-heap collection once per read (Medium) |
| `2715466` | 2 | aead | Reopened writer never reuses a committed nonce (High) |
| `51fa9c0` | 2 | agent | Strict open on every public path, SHA-256 provenance (Medium × 2) |
| `59190aa` | 2 | io | Percent-encode HSDS URLs, document S3 proof cost (Medium, open item) |
| `e170ef9` | 2 | agent | Embedding normalization, text-only reopen, BLAS bounds (High × 2, Medium) |
| `3570a73` | 2 | ffi | Android integrity + panic and null guards; Node `catch_unwind` (High × 2, Medium × 2) |
| `b2cc968` | 2 | harness | Honest arms for T3/T4a/T5/T6a (Low) |

## Verification status

Every commit was verified against the relevant test matrix before being
pushed; its message records the run. Final state on `review-round3` after
`b2cc968`:

- ✓ `clawhdf5-sign` (all features): 41 tests
- ✓ `clawhdf5-format` (`merkle,zstd,lz4,parallel`): 878 tests, made up
  of library, integration, and example tests plus doctests. One bench
  target, `contiguous_bao_baseline`, cannot run under `cargo test` without
  a pre-generated 1000³ dataset file. That is environmental and unrelated
  to these changes.
- ✓ `clawhdf5-filters` (`chacha20`): 118 tests
- ✓ `clawhdf5-agent` (`async,merkle-provenance`): 609 tests
- ✓ `clawhdf5-io` (`s3,hsds,async,mmap`): 134 tests
- ✓ `clawhdf5` facade: 184 tests
- ✓ `clawhdf5-android`: 3 C-ABI tests, covering panics, null pointers and
  wrong dimensions, and strict open
- ✓ `clawhdf5-secure-demo`: 6/6 stages
- ✓ `clawhdf5-attack-harness`: 30/33 detected, 0/29 false positives. The
  three undetected rows are T1d and T6b, the unsigned-path limits whose
  signed counterparts T1f and T6c are detected, and T8, structure hiding,
  which is not implemented.
- ✓ `cargo clippy -D warnings` is clean, per commit, on every crate each
  commit touched. The exceptions are the pre-existing failures listed
  under open items.
