# Red-Team Engagement 2026-10 — ClawHDF5

**Repo:** clawhdf5 · **Branch:** main · **Window:** 2026-10-08 → 2026-10-10
**Commits:** 4 · **Range:** `5b84998..6c89a91`

A second adversarial pass over the Merkle/signing/AEAD stack and the
pure-Rust HDF5 parser, three months after the P2.4 engagement. Scope was
the S2-D2 "schema" (the on-disk layout the design document, the C
integration spec, and `clawhdf5-format`'s `MerkleAttr`/`canonical_payload`
all describe) *and* every public verifier it feeds. The P2.4 report
(`P2.4_REDTEAM_REPORT.md`) is the baseline this engagement started from;
nothing listed there is re-reported here.

| | |
|---|---|
| Findings fixed | 16 |
| Critical | 1 (signed-config leaf-format downgrade with chunk re-framing) |
| High | 4 (grid rewrite under signature, agent loader non-injective root, WAL strip-to-v1 under `open_strict`, Datatype::parse stack overflow) |
| Tests landed | 824 → 832 format, 568 → 576 agent, +16 new regressions, +3 payload-encoding tests |
| Harness detection | 29/33, 0 false positives (unchanged — ceiling for a prototype that leaves four by design) |

## Executive summary

Four classes of real defect showed up, each confirmed by a reproduction
against the actual code before any fix: the hybrid-signed payload left out
two attribute fields that *change what the root means*, the agent-crate
loader panicked on crafted on-disk state before any integrity check could
run, the agent WAL's strict mode did not reach the sidecar, and the
untrusted-HDF5 parser had the July engagement's bug class in files the
July engagement did not touch.

The signing finding is the sharpest. The canonical signed payload R was
exactly what the design equation specified — `root || companion_hash ||
version || timestamp || alg_id`, 81 bytes — but the attribute had
grown two more fields since: `leaf_format` (which preimage the tree was
built with) and `grid_hash` (the chunk-grid binding). An attacker with
file write access and no signing key could flip a `Bound` dataset to
`Content`, re-frame every stored chunk as the bound preimage suffix
`be32(8) || le64(k) || be32(len) || data || le64(v)`, and watch every
verifier accept bytes the signer never committed to while the signature
still verified. Payload v1 (114 bytes) brings every field a verifier reads
out of the attribute under the signature; signatures over the v0 layout
are now rejected at parse time rather than verified against a payload they
never bound.

The agent-crate and parser findings were the P2.4 report's own posture
turned on files it had marked out of scope: the agent loader is called
*before* `integrity::verify_content`, so a crafted `/sessions/summaries`
shorter than `/sessions/ids` panicked the open path with no integrity
check even possible; the WAL `open_strict` flag didn't thread to the
sidecar, so an attacker rewriting the WAL header's version byte from 2 to
1 stripped the chain hash with less effort than recomputing it; the
pure-Rust `Datatype::parse` recursed without any depth guard, and a 24 KiB
chain of nested VL types SIGABRTed the process on any public `open`.
Each of these has a regression test that is the exact crafted-bytes
reproduction, in the test module itself.

Everything below is fixed, tested, pushed to `main`, and running green in
the attack harness (29/33 detected, same as before — the four remaining
are each a documented design limit, not a silent gap), the secure demo
(6/6 stages), and the format crate's 832-test matrix.

## Method

Three guard-rails from the P2.4 engagement, kept in force:

- **Re-verify every "confirmed" finding against the actual code before
  fixing it.** Four of the sixteen final findings came from leads that
  turned out to be sharper than their initial description; two others were
  refuted during that pass and are listed in "Investigated and refuted."
- **Reproduce the attack as a test *first*, then fix.** Every fix in this
  engagement has a regression test that fails on the pre-fix tree
  (verified by hand before each commit) and passes on the post-fix tree.
- **Fix as you go.** One commit per fix batch, each with the attack
  scenario and the root cause in the message at the same level of detail
  as the code comments. The repository's commit log is the audit trail.

Lanes were run one at a time this round. Parallel sweeps exhausted the
model's session limit three times at the start of the engagement before
anything was written to disk; a serial pass finished with the budget to
spare and produced findings files that survive future continuation.

## Findings

### Hybrid-signed payload omitted `leaf_format` and `grid_hash`
**Critical · Fixed** · `clawhdf5-sign/src/lib.rs` (`canonical_payload`,
`HybridSignature`), `clawhdf5-format/src/merkle.rs` (`SignedRootVerifier`,
`verify_signed_root`)

The canonical signed payload R was exactly the design equation —
`root || companion_hash || version || timestamp || alg_id`, 81 bytes —
but the `merkle_root` attribute had grown two more fields that give the
root its meaning. `leaf_format` (added in commit `87f6b08`, 2026-08-27)
is protected by the attribute's unkeyed integrity hash, which a storage-
level attacker recomputes with no key material at all; `grid_hash`
(added in P1.5) is in nothing at all and its preimage is public. Two
attacks followed:

- **Leaf-format downgrade with re-framing.** Flip a `Bound` dataset to
  `Content` (recomputing the unkeyed integrity hash via the public
  constructor) and rewrite each stored chunk as the bound preimage suffix
  `be32(8) || le64(k) || be32(len) || data || le64(v)`. Under `Content`,
  `H(0x00 || framed)` equals the original bound leaf, so the signed root
  is untouched: `verify_signed_root`, `verify_dataset`, and `verify_chunk`
  all accept bytes the signer never committed to. The attack strips the
  index/version binding the `Bound` format exists to provide, without a
  hash collision or a signature forgery.
- **Grid rewrite.** Rewrite `dims`/`chunk_shape` in the (unsigned) object
  header and recompute `grid_hash`. The root is unchanged, the signature
  passes, and `verify_subset` — which trusts the attribute's `grid_hash`,
  exactly as the P2.3 security review intended — maps the same requested
  hyperslab to a different leaf and delivers the wrong chunk under a
  valid proof.

Both were reproduced against the real Ed25519+ML-DSA-65 signer before the
fix; the first ran end-to-end in `verify_dataset`, the second in
`verify_subset`.

**Fix:** Payload v1 is 114 bytes (adds `leaf_format(1)` and
`grid_hash(32)` after `alg_id`). `SignedRootVerifier` now receives the
whole `MerkleAttr` so any field added later is at least available to the
signer. `HybridSignature::VERSION` was bumped `0 → 1` and `from_bytes`
checks it: a stored v0 signature is rejected at parse time rather than
verified against a payload it never bound. Spec table (`docs/c-hdf5-
integration-spec.md`) and the hybrid-signature test vector were updated
to match. Three new regressions pin both attacks and the v0-reject.

### Agent loader panicked on crafted parallel-dataset lengths
**High · Fixed** · `clawhdf5-agent/src/schema.rs`,
`clawhdf5-agent/src/integrity.rs`

The content Merkle root's leaf for entry `i` is built from
`X.get(i).unwrap_or(default)`, so it cannot tell a missing trailing
element from a present default one: a `/memory/tombstones` dataset
shortened from `[n]` to `[1]` hashed to the honest root, loaded under
`open_strict`, undercounted active entries, made `delete()` report
`NotFound`, and panicked `get_chunk()` and the `compact()` that
`delete()` auto-triggers once the tombstone fraction crosses its
threshold. Other datasets were worse: a short `/sessions/summaries`, or
short entity/relation arrays, indexed out of bounds before
`verify_content` could even run.

**Fix:** The loader rejects any per-entry dataset whose length disagrees
with its driver count with `MemoryError::Schema` (`/memory`, `/sessions`,
entities, relations, aliases, and the flat embeddings array against
`n * embedding_dim`). `integrity.rs`'s module doc records length
agreement as the loader's invariant, not the root's.

### Untrusted `/meta` attributes reached arithmetic before verification
**High · Fixed** · `clawhdf5-agent/src/schema.rs`

`embedding_dim` is not covered by the root (it says how to parse what the
root covers), but was read as `extract_i64_attr(..) as usize`: a value of
0 reached `slice::chunks(0)` and panicked, a negative value became a
huge usize. Entity ids are stored as `u64 as i64`, so an id ≥ 2^63 read
back negative and `max + 1` overflowed (debug panic; release wrapped
`next_id` to 0 so new entities collided with existing ones).

**Fix:** Counts parse through a new `extract_count_attr` with a minimum
(1 for `embedding_dim`); ids pass through `non_negative` + `checked_add`;
a `/meta` value out of range is a `MemoryError::Schema`.

### WAL allocations driven by u32 length prefix
**High · Fixed** · `clawhdf5-agent/src/wal.rs`

`vec![0u8; len]` and `Vec::<f32>::with_capacity(count)` were sized
straight from a u32 prefix in the sidecar: a 4-byte edit to the WAL made
every `open`/`open_strict`/`WalFile::open` attempt allocate 4 GiB
(string) or 16 GiB (embedding) before reading a single byte of payload.
Under `ulimit -v 2G` the Rust allocator SIGABRTs; without a limit the
allocation "succeeded" lazily and the entry was then dropped as if
truncated.

**Fix:** Every prefix is bounded by the bytes actually remaining in the
file before any allocation; a larger prefix is handled exactly like a
truncation (the module's existing crash-safe path).

### `open_strict` accepted a version-1 (unchained) WAL sidecar
**High · Fixed** · `clawhdf5-agent/src/lib.rs`, `wal.rs`

The strict flag only reached `storage::read_from_disk_strict`; the WAL
branch was identical for strict and non-strict and called the
version-tolerant `WalFile::read_entries`. Rewriting the sidecar's header
version byte from `2` to `1` and dropping the chain trailers stripped
the v2 chain with less effort than recomputing it; forged `Save` and
`Tombstone` entries replayed under `open_strict`, and the process then
kept appending unchained v1 records.

**Fix:** New `WalFile::open_strict` / `read_entries_strict` reject a v1
sidecar with `MemoryError::Integrity`. `HDF5Memory::open_strict` uses
them. The CLI gains a global `--strict` flag (`CLAWHDF5_STRICT`) so an
operator can opt in; previously every subcommand used the fail-open
`open()` with no alternative.

### Chain-coverage gaps in the v2 WAL
**Medium · Fixed** · `clawhdf5-agent/src/wal.rs`

Two records escaped the v2 chain. `ActivationUpdate` (`0x03`) records
`continue`d before the trailer read, so unchained 9-byte blobs could be
spliced anywhere in a chained file; a present-but-unrecognized type byte
silently ended replay, dropping every later entry, contradicting the
module's own truncation-vs-corruption rule. Both are now chain failures
in v2.

### `restore_to_version` staged the journal's `snapshot_ref` before any check
**Medium · Fixed** · `clawhdf5-agent/src/storage.rs`

The signed payload binds root/version/timestamp, not the snapshot path,
so a record naming a FIFO, device, or multi-GB file was copied in full
next to the live file before any gate ran. A failed copy left the
partial temp behind. Fixed: only regular files are staged, and the copy
failure path cleans up.

### `Datatype::parse` recursed with no depth guard
**High · Fixed** · `clawhdf5-format/src/datatype.rs`

Seven recursive callsites across Compound members, Enum base, VL base,
and Array base (both v1/v2 and v3-5 encodings) with no depth cap. A
~24 KiB datatype of 16000 nested VL types SIGABRTed the process on any
public `open` or `shape` call. Array and Compound chains gave similar
reach at ~5–13 bytes per level. Reproduction killed the test runner
process; after the fix the same input returns `FormatError::
NestingDepthExceeded`.

**Fix:** Private `parse_with_depth` seams in the recursion. New
`MAX_NEST = 32` (real HDF5 datatypes nest 2–3 levels; 10× safety
margin), matching the sibling parsers' depth caps.

### B-tree v2 recursion bounded only by the file's `depth` header
**High · Fixed** · `clawhdf5-format/src/btree_v2.rs`

`collect_internal_records`' recursion was bounded only by the file's own
`u16 depth` field, so a crafted header with `depth = 65535` and a BTIN
node that pointed its child at itself cascaded 64K recursive calls and
stack-overflowed. Same class as the July engagement's `btree_v1`/
`chunked_read` fix; the sibling parsers already had `MAX_BTREE_DEPTH`/
`MAX_CHUNK_BTREE_DEPTH` of 64, this parser did not. New
`MAX_BTREE_V2_DEPTH = 64`.

### `chunked_read` rank underflow on empty `chunk_dimensions`
**Medium · Fixed** · `clawhdf5-format/src/chunked_read.rs` (4 sites)

`rank = ndims - 1` from a zero-length chunk_dimensions wrapped to
`usize::MAX`; the subsequent `chunk_dimensions[..rank]` slice panicked
in release and the arithmetic in debug. Rejected before the subtraction
at every site.

### Five unchecked `c_addr + size` sites in `chunked_read`
**Medium · Fixed** · `clawhdf5-format/src/chunked_read.rs` (5 sites)

Same bug class as the July engagement's `offset + N` fix in the same
file: a crafted B-tree key or fixed/extensible-array entry with an
address near `usize::MAX` wrapped past the bound check and panicked the
subsequent slice. All five now use `checked_add`; failure is a
`FormatError::Overflow`.

### `generate_implicit_chunks` had five arithmetic hazards
**Medium · Fixed** · `clawhdf5-format/src/chunked_read.rs`

Zero chunk dimension became a div-by-zero in `div_ceil`; chunk-byte-size
product overflowed `u64`; `total_chunks` product overflowed `u64`;
`Vec::with_capacity(total_chunks)` allocated terabyte-scale buffers from
the attacker-controlled count; `base_address + linear_idx *
chunk_byte_size` wrapped past `u64::MAX`. All five are now checked;
the function returns `Result`; the four production callers and the one
`data_read` caller were updated.

### Space-filling-curve bit-budget overflow on skewed grids
**High · Fixed** · `clawhdf5-format/src/subset_proof.rs`

`morton_index` gives each axis `floor(64/rank)` bits and drops the rest;
`hilbert_index` needs `hilbert_bits * rank` bits and its doc claimed the
padded-leaf-count check guaranteed that. It does not: the bound caps the
total chunk count (≤ 2^39), not the per-axis extent, so dims
`[2^20,1,1,1]` pass it yet need 80 Hilbert bits, and `[2^33,1]` passes
yet needs 33 Morton bits per axis against a budget of 32. Past the
budget two distinct chunk coordinates map onto one leaf, the verifier's
expected chunk set for a hyperslab covering both is one index short, and
a proof delivering one chunk for two logical regions passes as complete
(violating R5b). Not attacker-triggerable once `grid_hash` is signed,
but a writer who chose such a curve for such a grid would get a tree
that verifies against itself and certifies the wrong coverage.

**Fix:** New `MerkleError::LeafOrderOverflow` and
`check_order_bit_budget`, called in `compute_expected_chunk_indices`
before any coordinate is mapped. Row-major is unaffected.

### Saturating selection and chunk-space bounds
**Medium · Fixed** · `clawhdf5-format/src/subset_proof.rs`

`selection_chunk_bounds` wrapped on extreme operands: `n_per_dim *
chunk_shape` wrapped with a `chunk_shape` near `2^64`; the hyperslab end
`start + (count-1)*stride + block` and the points-set `max + 1` could
wrap too. A wrap turned "inside the grid" into "no chunks" — an empty
expected set is exactly what an omission attack needs the verifier to
derive. All three now saturate.

### `contiguous_layout` returned mutually inconsistent untrusted fields
**Medium · Fixed** · `clawhdf5-format/src/subset_proof.rs`

The layout message's `size`, the dataspace `dims`, and the datatype size
are independent fields of an untrusted object header. The function's
contract tells callers to slice `file[data_addr..data_addr + nbytes]`
and hand the buffer, with `dims` and `elem_size`, to the grid builders,
which tile it assuming `len == product(dims) * elem_size`. A `size` that
disagreed with the dataspace, or an address range wrapping past
`u64::MAX`, was therefore a crafted-file panic in the caller or the
builder (T7) rather than a verification failure. Rejected now as
`MalformedHeader` before any data byte is touched.

### Spec document stale after `87f6b08`
**Low · Fixed** · `docs/c-hdf5-integration-spec.md`

Section 1.2's attribute table was 129 bytes with the pre-`leaf_format`
layout even after commit `87f6b08` grew the attribute to 130; section
1.5's payload table omitted `leaf_format` and `grid_hash` even after
this engagement's commit 1 made the hybrid signature cover them. Both
tables, plus §7's "Leaf hash formula" row, were updated to match the
code.

## Investigated and refuted

A sweep that only lists confirmed findings reads as a rubber stamp;
these are the leads that survived to the verification step and did not
survive it. Listed here so the engagement reads as scrutiny.

- `hash_chunk_bound`'s `u32` length saturation in release builds —
  refuted: the fixed-width `v_k` suffix after the data makes the
  preimage unambiguous regardless of the length field, so a saturated
  length never produces a collision.
- `verify_subset_mesh` fold-from-forged-fill-in attack — refuted: the
  verifier folds from its own recomputed leaves, and the proof's
  fill-in only resolves members absent from the delivered set.
- `LeafOrder`/`MeshLevel`/`ProofConstruction` taken from the proof —
  refuted: all three are caller parameters, never read from
  `SubsetProof` or `MeshProof`.
- `MeshLevel::new` upper bound driving `leaf_range` overflow — refuted:
  the level is bounded by the tree height before any range arithmetic.
- Mesh `proof_nodes`/`fill_in` unbounded sweep — refuted: both are only
  looked up by key, never iterated.
- Agent-WAL `read_u8_dataset` sign confusion — refuted: the honest writer
  only ever writes 0/1, and the loaded `u8` is both what is hashed and
  what consumers use, so no verifier/consumer divergence exists.
- Agent-WAL `tombstone_index` beyond cache length — refuted:
  `cache.mark_deleted` is bounds-checked and returns `false` cleanly.
- Agent `restore_to_version` non-atomic swap — refuted: rename after
  fsync, with cleanup on every failure path.

## Open items (documented, not fixed)

- **Signing and encryption remain unreachable from `clawhdf5-agent`.**
  Same as the P2.4 engagement's open item 1. Every real ClawHDF5 file is
  unsigned and unencrypted today; the design document scopes production
  wiring to the C library, not this prototype.
- **Default agent `open()` is still fail-open on a missing `_merkle_root`
  attribute.** `open_strict` and the new CLI `--strict` flag are
  available for callers that need the guarantee; the default stays
  backward-compatible.
- **Subset proofs don't yet work with `LeafFormat::Bound` datasets.**
  `verify_subset`/`extract_subset`/`verify_subset_mesh` compute only the
  `Content` leaf preimage, so a `Bound` dataset's subset proofs fail
  closed with `HashMismatch`. A functional gap, not a bypass; threading
  `leaf_format` + per-chunk versions through the subset API is a
  larger change out of scope for this engagement.
- **Fractal-heap and local-heap `debug_assert`/debug-panic sites;
  object-header v2 continuation copy amplification.** Reviewed in the
  untrusted-parse lane; each is lower severity than the issues fixed in
  this pass (debug-only or quadratic rather than crash on open) and left
  for the next engagement.
- **Per-chunk version counters still in-memory + WAL only.** Same as the
  spec's own §7 "Known Gaps" item 3; not altered by this engagement.
- **WAL `flush()` not `fsync()`** (`TODO(P3)`, pre-existing). Same.

## Commit ledger

| Commit | Scope | Description |
|---|---|---|
| `5b84998` | sign | Bind `leaf_format` and `grid_hash` into the hybrid-signed payload (Critical, 2 reproductions) |
| `e892c03` | agent | Close loader panics, WAL allocation / strip-to-v1 gaps, and restore staging (High × 3 + Medium × 4) |
| `7b2bca6` | subset | Reject SFC bit-budget overflow, saturate selection bounds, validate contiguous layout size (High + Medium × 2) |
| `6c89a91` | parse | Cap datatype recursion, bound btree_v2 depth, harden implicit chunks (High × 2 + Medium × 3) |

## Verification status

Every commit was verified against the full relevant test matrix before
being pushed; the harness and secure demo were re-run after the last
commit.

- ✓ `clawhdf5-sign`: 33 tests (+ 3 payload-encoding regressions)
- ✓ `clawhdf5-format`: 832 tests across all targets (+ 16 regressions)
- ✓ `clawhdf5-agent`: 576 tests (+ 8 regressions)
- ✓ `clawhdf5-secure-demo`: 6/6 stages
- ✓ `clawhdf5-attack-harness`: 29/33 detected, 0 false positives
  (unchanged — the four undetected rows are each a documented design
  limit, not a silent gap)
- ✓ `cargo clippy -D warnings` clean on `clawhdf5-sign`,
  `clawhdf5-format`, `clawhdf5-filters`, `clawhdf5-agent`,
  `clawhdf5-cli`, `clawhdf5-attack-harness`, `clawhdf5-secure-demo`,
  `clawhdf5`
