# C HDF5 Integration Design Document (P2.6)

**Status:** Draft, pending senior-engineer review (P2.6 step 7). Not yet reviewed —
see the note at the end of this document instead of a
`docs/c-hdf5-integration-review.md` file, which this repo's convention
(cf. the still-absent `docs/security-review-notes.md` for P2.3) is to create
only once an actual review has happened, not as a pre-filled placeholder.

**Scope.** This document specifies how the Merkle-tree tamper-detection system
— prototyped in Rust in this repository (`clawhdf5-format`, `clawhdf5-filters`,
`clawhdf5-sign`) — would be added to the official C HDF5 library. It covers
the six items S2-D2-Yr2 §7 (P2.6) requires: on-disk layout, public C API,
copy/repack preservation semantics, crash-consistency enforcement (with a
chosen mechanism), the encrypted-dataset nonce scheme (with a chosen default
per workload class), and filter-pipeline / signed-root layout.

**Grounding.** Every design choice below is checked against the actual Rust
prototype's source (file:line citations throughout), not just the S2-D2-Yr2
prose. Where the prototype diverges from the paper, or from itself, that is
called out explicitly rather than silently picking one — see §7.

---

## 1. On-Disk Layout

### 1.1 Naming convention (a decision this document makes)

S2-D2-Yr2 §5 "Merkle-Tree Provenance" (its "Storage of the Merkle Tree" and
"Signing Authority" subsections) writes the attribute and companion paths
with an underscore prefix: `_merkle_root`,
`/_merkle/{name}`. The Rust prototype does not follow this — `merkle.rs`
uses unprefixed names throughout (`MERKLE_ATTR_NAME = "merkle_root"`,
`MERKLE_GROUP_NAME = "merkle"`, `MERKLE_VERSION_ATTR_NAME = "merkle_version"`
— `clawhdf5-format/src/merkle.rs:1261,2040,1551`, with an explicit code
comment at `merkle.rs:1548-1550` noting the divergence from the paper's
naming), and the existing internal `docs/mpi-protocol.md` design document
also uses the unprefixed form (`/merkle/dataset_name`).

**This document adopts the unprefixed convention** (`merkle_root`,
`/merkle/{name}`, `merkle_version`, `merkle_journal`), for consistency with
the validated prototype and the one internal design document that already
exists. The paper's underscore-prefixed names should be read as earlier
draft naming superseded by the implementation. If underscore-prefixed names
are required for some external compatibility reason, that is a rename, not
a redesign — everything below applies unchanged either way.

### 1.2 Root attribute: `merkle_root`

A fixed-size attribute on the dataset, mirroring `MerkleAttr`
(`clawhdf5-format/src/merkle.rs:1312-1448`), **129 bytes**:

| Offset | Size | Field | Encoding |
|---|---|---|---|
| 0 | 32 | `root` | Merkle root hash, raw bytes |
| 32 | 1 | `algorithm` | `0x00`=SHA-256, `0x01`=BLAKE3, `0x02`=K12 |
| 33 | 32 | `integrity` | `H(0x03 \|\| root \|\| algorithm)` — binds the attribute's own fields together |
| 65 | 32 | `companion_hash` | SHA-256 of the flattened companion node array (all-zero if none) |
| 97 | 32 | `grid_hash` | `H(0x04 \|\| layout_class \|\| elem_size \|\| dims \|\| chunk_shape)`, binding the grid parameters (all-zero = unbound) — see §1.6 |

A reader recomputes `integrity` from the first 33 bytes on every read and
rejects (`H5E_MERKLE_INVALID_ATTR`, §2.4) on mismatch, size mismatch, or an
unrecognized algorithm byte — this is the C equivalent of `MerkleAttr::unpack`'s
constant-time integrity recheck (`merkle.rs:1389-1448`).

This attribute is intentionally small (129 bytes, well under HDF5's
object-header attribute limits) so it imposes negligible per-dataset
overhead and is always loaded with the dataset's object header — the
"instant root verification" property S2-D2-Yr2 §"Storage of the Merkle
Tree" calls out.

### 1.3 Companion dataset: `/merkle/{name}`

For datasets with more than `INLINE_CHUNK_THRESHOLD` (256) chunks, the full
tree is stored as a separate 1D dataset at `/merkle/{name}`, mirroring
`write_merkle_companion` (`merkle.rs:2223-2257`):

- **Layout**: flat array of 32-byte node hashes, level-order (root at index
  0, children of node *i* at `2i+1`/`2i+2`) — `merkle.rs:626,633,722-723`.
  Total size = `(2·padded_leaf_count − 1) × 32` bytes.
- **Storage class**: contiguous, not HDF5-chunked (`merkle.rs:2125-2126`,
  confirmed no `chunk_dims` set — `type_builders.rs:395`, `file_writer.rs:1014-1022`).
  A single node or a proof-path span is read via `H5Dread` with a
  hyperslab selection over this contiguous region — no full-tree load
  required.

**Departure from S2-D2-Yr2's chunked-companion recommendation.** §"Storage
of the Merkle Tree" recommends the companion dataset itself be HDF5-chunked
(e.g. 4 KB chunks holding 128 nodes each) specifically to support
independent I/O tuning and to bound write-amplification under concurrent
updates. The validated prototype instead uses a **contiguous** layout. For
this design document, **contiguous is retained as the default** for two
reasons grounded in what's actually implemented and tested:

1. A contiguous companion dataset is what P1.4/P1.5's partial-verification
   and subset-extraction paths were built and benchmarked against
   (`clawhdf5-format/src/subset_proof.rs`) — changing the on-disk layout in
   the C port without re-validating those paths against the new layout
   would be introducing an unverified change, not porting a verified one.
2. `clawhdf5-io`'s S3 range-GET path (P2.5, `crates/clawhdf5-io/src/s3.rs`)
   depends on the companion dataset having a single, predictable contiguous
   byte range resolvable from one HDF5 metadata parse — a chunked companion
   would need per-chunk address resolution (its own B-tree/chunk-index
   lookup) before any proof-node range-GET could be issued, adding
   complexity with no benchmarked benefit yet.

The write-amplification concern S2-D2-Yr2 raises for concurrent HPC writes
is real and unaddressed by a contiguous layout as stated — it is deferred
to the **lazy in-memory tree maintenance** mitigation the paper already
specifies (accumulate leaf hashes per-rank, flush the full tree delta at an
epoch boundary via one collective I/O pass), which sidesteps
per-chunk companion contention regardless of whether the companion is
chunked or contiguous. `docs/mpi-protocol.md`'s "Eager Collective Flush"
strategy already implements this pattern for the Rust prototype and applies
unchanged to the C port. If RQ3 benchmarking later shows contiguous layout
becomes the bottleneck at extreme chunk counts, a chunked companion is a
compatible follow-up (the node-index-to-byte-offset math is unaffected;
only the physical storage class changes) — not a reason to block this spec
on a change nothing in the prototype currently exercises.

### 1.4 Inline fallback: `merkle_nodes` attribute (N ≤ 256 chunks)

For small datasets, the flattened node array is stored directly as a second
attribute (`MERKLE_NODES_ATTR_NAME`, `merkle.rs:2037`) rather than a
separate dataset, avoiding companion-dataset overhead for cases where the
whole tree is already small enough to be attribute-sized (≤256 leaves ⇒
≤511 nodes ⇒ ≤ 16,352 bytes). `H5Dverify_chunk`/`H5Dextract_subset` (§2.1, §2.2)
must check for `merkle_nodes` before assuming a `/merkle/{name}` dataset
exists — `companion_hash` in the root attribute covers both cases
identically (SHA-256 of the same flattened byte array, regardless of which
container holds it), so the check for tampering is unaffected by which
storage class is in use.

### 1.5 Hybrid signature storage

Per S2-D2-Yr2 §"Post-quantum hybrid signing", the signature is applied
**once**, at file level, over a payload that aggregates all per-dataset
roots. The serialized signature (mirroring `HybridSignature::to_bytes()`,
`clawhdf5-sign/src/lib.rs:261-277`) is:

```
offset 0:  version        1 byte    (0x00)
offset 1:  ed25519_sig    64 bytes
offset 65: mldsa65_sig    3309 bytes
```
Total 3374 bytes, fixed-width, no separators — `SERIALIZED_SIZE`
(`clawhdf5-sign/src/lib.rs:242`).

The canonical signed payload `R` (mirroring `canonical_payload`,
`clawhdf5-sign/src/lib.rs:158-188`) is 81 bytes:

```
offset 0:  root            32 bytes  raw
offset 32: companion_hash  32 bytes  raw
offset 64: version         8 bytes   big-endian u64
offset 72: timestamp       8 bytes   big-endian u64 (Unix seconds)
offset 80: alg_id          1 byte
```

**Storage location — correction.** An earlier draft of this document
claimed no signature attribute exists in the prototype at all; that was
wrong, caught by an independent review pass, and is corrected here.
`clawhdf5-sign/src/lib.rs:439` already defines `SIG_ATTR_NAME = "merkle_sig"`,
with a working `write_sig_attr()`/`read_sig_attr()` pair
(`lib.rs:462-486`) and a passing round-trip test — the attribute is
written as the **bare** `HybridSignature::to_bytes()` output (the 3374
bytes above), with no AlgID/version/timestamp wrapper. (Separately, the
serialized signature *also* still travels through `ProvenanceRecord`/
`ProvenanceJournal` — `merkle_journal.rs:63-76` — for the journal/rollback
path in §4.3; that is a second, additional use of the same
`HybridSignature` bytes, not a contradiction with the attribute existing.)

This leaves a real, narrower gap than the one originally claimed:
verifying a stored `merkle_sig` requires reconstructing the full 81-byte
payload `R`, which needs `timestamp` in addition to `root`/`companion_hash`/
`version` (available from `merkle_root`, §1.2) and `alg_id`. There is a
`_provenance_timestamp` attribute constant (`ATTR_TIMESTAMP`,
`clawhdf5-format/src/provenance.rs:50`), but it belongs to a different,
unrelated feature (the SHA-256 "provenance" hashing path) and stores an
ISO-8601 *string*, not the `u64` Unix-seconds integer `canonical_payload`
requires — it is not reusable for this purpose as-is. No attribute
carrying the right value in the right encoding for `merkle_sig`
verification exists yet — a caller today must supply `timestamp` from some
out-of-band source (e.g. the provenance journal's own record, which does
carry one, `merkle_journal.rs`). This document proposes closing *that*
gap, not the attribute's existence:

- **Primary path**: extend the existing bare `merkle_sig` attribute to
  `AlgID(1) || version(8, BE) || timestamp(8, BE) || HybridSignature(3374)`
  = 3391 bytes — a superset of what `write_sig_attr`/`read_sig_attr`
  already produce/consume, so the C port (and a corresponding small Rust
  change) can add the three-field prefix without redesigning the existing
  function pair, only widening their output format. Plus a `merkle_pubkey`
  attribute (Ed25519 32 bytes + ML-DSA-65 ~1952 bytes ≈ 1984 bytes).
  Combined with the per-dataset `merkle_root` attribute (129 bytes) this
  stays comfortably under the paper's provisional 32 KB fallback threshold
  for any single dataset, but a file with many datasets accumulates one
  `merkle_root` per dataset (129 B × D) plus exactly one file-level
  signature pair — not one signature per dataset, matching the paper's
  "one signature, all datasets" requirement.
- **Fallback path** (RQ7-gated, per S2-D2-Yr2 §"Why this still needs to be
  measured at HDF5 scale"): if the file-level attribute payload (root,
  companion hash, version, timestamp, AlgID, both signatures, both public
  keys, and optionally a Rekor bundle + TSA token) approaches HDF5's
  ~64 KB object-header ceiling, fall back to writing that payload as a
  dedicated dataset (`/merkle_signature`), with the file-level attribute
  reduced to a single 32-byte hash of that dataset's contents
  (`H_sig_ds`). Verification order: read `H_sig_ds` from the attribute →
  read `/merkle_signature` → confirm its hash matches `H_sig_ds` → extract
  and verify `σ` against `R`. `R` deliberately excludes `H_sig_ds` itself
  (no circular dependency — `H_sig_ds` commits to the dataset *containing*
  `σ`, so it cannot also be an input to `σ`).
- The exact 32 KB threshold is provisional per the paper and should be
  confirmed or adjusted once RQ7's C-HDF5-scale attribute-size
  measurements exist; this spec does not have those numbers yet (see §5.2
  for the analogous honesty constraint on throughput numbers).

### 1.6 Layout class and contiguous-dataset support (P2.8)

HDF5's *default* dataset layout is contiguous, not chunked, so a mechanism
covering only chunked datasets would not satisfy R13. P2.8 extends coverage
to contiguous datasets by deriving a **verification grid** over the
unmodified byte stream — a partition of the dataspace chosen so every leaf
is a single contiguous byte run — without rewriting the file or changing
its `H5D_CONTIGUOUS` layout message.

The `grid_hash` preimage (§1.2) is:

```
grid_hash = H(0x04 || layout_class || elem_size || dims || chunk_shape)
```

with `layout_class` one byte (`0x00` = chunked, `0x01` = contiguous) and
`elem_size` the file datatype size as a 4-byte little-endian integer. `0x04`
continues the domain-separation prefix scheme (`0x00` leaf, `0x01` internal,
`0x02` unallocated, `0x03` attribute integrity — note `0x03` was already
spent on the `MerkleAttr` integrity hash, so the grid preimage continues at
`0x04`). The prefix is applied to the *raw* hash rather than through the
leaf-hashing helper: wrapping it as `H(0x00 || 0x04 || …)` would collide
with the leaf hash of chunk data that merely begins with `0x04`. A C
implementation must reproduce this preimage byte-for-byte, including the
prefix and the little-endian integer encodings, or its `grid_hash` will not
match one written by the Rust prototype.

Three attacks are closed by what the preimage covers, each with a named
regression test in `subset_proof.rs`:

| Attack | Closed by |
|---|---|
| Cross-layout replay — a contiguous dataset's proof presented as a chunked one with the same extents | binding `layout_class` |
| Reinterpretation — the same bytes re-presented under a different element width (f32 claimed as f64, halving the apparent extent) | binding `elem_size` |
| Grid substitution — a verifier re-deriving the grid at a different target size than the prover used | the derived `chunk_shape` is itself in the preimage |

**No API branches on layout class.** The verification grid makes a
contiguous dataset indistinguishable from a chunked one to the proof
system: `H5Dverify_chunk` and `H5Dextract_subset` (§2.1, §2.2) take the
same arguments and return the same structures for both. The layout class
reaches the verifier only through the `grid_hash` preimage, never as a code
path. In the prototype this is enforced mechanically — a test greps
`verify_subset`'s own source and fails if `LayoutClass` appears as control
flow — and a C port should carry the same constraint, since it is what
makes "contiguous reduces to chunked" structural rather than incidental.

**Grid derivation is caller-parameterized, never environment-derived.** The
target leaf size is an explicit parameter (`H5Pset_merkle_leaf_target` or
equivalent; default 1 MiB). It must not be read from an environment
variable: a verifier that re-derived the grid from its own configuration
would accept whatever grid its environment happened to produce, defeating
the point of binding the grid at all. Only the *derivation* is automatic;
the derived grid must be stored and authenticated.

**Rejected layouts.** These are hard errors (`H5E_MERKLE_UNSUPPORTED_LAYOUT`,
§2.4), never warnings — emitting a proof over any of them would certify
something other than the data:

| Case | Why rejected |
|---|---|
| Compact, chunked, or virtual layout | Not a single byte range (chunked datasets already have a real grid) |
| Unallocated contiguous storage | Storage is allocated as a unit but need not be *written*; under `H5D_FILL_TIME_NEVER` the unwritten remainder holds whatever the filesystem supplies, so semantically identical datasets can yield different roots |
| External storage (`H5Pset_external`) | Raw data is split across external files, breaking the single-address assumption |
| Variable-length and reference datatypes | The contiguous stream stores global-heap identifiers while payloads live outside the byte range — a proof over it would **certify pointers rather than values** |

Full rationale, the derivation rule, and the known gaps (fill-policy binding
and user-block offset are not yet in any hash) are recorded in
`docs/contiguous-adapter.md`, which is the normative compatibility reference
for this section.

---

## 2. Proposed Public C API

The three functions S2-D2-Yr2 §7 names — `H5Dverify_chunk`,
`H5Dextract_subset`, `H5Fget_integrity_root` — plus two small companion
functions needed to make `H5Dextract_subset`'s opaque proof handle usable
(`H5Dsubset_proof_verify`, `H5Dsubset_proof_close`). All follow existing
HDF5 C API conventions (`hid_t` handles, `herr_t` returns, `H5E`-stack
error reporting).

### 2.1 `H5Dverify_chunk`

```c
typedef enum {
    H5D_MERKLE_VERIFIED   = 0,  /* chunk hash matches its authenticated leaf */
    H5D_MERKLE_TAMPERED   = 1,  /* hash mismatch: corruption or attack */
    H5D_MERKLE_UNSIGNED   = 2,  /* dataset has no merkle_root attribute */
    H5D_MERKLE_PENDING    = 3   /* uncommitted WAL entry for this chunk (write in progress) */
} H5D_merkle_status_t;

herr_t H5Dverify_chunk(hid_t dset_id, const hsize_t *chunk_coords,
                        H5D_merkle_status_t *status /*out*/);
```

Reads the chunk's current on-disk bytes (post-filter-pipeline, i.e. the
same ciphertext-or-plaintext the leaf hash actually covers — see §6.2 for
which), computes the O(log N) Merkle path from the companion
dataset/attribute, recomputes the root, and compares against the trusted
`merkle_root` attribute using constant-time comparison. `chunk_coords` uses
the same N-dimensional coordinate convention as `H5Dget_chunk_info_by_coord`
(**correction**: an earlier draft cited `H5Dget_chunk_info` here, which is
wrong — that function takes a *linear* chunk index as input and returns
N-D coordinates as an output parameter; `H5Dget_chunk_info_by_coord` is the
one whose input actually is an N-D coordinate array, matching what
`chunk_coords` needs to be). Returns `FAIL` (negative) only for operational
errors (I/O failure, malformed attribute, out-of-bounds coordinate) — a
*successful* tamper detection is a `SUCCEED` return with
`*status = H5D_MERKLE_TAMPERED`, not a failure return; callers must check
`*status`, not just the return code, exactly as `clawhdf5-format`'s
`verify_chunk` returns `Result<VerifyResponse, MerkleError>` rather than
folding tamper-detected into an error path (`merkle.rs`
`VerifyResponse`/`MerkleError` split).

**Chunk address resolution — a real implementation dependency this spec
should name.** Reading "the chunk's current on-disk bytes" is not a single
lookup in the real library: HDF5 resolves a chunk's on-disk address (and,
for filtered chunks, its variable on-disk size) via `H5D__chunk_lookup`,
which dispatches across five different on-disk chunk-index
representations (`H5D_CHUNK_IDX_BTREE`/`EARRAY`/`FARRAY`/`SINGLE`/`NONE`).
An `H5Dverify_chunk` implementation must call through this existing
machinery rather than assuming a single uniform addressing scheme — the
index-type variety is orthogonal to (and must compose with) the Merkle
verification logic, not something this design can abstract away.

### 2.2 `H5Dextract_subset`

```c
typedef struct H5D_subset_proof_t H5D_subset_proof_t;  /* opaque */

herr_t H5Dextract_subset(hid_t dset_id, hid_t space_id,
                          void *buf, size_t buf_size,
                          H5D_subset_proof_t **proof /*out*/);

herr_t H5Dsubset_proof_verify(const H5D_subset_proof_t *proof,
                               const void *buf, size_t buf_size,
                               hbool_t *valid /*out*/);

herr_t H5Dsubset_proof_close(H5D_subset_proof_t *proof);
```

`space_id` is a dataspace with a hyperslab selection (`H5Sselect_hyperslab`),
mirroring the `Selection`/`LeafOrder` inputs to `extract_subset`
(`clawhdf5-format/src/subset_proof.rs:572-619`). `H5Dextract_subset` reads
the requested chunks into `buf` and produces an opaque `H5D_subset_proof_t`
carrying the same fields as Rust's `SubsetProof` (sorted chunk-index set,
leaf hashes, deduplicated sibling nodes, grid params, coverage
certificate). `H5Dsubset_proof_verify` is the C equivalent of
`verify_subset` (`subset_proof.rs:721-...`): it independently recomputes
the expected chunk-index set from the caller's *own* trusted
`space_id`/grid knowledge — never from fields inside `proof` itself — and
rejects (`*valid = FALSE`) on any mismatch (wrong region, omitted chunk,
substituted chunk, or corrupted coverage certificate). This mirrors the
trust-boundary design already documented on `verify_subset`
(`subset_proof.rs:668-719`): the proof is untrusted wire data; only the
caller's own `space_id` and the file's authenticated `grid_hash` (from
`merkle_root`, §1.2) are trusted inputs.

Both functions apply unchanged to **contiguous** datasets (§1.6): the
verification grid supplies leaf granularity, and neither the arguments nor
the proof structure differ by layout class. `H5Dextract_subset` returns
`H5E_MERKLE_UNSUPPORTED_LAYOUT` for the layouts and datatypes §1.6 rejects.

### 2.3 `H5Fget_integrity_root`

```c
typedef struct {
    uint8_t  root[32];
    uint8_t  algorithm;          /* 0=SHA-256, 1=BLAKE3, 2=K12 */
    uint64_t version;
    uint64_t timestamp;          /* Unix seconds, from the signed payload */
    hbool_t  signature_present;
    hbool_t  signature_valid;    /* only meaningful if signature_present */
} H5F_integrity_info_t;

herr_t H5Fget_integrity_root(hid_t loc_id, H5F_integrity_info_t *info /*out*/);
```

`loc_id` may be a file or a dataset (a dataset ID resolves to its own
`merkle_root`; a file ID resolves to the file-level signed aggregate root,
§1.5). Signature verification requires a public key from a source
independent of the file itself — per S2-D2-Yr2's trust assumption 2
(§"Signing Authority and Key Management"), `H5Fget_integrity_root` **does
not** trust a file-embedded `merkle_pubkey` attribute for the
`signature_valid` determination unless the caller has explicitly opted into
that convenience mode via a property list (`H5Pset_integrity_verify_mode`,
out of scope for this document's core three functions but noted so the
trust boundary is not silently collapsed by an oversight in a later
implementation pass) — the default path requires a public key supplied by
the caller (from the KeyStore or a prior cached observation), matching
`verify_sig`'s signature (`clawhdf5-sign/src/lib.rs:415-429`), which takes
`ed_pub`/`ml_pub` as explicit parameters rather than reading them from the
payload being verified.

### 2.4 Error code mapping

**Registration mechanism.** HDF5 has two unrelated ways to add an error
class, and this document specifies the first, not the second: (a) the
build-time path (`src/H5err.txt` entries, code-generated by `bin/make_err`
into `H5Epubgen.h`) that every existing built-in class/minor code actually
uses; (b) the runtime `H5Eregister_class()`/`H5Ecreate_msg()` public API,
intended for external plugins that register dynamic `hid_t`-valued classes
at load time, not for core-library additions. Since `H5E_MERKLE` is
proposed as core library API (not a plugin), it should be added via (a) —
an `H5err.txt` entry generating a compile-time constant, exactly like
`H5E_CANTOPENFILE` and the other existing minor codes — not via the
runtime registration API. This distinction matters for whoever implements
this table: reaching for `H5Eregister_class()` would produce a
class/codes that behave differently (dynamically registered, not
compile-time constants other core code can `switch` on) than every
existing `H5E_*` class in the library.

New major error class `H5E_MERKLE`, minor codes mapped from
`MerkleError` (`merkle.rs:83-167`):

| Rust `MerkleError` variant | Proposed `H5E` minor code | Fires when |
|---|---|---|
| `HashMismatch{chunk_idx}` | `H5E_MERKLE_HASH_MISMATCH` | recomputed leaf hash ≠ stored leaf hash |
| `CompanionTampered` | `H5E_MERKLE_COMPANION_TAMPERED` | companion node array's SHA-256 ≠ `companion_hash` |
| `SignatureInvalid` | `H5E_MERKLE_SIG_INVALID` | Ed25519+ML-DSA-65 strict-AND verification failed |
| `MissingChunkGridMetadata` | `H5E_MERKLE_NO_ROOT` | dataset has no `merkle_root` attribute |
| `HyperslabOutOfBounds{idx}` | `H5E_MERKLE_OUT_OF_BOUNDS` | requested chunk index outside the grid |
| `TreeTooDeep{depth}` | `H5E_MERKLE_TREE_TOO_DEEP` | implied tree depth > 40 (overflow/DoS guard) |
| `NoncePending` | `H5E_MERKLE_WAL_PENDING` | uncommitted WAL entry for the chunk (maps to `H5D_MERKLE_PENDING`, not an error return) |
| `InvalidAttribute{reason}` | `H5E_MERKLE_INVALID_ATTR` | attribute unpack failed (wrong size / unknown alg / integrity mismatch) |
| `SelectionMismatch` | `H5E_MERKLE_SELECTION_MISMATCH` | subset proof covers the wrong region |
| `VersionRollback{observed,highest_seen}` | `H5E_MERKLE_ROLLBACK` | file version lower than previously observed (T4) |
| `JournalCorrupt` | `H5E_MERKLE_JOURNAL_CORRUPT` | provenance journal malformed/truncated |
| `JournalUnsupportedVersion{found}` | `H5E_MERKLE_JOURNAL_VERSION` | journal format newer than this build understands |
| `JournalNonMonotonic{appended,last}` | `H5E_MERKLE_JOURNAL_ORDER` | journal append version doesn't strictly increase (API misuse) |
| `GridHashMismatch` | `H5E_MERKLE_GRID_MISMATCH` | declared grid parameters don't match authenticated `grid_hash` (shape, element size, or layout class) |
| `UnsupportedLayout{reason}` | `H5E_MERKLE_UNSUPPORTED_LAYOUT` | layout/datatype outside contiguous-verification scope: non-contiguous, unallocated, external storage, or variable-length/reference datatype (§1.6) |

`H5D_MERKLE_PENDING`/`NoncePending` deliberately does **not** map to a
`FAIL` return from `H5Dverify_chunk` — an in-progress write is not a
tampering signal, and treating it as a hard error would make ordinary
concurrent write/verify races look like attacks.

---

## 3. Copy / Repack Preservation Semantics

The companion dataset (or inline attribute) and the root attribute must
move together, atomically, under every tool that can produce a copy or
partial copy of a dataset. Per S2-D2-Yr2's "Portability consideration": a
tool unaware of the convention will silently drop `/merkle/{name}`, leaving
a `merkle_root` attribute with no companion to back it — not a crash, but a
verification that will fail with `H5E_MERKLE_COMPANION_TAMPERED` (companion
missing ⇒ its hash can't be recomputed ⇒ treated as tampering, fail-closed
by construction) the next time anyone tries to verify.

- **`h5copy`**: when copying a dataset that carries a `merkle_root`
  attribute, `h5copy` must also copy `/merkle/{name}` (or the
  `merkle_nodes` attribute, for the inline case) as part of the same copy
  operation, not as a separately-discoverable sibling the user must
  remember to include. Proposed mechanism: extend `h5copy`'s existing
  object-dependency resolution (already used for copying a dataset's
  external links and referenced objects) to treat `/merkle/{name}` as a
  copy-dependency of any dataset whose `merkle_root.companion_hash` is
  non-zero. A `--no-merkle` flag opts out for users who explicitly want an
  unverifiable copy (e.g. extracting non-sensitive data for a demo).
- **`h5repack`**: repacking changes chunk layout, compression, or storage
  class — any of which changes the on-disk bytes the leaf hashes cover.
  `h5repack` must treat a dataset with a `merkle_root` attribute as
  requiring **tree regeneration**, not companion preservation: rehash the
  repacked chunks, rebuild the companion, and update `merkle_root`'s
  `root`/`companion_hash`. If the original dataset was **signed** (§1.5),
  `h5repack` must refuse by default (mirroring the crash-recovery
  "signed datasets fail closed" rule in §4.3 — repacking a signed dataset
  and silently re-deriving a new root is the same laundering risk as
  auto-rebuilding after a crash) unless invoked with an explicit
  `--resign-with=<keyfile>` flag that re-signs the new root under an
  operator-supplied key, or `--allow-unsigned-repack` that strips the
  signature and downgrades the dataset to unsigned-but-still-hashed.
- **Hyperslab-copy operations** (`H5Ocopy` with a source dataspace
  selection, or application code building a subset file): must use
  `H5Dextract_subset` (§2.2) rather than a raw `H5Dread` + `H5Dwrite`, so
  the destination file receives a `SubsetProof`-equivalent it can store
  (e.g. as a `merkle_subset_proof` attribute on the new dataset) rather
  than silently losing the parent-file provenance link. A subset copy
  produced this way is verifiable against the *original* file's root
  without needing the original file present, by design (that's the whole
  point of P1.5's coverage certificate) — but only if the copy path
  actually calls the extraction API instead of bypassing it.

---

## 4. Crash-Consistency Write Order and Enforcement Mechanism

### 4.1 Required write order

Mirroring `EncryptedChunkWriter::commit_with_write_order`
(`clawhdf5-filters/src/write_order.rs:174-225`):

1. Write chunk data (through the filter pipeline).
2. Write the updated companion Merkle nodes (leaf hash, then the
   recomputed path to root).
3. Update the `merkle_root` attribute (root, companion hash, and the
   dataset-level version counter — the strictly-per-commit counter, not
   any per-chunk value).

Writes must never be observed on-disk in the reverse order. Each step must
be durable (flushed) before the next begins.

### 4.2 Chosen mechanism: `H5Fflush` barriers (default), MDC flush dependencies as an opt-in optimization

S2-D2-Yr2 names two candidate mechanisms and asks this document to choose.
**This document specifies `H5Fflush(fid, H5F_SCOPE_LOCAL)` barriers after
each step as the default enforcement mechanism**, with MDC parent-child
flush dependencies available as an opt-in performance path behind a
property-list flag. Reasoning:

- **Correctness auditability matters more than throughput for a
  security-critical write path at this stage.** `H5Fflush` uses only
  public API — its correctness is trivial to reason about and to test
  (flush happened or it didn't; there's no cache-eviction-heuristic
  behavior to reverse-engineer). MDC flush dependencies require
  `H5AC_create_flush_dependency`, an internal API not exposed in public
  headers, meaning correctness depends on understanding and keeping pace
  with the MDC's internal eviction-ordering guarantees — a much larger
  surface to get right and to keep right across HDF5 library versions.
  For the initial C port, where the priority is a verifiably-correct
  crash-consistency guarantee (not yet a benchmarked one), the simpler
  mechanism is the safer default.
- **The Rust prototype's own choice supports this.** The prototype enforces
  this exact ordering via explicit sync calls at the I/O layer
  (`write_order.rs`'s `sink.sync()` after each of the three steps) — the
  same "explicit barrier per step" shape as `H5Fflush`, not an
  eviction-heuristic-dependent mechanism. Porting the validated approach
  rather than a novel one is lower-risk.
- **The throughput cost is real but bounded and known.** `H5Fflush`
  barriers add I/O round trips and prevent MDC coalescing across the three
  steps — S2-D2-Yr2 states this trade-off directly. For interactive/SWMR
  workloads (§5.3) this cost is acceptable (writes are infrequent relative
  to the flush cost). For high-frequency collective MPI-IO checkpoint
  workloads, this document's nonce-scheme recommendation (§5) already
  routes around the worst of the per-write sync cost by defaulting to
  random 96-bit nonces (eliminating the WAL, not just batching it) — so
  the marginal cost `H5Fflush` barriers add on top is the three-flush
  crash-consistency sequence itself, not compounded with a per-chunk WAL
  sync.
- **MDC flush dependencies remain the specified upgrade path** once (a) a
  performance benchmark on real collective-write workloads shows
  `H5Fflush` barriers are the actual bottleneck (not yet measured — this
  document does not have those numbers, consistent with the honesty
  constraint in §5.2), and (b) an HDF5-internals-fluent reviewer has
  audited the flush-dependency registration against the specific MDC
  version in use. Gate it behind a property-list selector,
  `H5Pset_merkle_flush_strategy(plist, strategy)` where `strategy` is one
  of `H5D_MERKLE_FLUSH_BARRIER` (default) or `H5D_MERKLE_FLUSH_DEPENDENCY`,
  so both implementations can ship in the same release — defaulting to
  the barrier — without blocking the flush-dependency work on this
  document's approval.

### 4.3 Recovery behavior

On open, if the companion dataset's recomputed hash doesn't match
`merkle_root.companion_hash` (or a chunk's leaf doesn't match its stored
value), the file is in a post-crash-or-tamper inconsistent state.
Recovery branches on signing status, mirroring
S2-D2-Yr2's "Crash consistency" recovery rule and `merkle_recovery.rs`'s
gates:

- **Unsigned dataset**: may auto-rebuild by rehashing on-disk chunk data —
  no authenticity guarantee was in force, so there is nothing to launder.
- **Signed dataset**: must **fail closed**. A verifier without the private
  key reports the inconsistency and halts. A writer with signing
  capability must *not* auto-rehash-and-resign — doing so would sign a new
  root over potentially-tampered chunks, institutionalizing the
  corruption (the "laundering risk" S2-D2-Yr2 names explicitly). The
  affected dataset is flagged unverified pending explicit operator
  action: `H5Drestore_to_version` (C equivalent of `restore_to_version`,
  `merkle_recovery.rs:212-225`) rolling back to the last journaled version
  whose signature verifies, or a manual re-sign from a trusted source.
  `H5Drestore_to_version` enforces the same two gates the Rust
  implementation does before accepting a restore: (1) the target journal
  record's signature must verify against `R` (`select_restore_record`,
  `merkle_recovery.rs:131-157`), and (2) after physically restoring the
  snapshot, `verify_dataset` must succeed **and** the restored dataset's
  live root must constant-time-equal the journaled `signed_root`
  (`verify_restored_dataset`, `merkle_recovery.rs:170-199`) — only then is
  the restore committed (atomic rename over the live file).

---

## 5. Nonce Scheme for Encrypted Datasets

### 5.1 The three options (recap, per S2-D2-Yr2 §"Nonce Derivation")

1. **Per-chunk WAL** (prototype default, `clawhdf5-filters/src/version_wal.rs`):
   journal `(chunk_idx, v_new)` before deriving the nonce, encrypt, then
   promote the journal record into the companion dataset and mark
   committed. Correct, but one crash-durable sync per chunk write.
2. **Epoch-based WAL batching**: buffer version-counter increments for a
   bounded epoch in memory, flush once as a collective record at the epoch
   boundary before any chunk in the epoch is written. Amortizes sync cost
   across the batch; still stateful/WAL-based.
3. **Random 96-bit nonces**: generate a fresh random 96-bit nonce per
   write, store it alongside the ciphertext (12 bytes/chunk overhead), no
   counter or WAL at all. Collision probability negligible up to 2³²
   writes per dataset. Loses the per-chunk version counter's *secondary*
   role in rollback detection — primary rollback defense (the dataset-level
   version counter + timestamp in the signed root, T4) is unaffected,
   since that lives in `merkle_root`/the signed payload regardless of
   which nonce scheme is in use.

### 5.2 Throughput characterization — what this document can and cannot claim

S2-D2-Yr2 asks this document to "characterize the throughput impact of
each option on representative HPC checkpoint workloads." **This document
does not have empirical C-HDF5-scale throughput numbers for any of the
three options** — no C implementation exists yet to benchmark, and
extrapolating the Rust prototype's numbers to the C library's different
I/O stack (MDC, VFD layer, MPI-IO driver) would overstate the confidence
of the claim. What can be stated analytically, from the sync-cost model
each option implies:

- Option 1's cost scales as *O(chunks written)* disk syncs — the WAL sync
  is on the write's critical path per chunk, independent of batch size.
- Option 2's cost scales as *O(epochs)* disk syncs — amortized, but still
  requires bounding epoch size against the in-memory buffer's crash-loss
  window (a crash mid-epoch loses the whole epoch's *un-flushed*
  increments, though the WAL replay on recovery for the epoch's own
  flushed record is still correct; the buffered-but-unflushed portion
  simply reverts to pre-epoch state, which is safe — a nonce that was
  never used is not a reuse).
- Option 3's cost is *O(1)* extra disk state per write (12 bytes appended
  to the chunk's existing write, no separate sync) — the elimination of
  the WAL's *separate* fsync is the whole point.

The relative ordering these models imply (3 fastest, 2 middle, 1 slowest
under write-dominated HPC checkpoint I/O) matches S2-D2-Yr2's own framing
("recommended for high-frequency parallel writes"). Turning this into an
actual measured throughput table (S2-D2-Yr2's RQ-numbered benchmarks
elsewhere in this repo, e.g. `crates/clawhdf5-format/benches/results/`,
follow the project's established "no fabricated numbers" discipline) is
out of scope for this design document and should be tracked as a follow-up
empirical task once a C prototype of at least option 3 exists to measure
against options 1/2 already implemented in Rust.

### 5.3 Chosen defaults (a decision this document makes)

- **Collective MPI-IO / HPC checkpoint workloads: random 96-bit nonces
  (option 3), default.** This is the workload class S2-D2-Yr2 explicitly
  motivates option 3 for ("HPC deployments where checkpoint I/O throughput
  is paramount"), and it is the only option that removes the WAL's
  per-write critical-path sync entirely rather than amortizing it — the
  only property that actually matters once rank counts reach 10⁴+.
  Configured via `H5Pset_merkle_nonce_scheme(plist, H5D_MERKLE_NONCE_RANDOM96)`
  on the collective I/O property list; this should be the value HDF5's
  MPI-IO VFD path sets by default rather than requiring every checkpoint
  application to opt in explicitly.
- **Interactive / SWMR workloads: per-chunk WAL (option 1), default.**
  These workloads write at a much lower rate, where the per-chunk sync
  cost is not the bottleneck it is under collective HPC I/O, and where the
  monotonic per-chunk version counter's secondary rollback-detection value
  (an extra signal beyond the dataset-level counter, useful for granular
  forensic triage per S2-D2-Yr2's `HashMismatch`-vs-`CompanionTampered`
  triage discussion) is worth keeping given there's no throughput pressure
  forcing the trade-off. This also matches the prototype's existing
  default (`version_wal.rs` is what's implemented and tested today), so
  the interactive path requires no new C-side WAL design — it's a direct
  port.
- **Epoch-based batching (option 2)** is retained as an explicit
  configuration choice (`H5D_MERKLE_NONCE_EPOCH_WAL`) for workloads between
  these two extremes — moderate-throughput parallel writers (e.g.
  thread-pool parallelism within a single process, or MPI jobs at ranks
  low enough that per-chunk WAL sync isn't yet the bottleneck) — but is
  not the default for either named category, matching how S2-D2-Yr2 itself
  frames it as a middle-ground mitigation rather than an endpoint
  recommendation.

---

## 6. Filter Pipeline Ordering and Signed-Root Attribute Layout

### 6.1 Filter order: shuffle → compress → encrypt → hash

**Correction to an earlier draft of this document**, caught by an
independent review pass: S2-D2-Yr2's actual design section on this,
§"Filter Pipeline Interaction" (`\label{sec:filter-order}`), states the
pipeline as shuffle → compress → encrypt → hash — the *same* order this
document and the implementation use, not the reverse. The
"compress → shuffle → encrypt" phrasing instead appears in a *different*
place: the P2.2 task-instructions paragraph in §7 "Implementation Plan"
(which itself cross-references `sec:filter-order` for the authoritative
order while restating it inconsistently). The real inconsistency this
section should flag is therefore *within S2-D2-Yr2 itself* — between its
own design section and its own later task-instruction restatement — not
between "the paper" (as a whole) and the implementation. This document
follows §"Filter Pipeline Interaction" (shuffle → compress → encrypt),
which is both the design section's own stated order and what
`clawhdf5-filters/src/filter_pipeline.rs` actually implements and tests
(`filter_pipeline.rs:391-433`, module doc at `filter_pipeline.rs:1-8,57-61`)
— shuffle-before-compress is also the standard HDF5 filter ordering for
unrelated reasons (byte-shuffling exposes more redundancy to a subsequent
compressor), so the task-paragraph's restatement is most likely the error,
and this design should not "correct" the validated implementation to match
an inconsistent restatement elsewhere in the same source document.
Hash computation stays last, over the final on-disk bytes (ciphertext, if
encryption is enabled) — this is the Encrypt-then-MAC property
S2-D2-Yr2 correctly identifies as required for third-party verification
without decryption and early rejection of tampered ciphertext before
spending CPU on decryption.

### 6.2 Leaf hash construction — reconciling two prototype formulas

The Rust prototype currently has **two different, unreconciled leaf-hash
formulas** (flagged as a known gap by the code's own `TODO(P2.4)` comment,
`clawhdf5-filters/src/filter_pipeline.rs:158-165`):

- `clawhdf5-format`'s `HashAlg::hash_leaf` (`merkle.rs:516-597`): plain
  `H(0x00 || chunk_data)`, no length prefix, no ciphertext/tag/version
  binding.
- `clawhdf5-filters`'s `compute_leaf_hash` (`filter_pipeline.rs:142-201`):
  the construction that actually matches S2-D2-Yr2's tuple-hash design,
  with a concrete BLAKE3 byte layout:
  ```
  H_leaf(k) = BLAKE3( 0x00 || be32(8) || le64(chunk_idx)
                      || be32(len(ciphertext_with_tag)) || ciphertext_with_tag
                      || le64(version) )
  ```
  (`ciphertext_with_tag` = ChaCha20-Poly1305 ciphertext with the 16-byte
  Poly1305 tag already appended, treated as one length-prefixed field.)

**This document specifies the `filters`-crate formula as the canonical
leaf hash for the C port** — it is the one that actually binds chunk index,
ciphertext, AEAD tag, and version counter together (closing the
position-swapping and version-rollback gaps S2-D2-Yr2's AEAD-integration
discussion requires), which the `format`-crate's simpler formula does not
do at all. **This is a prerequisite fix, not just a porting note**: before
`H5Dverify_chunk` (§2.1) can be implemented against real encrypted
datasets, `clawhdf5-format`'s verification path must be updated to compute
this same formula (or accept it as a pluggable leaf-hash function),
otherwise a `Dataset`-companion tree built by the encrypted write path
cannot be verified by the same verifier that checks unencrypted data. Filed
here as a blocking dependency for whoever picks up the C `H5Dverify_chunk`
implementation, not something this design document can silently paper
over by picking whichever formula is more convenient to describe.

For plaintext (unencrypted) chunks, the same shape applies without the
ciphertext/tag: `H(0x00 || len(k) || k || len(chunk) || chunk || version)`
— `compute_leaf_hash_plaintext`, `filter_pipeline.rs:224-247`.

Per S2-D2-Yr2's "Implementation note: formalized tuple serialization", a
conforming implementation may use NIST SP 800-185 TupleHash (or an
equivalent vetted multi-part hash API) to produce this same logical tuple
without hand-rolled length-prefix code; the byte layout above is the
*logical* contract, not a mandate to hand-roll it exactly as shown if a
canonicalized library call achieves the same binding.

### 6.3 Nonce derivation

Mirroring `derive_nonce` (`clawhdf5-filters/src/chacha20_filter.rs:167-184`):
key-separated subkeys (`DEK_enc`, `DEK_kdf` — distinct BLAKE3-derived keys
from the same DEK, so encryption and nonce-derivation never share key
material), then
`BLAKE3::new_derive_key("clawhdf5 chacha20 nonce v1").update(DEK_kdf).update(chunk_idx.to_le_bytes() || version.to_le_bytes()).finalize()[..12]`.
Both `chunk_idx` and `version` are little-endian u64 in the 16-byte context
buffer. This is a direct, unmodified port — no gap found here between the
prototype and what a C implementation needs to do.

---

## 7. Known Gaps Between This Spec and the Current Rust Prototype

Collected here for a reviewer's convenience — everything below is also
called out inline at its point of relevance above, this is the summary:

1. **Naming**: paper uses `_merkle`-prefixed names; prototype and this spec
   use unprefixed names (§1.1). Also, `merkle_journal.rs`'s own doc comment
   (line 25) describes a `/_merkle/journal` path while its actual attribute
   name constant is `merkle_journal` (unprefixed); and separately,
   `merkle.rs`'s own `MerkleError::MissingChunkGridMetadata` doc comment and
   `Display` impl (lines 84-85, 201-202) still say `` `_merkle_root` ``
   underscore-prefixed, even though the actual constant is unprefixed
   `merkle_root` — two independent instances of the *same* prototype
   documentation lagging behind its own code, not just against the paper.
2. **Companion dataset chunking**: paper recommends HDF5-chunked companion
   storage; prototype uses contiguous. This spec keeps contiguous (§1.3)
   with reasoning, not a silent adoption of whichever the paper said.
3. **Per-chunk version counter storage**: paper states `v_k` is stored in
   the companion Merkle dataset (8 bytes/chunk). The prototype's
   `VersionCounterStore` is currently in-memory-plus-WAL only — no
   on-disk companion field exists yet (`version_wal.rs:589-591`, an
   explicit code comment admitting this is a P2.2-step-3 shortcut). A C
   implementation following this design document should implement the
   on-disk companion field the paper specifies rather than porting the
   Rust prototype's current in-memory shortcut. Separately, the WAL that
   *is* implemented has its own known durability gap: journal writes
   currently use `Write::flush()`, not `fsync`/`sync_all`
   (`version_wal.rs:29-36`, explicit `TODO(P3)`) — a power loss between
   flush and the OS writing the page back can lose the most recent journal
   record. §5.3 recommends this WAL as the interactive/SWMR default
   partly *because* it is "what's implemented and tested today"; that
   framing should be read with this durability caveat attached; a C
   implementation must close the `fsync` gap, not just port the WAL as-is.
4. **Hybrid signature attachment**: `merkle_sig` already exists as a bare
   `HybridSignature`-bytes attribute in the current prototype
   (`clawhdf5-sign/src/lib.rs:439-486`, `write_sig_attr`/`read_sig_attr`,
   with a passing round-trip test) — an earlier draft of this document
   claimed otherwise and has been corrected in §1.5. The real, narrower
   gap is that verifying against the full canonical payload `R` needs a
   `timestamp` field that has no existing attribute home; §1.5 proposes
   widening `merkle_sig` to a 3391-byte `AlgID||version||timestamp||sig`
   layout that is a strict superset of the existing bare format, not a
   redesign of the existing function pair.
5. **Leaf hash formula**: two different, unreconciled formulas exist in
   the Rust prototype today (§6.2) — reconciling them is a prerequisite,
   not a detail this spec can leave implicit.
6. **Filter order**: not a paper-vs-implementation gap after all — an
   earlier draft of this document misattributed this (corrected in §6.1).
   S2-D2-Yr2's own design section (§"Filter Pipeline Interaction") already
   states shuffle→compress→encrypt, matching the validated implementation;
   the inconsistent compress→shuffle→encrypt phrasing is a restatement
   error in a *different* part of S2-D2-Yr2 itself (the P2.2 task
   paragraph in §7), not in this design or the code it describes.

7. **Contiguous-dataset support (P2.8), with two pieces still unbuilt.**
   The prototype implements grid derivation, `grid_hash` binding (§1.6),
   the layout-class/element-size negative tests, and proof reuse over
   contiguous datasets. Two gaps remain, both recorded in
   `docs/contiguous-adapter.md` rather than silently deferred: the leaf
   preimage does not yet bind the **fill value and fill-time policy** (the
   adapter refuses unallocated storage outright instead, which is safe but
   narrower than the eventual design), and the **user-block offset** is not
   itself bound into any hash (addresses are read from the layout message,
   so reads are correct today, but a shifted user block is not detected as
   tampering). A C implementation should close both rather than porting the
   current narrower behavior.

None of these are reasons to delay this design document — P2.6's job is to
specify the target C behavior precisely, including where that target is
ahead of (items 3, 4, 7) or corrects (item 2) the current Rust prototype.
Item 6 turned out not to be a prototype gap at all, on closer inspection —
see its corrected entry above. Items 1, 3, 4, 7, and the `version_wal.rs`
durability caveat folded into item 3 should be filed as Rust-prototype
follow-up work (cross-referenced from this document) so the C port and the
Rust prototype converge on one design rather than the C port silently
becoming the more authoritative one without anyone deciding that on
purpose.

---

## Review status

Per P2.6 step 7, this document requires an internal review from a senior
team member before it is considered final; per P2.6's own "Done when"
criterion, that has not yet happened. `docs/c-hdf5-integration-review.md`
(the artifact for reviewer comments and responses) is intentionally not
created by this commit — following this repository's existing convention
for P2.3's equivalent artifact (`docs/security-review-notes.md`, also not
yet created), that file is created once an actual review occurs, not
pre-filled as a placeholder.
