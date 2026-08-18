# Contiguous-Dataset Adapter: Supported and Rejected Layouts

HDF5's *default* dataset layout is contiguous, not chunked. The Merkle
machinery built in P1.5 needs a chunk grid to supply leaf granularity, and a
contiguous dataset has none. P2.8 closes that gap by deriving a
**verification grid** over the unmodified byte stream — a partition of the
dataspace chosen so every leaf is a single contiguous byte run — without
rewriting the file or changing its `H5D_CONTIGUOUS` layout message.

This document records which HDF5 layout and datatype features the adapter
accepts, which it rejects and why, and what a caller must supply. It is the
source for the compatibility section of the C HDF5 integration design
(`c-hdf5-integration-spec.md`).

## The reuse claim, and how it is enforced

The load-bearing design claim is that a contiguous dataset, once given a
verification grid, is *indistinguishable from a chunked one to the proof
system*. That claim is falsifiable, and it is tested as such rather than
assumed:

| Claim | Enforced by |
|---|---|
| The verifier never branches on layout class | `test_verify_subset_never_branches_on_layout_class` greps `verify_subset`'s own source and fails if `LayoutClass` appears as control flow |
| `SubsetProof` is unchanged from P1.5 — no new field, no byte-range variant | `test_contiguous_and_chunked_proofs_are_identical_except_grid_hash` |
| A contiguous proof and a chunked proof over the same bytes under the same grid are identical except `layout_class`/`grid_hash` | same test — asserts equal root, leaf hashes, proof nodes, and indices |

The layout class reaches the verifier only through the `grid_hash` preimage,
never as a code path. That is what makes "contiguous reduces to chunked" a
structural property rather than a behavioral coincidence.

## What is bound into `grid_hash`

```
grid_hash = H(0x04 || layout_class || elem_size || dims || chunk_shape)
```

`0x04` continues the domain-separation prefix scheme (`0x00` leaf, `0x01`
internal, `0x02` unallocated, `0x03` attribute integrity), applied to the raw
hash rather than through the leaf helper — wrapping it as
`H(0x00 || 0x04 || …)` would collide with the leaf hash of chunk data that
merely begins with `0x04`.

Three attacks are closed by what this preimage covers:

- **Cross-layout replay** — a contiguous dataset's proof presented as a
  chunked one with the same extents. Closed by binding `layout_class`.
- **Reinterpretation** — the same bytes re-presented under a different
  element width (f32 claimed as f64, halving the apparent extent). Closed by
  binding `elem_size`.
- **Grid substitution** — a verifier that re-derives the grid at a different
  target size than the prover used. Closed because the derived
  `chunk_shape` is itself in the preimage.

Each has a named negative test asserting the specific error variant
(`MerkleError::GridHashMismatch`), not merely that verification fails.

## Grid derivation

`verification_grid(dims, elem_size, target_bytes) -> Option<Vec<u64>>` ports
the DAOS VOL connector's automatic grid-selection rule (its *selection* rule
only — nothing here converts storage or rewrites the file). `None` means the
dataset is too small to be worth tiling and should stay a single leaf; it is
not an error.

**`target_bytes` is always an explicit parameter, never read from an
environment variable.** A verifier that re-derived the grid from its own
configuration would accept whatever grid its environment happened to
produce, defeating the point of binding the grid. Only the *derivation* is
automatic; the derived grid must be stored and authenticated. The shipped
default is `verification_grid::DEFAULT_TARGET_BYTES` (1 MiB), and both
`write_merkle_attr` and `contiguous_tree_and_grid` must use the same value
or the bound `grid_hash` describes a different tiling than the tree it
authenticates.

Every grid this rule returns has the form `(1, …, 1, T_k, n_{k+1}, …)`:
axes faster than the pivot are full extent (so their rows merge into one
run) and axes slower than it are 1 (so no gather occurs). That is what makes
each leaf a single contiguous byte run, and therefore what makes
`leaf_byte_range` arithmetic rather than a stored index. It is property-
tested over 10,000 generated dataspaces.

## Accepted

| Feature | Notes |
|---|---|
| Contiguous layout with an allocated address | The supported case |
| Fixed-point, floating-point, string, bitfield, opaque, compound, enum, array datatypes | Any datatype whose values live inside the dataset's own byte range |
| Any rank 1–4 (and beyond) | Derivation and `leaf_byte_range` are rank-generic |
| Row-major and Morton leaf order | Per P1.5; Morton changes numbering only |

## Rejected — hard errors, never warnings

All rejections surface as `MerkleError::UnsupportedLayout { reason }` from
`subset_proof::contiguous_layout`. They are hard errors by design: emitting a
proof over any of these would certify something *other than the data*, which
is a worse outcome than refusing.

| Reason | Case | Why rejected |
|---|---|---|
| `NotContiguous` | Compact, chunked, or virtual layout | Not a single byte range; chunked datasets already have a real grid and use the P1.5 path |
| `Unallocated` | Contiguous layout with the undefined-address sentinel | Storage is allocated as a unit but need not be *written*. Under `H5D_FILL_TIME_NEVER` the unwritten remainder holds whatever the filesystem supplies, so two runs producing semantically identical datasets can yield different roots — fatal for reproducible provenance. Such a dataset must commit to a distinguished "unallocated, fill value *f*" state rather than to bytes; that state is **not yet implemented**, so the case is refused rather than guessed |
| `ExternalStorage` | `H5Pset_external` | Raw data is split across external files, breaking the single-address assumption. Detected by the presence of the External Data Files message (`0x0007`), which this parser recognizes only well enough to reject |
| `IndirectDatatype` | Variable-length and reference datatypes | The contiguous stream stores *global-heap identifiers*, while the payloads live in heap collections outside the dataset's byte range. Leaves over the stream would authenticate the identifiers but not the data they point at — **a proof that certifies pointers rather than values** |
| `MalformedHeader` | Dataspace, Datatype, or DataLayout message missing or unparseable | Nothing trustworthy to derive extents from |

## Known gaps

These are recorded rather than silently deferred:

- **Fill policy is not yet in the leaf preimage.** The design requires the
  preimage to bind the fill value and fill-time policy so that an unwritten
  remainder cannot vary the root. Today the adapter refuses unallocated
  storage outright (`Unallocated`) instead, which is safe but narrower than
  the eventual design.
- **User block offset is not bound.** A nonzero user block shifts every file
  address. The adapter reads the address from the layout message rather than
  assuming a fixed origin, so it reads correctly today, but the user-block
  size is not itself part of any hash.
- **Byte order is enforced structurally, not by a datatype check.** Leaves
  are computed over raw file bytes because the API takes `&[u8]` and never
  converts — `test_root_is_over_raw_file_bytes_not_converted_values` pins
  that a big-endian and a little-endian encoding of the same logical values
  produce different roots, as they must. There is no separate check that a
  caller *hasn't* pre-converted; the type signature is the enforcement.
- **No production writer calls this yet.** `contiguous_tree_and_grid` and
  `contiguous_layout` are exercised by tests and by
  `examples/gen_contiguous_subset_vectors.rs`; wiring them into a real write
  path in a consumer crate remains open.

## Artifacts

- `test-vectors/contiguous-subset-vectors.json` — three serialized
  `SubsetProof`s over a contiguous dataset (compact sub-cube, plane, 1-D
  range), in the same schema as P1.5's `subset-vectors.json` so both can be
  fed to one verifier harness. Regenerate with
  `cargo run --features merkle,blake3 --example gen_contiguous_subset_vectors`.
- `test-vectors/verification-grid-shapes.json` — the derivation fixture
  table pinning `(dims, elem_size, target) -> grid`.
