# Merkle Tree Hashing: Two Independent Layers of Parallelism

`clawhdf5-format`'s Merkle tree implementation (`src/merkle.rs`) has two
distinct, independent parallelism mechanisms. They're easy to conflate
since both ultimately make hashing faster, but they operate at different
levels and neither implies the other.

## 1. Multi-core, across chunks (`parallel` Cargo feature)

`MerkleTree::from_chunks_parallel()` (`src/merkle.rs`, gated behind
`#[cfg(feature = "parallel")]`, with `parallel = ["rayon", "std"]` in
`Cargo.toml`) uses `rayon`'s `par_iter()` to hash all **leaf chunks**
across multiple CPU cores simultaneously:

```rust
#[cfg(feature = "parallel")]
pub fn from_chunks_parallel(chunks: &[&[u8]], alg: HashAlg) -> Self {
    use rayon::prelude::*;
    let leaf_hashes: Vec<[u8; HASH_SIZE]> = chunks
        .par_iter()
        .map(|chunk| alg.hash_leaf(chunk))
        .collect();
    Self::from_leaf_hashes(&leaf_hashes, alg)
}
```

This is the right tool when hashing many chunks of a dataset (e.g.
building the tree over thousands of HDF5 chunks) — leaf hashing is
embarrassingly parallel across chunks.

**Limitation**: only leaf hashing is parallelized. The internal-node
combining step in `from_leaf_hashes` walks the tree bottom-up on a
single thread:

```rust
for i in (0..internal_nodes).rev() {
    let left_idx = 2 * i + 1;
    let right_idx = 2 * i + 2;
    nodes[i] = alg.hash_pair(&nodes[left_idx], &nodes[right_idx]);
}
```

For a large chunk count this is fine, since leaf hashing dominates the
total work, but it means `from_chunks_parallel` is not end-to-end
parallel — only the leaf stage is.

## 2. Single-core SIMD, within one hash call

Independently of the above, a *single* `hash_chunk()` call (the function
exercised by the P1.2 hash benchmark, see
`crates/clawhdf5-format/benches/results/hash-bench-explanatory-note-localhost.localdomain.md`)
can itself be very fast on one core, with no `rayon` or multithreading
involved at all. Confirmed via `Cargo.lock`: the `blake3` crate dependency
here has no `rayon` feature enabled.

BLAKE3 measured ~10-12.5 GiB/s on a single core at 1 MB chunk size on this
host (AMD Ryzen 9 9950X3D). That throughput comes from BLAKE3's internal
design: it splits input into 1024-byte sub-chunks and hashes multiple of
them **in parallel using SIMD lanes within one core** (8-wide on AVX2,
16-wide on AVX-512) — intra-call SIMD parallelism, not multithreading.
This CPU's full-width AVX-512 support and high boost clock are a
particularly favorable combination for this mechanism. SHA-256 and K12
don't have an equivalent internal-tree structure, so they don't benefit
from this effect and stay flat relative to chunk size.

## Summary

| Mechanism | Scope | Enabled by | Benefits |
|---|---|---|---|
| `from_chunks_parallel` (rayon) | Across chunks, multi-core | `parallel` feature, explicit call | All algorithms, when hashing many chunks |
| BLAKE3 internal SIMD | Within one call, single-core | Always on (BLAKE3 crate default), automatic | BLAKE3 only, scales with chunk size |

The hash-bench harness (`examples/hash_bench_harness.rs`) only exercises
mechanism 2 — it calls `hash_chunk()` once per trial and never invokes
`from_chunks_parallel()`. The two mechanisms are independent and would
compose: hashing many large chunks with the `parallel` feature would get
both multi-core leaf distribution *and* per-chunk SIMD throughput from
BLAKE3.
