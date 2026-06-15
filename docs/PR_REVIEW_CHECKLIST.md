# clawhdf5 — Condensed Review Checklist

> Short form of [PR_REVIEW_PROMPT.md](PR_REVIEW_PROMPT.md), tuned for the built-in
> `/review` skill or a quick pass. Use the full prompt for parser/`unsafe`/FFI-heavy PRs.

**Context:** pure-Rust HDF5 (edition 2024), parses untrusted binary; WAL + HNSW; PyO3/napi/JNI
bindings; SIMD. CI gate = `scripts/ci-test.sh` (fmt → clippy `-D warnings` → test → no_std).

Run `scripts/review-pr.sh main` to assemble the diff + risk signals first.

### Blocking — must hold
- [ ] **No panic on untrusted bytes.** Bounds-check before slicing (`ensure_len` pattern);
      `checked_*`/`saturating_*` on all offset/size/stride math; recursion depth-limited.
      No `unwrap`/`expect`/`panic!`/indexing/`as` truncation on any parse path.
- [ ] **`unsafe` is sound.** `// SAFETY:` comment present and true; `from_raw_parts` len +
      alignment + lifetime valid; SIMD has runtime detection + scalar fallback + length checks.
- [ ] **FFI doesn't UB.** No panic unwinding across the boundary; JNI handles no double-free/
      UAF; null + non-null-terminated C strings handled; errors converted, not `unwrap`ped.
- [ ] **No data corruption.** WAL `entry_count` consistent; replay tolerates a torn final
      entry; HNSW mirror invariant preserved on insert/delete/update.
- [ ] **`clawhdf5-format` stays `no_std`-clean** (std behind `#[cfg(feature = "std")]`).

### Should hold
- [ ] New parser → new/updated **fuzz target**; format change → **h5py interop test**.
- [ ] New logic has unit + integration tests; passes `scripts/ci-test.sh`.
- [ ] Specific error variants (not stringly-typed); `From`/`source()` chains intact.
- [ ] Default features unchanged unless intended (`hnsw`, `gpu-wgpu` on by default).
- [ ] No needless alloc/clone in hot paths; endianness explicit (`byteorder`).
- [ ] Conventional commit (`feat(crate):` / `fix:` / `harden:` / `test:`).

**For each blocking finding:** `file:line` + concrete triggering input + fix. Assert a panic/UB
only if you can name the input that triggers it; otherwise ask.
