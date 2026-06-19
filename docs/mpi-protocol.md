# MPI-IO Merkle Tree Protocol

This document describes distributed Merkle tree construction protocols for parallel HDF5 workloads using MPI-IO. These strategies are designed for Phase 3 (C HDF5 integration) where multiple MPI ranks write disjoint chunk subsets concurrently.

## Overview

In a parallel HDF5 scenario with P ranks, each rank owns a disjoint subset of chunks. The challenge is constructing a globally consistent Merkle tree without:
- Global locks that serialize writes
- Per-chunk companion I/O that creates contention hotspots
- Excessive memory overhead on any single rank

We describe two complementary strategies optimized for different workload characteristics.

---

## Strategy 1: Eager Collective Flush

**Best for:** Frequent checkpoints, bounded memory, moderate rank counts (P ≤ 1024)

### Protocol

```
Phase 1: Local Leaf Hashing (parallel, no communication)
┌─────────────────────────────────────────────────────────────┐
│  Rank 0          Rank 1          Rank 2          Rank 3    │
│  ┌─────┐         ┌─────┐         ┌─────┐         ┌─────┐   │
│  │L0-L3│         │L4-L7│         │L8-L11│        │L12-L15│ │
│  └──┬──┘         └──┬──┘         └──┬──┘         └──┬──┘   │
│     │               │               │               │      │
│  hash_chunk()    hash_chunk()    hash_chunk()    hash_chunk()
│     │               │               │               │      │
│  ┌─────┐         ┌─────┐         ┌─────┐         ┌─────┐   │
│  │H0-H3│         │H4-H7│         │H8-H11│        │H12-H15│ │
│  └─────┘         └─────┘         └─────┘         └─────┘   │
└─────────────────────────────────────────────────────────────┘

Phase 2: MPI Reduce Tree (⌈log₂ P⌉ rounds)
┌─────────────────────────────────────────────────────────────┐
│  Round 1: Pairwise exchange                                 │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐│
│  │ Rank 0  │────▶│ Rank 0  │     │ Rank 2  │────▶│ Rank 2  ││
│  │ H0-H3   │     │ H0-H7   │     │ H8-H11  │     │ H8-H15  ││
│  └─────────┘     └─────────┘     └─────────┘     └─────────┘│
│       ▲               │               ▲               │     │
│  ┌────┴────┐          │          ┌────┴────┐          │     │
│  │ Rank 1  │          │          │ Rank 3  │          │     │
│  │ H4-H7   │          │          │ H12-H15 │          │     │
│  └─────────┘          │          └─────────┘          │     │
│                       ▼                               ▼     │
│  Round 2: Final aggregation                                 │
│                  ┌─────────┐                                │
│                  │ Rank 0  │                                │
│                  │ H0-H15  │  ◀── Full leaf array           │
│                  └────┬────┘                                │
│                       │                                     │
│                  build_tree()                               │
│                       │                                     │
│                  ┌────▼────┐                                │
│                  │  Root   │                                │
│                  │ + nodes │                                │
│                  └─────────┘                                │
└─────────────────────────────────────────────────────────────┘

Phase 3: Collective I/O Write (single pass)
┌─────────────────────────────────────────────────────────────┐
│  Rank 0 writes companion dataset via MPI_File_write_at_all  │
│  Other ranks participate with zero-length writes            │
│                                                             │
│  /merkle/dataset_name: [node0][node1]...[nodeN]            │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Pseudocode

```c
// Phase 1: Local leaf hashing
int my_start = (rank * total_chunks) / size;
int my_end = ((rank + 1) * total_chunks) / size;
int my_count = my_end - my_start;

uint8_t *my_leaves = malloc(my_count * 32);
for (int i = 0; i < my_count; i++) {
    hash_chunk(chunks[my_start + i], chunk_sizes[my_start + i],
               alg, &my_leaves[i * 32]);
}

// Phase 2: MPI Reduce to collect all leaves at root
int *recv_counts = NULL, *displs = NULL;
uint8_t *all_leaves = NULL;

if (rank == 0) {
    recv_counts = malloc(size * sizeof(int));
    displs = malloc(size * sizeof(int));
    all_leaves = malloc(total_chunks * 32);
}

// Gather counts
MPI_Gather(&my_count, 1, MPI_INT, recv_counts, 1, MPI_INT, 0, comm);

// Compute displacements (in bytes)
if (rank == 0) {
    displs[0] = 0;
    for (int i = 1; i < size; i++) {
        displs[i] = displs[i-1] + recv_counts[i-1] * 32;
    }
    // Convert counts to bytes
    for (int i = 0; i < size; i++) {
        recv_counts[i] *= 32;
    }
}

// Gather all leaves
MPI_Gatherv(my_leaves, my_count * 32, MPI_BYTE,
            all_leaves, recv_counts, displs, MPI_BYTE, 0, comm);

// Phase 3: Root builds tree and writes
if (rank == 0) {
    merkle_tree_t tree;
    merkle_build_from_leaves(&tree, all_leaves, total_chunks, alg);

    // Write companion dataset
    uint8_t *packed_nodes = merkle_pack_nodes(&tree);
    size_t packed_size = tree.node_count * 32;

    // Collective write (root writes, others no-op)
    MPI_File_write_at_all(fh, companion_offset, packed_nodes,
                          packed_size, MPI_BYTE, &status);

    // Write attribute with root hash + companion hash
    uint8_t attr[97];
    merkle_pack_attr(&tree, packed_nodes, packed_size, attr);
    // ... write attr to dataset
}

MPI_Barrier(comm);  // Ensure companion is visible before proceeding
```

### Complexity Analysis

| Metric | Cost |
|--------|------|
| Communication rounds | ⌈log₂ P⌉ |
| Data transferred per rank | O(N/P × 32) bytes |
| Memory at root | O(N × 32) bytes for full leaf array |
| I/O operations | 1 collective write |

### Advantages

- No global locks required (each rank owns disjoint leaves)
- Single collective I/O pass for companion dataset
- Bounded communication rounds (logarithmic in P)
- Works with existing MPI-IO collective infrastructure

### Limitations

- Root rank must hold full leaf array (O(N × 32) memory)
- Synchronization point at each flush
- Not suitable for P > 1024 with very large N

---

## Strategy 2: Lazy In-Memory Tree Maintenance

**Best for:** High-concurrency checkpoint workloads, large rank counts, streaming writes

**Recommended for:** Production checkpoint/restart scenarios per §5.5 analysis

### Protocol

```
Write Epoch (no companion I/O)
┌─────────────────────────────────────────────────────────────┐
│  Rank 0          Rank 1          Rank 2          Rank 3    │
│  ┌─────┐         ┌─────┐         ┌─────┐         ┌─────┐   │
│  │Chunk│────────▶│Chunk│────────▶│Chunk│────────▶│Chunk│   │
│  │Write│         │Write│         │Write│         │Write│   │
│  └──┬──┘         └──┬──┘         └──┬──┘         └──┬──┘   │
│     │               │               │               │      │
│     ▼               ▼               ▼               ▼      │
│  ┌─────┐         ┌─────┐         ┌─────┐         ┌─────┐   │
│  │Local│         │Local│         │Local│         │Local│   │
│  │Delta│         │Delta│         │Delta│         │Delta│   │
│  │Buffer│        │Buffer│        │Buffer│        │Buffer│  │
│  └─────┘         └─────┘         └─────┘         └─────┘   │
│                                                             │
│  ═══════════════════════════════════════════════════════   │
│                    No companion I/O                         │
│                    No lock contention                       │
│  ═══════════════════════════════════════════════════════   │
└─────────────────────────────────────────────────────────────┘

Epoch Close (file flush or explicit barrier)
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  1. Local: Compute leaf hashes for buffered chunks          │
│                                                             │
│  2. Collective: MPI_Allgather leaf hash ranges              │
│     ┌─────────────────────────────────────────────┐        │
│     │  [R0: L0-L99] [R1: L100-L199] [R2: L200-L299] ...   │ │
│     └─────────────────────────────────────────────┘        │
│                                                             │
│  3. Local: Each rank builds full tree from gathered leaves  │
│     (Replicated computation, avoids communication)          │
│                                                             │
│  4. Collective: Single MPI_File_write_at_all for companion  │
│     - Rank 0 writes full tree                               │
│     - Other ranks write zero bytes (collective semantics)   │
│                                                             │
│  5. Collective: Update merkle_root attribute                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Pseudocode

```c
// Per-rank state during write epoch
typedef struct {
    uint8_t *pending_leaves;   // Leaf hashes for chunks written this epoch
    int *pending_indices;       // Global chunk indices
    int pending_count;
    int pending_capacity;
} epoch_state_t;

// Called on each chunk write (no I/O, just buffer)
void merkle_lazy_record_chunk(epoch_state_t *state,
                               int global_chunk_idx,
                               const void *chunk_data,
                               size_t chunk_size,
                               hash_alg_t alg) {
    // Grow buffer if needed
    if (state->pending_count >= state->pending_capacity) {
        state->pending_capacity *= 2;
        state->pending_leaves = realloc(state->pending_leaves,
                                         state->pending_capacity * 32);
        state->pending_indices = realloc(state->pending_indices,
                                          state->pending_capacity * sizeof(int));
    }

    // Hash and store locally
    hash_chunk(chunk_data, chunk_size, alg,
               &state->pending_leaves[state->pending_count * 32]);
    state->pending_indices[state->pending_count] = global_chunk_idx;
    state->pending_count++;
}

// Called at epoch close (file flush)
void merkle_lazy_flush(epoch_state_t *state,
                       MPI_File fh,
                       MPI_Comm comm,
                       int total_chunks,
                       hash_alg_t alg) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // 1. Gather counts from all ranks
    int *all_counts = malloc(size * sizeof(int));
    MPI_Allgather(&state->pending_count, 1, MPI_INT,
                  all_counts, 1, MPI_INT, comm);

    // 2. Compute total and allocate full leaf array
    int total_pending = 0;
    for (int i = 0; i < size; i++) {
        total_pending += all_counts[i];
    }

    // 3. Gather all (index, hash) pairs
    // Using custom datatype for (int, uint8_t[32]) pairs
    typedef struct { int idx; uint8_t hash[32]; } leaf_entry_t;

    // NOTE: Use MPI_Type_create_struct to define leaf_entry_type, NOT
    // MPI_Type_contiguous. The compiler may insert padding between
    // `int idx` and `uint8_t hash[32]` for alignment. Example:
    //
    //   int blocklengths[2] = {1, 32};
    //   MPI_Aint displacements[2];
    //   MPI_Datatype types[2] = {MPI_INT, MPI_BYTE};
    //   leaf_entry_t dummy;
    //   MPI_Get_address(&dummy.idx, &displacements[0]);
    //   MPI_Get_address(&dummy.hash, &displacements[1]);
    //   displacements[1] -= displacements[0];
    //   displacements[0] = 0;
    //   MPI_Type_create_struct(2, blocklengths, displacements, types,
    //                          &leaf_entry_type);
    //   MPI_Type_commit(&leaf_entry_type);

    leaf_entry_t *my_entries = malloc(state->pending_count * sizeof(leaf_entry_t));
    for (int i = 0; i < state->pending_count; i++) {
        my_entries[i].idx = state->pending_indices[i];
        memcpy(my_entries[i].hash, &state->pending_leaves[i * 32], 32);
    }

    int *displs = malloc(size * sizeof(int));
    displs[0] = 0;
    for (int i = 1; i < size; i++) {
        displs[i] = displs[i-1] + all_counts[i-1];
    }

    leaf_entry_t *all_entries = malloc(total_pending * sizeof(leaf_entry_t));
    MPI_Allgatherv(my_entries, state->pending_count, leaf_entry_type,
                   all_entries, all_counts, displs, leaf_entry_type, comm);

    // 4. Build full leaf array (each rank does this independently)
    //    Pre-fill with null sentinel, then overwrite with actual hashes
    uint8_t null_sentinel[32];
    compute_null_sentinel(alg, null_sentinel);

    uint8_t *full_leaves = malloc(total_chunks * 32);
    for (int i = 0; i < total_chunks; i++) {
        memcpy(&full_leaves[i * 32], null_sentinel, 32);
    }

    // Overwrite only positions with actual written chunks
    for (int i = 0; i < total_pending; i++) {
        memcpy(&full_leaves[all_entries[i].idx * 32],
               all_entries[i].hash, 32);
    }

    // 5. Build tree (replicated on all ranks for consistency check)
    merkle_tree_t tree;
    merkle_build_from_leaves(&tree, full_leaves, total_chunks, alg);

    // 6. Collective write (rank 0 writes, others no-op)
    size_t tree_bytes = tree.node_count * 32;
    uint8_t *packed = merkle_pack_nodes(&tree);

    if (rank == 0) {
        MPI_File_write_at(fh, companion_offset, packed, tree_bytes,
                          MPI_BYTE, &status);
    }
    MPI_Barrier(comm);

    // 7. Clear epoch state
    state->pending_count = 0;

    // Cleanup
    free(my_entries);
    free(all_entries);
    free(full_leaves);
    free(all_counts);
    free(displs);
}
```

### Memory Budget

Per-rank memory during epoch:
```
M_rank = pending_chunks × (32 + sizeof(int))
       = pending_chunks × 36 bytes
```

At flush time (temporary):
```
M_flush = total_chunks × 32           // full leaf array
        + total_pending × 36          // gathered entries
        + tree_nodes × 32             // built tree
        ≈ 3 × total_chunks × 32       // worst case
```

### Complexity Analysis

| Metric | Cost |
|--------|------|
| I/O during epoch | 0 (companion I/O eliminated) |
| Communication at flush | 1 Allgather + 1 Allgatherv |
| Memory per rank (epoch) | O(local_chunks × 36) bytes |
| Memory at flush | O(N × 32) bytes (temporary) |
| I/O at flush | 1 collective write |

### Advantages

- **Zero companion I/O during writes** - eliminates §5.5 contention hotspots
- **No locks** - each rank buffers independently
- **Single sync point** - only at epoch close
- **Scales to large P** - communication is O(N) total, not O(N × P)

### Limitations

- Memory overhead proportional to chunks written per epoch
- All ranks must participate in flush (collective semantics)
- Not suitable for unbounded streaming (memory grows without bound)

### Recommended Configuration

```c
// Tune based on available memory and checkpoint frequency
#define MAX_PENDING_CHUNKS_PER_RANK  10000  // ~360 KB buffer
#define FLUSH_THRESHOLD_BYTES        (100 * 1024 * 1024)  // 100 MB

// Auto-flush when threshold exceeded
void merkle_lazy_record_chunk_with_autoflush(...) {
    merkle_lazy_record_chunk(state, ...);

    size_t buffered = state->pending_count * 36;
    if (buffered > FLUSH_THRESHOLD_BYTES / size) {
        merkle_lazy_flush(state, ...);
    }
}
```

---

## Strategy Comparison

| Criterion | Eager Collective | Lazy In-Memory |
|-----------|------------------|----------------|
| I/O during writes | None | None |
| Sync points | Per-flush | Per-epoch |
| Memory (epoch) | O(1) | O(local_writes) |
| Memory (flush) | O(N) at root | O(N) all ranks |
| Communication | log₂P rounds | 2 Allgather |
| Best for | Frequent small flushes | Large batch checkpoints |
| Lock-free | Yes | Yes |

## Selecting a Strategy

```
if (checkpoint_frequency == HIGH && chunks_per_checkpoint < 1000) {
    use EAGER_COLLECTIVE;
} else if (checkpoint_frequency == LOW && chunks_per_checkpoint > 10000) {
    use LAZY_IN_MEMORY;  // Recommended for production
} else {
    // Hybrid: eager for small writes, lazy for large epochs
    use ADAPTIVE;
}
```

---

## Integration with HDF5 Virtual Object Layer (VOL)

For Phase 3 C HDF5 integration, these protocols can be implemented as a VOL connector:

```c
// VOL callback for dataset write
herr_t merkle_vol_dataset_write(void *dset, hid_t mem_type_id,
                                 hid_t mem_space_id, hid_t file_space_id,
                                 hid_t dxpl_id, const void *buf) {
    // 1. Perform actual HDF5 write via passthrough
    herr_t ret = H5VLdataset_write(underlying_vol, ...);

    // 2. Record chunk hash in lazy buffer (no I/O)
    merkle_lazy_record_chunk(&epoch_state, chunk_idx, buf, size, alg);

    return ret;
}

// VOL callback for file flush
herr_t merkle_vol_file_flush(void *file, hid_t dxpl_id) {
    // 1. Flush underlying file
    herr_t ret = H5VLfile_flush(underlying_vol, ...);

    // 2. Perform lazy Merkle flush
    merkle_lazy_flush(&epoch_state, ...);

    return ret;
}
```

---

## §7 Performance Analysis: Parallel Tree Construction

### Benchmark Methodology

**Reproduction command:**
```bash
cargo run --example parallel_build_bench --release --features "merkle,parallel"
```

**Hardware:** 24-thread workstation (12-core CPU with SMT), DDR4-3200 RAM, Linux 6.6 kernel (WSL2).

**What is measured:** Wall-clock time for `MerkleTree::from_chunks` (sequential) and `MerkleTree::from_chunks_parallel` (Rayon parallel) over synthetic 1 KB chunks. Each configuration runs 3 warmup iterations followed by 30 timed trials; the table reports median wall time. BLAKE3 is used as the representative algorithm.

### Results

The `from_chunks_parallel` implementation uses Rayon to parallelize leaf hashing across available cores:

| N (chunks) | Data Size | Sequential | Parallel | Speedup | Parallel Throughput |
|------------|-----------|------------|----------|---------|---------------------|
| 10⁴        | 9.8 MiB   | 10.6 ms    | 5.6 ms   | 1.9×    | 1.7 GiB/s           |
| 10⁵        | 97.7 MiB  | 103.5 ms   | 24.2 ms  | 4.3×    | 4.0 GiB/s           |
| 10⁶        | 976.6 MiB | 1021.5 ms  | 202.2 ms | 5.1×    | 4.8 GiB/s           |

### Trend Analysis

**Sublinear speedup at small N (1.9× at 10⁴):** The modest speedup at 10⁴ chunks is attributable to Rayon's thread pool initialization and work-stealing overhead dominating the ~10 ms total runtime. With only ~0.4 ms of useful work per thread at 24 threads, synchronization costs (thread wake-up, task queue contention, cache line bouncing) consume a significant fraction of execution time. This is consistent with Amdahl's Law: when parallel work is small relative to coordination overhead, speedup is bounded by the serial fraction.

**Improving speedup at larger N (4.3× at 10⁵, 5.1× at 10⁶):** As chunk count increases, per-thread work grows proportionally while coordination overhead remains roughly constant. At 10⁶ chunks, each thread processes ~42,000 chunks, amortizing startup costs to negligible levels. The 5.1× speedup on 24 threads (21% parallel efficiency) reflects memory bandwidth saturation: BLAKE3 at 4.8 GiB/s approaches DDR4-3200's theoretical ~25 GiB/s per-channel bandwidth, with cache effects reducing effective throughput.

**Practical implications for MPI-IO:** In distributed workloads, each MPI rank's local leaf-hashing phase can exploit all available cores. The sublinear small-N speedup is acceptable because (1) small datasets complete quickly regardless, and (2) production checkpoint workloads typically operate at 10⁵–10⁶ chunks per rank where parallel efficiency is 4–5×.

### Correctness Verification

The `test_parallel_build_correctness` unit test (gated on `features = ["parallel", "blake3"]`) verifies that `from_chunks_parallel` and `from_chunks` produce identical Merkle roots across all three hash algorithms (SHA-256, BLAKE3, K12) on a 1,024-chunk synthetic dataset with 4+ Rayon threads.

---

## References

- §5.5: Contention Analysis for Parallel Merkle Companion Writes
- HDF5 MPI-IO Documentation: https://docs.hdfgroup.org/hdf5/develop/group___f_a_p_l.html
- MPI-3.1 Specification: Collective I/O (§13.2)
