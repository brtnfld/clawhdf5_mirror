# Merkle Tree Companion Dataset Implementation

## Overview

This changeset implements companion dataset storage for Merkle trees with integrity verification, enabling efficient chunk-level verification for large HDF5 datasets.

## Changes Summary

### 1. Core Merkle Module (`crates/clawhdf5-format/src/merkle.rs`)

#### New Types

- **`MerkleCompanionResult`** - Enum indicating storage location:
  ```rust
  pub enum MerkleCompanionResult {
      Inline { nodes: Vec<u8>, companion_hash: [u8; 32] },
      Dataset { companion_hash: [u8; 32] },
  }
  ```

#### Extended Attribute Layout (65 → 97 bytes)

| Offset | Size | Field |
|--------|------|-------|
| 0 | 32 | Merkle root hash |
| 32 | 1 | Algorithm ID (0=SHA-256, 1=BLAKE3, 2=K12) |
| 33 | 32 | Integrity hash (H(0x03 \|\| root \|\| alg)) |
| 65 | 32 | **NEW:** Companion hash (SHA-256 of nodes data) |

#### New Functions

- **`write_merkle_companion(file, name, tree)`** - Writes Merkle tree nodes:
  - ≤256 chunks: Returns `Inline` with packed nodes
  - >256 chunks: Creates `/merkle/{name}` dataset, returns `Dataset`

- **`compute_sha256(data)`** - Helper for companion integrity hash

#### New Methods on `MerkleAttr`

- `from_tree_with_companion(tree, companion_hash)` - Create attr with companion hash
- `verify_companion(nodes_data)` - Verify companion data integrity
- `has_companion()` - Check if companion hash is present (non-zero)
- `version()` - Returns the attribute format version (currently 0)

#### Versioned Struct with Zero-Copy Support

Added `MerkleAttrRef<'a>` for zero-copy attribute reading:

```rust
pub struct MerkleAttrRef<'a> {
    data: Cow<'a, [u8]>,  // Borrowed for zero-copy, Owned when needed
    version: u8,
}
```

**Version Constants:**
- `MERKLE_ATTR_VERSION_0 = 0` - Current version
- `MERKLE_ATTR_SIZE_V0 = 97` - Size of v0 attribute (32 + 1 + 32 + 32)

**Methods on `MerkleAttrRef`:**
- `from_slice(data)` - Zero-copy construction from borrowed slice
- `from_vec(data)` - Construct from owned Vec
- `from_attr(attr)` - Create from existing MerkleAttr (always owned)
- `version()` - Get format version
- `as_bytes()` - Get raw bytes
- `root()` / `root_array()` - Get root hash (slice or array)
- `algorithm_id()` / `algorithm()` - Get hash algorithm
- `integrity()` - Get integrity hash
- `companion_hash()` - Get companion hash
- `has_companion()` - Check if companion present
- `verify_integrity()` - Validate integrity hash
- `verify_companion(nodes_data)` - Validate companion data
- `to_owned_attr()` - Convert to owned MerkleAttr
- `into_owned()` - Consume and return owned data
- `is_borrowed()` - Check if data is borrowed (zero-copy)

**Zero-Copy Usage Example:**
```rust
// Read attribute bytes directly from file (no copy)
let attr_bytes: &[u8] = read_attribute_from_hdf5();

// Create zero-copy reference
let attr_ref = MerkleAttrRef::from_slice(attr_bytes)?;

// Verify without allocating
attr_ref.verify_integrity()?;

// Access fields directly from borrowed data
let root = attr_ref.root_array();
let alg = attr_ref.algorithm()?;
```

#### New Tests

- `test_merkle_attr_with_companion_hash` - Companion hash storage
- `test_merkle_attr_verify_companion` - Companion verification
- `test_write_merkle_companion_inline` - Inline storage (≤256 chunks)
- `test_write_merkle_companion_dataset` - Dataset storage (>256 chunks)
- `test_write_merkle_companion_threshold_boundary` - Exactly 256 chunks
- `test_write_merkle_companion_just_over_threshold` - 257 chunks
- **`test_merkle_roundtrip_1024_chunks`** - Full round-trip verification
- `test_merkle_attr_version` - Version field accessor
- `test_merkle_attr_ref_zero_copy` - Zero-copy construction verification
- `test_merkle_attr_ref_from_vec` - Owned data construction
- `test_merkle_attr_ref_invalid_size` - Invalid size rejection
- `test_merkle_attr_ref_verify_integrity` - Integrity verification via ref
- `test_merkle_attr_ref_verify_companion` - Companion verification via ref
- `test_merkle_attr_ref_to_owned` - Conversion to owned MerkleAttr

### 2. Python Bindings (`crates/clawhdf5-py/src/lib.rs`)

- Added `AttrValue::Bytes` handling in `attr_value_to_py()`:
  ```rust
  clawhdf5_rs::AttrValue::Bytes(b) => {
      pyo3::types::PyBytes::new(py, b).into_any().unbind()
  }
  ```

### 3. Cargo Configuration (`crates/clawhdf5-format/Cargo.toml`)

- Added `required-features = ["merkle"]` for examples:
  - `gen_merkle_vectors`
  - `write_merkle_test`

### 4. New Files

- **`examples/write_merkle_test.rs`** - Example generating test HDF5 with companion
- **`test-vectors/p1.3-h5dump.txt`** - Reference h5dump output:
  ```
  HDF5 "merkle_test.h5" {
  FILE_CONTENTS {
   group      /
   group      /merkle
   dataset    /merkle/sensor_data
   dataset    /sensor_data
   }
  }
  ```

## On-Disk Layout

```
/                           # Root group
├── sensor_data             # Main dataset with chunk data
│   └── @merkle_root        # 97-byte attribute with root + companion hash
└── merkle/                 # Merkle companion group
    └── sensor_data         # Flat u8 array of tree nodes (2047 × 32 bytes)
```

## Verification Flow

1. Read `merkle_root` attribute from dataset
2. Extract companion hash from bytes 65-97
3. Read `/merkle/{dataset_name}` companion dataset
4. Compute SHA-256 of companion data
5. Compare with stored companion hash (constant-time)
6. If match, walk tree to verify individual chunks

## Test Results

```
running 47 tests
test merkle::tests::test_merkle_roundtrip_1024_chunks ... ok
test merkle::tests::test_merkle_attr_ref_zero_copy ... ok
test merkle::tests::test_merkle_attr_ref_verify_integrity ... ok
... (all pass)
test result: ok. 47 passed; 0 failed
```

## Workspace Compatibility

All existing tests pass with 0 regressions:
- clawhdf5-format: 47 merkle tests + 485 other tests
- Full workspace: All test suites pass
