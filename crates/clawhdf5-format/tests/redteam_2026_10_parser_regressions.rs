//! Regression tests for 2026-10 red-team findings against the pure-Rust
//! HDF5 parser (T7: a crafted file must not stack-overflow, abort, or wrap
//! out of a `Result` on open). Attacker model: R/W on the file, no key.

use clawhdf5_format::chunked_read::generate_implicit_chunks;
use clawhdf5_format::datatype::Datatype;
use clawhdf5_format::error::FormatError;

/// Datatype::parse used to recurse one level per nested VL/Array/Compound/
/// Enum base, with no depth guard; a crafted 200 KiB datatype of 16000
/// nested VL types SIGABRTed the process on open (confirmed by direct
/// reproduction before the fix).
#[test]
fn deeply_nested_variable_length_datatype_is_rejected() {
    const DEPTH: usize = 2000; // ~24 KiB, far above the 32-level cap
    let mut data = Vec::with_capacity(DEPTH * 12 + 16);
    for _ in 0..DEPTH {
        data.push((1 << 4) | 9); // class 9 (VL), version 1
        data.push(0x00);
        data.push(0x00);
        data.push(0x00);
        data.extend_from_slice(&16u32.to_le_bytes());
    }
    // Innermost base: valid f64.
    data.push((1 << 4) | 1);
    data.push(0x20);
    data.push(0x1F);
    data.push(0x00);
    data.extend_from_slice(&8u32.to_le_bytes());
    data.extend_from_slice(&[0, 0, 0x34, 0, 0, 0x40, 0x0d, 0xff, 0, 0, 0, 0]);
    let err = Datatype::parse(&data).unwrap_err();
    assert!(
        matches!(err, FormatError::NestingDepthExceeded),
        "expected NestingDepthExceeded, got {err:?}"
    );
}

/// Same class of bug with class-10 (Array) wrapping Array wrapping ... :
/// each level is ~5 bytes, so this is a *much* denser attack than VL.
#[test]
fn deeply_nested_array_datatype_is_rejected() {
    // Array v3: class(1) | version(1) | <2 reserved bitfield bytes> | size(4)
    // | ndims(1) | dims (ndims*4) | base. Minimum 13 bytes per level with
    // ndims=1, dims=[1].
    const DEPTH: usize = 2000;
    let mut data = Vec::new();
    for _ in 0..DEPTH {
        data.push((3 << 4) | 10); // class 10 (Array), version 3
        data.push(0);
        data.push(0);
        data.push(0);
        data.extend_from_slice(&8u32.to_le_bytes()); // size
        data.push(1); // ndims = 1
        data.extend_from_slice(&1u32.to_le_bytes()); // dim = 1
    }
    data.push((1 << 4) | 1);
    data.push(0x20);
    data.push(0x1F);
    data.push(0x00);
    data.extend_from_slice(&8u32.to_le_bytes());
    data.extend_from_slice(&[0, 0, 0x34, 0, 0, 0x40, 0x0d, 0xff, 0, 0, 0, 0]);
    let err = Datatype::parse(&data).unwrap_err();
    assert!(
        matches!(err, FormatError::NestingDepthExceeded),
        "expected NestingDepthExceeded, got {err:?}"
    );
}

/// A real two-level compound still works (sanity: the guard is nowhere near
/// production datatypes).
#[test]
fn ordinary_two_level_compound_still_parses() {
    // Compound v3 with 1 member named "x", f64 base.
    let f64_dt = {
        let mut d = vec![(1u8 << 4) | 1, 0x20, 0x1F, 0x00];
        d.extend_from_slice(&8u32.to_le_bytes());
        d.extend_from_slice(&[0, 0, 0x34, 0, 0, 0x40, 0x0d, 0xff, 0, 0, 0, 0]);
        d
    };
    let mut compound = vec![(3u8 << 4) | 6, 1, 0, 0]; // class 6 (Compound), 1 member in bf0
    compound.extend_from_slice(&8u32.to_le_bytes()); // size
    compound.extend_from_slice(b"x\0"); // null-terminated name
    compound.push(0); // offset byte 0 (ob = 1 for size 8)
    compound.extend_from_slice(&f64_dt);
    let (dt, _) = Datatype::parse(&compound).expect("ordinary compound parses");
    assert!(matches!(dt, Datatype::Compound { .. }));
}

/// generate_implicit_chunks used to divide by zero on a zero chunk dim,
/// multiply unchecked u64 products past overflow, add raw addresses past
/// u64::MAX, and allocate a terabyte-sized Vec for an attacker-crafted
/// total_chunks. Each is now a FormatError.
#[test]
fn implicit_chunks_rejects_zero_chunk_dimension() {
    let err = generate_implicit_chunks(0x1000, &[100], &[0], 8).unwrap_err();
    assert!(matches!(err, FormatError::ChunkedReadError(_)), "got {err:?}");
}

#[test]
fn implicit_chunks_rejects_overflow_chunk_byte_size() {
    // chunk_byte_size = 2^20 * 2^20 * 2^20 * elem_size overflows u64.
    let err = generate_implicit_chunks(0, &[1 << 20; 3], &[1 << 20; 3], 8).unwrap_err();
    assert!(matches!(err, FormatError::Overflow(_)), "got {err:?}");
}

#[test]
fn implicit_chunks_rejects_overflow_total_chunks() {
    // 2^32 x 2^32 chunks of a tiny chunk size: total_chunks product overflows u64.
    let err = generate_implicit_chunks(0, &[1 << 32, 1 << 32], &[1, 1], 1).unwrap_err();
    assert!(matches!(err, FormatError::Overflow(_)), "got {err:?}");
}

#[test]
fn implicit_chunks_rejects_rank_mismatch() {
    // 1 chunk dim, 0 dataspace dims
    let err = generate_implicit_chunks(0, &[], &[10], 4).unwrap_err();
    assert!(matches!(err, FormatError::ChunkedReadError(_)), "got {err:?}");
}

#[test]
fn implicit_chunks_rejects_address_overflow() {
    // base_address near u64::MAX plus even one chunk's offset wraps.
    let err = generate_implicit_chunks(u64::MAX - 10, &[100], &[20], 8).unwrap_err();
    assert!(matches!(err, FormatError::Overflow(_)), "got {err:?}");
}
