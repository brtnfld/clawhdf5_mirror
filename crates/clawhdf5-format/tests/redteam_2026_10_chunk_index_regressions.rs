//! Regression tests for the 2026-10 red-team finding that the v4 chunk-index
//! parsers (fixed array, extensible array) checked file-supplied block
//! addresses with plain `a + b > len`. Debug builds panicked on the overflow;
//! release builds wrapped past the check and panicked on the slice. Both
//! must return a clean error for an address near `usize::MAX`.

use clawhdf5_format::error::FormatError;
use clawhdf5_format::extensible_array::{ExtensibleArrayHeader, read_extensible_array_chunks};
use clawhdf5_format::fixed_array::{FixedArrayHeader, read_fixed_array_chunks};

#[test]
fn fixed_array_with_wrapping_data_block_address_is_an_error() {
    let hdr = FixedArrayHeader {
        client_id: 0,
        element_size: 8,
        max_nelmts_bits: 8,
        num_elements: 4,
        data_block_address: u64::MAX - 4,
    };
    let file = vec![0u8; 4096];
    let err = read_fixed_array_chunks(&file, &hdr, &[40], &[10], 4, 8, 8).unwrap_err();
    assert!(matches!(err, FormatError::UnexpectedEof { .. }), "got {err:?}");
}

#[test]
fn extensible_array_with_wrapping_index_block_address_is_an_error() {
    let hdr = ExtensibleArrayHeader {
        client_id: 0,
        element_size: 8,
        max_nelmts_bits: 32,
        idx_blk_elmts: 4,
        min_dblk_nelmts: 16,
        super_blk_min_nelmts: 2,
        max_dblk_nelmts_bits: 10,
        num_elements: 4,
        index_block_address: u64::MAX - 4,
    };
    let file = vec![0u8; 4096];
    let err = read_extensible_array_chunks(&file, &hdr, &[40], &[10], 4, 8, 8).unwrap_err();
    assert!(matches!(err, FormatError::UnexpectedEof { .. }), "got {err:?}");
}

/// `super_blk_min_nelmts` is a u8 from the file, and the reader used to push
/// one entry per direct data block (`2^level` per level) before reading
/// anything: a value of 40 in a 4 KiB file aborted the process when the
/// vector's growth failed. The count must be checked against the bytes the
/// index block can actually hold.
#[test]
fn extensible_array_with_impossible_direct_block_count_is_an_error() {
    let mut file = vec![0u8; 4096];
    file[0..4].copy_from_slice(b"EAIB");
    for sblk_min in [40u8, 63, 64, 200, 255] {
        let hdr = ExtensibleArrayHeader {
            client_id: 0,
            element_size: 8,
            max_nelmts_bits: 32,
            idx_blk_elmts: 0,
            min_dblk_nelmts: 16,
            super_blk_min_nelmts: sblk_min,
            max_dblk_nelmts_bits: 10,
            num_elements: 1 << 40,
            index_block_address: 0,
        };
        let err = read_extensible_array_chunks(&file, &hdr, &[1 << 20], &[1], 4, 8, 8).unwrap_err();
        assert!(matches!(err, FormatError::Overflow(_)), "sblk_min={sblk_min}: got {err:?}");
    }
}
