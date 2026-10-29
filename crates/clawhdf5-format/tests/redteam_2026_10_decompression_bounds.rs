//! Regression tests for the 2026-10 red-team finding that the filter
//! pipeline's decoders were unbounded (T7). The P2.4 engagement recorded the
//! deflate case as refuted ("allocation is bounded elsewhere"); a direct
//! reproduction showed a 261 KB deflate chunk declared as 1 KiB decoding to
//! 256 MiB. Each decoder is now bounded by the chunk's uncompressed size.

use clawhdf5_format::error::FormatError;
use clawhdf5_format::filter_pipeline::{FILTER_DEFLATE, FilterDescription, FilterPipeline};
use clawhdf5_format::filters::{compress_chunk, decompress_chunk};

fn pipeline(id: u16) -> FilterPipeline {
    FilterPipeline {
        version: 2,
        filters: vec![FilterDescription {
            filter_id: id,
            name: None,
            flags: 0,
            client_data: vec![],
        }],
    }
}

fn assert_bomb_rejected(r: Result<Vec<u8>, FormatError>, what: &str) {
    match r {
        Err(FormatError::DecompressionError(msg)) => eprintln!("{what}: rejected: {msg}"),
        Err(e) => panic!("{what}: expected DecompressionError, got {e:?}"),
        Ok(v) => panic!("{what}: decoded {} bytes past the bound", v.len()),
    }
}

/// 64 MiB of zeros, declared as a 1 KiB chunk.
#[test]
fn deflate_bomb_is_rejected_at_the_chunk_bound() {
    let zeros = vec![0u8; 64 << 20];
    let bomb = compress_chunk(&zeros, &pipeline(FILTER_DEFLATE), 1).unwrap();
    assert!(bomb.len() < 256 << 10, "precondition: the bomb is small ({} bytes)", bomb.len());
    assert_bomb_rejected(decompress_chunk(&bomb, &pipeline(FILTER_DEFLATE), 1024, 1), "deflate");
}

/// An honest chunk exactly at the bound still decodes.
#[test]
fn deflate_honest_chunk_at_the_bound_decodes() {
    let data: Vec<u8> = (0..4096u32).map(|i| (i * 31 % 251) as u8).collect();
    let packed = compress_chunk(&data, &pipeline(FILTER_DEFLATE), 1).unwrap();
    let out = decompress_chunk(&packed, &pipeline(FILTER_DEFLATE), data.len(), 1).unwrap();
    assert_eq!(out, data);
}

#[cfg(feature = "zstd")]
#[test]
fn zstd_bomb_is_rejected_at_the_chunk_bound() {
    use clawhdf5_format::filter_pipeline::FILTER_ZSTD;
    let zeros = vec![0u8; 64 << 20];
    let bomb = compress_chunk(&zeros, &pipeline(FILTER_ZSTD), 1).unwrap();
    assert!(bomb.len() < 64 << 10, "precondition: the bomb is small ({} bytes)", bomb.len());
    assert_bomb_rejected(decompress_chunk(&bomb, &pipeline(FILTER_ZSTD), 1024, 1), "zstd");
    let data: Vec<u8> = (0..4096u32).map(|i| (i * 7 % 253) as u8).collect();
    let packed = compress_chunk(&data, &pipeline(FILTER_ZSTD), 1).unwrap();
    assert_eq!(decompress_chunk(&packed, &pipeline(FILTER_ZSTD), data.len(), 1).unwrap(), data);
}

/// LZ4's in-band header is an attacker-controlled u32 that lz4_flex used to
/// allocate up front; a 4 GiB claim must be rejected before decoding.
#[cfg(feature = "lz4")]
#[test]
fn lz4_oversized_header_is_rejected_before_allocation() {
    use clawhdf5_format::filter_pipeline::FILTER_LZ4;
    let mut bomb = u32::MAX.to_le_bytes().to_vec();
    bomb.extend_from_slice(&[0u8; 16]);
    assert_bomb_rejected(decompress_chunk(&bomb, &pipeline(FILTER_LZ4), 1024, 1), "lz4");
    let data = vec![7u8; 4096];
    let packed = compress_chunk(&data, &pipeline(FILTER_LZ4), 1).unwrap();
    assert_eq!(decompress_chunk(&packed, &pipeline(FILTER_LZ4), data.len(), 1).unwrap(), data);
}
