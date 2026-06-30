//! SZIP (libaec Adaptive Entropy Coding) decompression.
//!
//! Gated by the `szip` feature which links against the system libaec library.

use crate::error::FormatError;

/// Decompress SZIP-compressed data using libaec.
///
/// `cd` is the HDF5 filter client data:
///   cd[0] = options mask (NN flag = 0x04, LSB = 0x40, allow_k13 = 0x100)
///   cd[1] = pixels per block (8, 10, 16, or 32)
///   cd[2] = pixels per scan line
///   cd[4] = bits per sample (element bit width)
pub(crate) fn szip_decompress(
    _data: &[u8],
    _cd: &[u32],
    _chunk_size: usize,
) -> Result<Vec<u8>, FormatError> {
    #[cfg(feature = "szip")]
    {
        szip_decode_impl(_data, _cd, _chunk_size)
    }
    #[cfg(not(feature = "szip"))]
    {
        Err(FormatError::UnsupportedFilter(
            crate::filter_pipeline::FILTER_SZIP,
        ))
    }
}

#[cfg(feature = "szip")]
fn szip_decode_impl(data: &[u8], cd: &[u32], chunk_size: usize) -> Result<Vec<u8>, FormatError> {
    if cd.len() < 5 {
        return Err(FormatError::ChunkedReadError(
            "szip: missing client data".into(),
        ));
    }
    let options = cd[0];
    let pixels_per_block = cd[1];
    let bits_per_sample = cd[4];
    if bits_per_sample == 0 || bits_per_sample > 32 {
        return Err(FormatError::ChunkedReadError(
            "szip: invalid bits per sample".into(),
        ));
    }
    if chunk_size == 0 {
        return Err(FormatError::ChunkedReadError(
            "szip: unknown output size".into(),
        ));
    }
    if data.is_empty() {
        return Err(FormatError::ChunkedReadError(
            "szip: empty input".into(),
        ));
    }

    // Map HDF5 options mask to libaec flags.
    // bit 2 (0x04): NN (nearest-neighbor) preprocessing
    // bit 6 (0x40): LSB order; absence means MSB
    // bit 8 (0x100): allow k=13
    let mut flags: u32 = 0;
    if options & 0x04 != 0 {
        flags |= libaec_sys::AEC_DATA_PREPROCESS;
    }
    if options & 0x40 == 0 {
        flags |= libaec_sys::AEC_DATA_MSB;
    }
    if options & 0x100 != 0 {
        flags |= libaec_sys::AEC_ALLOW_K13;
    }

    let mut out = vec![0u8; chunk_size];
    let mut strm = libaec_sys::AecStream::zeroed();
    strm.next_in = data.as_ptr();
    strm.avail_in = data.len();
    strm.next_out = out.as_mut_ptr();
    strm.avail_out = chunk_size;
    strm.bits_per_sample = bits_per_sample;
    strm.block_size = pixels_per_block;
    strm.rsi = 128; // HDF5 default: 128 blocks per reference sample interval
    strm.flags = flags;

    let result = unsafe { libaec_sys::aec_buffer_decode(&mut strm) };
    if result != 0 {
        return Err(FormatError::DecompressionError(format!(
            "szip: libaec error {result}"
        )));
    }
    let decoded_len = chunk_size - strm.avail_out;
    out.truncate(decoded_len);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn szip_disabled_returns_unsupported() {
        #[cfg(not(feature = "szip"))]
        {
            let result = szip_decompress(&[], &[4, 8, 10, 0, 8], 64);
            assert!(
                matches!(result, Err(FormatError::UnsupportedFilter(4))),
                "expected UnsupportedFilter(4), got {result:?}"
            );
        }
        #[cfg(feature = "szip")]
        {
            // When szip IS enabled, an empty buffer should error but not panic.
            let result = szip_decompress(&[], &[4, 8, 10, 0, 8], 64);
            assert!(result.is_err(), "empty buffer must not succeed");
        }
    }
}
