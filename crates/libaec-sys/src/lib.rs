//! Raw FFI bindings to libaec (Adaptive Entropy Coding library).
//!
//! Provides the `aec_buffer_decode` convenience function for one-shot decompression.

// AEC flag constants matching aec.h
pub const AEC_DATA_PREPROCESS: u32 = 1; // NN preprocessing
pub const AEC_DATA_MSB: u32 = 2; // big-endian sample order
pub const AEC_RESTRICTED: u32 = 4; // restricted coding set
pub const AEC_ALLOW_K13: u32 = 8; // allow k=13 option

unsafe extern "C" {
    /// One-shot decompression. Returns 0 on success.
    ///
    /// # Safety
    /// `src` must be valid for `src_len` bytes; `dst` must be valid for `*dst_len` bytes.
    pub fn aec_buffer_decode(
        src: *const u8,
        src_len: usize,
        dst: *mut u8,
        dst_len: *mut usize,
        bits_per_sample: u32,
        block_size: u32,
        flags: u32,
    ) -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_are_correct() {
        assert_eq!(AEC_DATA_PREPROCESS, 1);
        assert_eq!(AEC_DATA_MSB, 2);
    }
}
