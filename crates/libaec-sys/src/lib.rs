//! Raw FFI bindings to libaec (Adaptive Entropy Coding library).
//!
//! Exposes the `aec_buffer_decode` one-shot convenience function via
//! the `AecStream` control structure, matching the libaec C API.

use std::os::raw::c_void;

// AEC flag constants matching aec.h
pub const AEC_DATA_PREPROCESS: u32 = 1; // NN preprocessing
pub const AEC_DATA_MSB: u32 = 2; // big-endian sample order
pub const AEC_RESTRICTED: u32 = 4; // restricted coding set
pub const AEC_ALLOW_K13: u32 = 8; // allow k=13 option

/// Mirror of `struct aec_stream` from `<libaec.h>`.
///
/// Must match the C layout exactly — all fields are C ABI integers/pointers.
#[repr(C)]
pub struct AecStream {
    pub next_in: *const u8,
    pub avail_in: usize,
    pub total_in: usize,
    pub next_out: *mut u8,
    pub avail_out: usize,
    pub total_out: usize,
    pub bits_per_sample: u32,
    pub block_size: u32,
    pub rsi: u32,
    pub flags: u32,
    /// Opaque internal state; initialised to null, set by libaec on first call.
    pub state: *mut c_void,
}

impl AecStream {
    /// Return a zero-initialised stream safe to pass to libaec.
    pub fn zeroed() -> Self {
        Self {
            next_in: std::ptr::null(),
            avail_in: 0,
            total_in: 0,
            next_out: std::ptr::null_mut(),
            avail_out: 0,
            total_out: 0,
            bits_per_sample: 0,
            block_size: 0,
            rsi: 0,
            flags: 0,
            state: std::ptr::null_mut(),
        }
    }
}

unsafe extern "C" {
    /// One-shot decompression.  Returns `AEC_OK` (0) on success.
    ///
    /// # Safety
    /// `strm.next_in` must be valid for `strm.avail_in` bytes;
    /// `strm.next_out` must be valid for `strm.avail_out` bytes.
    pub fn aec_buffer_decode(strm: *mut AecStream) -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_are_correct() {
        assert_eq!(AEC_DATA_PREPROCESS, 1);
        assert_eq!(AEC_DATA_MSB, 2);
    }

    #[test]
    fn aec_stream_zeroed_has_null_ptrs() {
        let s = AecStream::zeroed();
        assert!(s.next_in.is_null());
        assert!(s.next_out.is_null());
        assert!(s.state.is_null());
    }
}
