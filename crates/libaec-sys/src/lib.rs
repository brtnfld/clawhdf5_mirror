//! Raw FFI bindings to libaec (Adaptive Entropy Coding library).
//!
//! Exposes `aec_buffer_encode` and `aec_buffer_decode` via the `AecStream`
//! control structure, matching the libaec C API defined in `<libaec.h>`.

use std::os::raw::c_void;

// AEC flag constants — values match <libaec.h> exactly.
pub const AEC_DATA_SIGNED: u32 = 1;
pub const AEC_DATA_3BYTE: u32 = 2;
pub const AEC_DATA_MSB: u32 = 4;
pub const AEC_DATA_PREPROCESS: u32 = 8;
pub const AEC_RESTRICTED: u32 = 16;

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
    /// One-shot compression.  Returns `AEC_OK` (0) on success.
    ///
    /// # Safety
    /// `strm.next_in` must be valid for `strm.avail_in` bytes;
    /// `strm.next_out` must be valid for `strm.avail_out` bytes.
    pub fn aec_buffer_encode(strm: *mut AecStream) -> i32;

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
    fn constants_match_libaec_header() {
        assert_eq!(AEC_DATA_SIGNED, 1);
        assert_eq!(AEC_DATA_3BYTE, 2);
        assert_eq!(AEC_DATA_MSB, 4);
        assert_eq!(AEC_DATA_PREPROCESS, 8);
        assert_eq!(AEC_RESTRICTED, 16);
    }

    #[test]
    fn aec_stream_zeroed_has_null_ptrs() {
        let s = AecStream::zeroed();
        assert!(s.next_in.is_null());
        assert!(s.next_out.is_null());
        assert!(s.state.is_null());
    }
}
