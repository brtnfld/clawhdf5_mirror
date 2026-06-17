#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Fuzz MerkleAttr::unpack - should handle any input without panicking
    let _ = clawhdf5_format::merkle::MerkleAttr::unpack(data);

    // Fuzz MerkleAttrRef::from_slice - should handle any input without panicking
    let _ = clawhdf5_format::merkle::MerkleAttrRef::from_slice(data);
});
