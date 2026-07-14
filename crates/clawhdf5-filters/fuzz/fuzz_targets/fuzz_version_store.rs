#![no_main]
use libfuzzer_sys::fuzz_target;

use clawhdf5_filters::VersionCounterStore;

// Fuzz VersionCounterStore::from_bytes with arbitrary bytes. This is the
// companion-dataset deserializer: the leading u64 `count` is attacker-controlled
// and must never cause a panic (integer overflow on `8 + count * 16`) or an
// unbounded allocation. Any input must return cleanly as Ok or Err.
fuzz_target!(|data: &[u8]| {
    if let Ok(store) = VersionCounterStore::from_bytes(data) {
        // A successful parse must round-trip back to an equivalent buffer.
        let reencoded = store.to_bytes();
        let reparsed =
            VersionCounterStore::from_bytes(&reencoded).expect("re-encoded store must parse");
        assert_eq!(reparsed.len(), store.len());
        assert_eq!(reparsed.max_version(), store.max_version());
    }
});
