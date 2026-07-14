#![no_main]
use libfuzzer_sys::fuzz_target;

use clawhdf5_filters::{WAL_RECORD_SIZE, WalRecord};

// Fuzz WalRecord::from_bytes with arbitrary bytes. Records are read back from a
// crash-durable log that may be torn or corrupted, so parsing must never panic:
// the CRC32 check, status-byte validation, and field decoding all have to reject
// bad input gracefully rather than aborting.
fuzz_target!(|data: &[u8]| {
    if data.len() < WAL_RECORD_SIZE {
        return;
    }

    let mut buf = [0u8; WAL_RECORD_SIZE];
    buf.copy_from_slice(&data[..WAL_RECORD_SIZE]);

    if let Ok(record) = WalRecord::from_bytes(&buf) {
        // A record that parses must serialize back to the identical bytes.
        assert_eq!(record.to_bytes(), buf);
    }
});
