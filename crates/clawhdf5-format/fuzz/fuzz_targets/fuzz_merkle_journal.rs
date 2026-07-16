#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Fuzz ProvenanceJournal::unpack (P2.2b step 3) - must handle any input
    // without panicking: bad magic, unknown format version, hostile record
    // counts, lengths running past the buffer, non-monotonic versions,
    // invalid UTF-8 snapshot refs, trailing bytes.
    if let Ok(journal) = clawhdf5_format::merkle_journal::ProvenanceJournal::unpack(data) {
        // A successfully parsed journal must round-trip: pack() of the parsed
        // records re-parses to the same journal (canonical encoding).
        let repacked = journal.pack();
        let reparsed = clawhdf5_format::merkle_journal::ProvenanceJournal::unpack(&repacked)
            .expect("pack() output must always unpack");
        assert_eq!(reparsed.records(), journal.records());
    }
});
