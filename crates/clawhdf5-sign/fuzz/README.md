# Fuzz Testing for clawhdf5-sign

This directory contains fuzz targets for `clawhdf5-sign` to ensure binary format parsers never panic on untrusted input.

## Targets

- **fuzz_hybrid_signature**: Fuzzes `HybridSignature::from_bytes()` to ensure it handles malformed input gracefully without panics

## Running

Requires `cargo-fuzz`:

```bash
cargo install cargo-fuzz
```

Run a specific target:

```bash
cd crates/clawhdf5-sign
cargo fuzz run fuzz_hybrid_signature -- -max_total_time=60
```

Run all targets:

```bash
cd crates/clawhdf5-sign
cargo fuzz list | xargs -I {} cargo fuzz run {} -- -max_total_time=10
```

## Coverage

The fuzz targets verify that:
- Binary deserialization never panics on arbitrary input
- All error paths return `Result<_, SignError>` properly
- Bounds checks prevent out-of-bounds reads
