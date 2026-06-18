# P1.2b Capability Matrix

Representative cell: chunk_size=256 KB, source=synthetic (time metrics averaged over all trials).

| metric | flat (SHA-256) | mac (HMAC-SHA-256) | sig_ed25519 | sig_mldsa (ML-DSA-65) |
|---|---|---|---|---|
| commit_time | 1273325271 ns | 1275097189 ns | 5856361231 ns | not implemented (Phase 2) |
| verify_chunk_latency | N/A — no partial access | 93910 ns | 229237 ns | not implemented (Phase 2) |
| verify_dataset_latency | 1273389596 ns | 1275006807 ns | 3081349476 ns | not implemented (Phase 2) |
| append_time | 1273729425 ns | 114676 ns | 432668 ns | not implemented (Phase 2) |
| update_time | 1273447814 ns | 93446 ns | 428336 ns | not implemented (Phase 2) |
| metadata_bytes | 32 bytes | 436864 bytes | 873728 bytes | not implemented (Phase 2) |
| public_verification | Yes | N/A — no public key | Yes | not implemented (Phase 2) |
| subset_proof_bytes | N/A | N/A | 87360 bytes | not implemented (Phase 2) |
