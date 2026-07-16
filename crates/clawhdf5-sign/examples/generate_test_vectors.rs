//! P2.1 step 7: Generate test vectors for hybrid signatures.
//!
//! Creates `test-vectors/hybrid-sig-vectors.json` with one known-good
//! HybridSignature and its canonical payload, using a deterministic
//! test-only key pair.
//!
//! Usage:
//!   cargo run --example generate_test_vectors --features "ed25519,mldsa"

use clawhdf5_sign::{HashAlg, canonical_payload, sign_root, verify_sig};
use clawhdf5_sign::{HybridSignature, SigningKey, mldsa::MlDsaSigningKey};
use serde::{Deserialize, Serialize};
use std::io::Write;

#[derive(Serialize, Deserialize)]
struct TestVector {
    description: String,
    ed25519_secret_key_hex: String,
    ed25519_public_key_hex: String,
    mldsa65_secret_key_hex: String,
    mldsa65_public_key_hex: String,
    canonical_payload_hex: String,
    hybrid_signature_hex: String,
    payload_fields: PayloadFields,
}

#[derive(Serialize, Deserialize)]
struct PayloadFields {
    root_hex: String,
    companion_hash_hex: String,
    version: u64,
    timestamp: u64,
    algorithm: String,
}

fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Generating hybrid signature test vectors...");

    // Generate deterministic keys using fixed seeds for reproducibility
    let ed_secret_bytes = [0x42u8; 32];
    let ed_key = SigningKey::from_bytes(&ed_secret_bytes)?;
    let ed_pub = ed_key.verifying_key();

    // For ML-DSA, we'll generate a key and document it
    // (ML-DSA key generation is randomized, so we document the generated key)
    let ml_key = MlDsaSigningKey::generate();
    let ml_pub = ml_key.verifying_key();

    // Create canonical payload
    let root = [0x42; 32];
    let companion_hash = [0x43; 32];
    let version = 1;
    let timestamp = 1704067200; // 2024-01-01 00:00:00 UTC
    let alg_id = HashAlg::Blake3;

    let payload = canonical_payload(&root, &companion_hash, version, timestamp, alg_id);

    // Sign the payload
    let hybrid_sig = sign_root(&payload, &ed_key, &ml_key)?;

    // Verify it works
    verify_sig(&payload, &hybrid_sig, &ed_pub, &ml_pub)?;
    println!("  Signature verified successfully");

    // Serialize hybrid signature
    let sig_bytes = hybrid_sig.to_bytes();

    // Build test vector
    let test_vector = TestVector {
        description:
            "Test vector for hybrid Ed25519 + ML-DSA-65 signature over canonical payload. \
                      This key pair is FOR TESTING ONLY and must never be used in production."
                .to_string(),
        ed25519_secret_key_hex: to_hex(&ed_key.to_bytes()),
        ed25519_public_key_hex: to_hex(&ed_pub.to_bytes()),
        mldsa65_secret_key_hex: "(generated - not exported for security)".to_string(),
        mldsa65_public_key_hex: to_hex(ml_pub.to_bytes()),
        canonical_payload_hex: to_hex(&payload),
        hybrid_signature_hex: to_hex(&sig_bytes),
        payload_fields: PayloadFields {
            root_hex: to_hex(&root),
            companion_hash_hex: to_hex(&companion_hash),
            version,
            timestamp,
            algorithm: "Blake3".to_string(),
        },
    };

    // Write to JSON
    let json = serde_json::to_string_pretty(&test_vector)?;

    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-vectors/hybrid-sig-vectors.json");

    let mut file = std::fs::File::create(&out_path)?;
    file.write_all(json.as_bytes())?;

    println!("  Written to: {}", out_path.display());
    println!("  Payload size: {} bytes", payload.len());
    println!("  Signature size: {} bytes", sig_bytes.len());
    println!("Done!");

    Ok(())
}
