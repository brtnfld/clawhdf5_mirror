//! Cross-backend ML-DSA-65 interoperability (2026-10 red-team lane:
//! signatures). The crate ships two ML-DSA-65 backends behind separate
//! features: `mldsa` (RustCrypto `ml-dsa`) and `pq-mldsa` (PQClean via
//! `pqcrypto-mldsa`). A file signed under one and verified under the other
//! must agree, or a deployment that mixes them (for instance a Rust writer
//! and a C reader) silently rejects every honest file. Both directions are
//! pinned here; both use the FIPS 204 pure variant with an empty context.
#![cfg(all(feature = "mldsa", feature = "pq-mldsa"))]

use clawhdf5_sign::mldsa::{MlDsaSigningKey, MlDsaVerifyingKey};
use clawhdf5_sign::pq_mldsa::{PqMlDsaSigningKey, PqMlDsaVerifyingKey};

#[test]
fn rustcrypto_signs_pqclean_verifies() {
    let k = MlDsaSigningKey::generate();
    let root = [0x5au8; 32];
    let sig = k.sign(&root).unwrap();
    let pk = PqMlDsaVerifyingKey::from_bytes(k.verifying_key().to_bytes()).unwrap();
    pk.verify(&root, &sig)
        .expect("PQClean must accept a RustCrypto ML-DSA-65 signature");
    let wrong = [0x5bu8; 32];
    assert!(pk.verify(&wrong, &sig).is_err());
}

#[test]
fn pqclean_signs_rustcrypto_verifies() {
    let k = PqMlDsaSigningKey::generate();
    let root = [0xa5u8; 32];
    let sig = k.sign(&root).unwrap();
    let pk = MlDsaVerifyingKey::from_bytes(k.verifying_key().to_bytes()).unwrap();
    pk.verify(&root, &sig)
        .expect("RustCrypto must accept a PQClean ML-DSA-65 signature");
    let wrong = [0xa6u8; 32];
    assert!(pk.verify(&wrong, &sig).is_err());
}
