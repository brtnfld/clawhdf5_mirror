//! Regression tests for the 2026-10 red-team findings in the signing lane.
//! Attacker model: file read/write, no signing key, public keys held by the
//! verifier out-of-band.

use clawhdf5_format::attribute::extract_attributes;
use clawhdf5_format::file_writer::FileWriter;
use clawhdf5_format::group_v2::resolve_path_any;
use clawhdf5_format::merkle::{HashAlg, LeafFormat, MerkleAttr, MerkleTree};
use clawhdf5_format::object_header::ObjectHeader;
use clawhdf5_format::signature::find_signature;
use clawhdf5_format::superblock::Superblock;
use clawhdf5_sign::mldsa::MlDsaSigningKey;
use clawhdf5_sign::{
    HybridSignature, SIG_ATTR_NAME, SIG_ATTR_SIZE, SignError, SignedRootMeta, SigningKey,
    VerifyingKey, canonical_payload, read_sig_attr, sign_root, verify_sig_attr, write_sig_attr,
};

/// A small-order (weak) Ed25519 public key validates a signature over almost
/// any message. The verifier's keys are trusted out-of-band, so this is a
/// configuration error rather than an attack, but refusing it at
/// construction is free and `verify_strict` refuses it at verification too.
#[test]
fn weak_ed25519_public_key_is_rejected() {
    // Compressed encoding of the identity point: y = 1, x sign bit 0.
    let mut identity = [0u8; 32];
    identity[0] = 1;
    assert!(matches!(VerifyingKey::from_bytes(&identity), Err(SignError::WeakKey)));
    // An honest key is unaffected.
    let honest = SigningKey::generate().verifying_key().to_bytes();
    assert!(VerifyingKey::from_bytes(&honest).is_ok());
}

fn attrs_of(file_bytes: &[u8], dataset: &str) -> Vec<clawhdf5_format::attribute::AttributeMessage> {
    let sig_offset = find_signature(file_bytes).unwrap();
    let sb = Superblock::parse(file_bytes, sig_offset).unwrap();
    let addr = resolve_path_any(file_bytes, &sb, dataset).unwrap();
    let hdr = ObjectHeader::parse(file_bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
    extract_attributes(&hdr, sb.length_size).unwrap()
}

fn signed_file() -> (Vec<u8>, MerkleAttr, SignedRootMeta, HybridSignature, VerifyingKey, clawhdf5_sign::mldsa::MlDsaVerifyingKey) {
    let chunks: Vec<&[u8]> = vec![b"chunk-a", b"chunk-b", b"chunk-c"];
    let tree = MerkleTree::from_chunks(&chunks, HashAlg::Blake3);
    let attr = MerkleAttr::from_tree(&tree);
    let meta = SignedRootMeta { alg_id: attr.algorithm, version: 7, timestamp: 1_700_000_000 };
    let ed = SigningKey::generate();
    let ml = MlDsaSigningKey::generate();
    let payload = canonical_payload(
        &attr.root,
        &attr.companion_hash,
        meta.version,
        meta.timestamp,
        attr.algorithm,
        attr.leaf_format,
        &attr.grid_hash,
    );
    let sig = sign_root(&payload, &ed, &ml).unwrap();
    let mut fw = FileWriter::new();
    let ds = fw.create_dataset("data");
    ds.with_u8_data(&[1, 2, 3]);
    ds.set_attr(
        clawhdf5_format::merkle::MERKLE_ATTR_NAME,
        clawhdf5_format::type_builders::AttrValue::Bytes(attr.pack().to_vec()),
    );
    write_sig_attr(ds, &meta, &sig);
    let bytes = fw.finish().unwrap();
    (bytes, attr, meta, sig, ed.verifying_key(), ml.verifying_key())
}

/// A stored signature must be verifiable from the file alone: `merkle_root`
/// plus `merkle_sig` plus the trusted keys, with no version or timestamp
/// retained from write time. Before the v1 attribute layout nothing in the
/// file carried the timestamp, so "verify the file" was not possible.
#[test]
fn stored_signature_verifies_from_file_alone() {
    let (bytes, attr, meta, sig, ed_pub, ml_pub) = signed_file();
    let attrs = attrs_of(&bytes, "data");
    // Everything comes back out of the file.
    let root_attr = clawhdf5_format::attribute::find_attribute(&attrs, clawhdf5_format::merkle::MERKLE_ATTR_NAME).unwrap();
    let attr_from_file = MerkleAttr::unpack(&root_attr.raw_data).unwrap();
    assert_eq!(attr_from_file, attr);
    let (meta_from_file, sig_from_file) = read_sig_attr(&attrs).unwrap();
    assert_eq!(meta_from_file, meta);
    assert_eq!(sig_from_file, sig);
    verify_sig_attr(&attr_from_file, &meta_from_file, &sig_from_file, &ed_pub, &ml_pub)
        .expect("file-only verification succeeds");

    // And the fields the attribute carries are genuinely bound: bump the
    // stored version or timestamp and the signature no longer verifies.
    let mut bumped = meta_from_file;
    bumped.version += 1;
    assert!(matches!(
        verify_sig_attr(&attr_from_file, &bumped, &sig_from_file, &ed_pub, &ml_pub),
        Err(SignError::InvalidSignature)
    ));
    let mut bumped = meta_from_file;
    bumped.timestamp += 1;
    assert!(matches!(
        verify_sig_attr(&attr_from_file, &bumped, &sig_from_file, &ed_pub, &ml_pub),
        Err(SignError::InvalidSignature)
    ));
}

/// The algorithm byte in `merkle_sig` is informational and must agree with
/// `merkle_root`; a disagreement is reported as such rather than silently
/// resolved by trusting either copy.
#[test]
fn sig_attr_algorithm_mismatch_is_reported() {
    let (_, attr, meta, sig, ed_pub, ml_pub) = signed_file();
    let wrong = SignedRootMeta { alg_id: HashAlg::Sha256, ..meta };
    assert!(matches!(
        verify_sig_attr(&attr, &wrong, &sig, &ed_pub, &ml_pub),
        Err(SignError::AlgorithmMismatch { attribute: 0x01, signature: 0x00 })
    ));
}

/// A bare pre-v1 `merkle_sig` (signature bytes only) is rejected, not read
/// as a signature with unknown version/timestamp.
#[test]
fn bare_pre_v1_sig_attr_is_rejected() {
    let ed = SigningKey::generate();
    let ml = MlDsaSigningKey::generate();
    let payload = canonical_payload(&[0x11; 32], &[0x22; 32], 1, 1, HashAlg::Blake3, LeafFormat::Content, &[0u8; 32]);
    let sig = sign_root(&payload, &ed, &ml).unwrap();
    let mut fw = FileWriter::new();
    let ds = fw.create_dataset("data");
    ds.with_u8_data(&[1]);
    ds.set_attr(SIG_ATTR_NAME, clawhdf5_format::type_builders::AttrValue::Bytes(sig.to_bytes().to_vec()));
    let bytes = fw.finish().unwrap();
    let attrs = attrs_of(&bytes, "data");
    assert_eq!(sig.to_bytes().len(), SIG_ATTR_SIZE - SignedRootMeta::SERIALIZED_SIZE);
    assert!(matches!(read_sig_attr(&attrs), Err(SignError::InvalidSignature)));
}
