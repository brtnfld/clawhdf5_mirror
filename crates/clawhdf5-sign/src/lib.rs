//! Cryptographic signing and verification for HDF5 files.
//!
//! This crate provides APIs for signing HDF5 file Merkle roots and verifying
//! signatures, enabling end-to-end data integrity from producer to consumer.
//!
//! # Overview
//!
//! The signing workflow:
//! 1. Build a Merkle tree over the HDF5 file's chunks
//! 2. Sign the Merkle root with a private key
//! 3. Attach the signature to the file (or distribute separately)
//!
//! The verification workflow:
//! 1. Rebuild the Merkle tree from the file
//! 2. Verify the signature over the root using the public key
//! 3. For subset requests, verify individual chunk paths to the signed root
//!
//! # Example
//!
//! ```ignore
//! use clawhdf5_sign::{SigningKey, VerifyingKey, sign_root, verify_root};
//!
//! // Producer: sign the Merkle root
//! let signing_key = SigningKey::generate();
//! let signature = sign_root(&signing_key, &merkle_root);
//!
//! // Consumer: verify the signature
//! let verifying_key = signing_key.verifying_key();
//! assert!(verify_root(&verifying_key, &merkle_root, &signature).is_ok());
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::string::String;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// Re-export HashAlg and LeafFormat from clawhdf5-format for canonical payload
// encoding: both are algorithm identifiers the signed payload must bind (T6).
pub use clawhdf5_format::merkle::{HashAlg, LeafFormat};

/// Hash size in bytes (Blake3/SHA-256).
pub const HASH_SIZE: usize = 32;

/// Size in bytes of the canonical signed payload produced by
/// [`canonical_payload`] (payload format v1).
pub const CANONICAL_PAYLOAD_SIZE: usize = 114;

/// Errors that can occur during signing or verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignError {
    /// The signature is invalid or does not match the data.
    InvalidSignature,

    /// The provided key is malformed.
    InvalidKey(String),

    /// The Merkle root hash is missing or malformed.
    InvalidRoot,

    /// Internal signing error (e.g., cryptographic library failure).
    SigningFailed,

    /// The `merkle_sig` attribute is absent from the dataset.
    AttributeMissing,

    /// The `merkle_sig` attribute's algorithm byte disagrees with the
    /// `merkle_root` attribute's. The signed payload binds the attribute's
    /// algorithm, so this is a tampered or mis-written file, never something
    /// to resolve by trusting one of the two.
    AlgorithmMismatch {
        /// Algorithm id carried by `merkle_root`.
        attribute: u8,
        /// Algorithm id carried by `merkle_sig`.
        signature: u8,
    },

    /// An Ed25519 public key of small order was offered as a verifying key.
    /// Such a key validates a signature over almost any message, so it is
    /// refused at construction rather than only at verification time.
    WeakKey,
}

impl core::fmt::Display for SignError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SignError::InvalidSignature => write!(f, "signature verification failed"),
            SignError::InvalidKey(msg) => write!(f, "invalid key format: {msg}"),
            SignError::InvalidRoot => write!(f, "missing or invalid Merkle root"),
            SignError::SigningFailed => write!(f, "signing operation failed"),
            SignError::AttributeMissing => write!(f, "merkle_sig attribute absent from dataset"),
            SignError::AlgorithmMismatch { attribute, signature } => write!(
                f,
                "merkle_sig algorithm id {signature:#04x} disagrees with merkle_root's {attribute:#04x}"
            ),
            SignError::WeakKey => write!(f, "Ed25519 public key has small order (weak key)"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SignError {}

/// A Merkle root hash that can be signed.
pub type MerkleRoot = [u8; HASH_SIZE];

/// Signature size in bytes (Ed25519).
pub const SIGNATURE_SIZE: usize = 64;

/// ML-DSA-65 signature size in bytes (NIST FIPS 204).
pub const MLDSA65_SIGNATURE_SIZE: usize = 3309;

/// A detached signature over a Merkle root.
pub type Signature = [u8; SIGNATURE_SIZE];

// ─────────────────────────────────────────────────────────────────────────────
// Canonical Payload Encoder (P2.1 step 2)
// ─────────────────────────────────────────────────────────────────────────────

/// Encode the canonical signed payload for hybrid post-quantum signatures.
///
/// The payload binds together every field of the `merkle_root` attribute that
/// gives the root its meaning — the Merkle root, companion integrity hash,
/// hash-algorithm identifier, leaf-preimage format, and chunk-grid hash —
/// plus the dataset version counter and timestamp. This canonical encoding
/// prevents algorithm-substitution attacks (T6) and ensures all critical
/// metadata is authenticated by both Ed25519 and ML-DSA-65 signatures.
///
/// # Format (payload v1)
///
/// The encoding is a big-endian fixed-width binary format with no separators:
///
/// | Field           | Offset | Size (bytes) | Encoding     | Description                     |
/// |-----------------|--------|--------------|--------------|---------------------------------|
/// | root            | 0      | 32           | raw bytes    | Merkle root hash                |
/// | companion_hash  | 32     | 32           | raw bytes    | Companion integrity hash        |
/// | version         | 64     | 8            | big-endian   | Dataset version counter         |
/// | timestamp       | 72     | 8            | big-endian   | Unix seconds                    |
/// | alg_id          | 80     | 1            | raw byte     | Hash algorithm identifier       |
/// | leaf_format     | 81     | 1            | raw byte     | Leaf-preimage format identifier |
/// | grid_hash       | 82     | 32           | raw bytes    | Chunk-grid binding hash         |
///
/// **Total: 114 bytes** ([`CANONICAL_PAYLOAD_SIZE`]).
///
/// # Why `leaf_format` and `grid_hash` are signed (2026-10 red-team finding)
///
/// Payload v0 (81 bytes) stopped after `alg_id`. Both omitted fields live in
/// the `merkle_root` attribute, where they are protected only by an *unkeyed*
/// integrity hash (`leaf_format`) or by nothing at all (`grid_hash`), so a
/// storage-level attacker without the signing key could rewrite them while the
/// v0 signature still verified:
///
/// - Downgrading `LeafFormat::Bound` to `Content` and re-framing each stored
///   chunk as the bound preimage suffix (`be32(8) || le64(k) || be32(len) ||
///   data || le64(v)`) leaves every leaf hash — and the signed root — unchanged
///   while every verifier accepts bytes the signer never committed to.
/// - Rewriting the declared dims/chunk shape and recomputing `grid_hash` with
///   its public preimage leaves the root unchanged while `verify_subset`
///   delivers a different chunk for the same requested hyperslab.
///
/// Signatures over the v0 layout are rejected outright: the
/// [`HybridSignature`] format version was bumped alongside this change, so a
/// stored v0 signature fails `HybridSignature::from_bytes` rather than being
/// re-verified against a payload it never bound. Files signed under v0 must be
/// re-signed.
///
/// # Arguments
///
/// * `root` - The 32-byte Merkle root hash
/// * `companion_hash` - The 32-byte companion dataset integrity hash
/// * `version` - Dataset-level version counter (increments on each update)
/// * `timestamp` - Unix timestamp in seconds when the signature was created
/// * `alg_id` - Hash algorithm used (from `HashAlg::to_id()`)
/// * `leaf_format` - Leaf-preimage format the tree was built with
///   (`MerkleAttr::leaf_format`)
/// * `grid_hash` - The attribute's 32-byte chunk-grid hash
///   (`MerkleAttr::grid_hash`; all-zero when unbound — that sentinel is signed
///   too, so an attacker cannot later *add* a grid binding)
///
/// # Returns
///
/// A fixed 114-byte array containing the canonical payload ready for signing.
///
/// # Example
///
/// ```ignore
/// use clawhdf5_sign::{canonical_payload, HashAlg};
///
/// let root = [0x42; 32];
/// let companion_hash = [0x43; 32];
/// let version = 1;
/// let timestamp = 1704067200; // 2024-01-01 00:00:00 UTC
/// let alg_id = HashAlg::Blake3;
///
/// let payload = canonical_payload(
///     &root, &companion_hash, version, timestamp, alg_id,
///     LeafFormat::Content, &[0u8; 32],
/// );
/// assert_eq!(payload.len(), CANONICAL_PAYLOAD_SIZE);
/// ```
///
/// # Spec Reference
///
/// S2-D2-Yr2 §"Post-quantum hybrid signing" (§sec:pq-hybrid), equation:
///
/// ```text
/// R = root || companion_hash || v_dataset || τ || AlgID
/// ```
///
/// where `||` denotes concatenation and all multi-byte integers are big-endian.
/// `AlgID` is realised here as the pair `(alg_id, leaf_format)` followed by
/// `grid_hash`, so that *every* identifier and parameter the verifier reads
/// out of the attribute is under the signature, as the design's threat T6
/// requires.
#[must_use]
pub fn canonical_payload(
    root: &[u8; 32],
    companion_hash: &[u8; 32],
    version: u64,
    timestamp: u64,
    alg_id: HashAlg,
    leaf_format: LeafFormat,
    grid_hash: &[u8; 32],
) -> [u8; CANONICAL_PAYLOAD_SIZE] {
    let mut payload = [0u8; CANONICAL_PAYLOAD_SIZE];
    let mut offset = 0;

    // 32 bytes: Merkle root
    payload[offset..offset + 32].copy_from_slice(root);
    offset += 32;

    // 32 bytes: companion integrity hash
    payload[offset..offset + 32].copy_from_slice(companion_hash);
    offset += 32;

    // 8 bytes: version counter (big-endian)
    payload[offset..offset + 8].copy_from_slice(&version.to_be_bytes());
    offset += 8;

    // 8 bytes: timestamp (big-endian)
    payload[offset..offset + 8].copy_from_slice(&timestamp.to_be_bytes());
    offset += 8;

    // 1 byte: algorithm identifier
    payload[offset] = alg_id.to_id();
    offset += 1;

    // 1 byte: leaf-preimage format identifier
    payload[offset] = leaf_format.to_id();
    offset += 1;

    // 32 bytes: chunk-grid binding hash (all-zero sentinel when unbound)
    payload[offset..offset + 32].copy_from_slice(grid_hash);

    payload
}

// ─────────────────────────────────────────────────────────────────────────────
// Hybrid Signature (P2.1 step 3)
// ─────────────────────────────────────────────────────────────────────────────

/// Hybrid post-quantum signature combining Ed25519 and ML-DSA-65.
///
/// This signature provides defense against harvest-now, forge-later attacks
/// by combining a classical signature (Ed25519) with a post-quantum signature
/// (ML-DSA-65). Both signatures are computed over the same canonical payload.
///
/// # Format
///
/// The serialized format is a fixed-width binary encoding:
///
/// | Field           | Size (bytes) | Description                          |
/// |-----------------|--------------|--------------------------------------|
/// | version         | 1            | Signature format version (currently 0)|
/// | ed25519_sig     | 64           | Ed25519 signature                    |
/// | mldsa65_sig     | 3309         | ML-DSA-65 signature                  |
///
/// **Total: 3374 bytes**
///
/// # Verification Policy
///
/// A verifier MUST accept the signature if and only if **both** components
/// verify successfully (strict-AND policy). This prevents downgrade attacks
/// where an adversary strips the post-quantum component.
///
/// # Spec Reference
///
/// S2-D2-Yr2 §"Post-quantum hybrid signing" (§sec:pq-hybrid), equation:
///
/// ```text
/// σ_root = (σ_Ed25519(R), σ_ML-DSA-65(R))
/// ```
///
/// where R is the canonical payload from [`canonical_payload`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridSignature {
    /// Signature format version (currently 0).
    pub version: u8,
    /// Ed25519 signature (64 bytes).
    pub ed25519_sig: [u8; SIGNATURE_SIZE],
    /// ML-DSA-65 signature (3309 bytes).
    pub mldsa65_sig: [u8; MLDSA65_SIGNATURE_SIZE],
}

impl HybridSignature {
    /// The current signature format version.
    ///
    /// Version 0 signed the 81-byte payload v0 (`root || companion_hash ||
    /// version || timestamp || alg_id`), which left `leaf_format` and
    /// `grid_hash` unsigned (see [`canonical_payload`]). Version 1 signs the
    /// 114-byte payload v1. The version byte is checked in
    /// [`from_bytes`](Self::from_bytes), so a stored v0 signature is rejected
    /// rather than verified against a payload it never covered — the only
    /// fail-closed option, since accepting it would re-open the downgrade.
    pub const VERSION: u8 = 1;

    /// Total serialized size: 1 (version) + 64 (Ed25519) + 3309 (ML-DSA-65).
    pub const SERIALIZED_SIZE: usize = 1 + SIGNATURE_SIZE + MLDSA65_SIGNATURE_SIZE;

    /// Create a new hybrid signature from Ed25519 and ML-DSA-65 components.
    #[must_use]
    pub fn new(
        ed25519_sig: [u8; SIGNATURE_SIZE],
        mldsa65_sig: [u8; MLDSA65_SIGNATURE_SIZE],
    ) -> Self {
        Self {
            version: Self::VERSION,
            ed25519_sig,
            mldsa65_sig,
        }
    }

    /// Serialize the hybrid signature to bytes.
    ///
    /// Format: `[version(1)] || [ed25519_sig(64)] || [mldsa65_sig(3309)]`
    #[must_use]
    pub fn to_bytes(&self) -> [u8; Self::SERIALIZED_SIZE] {
        let mut bytes = [0u8; Self::SERIALIZED_SIZE];
        let mut offset = 0;

        // 1 byte: version
        bytes[offset] = self.version;
        offset += 1;

        // 64 bytes: Ed25519 signature
        bytes[offset..offset + SIGNATURE_SIZE].copy_from_slice(&self.ed25519_sig);
        offset += SIGNATURE_SIZE;

        // 3309 bytes: ML-DSA-65 signature
        bytes[offset..offset + MLDSA65_SIGNATURE_SIZE].copy_from_slice(&self.mldsa65_sig);

        bytes
    }

    /// Deserialize a hybrid signature from bytes.
    ///
    /// # Errors
    ///
    /// Returns `SignError::InvalidSignature` if:
    /// - The data length is not exactly SERIALIZED_SIZE bytes
    /// - The version byte is not recognized
    pub fn from_bytes(data: &[u8]) -> Result<Self, SignError> {
        // Require exact length (no trailing bytes allowed)
        if data.len() != Self::SERIALIZED_SIZE {
            return Err(SignError::InvalidSignature);
        }

        let version = data[0];
        if version != Self::VERSION {
            return Err(SignError::InvalidSignature);
        }

        let ed25519_sig: [u8; SIGNATURE_SIZE] = data[1..1 + SIGNATURE_SIZE]
            .try_into()
            .map_err(|_| SignError::InvalidSignature)?;

        let mldsa65_sig: [u8; MLDSA65_SIGNATURE_SIZE] = data
            [1 + SIGNATURE_SIZE..1 + SIGNATURE_SIZE + MLDSA65_SIGNATURE_SIZE]
            .try_into()
            .map_err(|_| SignError::InvalidSignature)?;

        Ok(Self {
            version,
            ed25519_sig,
            mldsa65_sig,
        })
    }
}

/// Sign a canonical payload with hybrid Ed25519 + ML-DSA-65 signatures.
///
/// This function produces both signatures independently over the same payload
/// and combines them into a [`HybridSignature`]. Both signatures MUST verify
/// for the payload to be accepted.
///
/// # Arguments
///
/// * `payload` - The canonical payload from [`canonical_payload`]
/// * `ed_key` - Ed25519 signing key
/// * `ml_key` - ML-DSA-65 signing key
///
/// # Returns
///
/// A [`HybridSignature`] containing both signature components.
///
/// # Example
///
/// ```ignore
/// use clawhdf5_sign::{canonical_payload, sign_root, HashAlg};
/// use clawhdf5_sign::{SigningKey, mldsa::MlDsaSigningKey};
///
/// let ed_key = SigningKey::generate();
/// let ml_key = MlDsaSigningKey::generate();
///
/// let root = [0x42; 32];
/// let companion_hash = [0x43; 32];
/// let payload = canonical_payload(&root, &companion_hash, 1, 1704067200, HashAlg::Blake3, LeafFormat::Content, &[0u8; 32]);
///
/// let hybrid_sig = sign_root(&payload, &ed_key, &ml_key).unwrap();
/// ```
///
/// # Spec Reference
///
/// S2-D2-Yr2 P2.1 step 3: "Produce both signatures independently over the same
/// payload and store them concatenated with a 1-byte version tag."
///
/// # Errors
///
/// Returns `SignError::SigningFailed` if the ML-DSA signing operation fails
/// (should not happen under normal circumstances).
#[cfg(all(feature = "ed25519", feature = "mldsa"))]
pub fn sign_root(
    payload: &[u8],
    ed_key: &SigningKey,
    ml_key: &mldsa::MlDsaSigningKey,
) -> Result<HybridSignature, SignError> {
    let ed25519_sig = ed_key.sign_payload(payload);
    let mldsa65_sig = ml_key.sign_payload(payload)?;

    Ok(HybridSignature::new(ed25519_sig, mldsa65_sig))
}

/// Verify a hybrid Ed25519 + ML-DSA-65 signature over a canonical payload.
///
/// This function implements the **strict-AND policy**: the signature is accepted
/// if and only if **both** the Ed25519 and ML-DSA-65 components verify successfully.
/// If either signature fails, the entire verification fails.
///
/// # Strict-AND Policy Rationale
///
/// This prevents downgrade attacks where an adversary strips the post-quantum
/// component. Even before a CRQC exists, both signatures must be present and
/// valid. Once a CRQC exists, the ML-DSA-65 component preserves authenticity
/// while the Ed25519 component is considered compromised.
///
/// # Arguments
///
/// * `payload` - The canonical payload from [`canonical_payload`]
/// * `sig` - The hybrid signature containing both Ed25519 and ML-DSA-65 components
/// * `ed_pub` - Ed25519 public key (verifying key)
/// * `ml_pub` - ML-DSA-65 public key (verifying key)
///
/// # Returns
///
/// * `Ok(())` - Both signatures are valid
/// * `Err(SignError::InvalidSignature)` - At least one signature is invalid
///
/// # Example
///
/// ```ignore
/// use clawhdf5_sign::{canonical_payload, sign_root, verify_sig, HashAlg};
/// use clawhdf5_sign::{SigningKey, mldsa::MlDsaSigningKey};
///
/// // Signer: generate keys and sign
/// let ed_key = SigningKey::generate();
/// let ml_key = MlDsaSigningKey::generate();
/// let payload = canonical_payload(&[0x42; 32], &[0x43; 32], 1, 1704067200, HashAlg::Blake3, LeafFormat::Content, &[0u8; 32]);
/// let hybrid_sig = sign_root(&payload, &ed_key, &ml_key).unwrap();
///
/// // Verifier: verify with public keys
/// let ed_pub = ed_key.verifying_key();
/// let ml_pub = ml_key.verifying_key();
/// assert!(verify_sig(&payload, &hybrid_sig, &ed_pub, &ml_pub).is_ok());
/// ```
///
/// # Spec Reference
///
/// S2-D2-Yr2 P2.1 step 4: "Require both signatures to be valid; accept the
/// payload only if neither fails."
#[cfg(all(feature = "ed25519", feature = "mldsa"))]
pub fn verify_sig(
    payload: &[u8],
    sig: &HybridSignature,
    ed_pub: &VerifyingKey,
    ml_pub: &mldsa::MlDsaVerifyingKey,
) -> Result<(), SignError> {
    // Verify Ed25519 component first (faster, fail early)
    ed_pub.verify_payload(payload, &sig.ed25519_sig)?;

    // Verify ML-DSA-65 component (slower, but both must pass)
    ml_pub.verify_payload(payload, &sig.mldsa65_sig)?;

    // Both signatures are valid
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// HDF5 Attribute Storage (P2.1 step 3)
// ─────────────────────────────────────────────────────────────────────────────

/// Name of the HDF5 attribute storing the hybrid signature.
///
/// Mirrors the `merkle_root` attribute in `clawhdf5_format::merkle`, which is
/// likewise named without its spec-document leading underscore.
pub const SIG_ATTR_NAME: &str = "merkle_sig";

/// Size of the packed `merkle_sig` attribute (attribute format v1):
///
/// ```text
/// alg_id(1) || version(8, BE) || timestamp(8, BE) || HybridSignature(3374)
/// ```
///
/// = 3391 bytes, the "primary path" layout of the C integration spec §1.5.
pub const SIG_ATTR_SIZE: usize = 1 + 8 + 8 + HybridSignature::SERIALIZED_SIZE;

/// The signed-payload fields that live in neither the `merkle_root` attribute
/// nor the signature itself, carried in `merkle_sig` so that a stored
/// signature can be verified from the file alone.
///
/// Before this layout existed (2026-10 red-team finding), `merkle_sig` held
/// the bare signature bytes and nothing in the file carried the `timestamp`
/// (or, for a dataset without a `merkle_version` attribute, the `version`)
/// that [`canonical_payload`] binds. A verifier therefore could not rebuild
/// the payload from disk: every caller that "verified a file" was actually
/// re-using the version and timestamp it still held in memory from writing
/// it, which is not a verification of the file at all.
///
/// `alg_id` duplicates `MerkleAttr::algorithm`. It is informational — the
/// spec puts it first in the layout for the C port's benefit — and
/// [`verify_sig_attr`] requires it to match the attribute's value. The signed
/// payload binds the attribute's algorithm, so a disagreement is a tampered
/// or mis-written file, never something to resolve by trusting either copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignedRootMeta {
    /// Hash algorithm the root was computed with (equals `MerkleAttr::algorithm`).
    pub alg_id: HashAlg,
    /// Dataset-level version counter bound into the signature.
    pub version: u64,
    /// Unix timestamp (seconds) bound into the signature.
    pub timestamp: u64,
}

impl SignedRootMeta {
    /// Serialized size of the metadata prefix (`alg_id || version || timestamp`).
    pub const SERIALIZED_SIZE: usize = 1 + 8 + 8;

    fn pack(&self) -> [u8; Self::SERIALIZED_SIZE] {
        let mut out = [0u8; Self::SERIALIZED_SIZE];
        out[0] = self.alg_id.to_id();
        out[1..9].copy_from_slice(&self.version.to_be_bytes());
        out[9..17].copy_from_slice(&self.timestamp.to_be_bytes());
        out
    }

    fn unpack(data: &[u8; Self::SERIALIZED_SIZE]) -> Result<Self, SignError> {
        let alg_id = HashAlg::from_id(data[0]).ok_or(SignError::InvalidSignature)?;
        let version = u64::from_be_bytes(data[1..9].try_into().map_err(|_| SignError::InvalidSignature)?);
        let timestamp =
            u64::from_be_bytes(data[9..17].try_into().map_err(|_| SignError::InvalidSignature)?);
        Ok(Self {
            alg_id,
            version,
            timestamp,
        })
    }
}

/// Write a [`HybridSignature`] and its [`SignedRootMeta`] to a dataset as the
/// `merkle_sig` attribute (layout: [`SIG_ATTR_SIZE`]).
///
/// # Example
///
/// ```ignore
/// use clawhdf5_sign::{canonical_payload, sign_root, write_sig_attr, SignedRootMeta};
/// use clawhdf5_format::file_writer::FileWriter;
///
/// let meta = SignedRootMeta { alg_id: attr.algorithm, version: 1, timestamp };
/// let payload = canonical_payload(
///     &attr.root, &attr.companion_hash, meta.version, meta.timestamp,
///     attr.algorithm, attr.leaf_format, &attr.grid_hash,
/// );
/// let sig = sign_root(&payload, &ed_key, &ml_key).unwrap();
///
/// let mut fw = FileWriter::new();
/// let ds = fw.create_dataset("data");
/// ds.with_u8_data(&data);
/// write_sig_attr(ds, &meta, &sig);
/// ```
#[cfg(feature = "std")]
pub fn write_sig_attr(
    dataset: &mut clawhdf5_format::type_builders::DatasetBuilder,
    meta: &SignedRootMeta,
    sig: &HybridSignature,
) {
    let mut bytes = Vec::with_capacity(SIG_ATTR_SIZE);
    bytes.extend_from_slice(&meta.pack());
    bytes.extend_from_slice(&sig.to_bytes());
    debug_assert_eq!(bytes.len(), SIG_ATTR_SIZE);
    dataset.set_attr(
        SIG_ATTR_NAME,
        clawhdf5_format::type_builders::AttrValue::Bytes(bytes),
    );
}

/// Read the [`SignedRootMeta`] and [`HybridSignature`] back from a dataset's
/// parsed attributes.
///
/// A bare 3374-byte attribute written before attribute format v1 (signature
/// bytes only, no `alg_id`/`version`/`timestamp` prefix) is rejected with
/// [`SignError::InvalidSignature`]: it could never be verified from the file
/// alone, and the signature format it carried (v0) is rejected by
/// [`HybridSignature::from_bytes`] in any case.
///
/// # Errors
///
/// Returns [`SignError::AttributeMissing`] if no `merkle_sig` attribute is
/// present, or [`SignError::InvalidSignature`] if the attribute is present but
/// malformed (wrong length, unknown algorithm id, or unrecognized signature
/// version tag).
#[cfg(feature = "std")]
pub fn read_sig_attr(
    attrs: &[clawhdf5_format::attribute::AttributeMessage],
) -> Result<(SignedRootMeta, HybridSignature), SignError> {
    let attr = clawhdf5_format::attribute::find_attribute(attrs, SIG_ATTR_NAME)
        .ok_or(SignError::AttributeMissing)?;
    let data = &attr.raw_data;
    if data.len() != SIG_ATTR_SIZE {
        return Err(SignError::InvalidSignature);
    }
    let (meta_bytes, sig_bytes) = data.split_at(SignedRootMeta::SERIALIZED_SIZE);
    let meta_bytes: &[u8; SignedRootMeta::SERIALIZED_SIZE] =
        meta_bytes.try_into().map_err(|_| SignError::InvalidSignature)?;
    let meta = SignedRootMeta::unpack(meta_bytes)?;
    let sig = HybridSignature::from_bytes(sig_bytes)?;
    Ok((meta, sig))
}

/// Verify a stored `merkle_sig` against its dataset's `merkle_root` attribute,
/// using nothing but what was read from the file plus the trusted public keys.
///
/// Rebuilds the v1 canonical payload from `attr` (root, companion hash,
/// algorithm, leaf format, grid hash) and `meta` (version, timestamp), after
/// requiring `meta.alg_id` to equal `attr.algorithm`, and applies the
/// strict-AND hybrid check. This is the single entry point a reader should
/// use to answer "is the file I just opened the one the key-holder signed?".
///
/// # Errors
///
/// - [`SignError::AlgorithmMismatch`] if the two attributes disagree on the
///   hash algorithm.
/// - [`SignError::InvalidSignature`] if either signature component fails.
#[cfg(all(feature = "ed25519", feature = "mldsa"))]
pub fn verify_sig_attr(
    attr: &clawhdf5_format::merkle::MerkleAttr,
    meta: &SignedRootMeta,
    sig: &HybridSignature,
    ed_pub: &VerifyingKey,
    ml_pub: &mldsa::MlDsaVerifyingKey,
) -> Result<(), SignError> {
    if meta.alg_id != attr.algorithm {
        return Err(SignError::AlgorithmMismatch {
            attribute: attr.algorithm.to_id(),
            signature: meta.alg_id.to_id(),
        });
    }
    let payload = canonical_payload(
        &attr.root,
        &attr.companion_hash,
        meta.version,
        meta.timestamp,
        attr.algorithm,
        attr.leaf_format,
        &attr.grid_hash,
    );
    verify_sig(&payload, sig, ed_pub, ml_pub)
}

// ─────────────────────────────────────────────────────────────────────────────
// Ed25519 implementation
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "ed25519")]
mod ed25519_impl {
    use super::*;
    use ed25519_dalek::{
        Signature as DalekSignature, Signer, SigningKey as DalekSigningKey,
        VerifyingKey as DalekVerifyingKey,
    };

    /// Ed25519 signing key (private key).
    #[derive(Debug)]
    pub struct SigningKey(DalekSigningKey);

    impl SigningKey {
        /// Generate a new random signing key.
        pub fn generate() -> Self {
            let mut csprng = rand::rngs::OsRng;
            Self(DalekSigningKey::generate(&mut csprng))
        }

        /// Create a signing key from raw bytes.
        pub fn from_bytes(bytes: &[u8; 32]) -> Result<Self, SignError> {
            let key = DalekSigningKey::from_bytes(bytes);
            Ok(Self(key))
        }

        /// Export the signing key as raw bytes.
        pub fn to_bytes(&self) -> [u8; 32] {
            self.0.to_bytes()
        }

        /// Get the corresponding verifying (public) key.
        pub fn verifying_key(&self) -> VerifyingKey {
            VerifyingKey(self.0.verifying_key())
        }

        /// Sign a Merkle root, producing a detached signature.
        pub fn sign(&self, root: &MerkleRoot) -> Signature {
            let sig = self.0.sign(root);
            sig.to_bytes()
        }

        /// Sign an arbitrary payload (for canonical payload signing).
        ///
        /// This is used internally by [`sign_root`] to sign the canonical
        /// payload that includes metadata beyond just the Merkle root.
        pub fn sign_payload(&self, payload: &[u8]) -> Signature {
            let sig = self.0.sign(payload);
            sig.to_bytes()
        }
    }

    /// Ed25519 verifying key (public key).
    #[derive(Debug, Clone)]
    pub struct VerifyingKey(DalekVerifyingKey);

    impl VerifyingKey {
        /// Create a verifying key from raw bytes.
        ///
        /// # Errors
        ///
        /// [`SignError::InvalidKey`] if the bytes do not decode to a curve
        /// point; [`SignError::WeakKey`] if they decode to a small-order point.
        /// The verifier's public key is trusted out-of-band by design (S2-D2
        /// trust assumption 2), so a weak key here is a configuration error,
        /// but a weak key would validate a signature over almost any message,
        /// and refusing it costs nothing (2026-10 red-team hardening).
        pub fn from_bytes(bytes: &[u8; 32]) -> Result<Self, SignError> {
            let key = DalekVerifyingKey::from_bytes(bytes)
                .map_err(|e| SignError::InvalidKey(e.to_string()))?;
            if key.is_weak() {
                return Err(SignError::WeakKey);
            }
            Ok(Self(key))
        }

        /// Export the verifying key as raw bytes.
        pub fn to_bytes(&self) -> [u8; 32] {
            self.0.to_bytes()
        }

        /// Verify a signature over a Merkle root.
        ///
        /// Uses `ed25519_dalek`'s strict check: a non-canonical `R`, a small-
        /// order `R`, or a small-order public key is rejected in addition to
        /// the ordinary equation check. Honest signatures never trip any of
        /// those, so this costs nothing and removes the only malleability the
        /// lenient check leaves open (2026-10 red-team hardening).
        pub fn verify(&self, root: &MerkleRoot, signature: &Signature) -> Result<(), SignError> {
            let sig = DalekSignature::from_bytes(signature);
            self.0
                .verify_strict(root, &sig)
                .map_err(|_| SignError::InvalidSignature)
        }

        /// Verify a signature over an arbitrary payload (for canonical payload verification).
        ///
        /// This is used internally by [`verify_sig`] to verify signatures over
        /// the canonical payload that includes metadata beyond just the Merkle root.
        pub fn verify_payload(
            &self,
            payload: &[u8],
            signature: &Signature,
        ) -> Result<(), SignError> {
            let sig = DalekSignature::from_bytes(signature);
            self.0
                .verify_strict(payload, &sig)
                .map_err(|_| SignError::InvalidSignature)
        }
    }

    /// Sign a Merkle root with the given signing key.
    ///
    /// This is the single-algorithm Ed25519 signing function. When the `mldsa`
    /// feature is enabled, this is not re-exported at the crate level to avoid
    /// conflicts with the hybrid `sign_root` function, but remains available
    /// for testing via `ed25519_impl::sign_root`.
    #[cfg_attr(feature = "mldsa", allow(dead_code))]
    pub fn sign_root(key: &SigningKey, root: &MerkleRoot) -> Signature {
        key.sign(root)
    }

    /// Verify a signature over a Merkle root.
    ///
    /// This is the single-algorithm Ed25519 verification function. When the `mldsa`
    /// feature is enabled, this is not re-exported at the crate level to avoid
    /// conflicts with the hybrid `verify_sig` function, but remains available
    /// for testing via `ed25519_impl::verify_root`.
    #[cfg_attr(feature = "mldsa", allow(dead_code))]
    pub fn verify_root(
        key: &VerifyingKey,
        root: &MerkleRoot,
        signature: &Signature,
    ) -> Result<(), SignError> {
        key.verify(root, signature)
    }
}

// Export Ed25519 types and functions.
// When both ed25519 and mldsa features are enabled, don't export the
// single-algorithm sign_root/verify_root to avoid conflicts with the hybrid versions.
#[cfg(feature = "ed25519")]
pub use ed25519_impl::{SigningKey, VerifyingKey};

// Export single-algorithm sign_root and verify_root only when mldsa is NOT enabled
#[cfg(all(feature = "ed25519", not(feature = "mldsa")))]
pub use ed25519_impl::{sign_root, verify_root};

// ─────────────────────────────────────────────────────────────────────────────
// ML-DSA (pure Rust) implementation — post-quantum signatures (FIPS 204)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "mldsa")]
pub mod mldsa {
    use super::*;
    use ml_dsa::{
        EncodedSignature, EncodedVerifyingKey, Generate, Keypair as KeypairTrait, MlDsa65,
        Signature, Signer, SigningKey, Verifier, VerifyingKey,
    };

    /// ML-DSA-65 signature size in bytes.
    pub const MLDSA65_SIGNATURE_SIZE: usize = 3309;

    /// ML-DSA-65 public key size in bytes.
    pub const MLDSA65_PUBLIC_KEY_SIZE: usize = 1952;

    /// ML-DSA-65 secret key size in bytes.
    pub const MLDSA65_SECRET_KEY_SIZE: usize = 4032;

    /// ML-DSA-65 signing key (secret key) for post-quantum signatures.
    pub struct MlDsaSigningKey {
        signing_key: SigningKey<MlDsa65>,
        verifying_key: VerifyingKey<MlDsa65>,
    }

    impl std::fmt::Debug for MlDsaSigningKey {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MlDsaSigningKey").finish_non_exhaustive()
        }
    }

    impl MlDsaSigningKey {
        /// Generate a new random ML-DSA-65 signing key.
        pub fn generate() -> Self {
            let signing_key = SigningKey::<MlDsa65>::generate();
            let verifying_key = KeypairTrait::verifying_key(&signing_key);
            Self {
                signing_key,
                verifying_key,
            }
        }

        /// Get the corresponding verifying (public) key.
        pub fn verifying_key(&self) -> MlDsaVerifyingKey {
            let encoded = self.verifying_key.encode();
            let bytes: &[u8] = encoded.as_slice();
            MlDsaVerifyingKey {
                public_key_bytes: bytes.to_vec(),
            }
        }

        /// Sign a Merkle root, producing a detached ML-DSA signature.
        ///
        /// # Errors
        ///
        /// Returns `SignError::SigningFailed` if the signature is not exactly
        /// 3309 bytes (should never happen per NIST FIPS 204).
        pub fn sign(&self, root: &MerkleRoot) -> Result<[u8; MLDSA65_SIGNATURE_SIZE], SignError> {
            let sig: Signature<MlDsa65> = self.signing_key.sign(root);
            let encoded = sig.encode();
            let bytes: &[u8] = encoded.as_slice();
            bytes.try_into().map_err(|_| SignError::SigningFailed)
        }

        /// Sign an arbitrary payload (for canonical payload signing).
        ///
        /// This is used internally by [`sign_root`] to sign the canonical
        /// payload that includes metadata beyond just the Merkle root.
        ///
        /// # Errors
        ///
        /// Returns `SignError::SigningFailed` if the signature is not exactly
        /// 3309 bytes (should never happen per NIST FIPS 204).
        pub fn sign_payload(
            &self,
            payload: &[u8],
        ) -> Result<[u8; MLDSA65_SIGNATURE_SIZE], SignError> {
            let sig: Signature<MlDsa65> = self.signing_key.sign(payload);
            let encoded = sig.encode();
            let bytes: &[u8] = encoded.as_slice();
            bytes.try_into().map_err(|_| SignError::SigningFailed)
        }
    }

    /// ML-DSA-65 verifying key (public key) for post-quantum verification.
    #[derive(Debug, Clone)]
    pub struct MlDsaVerifyingKey {
        public_key_bytes: Vec<u8>,
    }

    impl MlDsaVerifyingKey {
        /// Create a verifying key from raw bytes.
        pub fn from_bytes(bytes: &[u8]) -> Result<Self, SignError> {
            if bytes.len() != MLDSA65_PUBLIC_KEY_SIZE {
                return Err(SignError::InvalidKey(format!(
                    "expected {} bytes, got {}",
                    MLDSA65_PUBLIC_KEY_SIZE,
                    bytes.len()
                )));
            }
            Ok(Self {
                public_key_bytes: bytes.to_vec(),
            })
        }

        /// Export the verifying key as raw bytes.
        pub fn to_bytes(&self) -> &[u8] {
            &self.public_key_bytes
        }

        /// Verify an ML-DSA signature over a Merkle root.
        pub fn verify(&self, root: &MerkleRoot, signature: &[u8]) -> Result<(), SignError> {
            let pk_bytes: [u8; MLDSA65_PUBLIC_KEY_SIZE] = self
                .public_key_bytes
                .as_slice()
                .try_into()
                .map_err(|_| SignError::InvalidKey("invalid public key length".into()))?;

            let encoded_vk = EncodedVerifyingKey::<MlDsa65>::from(pk_bytes);
            let vk = VerifyingKey::<MlDsa65>::decode(&encoded_vk);

            let sig_bytes: [u8; MLDSA65_SIGNATURE_SIZE] = signature
                .try_into()
                .map_err(|_| SignError::InvalidSignature)?;

            let encoded_sig = EncodedSignature::<MlDsa65>::from(sig_bytes);
            let sig =
                Signature::<MlDsa65>::decode(&encoded_sig).ok_or(SignError::InvalidSignature)?;

            vk.verify(root, &sig)
                .map_err(|_| SignError::InvalidSignature)
        }

        /// Verify an ML-DSA signature over an arbitrary payload (for canonical payload verification).
        ///
        /// This is used internally by [`verify_sig`] to verify signatures over
        /// the canonical payload that includes metadata beyond just the Merkle root.
        pub fn verify_payload(&self, payload: &[u8], signature: &[u8]) -> Result<(), SignError> {
            let pk_bytes: [u8; MLDSA65_PUBLIC_KEY_SIZE] = self
                .public_key_bytes
                .as_slice()
                .try_into()
                .map_err(|_| SignError::InvalidKey("invalid public key length".into()))?;

            let encoded_vk = EncodedVerifyingKey::<MlDsa65>::from(pk_bytes);
            let vk = VerifyingKey::<MlDsa65>::decode(&encoded_vk);

            let sig_bytes: [u8; MLDSA65_SIGNATURE_SIZE] = signature
                .try_into()
                .map_err(|_| SignError::InvalidSignature)?;

            let encoded_sig = EncodedSignature::<MlDsa65>::from(sig_bytes);
            let sig =
                Signature::<MlDsa65>::decode(&encoded_sig).ok_or(SignError::InvalidSignature)?;

            vk.verify(payload, &sig)
                .map_err(|_| SignError::InvalidSignature)
        }
    }

    /// Sign a Merkle root with an ML-DSA-65 signing key.
    ///
    /// # Errors
    ///
    /// Returns `SignError::SigningFailed` if the signing operation fails.
    pub fn sign_root_mldsa(
        key: &MlDsaSigningKey,
        root: &MerkleRoot,
    ) -> Result<[u8; MLDSA65_SIGNATURE_SIZE], SignError> {
        key.sign(root)
    }

    /// Verify an ML-DSA-65 signature over a Merkle root.
    pub fn verify_root_mldsa(
        key: &MlDsaVerifyingKey,
        root: &MerkleRoot,
        signature: &[u8],
    ) -> Result<(), SignError> {
        key.verify(root, signature)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn mldsa_sign_and_verify_roundtrip() {
            let signing_key = MlDsaSigningKey::generate();
            let verifying_key = signing_key.verifying_key();

            let root: MerkleRoot = [0x42; HASH_SIZE];
            let signature = sign_root_mldsa(&signing_key, &root).unwrap();

            assert!(verify_root_mldsa(&verifying_key, &root, &signature).is_ok());
        }

        #[test]
        fn mldsa_verify_rejects_wrong_signature() {
            let signing_key = MlDsaSigningKey::generate();
            let verifying_key = signing_key.verifying_key();

            let root: MerkleRoot = [0x42; HASH_SIZE];
            let mut signature = sign_root_mldsa(&signing_key, &root).unwrap();

            // Tamper with the signature
            signature[0] ^= 0xFF;

            assert!(matches!(
                verify_root_mldsa(&verifying_key, &root, &signature),
                Err(SignError::InvalidSignature)
            ));
        }

        #[test]
        fn mldsa_verify_rejects_wrong_root() {
            let signing_key = MlDsaSigningKey::generate();
            let verifying_key = signing_key.verifying_key();

            let root: MerkleRoot = [0x42; HASH_SIZE];
            let signature = sign_root_mldsa(&signing_key, &root).unwrap();

            let wrong_root: MerkleRoot = [0x43; HASH_SIZE];

            assert!(matches!(
                verify_root_mldsa(&verifying_key, &wrong_root, &signature),
                Err(SignError::InvalidSignature)
            ));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// pqcrypto-mldsa implementation — post-quantum signatures (C reference)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "pq-mldsa")]
pub mod pq_mldsa {
    use super::*;
    use pqcrypto_mldsa::mldsa65;
    use pqcrypto_traits::sign::{PublicKey, SignedMessage};

    /// ML-DSA-65 signing key using pqcrypto (C reference implementation).
    pub struct PqMlDsaSigningKey {
        secret_key: mldsa65::SecretKey,
        public_key: mldsa65::PublicKey,
    }

    impl std::fmt::Debug for PqMlDsaSigningKey {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("PqMlDsaSigningKey")
                .field("public_key_len", &self.public_key.as_bytes().len())
                .finish_non_exhaustive()
        }
    }

    impl PqMlDsaSigningKey {
        /// Generate a new random ML-DSA-65 keypair.
        pub fn generate() -> Self {
            let (public_key, secret_key) = mldsa65::keypair();
            Self {
                secret_key,
                public_key,
            }
        }

        /// Get the corresponding verifying (public) key.
        pub fn verifying_key(&self) -> PqMlDsaVerifyingKey {
            PqMlDsaVerifyingKey {
                public_key_bytes: self.public_key.as_bytes().to_vec(),
            }
        }

        /// Sign a Merkle root, producing a detached ML-DSA signature.
        ///
        /// # Errors
        ///
        /// Returns `SignError::SigningFailed` if the signature is not exactly
        /// 3309 bytes (should never happen per NIST FIPS 204).
        pub fn sign(&self, root: &MerkleRoot) -> Result<[u8; MLDSA65_SIGNATURE_SIZE], SignError> {
            let signed = mldsa65::sign(root, &self.secret_key);
            // Extract just the signature (signed message = signature || message)
            let sig_len = signed.as_bytes().len() - root.len();
            signed.as_bytes()[..sig_len]
                .try_into()
                .map_err(|_| SignError::SigningFailed)
        }
    }

    /// ML-DSA-65 verifying key using pqcrypto (C reference implementation).
    #[derive(Debug, Clone)]
    pub struct PqMlDsaVerifyingKey {
        public_key_bytes: Vec<u8>,
    }

    impl PqMlDsaVerifyingKey {
        /// Create a verifying key from raw bytes.
        pub fn from_bytes(bytes: &[u8]) -> Result<Self, SignError> {
            // Validate by trying to construct the key
            let _ = mldsa65::PublicKey::from_bytes(bytes)
                .map_err(|_| SignError::InvalidKey("invalid ML-DSA public key".into()))?;
            Ok(Self {
                public_key_bytes: bytes.to_vec(),
            })
        }

        /// Export the verifying key as raw bytes.
        pub fn to_bytes(&self) -> &[u8] {
            &self.public_key_bytes
        }

        /// Verify an ML-DSA signature over a Merkle root.
        pub fn verify(&self, root: &MerkleRoot, signature: &[u8]) -> Result<(), SignError> {
            let pk = mldsa65::PublicKey::from_bytes(&self.public_key_bytes)
                .map_err(|_| SignError::InvalidKey("invalid public key".into()))?;

            // Reconstruct signed message (signature || message)
            let mut signed_msg = signature.to_vec();
            signed_msg.extend_from_slice(root);

            let sm = mldsa65::SignedMessage::from_bytes(&signed_msg)
                .map_err(|_| SignError::InvalidSignature)?;

            mldsa65::open(&sm, &pk).map_err(|_| SignError::InvalidSignature)?;
            Ok(())
        }
    }

    /// Sign a Merkle root with a pqcrypto ML-DSA-65 signing key.
    ///
    /// # Errors
    ///
    /// Returns `SignError::SigningFailed` if the signing operation fails.
    pub fn sign_root_pq(
        key: &PqMlDsaSigningKey,
        root: &MerkleRoot,
    ) -> Result<[u8; MLDSA65_SIGNATURE_SIZE], SignError> {
        key.sign(root)
    }

    /// Verify a pqcrypto ML-DSA-65 signature over a Merkle root.
    pub fn verify_root_pq(
        key: &PqMlDsaVerifyingKey,
        root: &MerkleRoot,
        signature: &[u8],
    ) -> Result<(), SignError> {
        key.verify(root, signature)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn pq_mldsa_sign_and_verify_roundtrip() {
            let signing_key = PqMlDsaSigningKey::generate();
            let verifying_key = signing_key.verifying_key();

            let root: MerkleRoot = [0x42; HASH_SIZE];
            let signature = sign_root_pq(&signing_key, &root).unwrap();

            assert!(verify_root_pq(&verifying_key, &root, &signature).is_ok());
        }

        #[test]
        fn pq_mldsa_verify_rejects_wrong_root() {
            let signing_key = PqMlDsaSigningKey::generate();
            let verifying_key = signing_key.verifying_key();

            let root: MerkleRoot = [0x42; HASH_SIZE];
            let signature = sign_root_pq(&signing_key, &root).unwrap();

            let wrong_root: MerkleRoot = [0x43; HASH_SIZE];

            assert!(matches!(
                verify_root_pq(&verifying_key, &wrong_root, &signature),
                Err(SignError::InvalidSignature)
            ));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(feature = "ed25519")]
mod tests {
    use super::*;

    #[test]
    fn sign_and_verify_roundtrip() {
        let signing_key = SigningKey::generate();
        let verifying_key = signing_key.verifying_key();

        let root: MerkleRoot = [0x42; HASH_SIZE];
        let signature = ed25519_impl::sign_root(&signing_key, &root);

        assert!(ed25519_impl::verify_root(&verifying_key, &root, &signature).is_ok());
    }

    #[test]
    fn verify_rejects_wrong_signature() {
        let signing_key = SigningKey::generate();
        let verifying_key = signing_key.verifying_key();

        let root: MerkleRoot = [0x42; HASH_SIZE];
        let mut signature = ed25519_impl::sign_root(&signing_key, &root);

        // Tamper with the signature
        signature[0] ^= 0xFF;

        assert!(matches!(
            ed25519_impl::verify_root(&verifying_key, &root, &signature),
            Err(SignError::InvalidSignature)
        ));
    }

    #[test]
    fn verify_rejects_wrong_root() {
        let signing_key = SigningKey::generate();
        let verifying_key = signing_key.verifying_key();

        let root: MerkleRoot = [0x42; HASH_SIZE];
        let signature = ed25519_impl::sign_root(&signing_key, &root);

        // Different root
        let wrong_root: MerkleRoot = [0x43; HASH_SIZE];

        assert!(matches!(
            ed25519_impl::verify_root(&verifying_key, &wrong_root, &signature),
            Err(SignError::InvalidSignature)
        ));
    }

    #[test]
    fn verify_rejects_wrong_key() {
        let signing_key = SigningKey::generate();
        let other_key = SigningKey::generate();
        let wrong_verifying_key = other_key.verifying_key();

        let root: MerkleRoot = [0x42; HASH_SIZE];
        let signature = ed25519_impl::sign_root(&signing_key, &root);

        assert!(matches!(
            ed25519_impl::verify_root(&wrong_verifying_key, &root, &signature),
            Err(SignError::InvalidSignature)
        ));
    }

    #[test]
    fn key_serialization_roundtrip() {
        let signing_key = SigningKey::generate();
        let verifying_key = signing_key.verifying_key();

        // Round-trip signing key
        let signing_bytes = signing_key.to_bytes();
        let signing_key2 = SigningKey::from_bytes(&signing_bytes).unwrap();
        assert_eq!(signing_key.to_bytes(), signing_key2.to_bytes());

        // Round-trip verifying key
        let verifying_bytes = verifying_key.to_bytes();
        let verifying_key2 = VerifyingKey::from_bytes(&verifying_bytes).unwrap();
        assert_eq!(verifying_key.to_bytes(), verifying_key2.to_bytes());
    }

    #[test]
    fn canonical_payload_has_correct_size() {
        let root = [0x42; 32];
        let companion_hash = [0x43; 32];
        let version = 1;
        let timestamp = 1704067200; // 2024-01-01 00:00:00 UTC

        let payload_blake3 =
            canonical_payload(&root, &companion_hash, version, timestamp, HashAlg::Blake3, LeafFormat::Content, &[0u8; 32]);
        assert_eq!(payload_blake3.len(), CANONICAL_PAYLOAD_SIZE);

        let payload_sha256 =
            canonical_payload(&root, &companion_hash, version, timestamp, HashAlg::Sha256, LeafFormat::Content, &[0u8; 32]);
        assert_eq!(payload_sha256.len(), CANONICAL_PAYLOAD_SIZE);

        let payload_k12 =
            canonical_payload(&root, &companion_hash, version, timestamp, HashAlg::K12, LeafFormat::Content, &[0u8; 32]);
        assert_eq!(payload_k12.len(), CANONICAL_PAYLOAD_SIZE);
        assert_eq!(CANONICAL_PAYLOAD_SIZE, 114);
    }

    #[test]
    fn canonical_payload_encodes_fields_correctly() {
        let root = [0x01; 32];
        let companion_hash = [0x02; 32];
        let version = 0x0123456789ABCDEFu64;
        let timestamp = 0xFEDCBA9876543210u64;
        let grid_hash = [0x03; 32];

        let payload =
            canonical_payload(&root, &companion_hash, version, timestamp, HashAlg::Blake3, LeafFormat::Bound, &grid_hash);

        // Check leaf_format (byte 81) and grid_hash (bytes 82-113)
        assert_eq!(payload[81], LeafFormat::Bound.to_id());
        assert_eq!(&payload[82..114], &grid_hash);

        // Check root (bytes 0-31)
        assert_eq!(&payload[0..32], &root);

        // Check companion_hash (bytes 32-63)
        assert_eq!(&payload[32..64], &companion_hash);

        // Check version (bytes 64-71, big-endian)
        let version_bytes = &payload[64..72];
        assert_eq!(
            u64::from_be_bytes(version_bytes.try_into().unwrap()),
            version
        );

        // Check timestamp (bytes 72-79, big-endian)
        let timestamp_bytes = &payload[72..80];
        assert_eq!(
            u64::from_be_bytes(timestamp_bytes.try_into().unwrap()),
            timestamp
        );

        // Check alg_id (byte 80)
        assert_eq!(payload[80], HashAlg::Blake3.to_id());
    }

    #[test]
    fn canonical_payload_differs_by_algorithm() {
        let root = [0x42; 32];
        let companion_hash = [0x43; 32];
        let version = 1;
        let timestamp = 1704067200;

        let payload_blake3 =
            canonical_payload(&root, &companion_hash, version, timestamp, HashAlg::Blake3, LeafFormat::Content, &[0u8; 32]);
        let payload_sha256 =
            canonical_payload(&root, &companion_hash, version, timestamp, HashAlg::Sha256, LeafFormat::Content, &[0u8; 32]);
        let payload_k12 =
            canonical_payload(&root, &companion_hash, version, timestamp, HashAlg::K12, LeafFormat::Content, &[0u8; 32]);

        // Payloads should differ only in byte 80 (algorithm ID)
        assert_eq!(&payload_blake3[..80], &payload_sha256[..80]);
        assert_eq!(&payload_blake3[..80], &payload_k12[..80]);
        assert_ne!(payload_blake3[80], payload_sha256[80]);
        assert_ne!(payload_blake3[80], payload_k12[80]);
        assert_ne!(payload_sha256[80], payload_k12[80]);
        assert_eq!(&payload_blake3[81..], &payload_sha256[81..]);
        assert_eq!(&payload_blake3[81..], &payload_k12[81..]);
    }

    #[test]
    fn canonical_payload_differs_by_leaf_format() {
        // 2026-10 red-team regression: the leaf-preimage format must be
        // signed, or a Bound->Content downgrade with re-framed chunks passes.
        let root = [0x42; 32];
        let companion_hash = [0x43; 32];
        let content = canonical_payload(&root, &companion_hash, 1, 1704067200, HashAlg::Blake3, LeafFormat::Content, &[0u8; 32]);
        let bound = canonical_payload(&root, &companion_hash, 1, 1704067200, HashAlg::Blake3, LeafFormat::Bound, &[0u8; 32]);
        assert_ne!(content, bound);
        assert_eq!(&content[..81], &bound[..81]);
        assert_ne!(content[81], bound[81]);
        assert_eq!(&content[82..], &bound[82..]);
    }

    #[test]
    fn canonical_payload_differs_by_grid_hash() {
        // 2026-10 red-team regression: the grid hash must be signed, or a
        // rewritten dims/chunk_shape (with recomputed grid_hash) passes.
        let root = [0x42; 32];
        let companion_hash = [0x43; 32];
        let unbound = canonical_payload(&root, &companion_hash, 1, 1704067200, HashAlg::Blake3, LeafFormat::Content, &[0u8; 32]);
        let bound = canonical_payload(&root, &companion_hash, 1, 1704067200, HashAlg::Blake3, LeafFormat::Content, &[0x55u8; 32]);
        assert_ne!(unbound, bound);
        assert_eq!(&unbound[..82], &bound[..82]);
        assert_ne!(&unbound[82..], &bound[82..]);
    }

    #[test]
    #[cfg(feature = "mldsa")]
    fn hybrid_signature_from_bytes_rejects_payload_v0_signatures() {
        // A version-0 signature was computed over the 81-byte payload that
        // left leaf_format/grid_hash unsigned; it must be rejected at parse
        // time rather than verified against a v1 payload it never bound.
        let mut data = vec![0x00]; // v0 version byte
        data.extend_from_slice(&[0u8; SIGNATURE_SIZE]);
        data.extend_from_slice(&[0u8; MLDSA65_SIGNATURE_SIZE]);
        assert_eq!(data.len(), HybridSignature::SERIALIZED_SIZE);
        assert!(matches!(
            HybridSignature::from_bytes(&data),
            Err(SignError::InvalidSignature)
        ));
        assert_eq!(HybridSignature::VERSION, 1);
    }

    #[test]
    fn canonical_payload_differs_by_version() {
        let root = [0x42; 32];
        let companion_hash = [0x43; 32];
        let timestamp = 1704067200;

        let payload_v1 = canonical_payload(&root, &companion_hash, 1, timestamp, HashAlg::Blake3, LeafFormat::Content, &[0u8; 32]);
        let payload_v2 = canonical_payload(&root, &companion_hash, 2, timestamp, HashAlg::Blake3, LeafFormat::Content, &[0u8; 32]);

        // Should differ in the version field (bytes 64-71)
        assert_ne!(payload_v1, payload_v2);
        assert_eq!(&payload_v1[0..64], &payload_v2[0..64]); // root and companion_hash same
        assert_ne!(&payload_v1[64..72], &payload_v2[64..72]); // version differs
        assert_eq!(&payload_v1[72..], &payload_v2[72..]); // timestamp and alg_id same
    }

    #[test]
    fn canonical_payload_differs_by_timestamp() {
        let root = [0x42; 32];
        let companion_hash = [0x43; 32];
        let version = 1;

        let payload_t1 =
            canonical_payload(&root, &companion_hash, version, 1704067200, HashAlg::Blake3, LeafFormat::Content, &[0u8; 32]);
        let payload_t2 =
            canonical_payload(&root, &companion_hash, version, 1704067201, HashAlg::Blake3, LeafFormat::Content, &[0u8; 32]);

        // Should differ in the timestamp field (bytes 72-79)
        assert_ne!(payload_t1, payload_t2);
        assert_eq!(&payload_t1[0..72], &payload_t2[0..72]); // root, companion_hash, version same
        assert_ne!(&payload_t1[72..80], &payload_t2[72..80]); // timestamp differs
        assert_eq!(&payload_t1[80..], &payload_t2[80..]); // alg_id same
    }

    #[test]
    fn canonical_payload_no_separators() {
        // Verify that the encoding is truly fixed-width with no separators
        let root = [0xFF; 32];
        let companion_hash = [0x00; 32];
        let version = u64::MAX;
        let timestamp = 0;

        let payload =
            canonical_payload(&root, &companion_hash, version, timestamp, HashAlg::Blake3, LeafFormat::Content, &[0u8; 32]);

        // All 0xFF bytes from root should be preserved
        assert_eq!(&payload[0..32], &[0xFF; 32]);
        // All 0x00 bytes from companion_hash should be preserved
        assert_eq!(&payload[32..64], &[0x00; 32]);
        // u64::MAX should be encoded as all 0xFF bytes
        assert_eq!(&payload[64..72], &[0xFF; 8]);
        // 0 should be encoded as all 0x00 bytes
        assert_eq!(&payload[72..80], &[0x00; 8]);
    }

    #[test]
    #[cfg(feature = "mldsa")]
    fn hybrid_signature_serialization_roundtrip() {
        use crate::mldsa::MlDsaSigningKey;

        let ed_key = SigningKey::generate();
        let ml_key = MlDsaSigningKey::generate();

        let root = [0x42; 32];
        let companion_hash = [0x43; 32];
        let payload = canonical_payload(&root, &companion_hash, 1, 1704067200, HashAlg::Blake3, LeafFormat::Content, &[0u8; 32]);

        let hybrid_sig = sign_root(&payload, &ed_key, &ml_key).unwrap();

        // Test serialization
        let bytes = hybrid_sig.to_bytes();
        assert_eq!(bytes.len(), HybridSignature::SERIALIZED_SIZE);
        assert_eq!(bytes[0], HybridSignature::VERSION);

        // Test deserialization
        let hybrid_sig2 = HybridSignature::from_bytes(&bytes).unwrap();
        assert_eq!(hybrid_sig, hybrid_sig2);
    }

    #[test]
    #[cfg(feature = "mldsa")]
    fn hybrid_signature_has_correct_size() {
        let ed_sig = [0x42; SIGNATURE_SIZE];
        let ml_sig = [0x43; MLDSA65_SIGNATURE_SIZE];

        let hybrid = HybridSignature::new(ed_sig, ml_sig);
        let bytes = hybrid.to_bytes();

        assert_eq!(bytes.len(), HybridSignature::SERIALIZED_SIZE);
        assert_eq!(bytes.len(), 1 + 64 + 3309);
    }

    #[test]
    #[cfg(feature = "mldsa")]
    fn hybrid_signature_from_bytes_rejects_wrong_size() {
        // Too short
        let short_data = vec![0u8; 100];
        assert!(matches!(
            HybridSignature::from_bytes(&short_data),
            Err(SignError::InvalidSignature)
        ));

        // Wrong ML-DSA signature length (too short)
        let mut bad_data = vec![HybridSignature::VERSION];
        bad_data.extend_from_slice(&[0u8; SIGNATURE_SIZE]); // Ed25519 sig
        bad_data.extend_from_slice(&[0u8; 100]); // Wrong ML-DSA size
        assert!(matches!(
            HybridSignature::from_bytes(&bad_data),
            Err(SignError::InvalidSignature)
        ));

        // Trailing bytes (too long)
        let mut too_long = vec![HybridSignature::VERSION];
        too_long.extend_from_slice(&[0u8; SIGNATURE_SIZE]); // Ed25519 sig
        too_long.extend_from_slice(&[0u8; 3309]); // ML-DSA sig
        too_long.push(0x42); // Extra trailing byte
        assert!(matches!(
            HybridSignature::from_bytes(&too_long),
            Err(SignError::InvalidSignature)
        ));
    }

    #[test]
    #[cfg(feature = "mldsa")]
    fn hybrid_signature_from_bytes_rejects_wrong_version() {
        let mut data = vec![0xFF]; // Wrong version
        data.extend_from_slice(&[0u8; SIGNATURE_SIZE]); // Ed25519 sig
        data.extend_from_slice(&[0u8; 3309]); // ML-DSA sig

        assert!(matches!(
            HybridSignature::from_bytes(&data),
            Err(SignError::InvalidSignature)
        ));
    }

    #[test]
    #[cfg(feature = "mldsa")]
    fn sign_root_produces_valid_signatures() {
        use crate::mldsa::MlDsaSigningKey;

        let ed_key = SigningKey::generate();
        let ml_key = MlDsaSigningKey::generate();

        let root = [0x42; 32];
        let companion_hash = [0x43; 32];
        let payload = canonical_payload(&root, &companion_hash, 1, 1704067200, HashAlg::Blake3, LeafFormat::Content, &[0u8; 32]);

        let hybrid_sig = sign_root(&payload, &ed_key, &ml_key).unwrap();

        // Verify Ed25519 component
        let ed_vk = ed_key.verifying_key();
        assert!(
            ed_vk
                .verify_payload(&payload, &hybrid_sig.ed25519_sig)
                .is_ok()
        );

        // Verify ML-DSA component
        let ml_vk = ml_key.verifying_key();
        assert!(
            ml_vk
                .verify_payload(&payload, &hybrid_sig.mldsa65_sig)
                .is_ok()
        );
    }

    #[test]
    #[cfg(feature = "mldsa")]
    fn sign_root_with_different_payloads_produces_different_signatures() {
        use crate::mldsa::MlDsaSigningKey;

        let ed_key = SigningKey::generate();
        let ml_key = MlDsaSigningKey::generate();

        let root = [0x42; 32];
        let companion_hash = [0x43; 32];

        let payload1 = canonical_payload(&root, &companion_hash, 1, 1704067200, HashAlg::Blake3, LeafFormat::Content, &[0u8; 32]);
        let payload2 = canonical_payload(&root, &companion_hash, 2, 1704067200, HashAlg::Blake3, LeafFormat::Content, &[0u8; 32]);

        let sig1 = sign_root(&payload1, &ed_key, &ml_key).unwrap();
        let sig2 = sign_root(&payload2, &ed_key, &ml_key).unwrap();

        // Signatures should differ for different payloads
        assert_ne!(sig1.ed25519_sig, sig2.ed25519_sig);
        assert_ne!(sig1.mldsa65_sig, sig2.mldsa65_sig);
    }

    #[test]
    #[cfg(feature = "mldsa")]
    fn verify_sig_accepts_valid_hybrid_signature() {
        use crate::mldsa::MlDsaSigningKey;

        let ed_key = SigningKey::generate();
        let ml_key = MlDsaSigningKey::generate();

        let root = [0x42; 32];
        let companion_hash = [0x43; 32];
        let payload = canonical_payload(&root, &companion_hash, 1, 1704067200, HashAlg::Blake3, LeafFormat::Content, &[0u8; 32]);

        let hybrid_sig = sign_root(&payload, &ed_key, &ml_key).unwrap();

        let ed_pub = ed_key.verifying_key();
        let ml_pub = ml_key.verifying_key();

        // Should accept valid signature
        assert!(verify_sig(&payload, &hybrid_sig, &ed_pub, &ml_pub).is_ok());
    }

    #[test]
    #[cfg(feature = "mldsa")]
    fn verify_sig_rejects_tampered_ed25519_signature() {
        use crate::mldsa::MlDsaSigningKey;

        let ed_key = SigningKey::generate();
        let ml_key = MlDsaSigningKey::generate();

        let root = [0x42; 32];
        let companion_hash = [0x43; 32];
        let payload = canonical_payload(&root, &companion_hash, 1, 1704067200, HashAlg::Blake3, LeafFormat::Content, &[0u8; 32]);

        let mut hybrid_sig = sign_root(&payload, &ed_key, &ml_key).unwrap();

        // Tamper with Ed25519 signature
        hybrid_sig.ed25519_sig[0] ^= 0xFF;

        let ed_pub = ed_key.verifying_key();
        let ml_pub = ml_key.verifying_key();

        // Should reject due to invalid Ed25519 signature
        assert!(matches!(
            verify_sig(&payload, &hybrid_sig, &ed_pub, &ml_pub),
            Err(SignError::InvalidSignature)
        ));
    }

    #[test]
    #[cfg(feature = "mldsa")]
    fn verify_sig_rejects_tampered_mldsa_signature() {
        use crate::mldsa::MlDsaSigningKey;

        let ed_key = SigningKey::generate();
        let ml_key = MlDsaSigningKey::generate();

        let root = [0x42; 32];
        let companion_hash = [0x43; 32];
        let payload = canonical_payload(&root, &companion_hash, 1, 1704067200, HashAlg::Blake3, LeafFormat::Content, &[0u8; 32]);

        let mut hybrid_sig = sign_root(&payload, &ed_key, &ml_key).unwrap();

        // Tamper with ML-DSA signature
        hybrid_sig.mldsa65_sig[0] ^= 0xFF;

        let ed_pub = ed_key.verifying_key();
        let ml_pub = ml_key.verifying_key();

        // Should reject due to invalid ML-DSA signature
        assert!(matches!(
            verify_sig(&payload, &hybrid_sig, &ed_pub, &ml_pub),
            Err(SignError::InvalidSignature)
        ));
    }

    #[test]
    #[cfg(feature = "mldsa")]
    fn verify_sig_rejects_modified_payload() {
        use crate::mldsa::MlDsaSigningKey;

        let ed_key = SigningKey::generate();
        let ml_key = MlDsaSigningKey::generate();

        let root = [0x42; 32];
        let companion_hash = [0x43; 32];
        let payload = canonical_payload(&root, &companion_hash, 1, 1704067200, HashAlg::Blake3, LeafFormat::Content, &[0u8; 32]);

        let hybrid_sig = sign_root(&payload, &ed_key, &ml_key).unwrap();

        // Create different payload (different version)
        let modified_payload =
            canonical_payload(&root, &companion_hash, 2, 1704067200, HashAlg::Blake3, LeafFormat::Content, &[0u8; 32]);

        let ed_pub = ed_key.verifying_key();
        let ml_pub = ml_key.verifying_key();

        // Should reject due to payload mismatch
        assert!(matches!(
            verify_sig(&modified_payload, &hybrid_sig, &ed_pub, &ml_pub),
            Err(SignError::InvalidSignature)
        ));
    }

    #[test]
    #[cfg(feature = "mldsa")]
    fn verify_sig_rejects_wrong_ed25519_key() {
        use crate::mldsa::MlDsaSigningKey;

        let ed_key = SigningKey::generate();
        let ml_key = MlDsaSigningKey::generate();

        let root = [0x42; 32];
        let companion_hash = [0x43; 32];
        let payload = canonical_payload(&root, &companion_hash, 1, 1704067200, HashAlg::Blake3, LeafFormat::Content, &[0u8; 32]);

        let hybrid_sig = sign_root(&payload, &ed_key, &ml_key).unwrap();

        // Use wrong Ed25519 key
        let wrong_ed_key = SigningKey::generate();
        let wrong_ed_pub = wrong_ed_key.verifying_key();
        let ml_pub = ml_key.verifying_key();

        // Should reject due to wrong Ed25519 key
        assert!(matches!(
            verify_sig(&payload, &hybrid_sig, &wrong_ed_pub, &ml_pub),
            Err(SignError::InvalidSignature)
        ));
    }

    #[test]
    #[cfg(feature = "mldsa")]
    fn verify_sig_rejects_wrong_mldsa_key() {
        use crate::mldsa::MlDsaSigningKey;

        let ed_key = SigningKey::generate();
        let ml_key = MlDsaSigningKey::generate();

        let root = [0x42; 32];
        let companion_hash = [0x43; 32];
        let payload = canonical_payload(&root, &companion_hash, 1, 1704067200, HashAlg::Blake3, LeafFormat::Content, &[0u8; 32]);

        let hybrid_sig = sign_root(&payload, &ed_key, &ml_key).unwrap();

        // Use wrong ML-DSA key
        let ed_pub = ed_key.verifying_key();
        let wrong_ml_key = MlDsaSigningKey::generate();
        let wrong_ml_pub = wrong_ml_key.verifying_key();

        // Should reject due to wrong ML-DSA key
        assert!(matches!(
            verify_sig(&payload, &hybrid_sig, &ed_pub, &wrong_ml_pub),
            Err(SignError::InvalidSignature)
        ));
    }

    #[test]
    #[cfg(feature = "mldsa")]
    fn verify_sig_rejects_when_both_signatures_tampered() {
        use crate::mldsa::MlDsaSigningKey;

        let ed_key = SigningKey::generate();
        let ml_key = MlDsaSigningKey::generate();

        let root = [0x42; 32];
        let companion_hash = [0x43; 32];
        let payload = canonical_payload(&root, &companion_hash, 1, 1704067200, HashAlg::Blake3, LeafFormat::Content, &[0u8; 32]);

        let mut hybrid_sig = sign_root(&payload, &ed_key, &ml_key).unwrap();

        // Tamper with both signatures
        hybrid_sig.ed25519_sig[0] ^= 0xFF;
        hybrid_sig.mldsa65_sig[0] ^= 0xFF;

        let ed_pub = ed_key.verifying_key();
        let ml_pub = ml_key.verifying_key();

        // Should reject (fails on first check: Ed25519)
        assert!(matches!(
            verify_sig(&payload, &hybrid_sig, &ed_pub, &ml_pub),
            Err(SignError::InvalidSignature)
        ));
    }

    #[test]
    #[cfg(feature = "mldsa")]
    fn verify_sig_enforces_strict_and_policy() {
        use crate::mldsa::MlDsaSigningKey;

        // This test demonstrates that BOTH signatures must be valid.
        // Even if Ed25519 is valid, a bad ML-DSA signature causes rejection.

        let ed_key = SigningKey::generate();
        let ml_key = MlDsaSigningKey::generate();

        let root = [0x42; 32];
        let companion_hash = [0x43; 32];
        let payload = canonical_payload(&root, &companion_hash, 1, 1704067200, HashAlg::Blake3, LeafFormat::Content, &[0u8; 32]);

        let mut hybrid_sig = sign_root(&payload, &ed_key, &ml_key).unwrap();

        // Tamper ONLY with ML-DSA (Ed25519 remains valid)
        hybrid_sig.mldsa65_sig[100] ^= 0xFF;

        let ed_pub = ed_key.verifying_key();
        let ml_pub = ml_key.verifying_key();

        // Verify Ed25519 alone would pass
        assert!(
            ed_pub
                .verify_payload(&payload, &hybrid_sig.ed25519_sig)
                .is_ok()
        );

        // But hybrid verification MUST fail due to strict-AND policy
        assert!(matches!(
            verify_sig(&payload, &hybrid_sig, &ed_pub, &ml_pub),
            Err(SignError::InvalidSignature)
        ));
    }

    #[test]
    #[cfg(feature = "mldsa")]
    fn write_and_read_sig_attr_roundtrip() {
        use crate::mldsa::MlDsaSigningKey;
        use clawhdf5_format::attribute::extract_attributes;
        use clawhdf5_format::file_writer::FileWriter;
        use clawhdf5_format::group_v2::resolve_path_any;
        use clawhdf5_format::object_header::ObjectHeader;
        use clawhdf5_format::signature::find_signature;
        use clawhdf5_format::superblock::Superblock;

        let ed_key = SigningKey::generate();
        let ml_key = MlDsaSigningKey::generate();

        let root = [0x11; 32];
        let companion_hash = [0x22; 32];
        let payload = canonical_payload(&root, &companion_hash, 1, 1_700_000_000, HashAlg::Blake3, LeafFormat::Content, &[0u8; 32]);
        let hybrid_sig = sign_root(&payload, &ed_key, &ml_key).unwrap();

        // Write a file with the signature attached to a dataset.
        let meta = SignedRootMeta {
            alg_id: HashAlg::Blake3,
            version: 1,
            timestamp: 1_700_000_000,
        };
        let mut fw = FileWriter::new();
        let ds = fw.create_dataset("data");
        ds.with_u8_data(&[1, 2, 3, 4, 5, 6]);
        write_sig_attr(ds, &meta, &hybrid_sig);
        let file_bytes = fw.finish().expect("file should build");

        // Reopen the file from scratch and read the attribute back.
        let sig_offset = find_signature(&file_bytes).expect("signature not found");
        let sb = Superblock::parse(&file_bytes, sig_offset).expect("superblock parse failed");
        let data_addr = resolve_path_any(&file_bytes, &sb, "data").expect("dataset not found");
        let data_hdr = ObjectHeader::parse(
            &file_bytes,
            data_addr as usize,
            sb.offset_size,
            sb.length_size,
        )
        .expect("dataset header parse failed");
        let attrs = extract_attributes(&data_hdr, sb.length_size).expect("extract attrs failed");

        let (recovered_meta, recovered) =
            read_sig_attr(&attrs).expect("merkle_sig attribute not found");
        assert_eq!(recovered, hybrid_sig);
        assert_eq!(recovered_meta, meta);

        // The recovered signature must still verify against the original payload.
        let ed_pub = ed_key.verifying_key();
        let ml_pub = ml_key.verifying_key();
        assert!(verify_sig(&payload, &recovered, &ed_pub, &ml_pub).is_ok());
    }

    #[test]
    #[cfg(feature = "mldsa")]
    fn read_sig_attr_missing_returns_attribute_missing() {
        use clawhdf5_format::attribute::extract_attributes;
        use clawhdf5_format::file_writer::FileWriter;
        use clawhdf5_format::group_v2::resolve_path_any;
        use clawhdf5_format::object_header::ObjectHeader;
        use clawhdf5_format::signature::find_signature;
        use clawhdf5_format::superblock::Superblock;

        let mut fw = FileWriter::new();
        let ds = fw.create_dataset("data");
        ds.with_u8_data(&[1, 2, 3]);
        let file_bytes = fw.finish().expect("file should build");

        let sig_offset = find_signature(&file_bytes).expect("signature not found");
        let sb = Superblock::parse(&file_bytes, sig_offset).expect("superblock parse failed");
        let data_addr = resolve_path_any(&file_bytes, &sb, "data").expect("dataset not found");
        let data_hdr = ObjectHeader::parse(
            &file_bytes,
            data_addr as usize,
            sb.offset_size,
            sb.length_size,
        )
        .expect("dataset header parse failed");
        let attrs = extract_attributes(&data_hdr, sb.length_size).expect("extract attrs failed");

        assert!(matches!(
            read_sig_attr(&attrs),
            Err(SignError::AttributeMissing)
        ));
    }
}
