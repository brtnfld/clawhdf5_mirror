//! Rollback-based recovery: `restore_to_version` (P2.2b step 4).
//!
//! After tampering is detected, the primary recovery path is to revert to the
//! last known-good signed version recorded in the provenance journal (P2.2b
//! step 3). This module implements the *decision and verification* half of that
//! path, independent of file I/O:
//!
//! 1. **Select** the target record — an explicit version, or the default
//!    "last known-good" (the highest version whose signature verifies).
//! 2. **Signature gate:** the selected record's hybrid signature must verify
//!    against its journaled signed root.
//! 3. **Dataset gate:** after the caller physically reverts to the record's
//!    snapshot, the restored dataset must pass [`verify_dataset`] *and* its
//!    Merkle root must equal the journaled `signed_root`.
//!
//! Only when both gates pass is the restored state accepted.
//!
//! # Decoupling
//!
//! `clawhdf5-format` depends on neither the P2.1 signing crate nor any file I/O,
//! so hybrid-signature verification is injected through the [`SignatureVerifier`]
//! trait, and the physical snapshot revert is performed by the caller
//! (`clawhdf5-agent`) between the two gates.

use crate::merkle::{Dataset, MerkleError, constant_time_eq, verify_dataset};
use crate::merkle_journal::{ProvenanceJournal, ProvenanceRecord};

/// Which version `restore_to_version` should revert to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestoreTarget {
    /// Restore a specific journaled version.
    Version(u64),
    /// Restore the highest version whose signature verifies — the default
    /// "last known-good signed version".
    LastKnownGood,
}

/// Injected verifier for a record's hybrid signature over its signed root.
///
/// Implemented by the caller using the P2.1 signing machinery (not available to
/// `clawhdf5-format`). It must return `true` only if `record.hybrid_sig` is a
/// valid signature over the canonical payload binding `record.signed_root`,
/// `record.version`, and `record.timestamp`.
pub trait SignatureVerifier {
    /// Verify the record's hybrid signature.
    fn verify(&self, record: &ProvenanceRecord) -> bool;
}

/// Why a restore was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestoreError {
    /// No journaled record exists for the requested version — a snapshot with no
    /// journaled root is not a valid rollback target.
    NoJournaledRecord {
        /// The requested version.
        version: u64,
    },
    /// No version in the journal has a signature that verifies, so there is no
    /// known-good state to fall back to.
    NoKnownGoodVersion,
    /// The selected record's hybrid signature did not verify.
    SignatureInvalid {
        /// The version whose signature failed.
        version: u64,
    },
    /// The restored dataset failed Merkle verification.
    DatasetVerificationFailed {
        /// The version being restored.
        version: u64,
        /// The underlying verification error, if one was produced.
        source: Option<MerkleError>,
    },
    /// The restored dataset verified internally, but its root does not match the
    /// journaled signed root for this version (the snapshot is not the state the
    /// signature certifies).
    RootMismatch {
        /// The version being restored.
        version: u64,
    },
}

impl core::fmt::Display for RestoreError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RestoreError::NoJournaledRecord { version } => {
                write!(
                    f,
                    "no journaled record for version {version} (not a valid rollback target)"
                )
            }
            RestoreError::NoKnownGoodVersion => {
                write!(f, "no journaled version has a verifying signature")
            }
            RestoreError::SignatureInvalid { version } => {
                write!(f, "hybrid signature for version {version} did not verify")
            }
            RestoreError::DatasetVerificationFailed { version, source } => match source {
                Some(e) => write!(
                    f,
                    "restored dataset for version {version} failed verification: {e}"
                ),
                None => write!(
                    f,
                    "restored dataset for version {version} failed verification"
                ),
            },
            RestoreError::RootMismatch { version } => {
                write!(
                    f,
                    "restored dataset root does not match the journaled signed root for version {version}"
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for RestoreError {}

/// Select the record to restore, applying the signature gate.
///
/// For [`RestoreTarget::Version`], the record must exist and its signature must
/// verify. For [`RestoreTarget::LastKnownGood`], returns the highest-version
/// record whose signature verifies.
///
/// # Errors
///
/// - [`RestoreError::NoJournaledRecord`] if an explicit version is not journaled
/// - [`RestoreError::SignatureInvalid`] if an explicit version's signature fails
/// - [`RestoreError::NoKnownGoodVersion`] if no version verifies (last-known-good)
pub fn select_restore_record<'j, V: SignatureVerifier + ?Sized>(
    journal: &'j ProvenanceJournal,
    target: RestoreTarget,
    verifier: &V,
) -> Result<&'j ProvenanceRecord, RestoreError> {
    match target {
        RestoreTarget::Version(v) => {
            let record = journal
                .record_for_version(v)
                .ok_or(RestoreError::NoJournaledRecord { version: v })?;
            if verifier.verify(record) {
                Ok(record)
            } else {
                Err(RestoreError::SignatureInvalid { version: v })
            }
        }
        RestoreTarget::LastKnownGood => {
            // Records are ordered by ascending version; scan from the newest.
            journal
                .records()
                .iter()
                .rev()
                .find(|r| verifier.verify(r))
                .ok_or(RestoreError::NoKnownGoodVersion)
        }
    }
}

/// The dataset gate: verify the restored dataset and bind it to the journaled
/// signed root.
///
/// Runs [`verify_dataset`] on the restored dataset and then requires that its
/// Merkle root equals `record.signed_root` — proving the restored bytes are the
/// exact state the record's signature certifies.
///
/// # Errors
///
/// - [`RestoreError::DatasetVerificationFailed`] if `verify_dataset` fails
/// - [`RestoreError::RootMismatch`] if the root differs from the signed root
pub fn verify_restored_dataset(
    record: &ProvenanceRecord,
    dataset: &Dataset<'_>,
) -> Result<(), RestoreError> {
    match verify_dataset(dataset) {
        Ok(true) => {}
        Ok(false) => {
            return Err(RestoreError::DatasetVerificationFailed {
                version: record.version,
                source: None,
            });
        }
        Err(source) => {
            return Err(RestoreError::DatasetVerificationFailed {
                version: record.version,
                source: Some(source),
            });
        }
    }

    // Merkle roots are public, so this isn't secrecy-critical, but hash
    // comparisons in this crate are uniformly constant-time (house style).
    if constant_time_eq(&dataset.merkle_attr.root, &record.signed_root) {
        Ok(())
    } else {
        Err(RestoreError::RootMismatch {
            version: record.version,
        })
    }
}

/// Orchestrate a restore: select the target (signature gate), then run the
/// caller-supplied `apply` step, which physically reverts to the record's
/// snapshot and enforces the dataset gate via [`verify_restored_dataset`].
///
/// Returns the version that was restored. `apply` is invoked only after the
/// signature gate passes, so a snapshot for an unverifiable version is never
/// materialized.
///
/// # Errors
///
/// Any [`RestoreError`] from selection, or one returned by `apply`.
pub fn restore_to_version<V, F>(
    journal: &ProvenanceJournal,
    target: RestoreTarget,
    verifier: &V,
    apply: F,
) -> Result<u64, RestoreError>
where
    V: SignatureVerifier + ?Sized,
    F: FnOnce(&ProvenanceRecord) -> Result<(), RestoreError>,
{
    let record = select_restore_record(journal, target, verifier)?;
    apply(record)?;
    Ok(record.version)
}

#[cfg(all(test, feature = "blake3"))]
mod tests {
    use super::*;
    use crate::merkle::{HASH_SIZE, HashAlg, MerkleAttr, MerkleTree};

    #[cfg(not(feature = "std"))]
    use alloc::{string::String, vec::Vec};

    /// Build an owned Dataset from chunks whose stored root is correct.
    fn dataset_for(chunks: &[Vec<u8>]) -> (MerkleAttr, Vec<u8>) {
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let tree = MerkleTree::from_chunks(&refs, HashAlg::Blake3);
        let mut nodes = Vec::with_capacity(tree.nodes().len() * HASH_SIZE);
        for n in tree.nodes() {
            nodes.extend_from_slice(n);
        }
        (MerkleAttr::from_tree(&tree), nodes)
    }

    fn record(version: u64, signed_root: [u8; HASH_SIZE]) -> ProvenanceRecord {
        ProvenanceRecord {
            version,
            signed_root,
            hybrid_sig: Vec::new(),
            timestamp: version,
            snapshot_ref: String::from("snap"),
        }
    }

    /// Verifier accepting only versions in an allow-list.
    struct AllowList(Vec<u64>);
    impl SignatureVerifier for AllowList {
        fn verify(&self, r: &ProvenanceRecord) -> bool {
            self.0.contains(&r.version)
        }
    }

    #[test]
    fn last_known_good_skips_unverifiable_newer_versions() {
        let mut j = ProvenanceJournal::new();
        j.append(record(1, [1; 32])).unwrap();
        j.append(record(2, [2; 32])).unwrap();
        j.append(record(3, [3; 32])).unwrap();

        // Version 3's signature does NOT verify; 1 and 2 do. Last known-good = 2.
        let verifier = AllowList(vec![1, 2]);
        let rec = select_restore_record(&j, RestoreTarget::LastKnownGood, &verifier).unwrap();
        assert_eq!(rec.version, 2);
    }

    #[test]
    fn last_known_good_errors_when_nothing_verifies() {
        let mut j = ProvenanceJournal::new();
        j.append(record(1, [1; 32])).unwrap();
        let verifier = AllowList(vec![]); // nothing verifies
        assert!(matches!(
            select_restore_record(&j, RestoreTarget::LastKnownGood, &verifier),
            Err(RestoreError::NoKnownGoodVersion)
        ));
    }

    #[test]
    fn explicit_version_requires_journaled_record_and_valid_signature() {
        let mut j = ProvenanceJournal::new();
        j.append(record(5, [5; 32])).unwrap();

        // Not journaled.
        assert!(matches!(
            select_restore_record(&j, RestoreTarget::Version(4), &AllowList(vec![4])),
            Err(RestoreError::NoJournaledRecord { version: 4 })
        ));

        // Journaled but signature fails.
        assert!(matches!(
            select_restore_record(&j, RestoreTarget::Version(5), &AllowList(vec![])),
            Err(RestoreError::SignatureInvalid { version: 5 })
        ));

        // Journaled and signature verifies.
        let rec =
            select_restore_record(&j, RestoreTarget::Version(5), &AllowList(vec![5])).unwrap();
        assert_eq!(rec.version, 5);
    }

    #[test]
    fn dataset_gate_accepts_matching_root() {
        let chunks = vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()];
        let (attr, nodes) = dataset_for(&chunks);
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let ds = Dataset::from_owned(attr.clone(), nodes, refs);

        let rec = record(1, attr.root); // signed root matches the dataset
        assert!(verify_restored_dataset(&rec, &ds).is_ok());
    }

    #[test]
    fn dataset_gate_rejects_root_mismatch() {
        let chunks = vec![b"a".to_vec(), b"b".to_vec()];
        let (attr, nodes) = dataset_for(&chunks);
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let ds = Dataset::from_owned(attr, nodes, refs);

        // Journaled signed root is something else entirely.
        let rec = record(1, [0xEE; 32]);
        assert!(matches!(
            verify_restored_dataset(&rec, &ds),
            Err(RestoreError::RootMismatch { version: 1 })
        ));
    }

    #[test]
    fn dataset_gate_rejects_tampered_data() {
        let chunks = vec![b"a".to_vec(), b"b".to_vec()];
        let (attr, nodes) = dataset_for(&chunks);
        let signed_root = attr.root;

        // Tamper: swap chunk contents so leaf hashes no longer match the tree.
        let tampered = [b"X".to_vec(), b"b".to_vec()];
        let refs: Vec<&[u8]> = tampered.iter().map(|c| c.as_slice()).collect();
        let ds = Dataset::from_owned(attr, nodes, refs);

        let rec = record(1, signed_root);
        assert!(matches!(
            verify_restored_dataset(&rec, &ds),
            Err(RestoreError::DatasetVerificationFailed { version: 1, .. })
        ));
    }

    #[test]
    fn restore_to_version_runs_apply_only_after_signature_gate() {
        let mut j = ProvenanceJournal::new();
        j.append(record(1, [1; 32])).unwrap();
        j.append(record(2, [2; 32])).unwrap();

        // Signature for the requested version fails -> apply must NOT run.
        let mut applied = false;
        let res = restore_to_version(&j, RestoreTarget::Version(2), &AllowList(vec![1]), |_r| {
            applied = true;
            Ok(())
        });
        assert!(matches!(
            res,
            Err(RestoreError::SignatureInvalid { version: 2 })
        ));
        assert!(!applied, "apply ran despite the signature gate failing");

        // Happy path: last known-good = 2, apply runs, restored version returned.
        let mut applied_version = None;
        let v = restore_to_version(
            &j,
            RestoreTarget::LastKnownGood,
            &AllowList(vec![1, 2]),
            |r| {
                applied_version = Some(r.version);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(v, 2);
        assert_eq!(applied_version, Some(2));
    }
}
