//! Append-only provenance journal for Merkle-protected datasets (P2.2b step 3).
//!
//! The journal records one [`ProvenanceRecord`] per committed mutation:
//! `(version, signed_root, hybrid_sig, timestamp, snapshot_ref)`. It is the
//! durable history a verifier consults to recover after detected tampering —
//! `restore_to_version` (P2.2b step 4) reverts to the snapshot named by a
//! journaled record and re-checks the dataset against that record's signed root.
//!
//! # Decoupling
//!
//! `clawhdf5-format` does not depend on the P2.1 signing crate or on
//! `clawhdf5-agent`, so:
//!
//! - `hybrid_sig` is stored as **opaque bytes** (the serialized P2.1
//!   `HybridSignature`); the journal never interprets it.
//! - `snapshot_ref` is an **opaque UTF-8 handle** to a full-file snapshot — in
//!   practice the path returned by `clawhdf5-agent`'s `snapshot_file`. The
//!   binding rule is enforced structurally here
//!   ([`ProvenanceJournal::is_valid_rollback_target`]): *a snapshot with no
//!   journaled record is not a valid rollback target.*
//!
//! # On-disk format
//!
//! The journal serializes to a self-describing byte blob suitable for a
//! `/_merkle/journal` dataset or a sidecar file:
//!
//! ```text
//! [magic "MJRN" : 4][version : 1][reserved : 3, must be zero][record_count : u32 BE]
//! then record_count records, each:
//!   [version : u64 BE][signed_root : 32][timestamp : u64 BE]
//!   [sig_len : u32 BE][hybrid_sig : sig_len]
//!   [ref_len : u32 BE][snapshot_ref : ref_len UTF-8]
//! ```
//!
//! [`ProvenanceJournal::unpack`] validates every length against the remaining
//! buffer with checked arithmetic and never panics on malformed input.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use crate::merkle::{HASH_SIZE, MerkleError};

/// Magic bytes identifying a serialized provenance journal.
pub const MERKLE_JOURNAL_MAGIC: [u8; 4] = *b"MJRN";

/// On-disk journal format version.
pub const MERKLE_JOURNAL_VERSION: u8 = 1;

/// Size of the journal header preceding the record count: magic(4) + version(1)
/// + reserved(3) + record_count(4).
pub const MERKLE_JOURNAL_HEADER_SIZE: usize = 12;

/// Attribute / dataset name for the persisted provenance journal.
pub const MERKLE_JOURNAL_ATTR_NAME: &str = "merkle_journal";

/// Minimum serialized size of one record: version(8) + root(32) + timestamp(8)
/// + sig_len(4) + ref_len(4), with zero-length sig and ref.
const MIN_RECORD_SIZE: usize = 8 + HASH_SIZE + 8 + 4 + 4;

/// One provenance record: a version certified by a signed Merkle root and tied
/// to a full-file snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvenanceRecord {
    /// Dataset version this record certifies (monotonically increasing).
    pub version: u64,
    /// The Merkle root that was signed for this version.
    pub signed_root: [u8; HASH_SIZE],
    /// Serialized P2.1 hybrid signature over the canonical payload (opaque).
    pub hybrid_sig: Vec<u8>,
    /// Commit timestamp (seconds since the Unix epoch, matching the signed
    /// payload's `τ`). Opaque to the journal.
    pub timestamp: u64,
    /// Opaque handle to the full-file snapshot certifying this version — the
    /// path returned by `clawhdf5-agent`'s `snapshot_file`.
    pub snapshot_ref: String,
}

/// Append-only log of [`ProvenanceRecord`]s, ordered by strictly increasing
/// version.
#[derive(Debug, Clone, Default)]
pub struct ProvenanceJournal {
    records: Vec<ProvenanceRecord>,
}

impl ProvenanceJournal {
    /// Create an empty journal.
    #[must_use]
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }

    /// Append a record, enforcing the append-only, strictly-increasing-version
    /// invariant.
    ///
    /// # Errors
    ///
    /// - [`MerkleError::JournalNonMonotonic`] if `record.version` is not
    ///   strictly greater than the last appended version (append-only history
    ///   cannot regress or duplicate a version — one record per commit). This
    ///   is caller misuse, not the verifier-side T4 signal
    ///   ([`MerkleError::VersionRollback`]).
    /// - [`MerkleError::JournalCorrupt`] if the signature or snapshot reference
    ///   is too large to serialize (length exceeds `u32`).
    pub fn append(&mut self, record: ProvenanceRecord) -> Result<(), MerkleError> {
        if let Some(last) = self.records.last()
            && record.version <= last.version
        {
            return Err(MerkleError::JournalNonMonotonic {
                appended: record.version,
                last: last.version,
            });
        }
        if record.hybrid_sig.len() > u32::MAX as usize
            || record.snapshot_ref.len() > u32::MAX as usize
        {
            return Err(MerkleError::JournalCorrupt);
        }
        self.records.push(record);
        Ok(())
    }

    /// All records, oldest first.
    #[must_use]
    pub fn records(&self) -> &[ProvenanceRecord] {
        &self.records
    }

    /// The most recently appended (highest-version) record, if any.
    #[must_use]
    pub fn latest(&self) -> Option<&ProvenanceRecord> {
        self.records.last()
    }

    /// The record certifying exactly `version`, if journaled.
    #[must_use]
    pub fn record_for_version(&self, version: u64) -> Option<&ProvenanceRecord> {
        // Records are sorted by strictly increasing version.
        self.records
            .binary_search_by(|r| r.version.cmp(&version))
            .ok()
            .map(|idx| &self.records[idx])
    }

    /// The record whose snapshot matches `snapshot_ref`, if journaled.
    #[must_use]
    pub fn record_for_snapshot(&self, snapshot_ref: &str) -> Option<&ProvenanceRecord> {
        self.records.iter().find(|r| r.snapshot_ref == snapshot_ref)
    }

    /// Whether `snapshot_ref` is a valid rollback target.
    ///
    /// P2.2b step 3: *a snapshot with no journaled root is not a valid rollback
    /// target.* A snapshot qualifies only if the journal has a record naming it
    /// (which therefore also carries the signed root and signature certifying
    /// that state).
    #[must_use]
    pub fn is_valid_rollback_target(&self, snapshot_ref: &str) -> bool {
        self.record_for_snapshot(snapshot_ref).is_some()
    }

    /// Number of records in the journal.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the journal has no records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Serialize the journal to its self-describing on-disk byte layout.
    #[must_use]
    pub fn pack(&self) -> Vec<u8> {
        let mut buf =
            Vec::with_capacity(MERKLE_JOURNAL_HEADER_SIZE + self.records.len() * MIN_RECORD_SIZE);
        buf.extend_from_slice(&MERKLE_JOURNAL_MAGIC);
        buf.push(MERKLE_JOURNAL_VERSION);
        buf.extend_from_slice(&[0u8; 3]); // reserved
        #[expect(
            clippy::cast_possible_truncation,
            reason = "record count bounded by memory; a journal never approaches 4 billion records"
        )]
        buf.extend_from_slice(&(self.records.len() as u32).to_be_bytes());

        for r in &self.records {
            buf.extend_from_slice(&r.version.to_be_bytes());
            buf.extend_from_slice(&r.signed_root);
            buf.extend_from_slice(&r.timestamp.to_be_bytes());
            // Lengths validated to fit u32 in `append`.
            #[expect(
                clippy::cast_possible_truncation,
                reason = "sig/ref lengths validated <= u32::MAX in append()"
            )]
            {
                buf.extend_from_slice(&(r.hybrid_sig.len() as u32).to_be_bytes());
                buf.extend_from_slice(&r.hybrid_sig);
                buf.extend_from_slice(&(r.snapshot_ref.len() as u32).to_be_bytes());
                buf.extend_from_slice(r.snapshot_ref.as_bytes());
            }
        }
        buf
    }

    /// Parse a journal from its on-disk byte layout.
    ///
    /// Every length field is checked against the remaining buffer with checked
    /// arithmetic, so malformed or truncated input yields
    /// [`MerkleError::JournalCorrupt`] rather than a panic.
    ///
    /// # Errors
    ///
    /// - [`MerkleError::JournalUnsupportedVersion`] if the magic matches but
    ///   the format-version byte is one this build does not understand.
    /// - [`MerkleError::JournalCorrupt`] on bad magic, nonzero reserved
    ///   header bytes, a length that runs past the buffer, non-monotonic
    ///   record versions, invalid UTF-8 in a snapshot reference, or trailing
    ///   bytes.
    pub fn unpack(data: &[u8]) -> Result<Self, MerkleError> {
        if data.len() < MERKLE_JOURNAL_HEADER_SIZE {
            return Err(MerkleError::JournalCorrupt);
        }
        if data[0..4] != MERKLE_JOURNAL_MAGIC {
            return Err(MerkleError::JournalCorrupt);
        }
        if data[4] != MERKLE_JOURNAL_VERSION {
            return Err(MerkleError::JournalUnsupportedVersion { found: data[4] });
        }
        // Reserved bytes must be zero, matching this parser's otherwise-strict
        // posture (trailing bytes rejected, monotonic versions enforced). This
        // keeps them genuinely available for future format extensions: an old
        // reader refuses rather than silently ignoring semantics it predates.
        if data[5..8] != [0, 0, 0] {
            return Err(MerkleError::JournalCorrupt);
        }

        let count = u32::from_be_bytes([data[8], data[9], data[10], data[11]]) as usize;

        // Reject an implausible count before allocating: each record needs at
        // least MIN_RECORD_SIZE bytes.
        let body_len = data.len() - MERKLE_JOURNAL_HEADER_SIZE;
        if count > body_len / MIN_RECORD_SIZE {
            return Err(MerkleError::JournalCorrupt);
        }

        let mut records = Vec::with_capacity(count);
        let mut off = MERKLE_JOURNAL_HEADER_SIZE;
        let mut prev_version: Option<u64> = None;

        for _ in 0..count {
            let version = read_u64(data, &mut off)?;
            let signed_root = read_array32(data, &mut off)?;
            let timestamp = read_u64(data, &mut off)?;
            let hybrid_sig = read_var_bytes(data, &mut off)?.to_vec();
            let ref_bytes = read_var_bytes(data, &mut off)?;
            let snapshot_ref =
                String::from_utf8(ref_bytes.to_vec()).map_err(|_| MerkleError::JournalCorrupt)?;

            // Enforce strictly increasing versions.
            if let Some(prev) = prev_version
                && version <= prev
            {
                return Err(MerkleError::JournalCorrupt);
            }
            prev_version = Some(version);

            records.push(ProvenanceRecord {
                version,
                signed_root,
                hybrid_sig,
                timestamp,
                snapshot_ref,
            });
        }

        // No trailing bytes allowed.
        if off != data.len() {
            return Err(MerkleError::JournalCorrupt);
        }

        Ok(Self { records })
    }
}

/// Read a big-endian `u64`, advancing `off`.
fn read_u64(data: &[u8], off: &mut usize) -> Result<u64, MerkleError> {
    let end = off.checked_add(8).ok_or(MerkleError::JournalCorrupt)?;
    let slice = data.get(*off..end).ok_or(MerkleError::JournalCorrupt)?;
    let arr: [u8; 8] = slice.try_into().map_err(|_| MerkleError::JournalCorrupt)?;
    *off = end;
    Ok(u64::from_be_bytes(arr))
}

/// Read a fixed 32-byte array, advancing `off`.
fn read_array32(data: &[u8], off: &mut usize) -> Result<[u8; HASH_SIZE], MerkleError> {
    let end = off
        .checked_add(HASH_SIZE)
        .ok_or(MerkleError::JournalCorrupt)?;
    let slice = data.get(*off..end).ok_or(MerkleError::JournalCorrupt)?;
    let arr: [u8; HASH_SIZE] = slice.try_into().map_err(|_| MerkleError::JournalCorrupt)?;
    *off = end;
    Ok(arr)
}

/// Read a `u32`-length-prefixed byte slice, advancing `off`.
fn read_var_bytes<'a>(data: &'a [u8], off: &mut usize) -> Result<&'a [u8], MerkleError> {
    let len_end = off.checked_add(4).ok_or(MerkleError::JournalCorrupt)?;
    let len_slice = data.get(*off..len_end).ok_or(MerkleError::JournalCorrupt)?;
    let len = u32::from_be_bytes([len_slice[0], len_slice[1], len_slice[2], len_slice[3]]) as usize;
    let data_end = len_end
        .checked_add(len)
        .ok_or(MerkleError::JournalCorrupt)?;
    let bytes = data
        .get(len_end..data_end)
        .ok_or(MerkleError::JournalCorrupt)?;
    *off = data_end;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(version: u64, snapshot_ref: &str) -> ProvenanceRecord {
        ProvenanceRecord {
            version,
            signed_root: [version as u8; HASH_SIZE],
            hybrid_sig: vec![0xAB; 3374], // realistic hybrid-sig size
            timestamp: 1_700_000_000 + version,
            snapshot_ref: String::from(snapshot_ref),
        }
    }

    #[test]
    fn append_enforces_strictly_increasing_versions() {
        let mut j = ProvenanceJournal::new();
        j.append(rec(1, "snap-1")).unwrap();
        j.append(rec(2, "snap-2")).unwrap();

        // Equal version is rejected (one record per commit).
        assert!(matches!(
            j.append(rec(2, "dup")),
            Err(MerkleError::JournalNonMonotonic {
                appended: 2,
                last: 2
            })
        ));
        // Lower version is rejected.
        assert!(matches!(
            j.append(rec(1, "older")),
            Err(MerkleError::JournalNonMonotonic {
                appended: 1,
                last: 2
            })
        ));
        assert_eq!(j.len(), 2);
    }

    #[test]
    fn lookup_by_version_and_snapshot() {
        let mut j = ProvenanceJournal::new();
        j.append(rec(1, "snap-1")).unwrap();
        j.append(rec(5, "snap-5")).unwrap();
        j.append(rec(9, "snap-9")).unwrap();

        assert_eq!(j.record_for_version(5).unwrap().snapshot_ref, "snap-5");
        assert!(j.record_for_version(4).is_none());
        assert_eq!(j.latest().unwrap().version, 9);

        assert_eq!(j.record_for_snapshot("snap-9").unwrap().version, 9);
        assert!(j.record_for_snapshot("snap-4").is_none());
    }

    #[test]
    fn snapshot_without_journaled_root_is_not_a_valid_rollback_target() {
        let mut j = ProvenanceJournal::new();
        j.append(rec(3, "certified-snapshot")).unwrap();

        assert!(j.is_valid_rollback_target("certified-snapshot"));
        assert!(!j.is_valid_rollback_target("random-backup.h5"));
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let mut j = ProvenanceJournal::new();
        j.append(rec(1, "snap-1")).unwrap();
        j.append(rec(2, "snap-two-with-longer-name")).unwrap();
        j.append(rec(100, "")).unwrap(); // empty snapshot_ref is allowed

        let bytes = j.pack();
        let parsed = ProvenanceJournal::unpack(&bytes).unwrap();

        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed.records(), j.records());
    }

    #[test]
    fn unpack_rejects_bad_magic() {
        let mut bytes = ProvenanceJournal::new().pack();
        bytes[0] = b'X';
        assert!(matches!(
            ProvenanceJournal::unpack(&bytes),
            Err(MerkleError::JournalCorrupt)
        ));
    }

    #[test]
    fn unpack_rejects_unknown_version() {
        let mut bytes = ProvenanceJournal::new().pack();
        bytes[4] = 0xFF;
        assert!(matches!(
            ProvenanceJournal::unpack(&bytes),
            Err(MerkleError::JournalUnsupportedVersion { found: 0xFF })
        ));
    }

    #[test]
    fn unpack_rejects_nonzero_reserved_bytes() {
        for reserved_idx in 5..8 {
            let mut bytes = ProvenanceJournal::new().pack();
            bytes[reserved_idx] = 0x01;
            assert!(
                matches!(
                    ProvenanceJournal::unpack(&bytes),
                    Err(MerkleError::JournalCorrupt)
                ),
                "nonzero reserved byte at offset {reserved_idx} must be rejected"
            );
        }
    }

    #[test]
    fn unpack_rejects_truncated_and_oversized_lengths() {
        let mut j = ProvenanceJournal::new();
        j.append(rec(1, "snap-1")).unwrap();
        let good = j.pack();

        // Truncated mid-record.
        assert!(matches!(
            ProvenanceJournal::unpack(&good[..good.len() - 5]),
            Err(MerkleError::JournalCorrupt)
        ));

        // A record count far larger than the body can hold is rejected before
        // allocating (guards against an allocation-size DoS / overflow).
        let mut hostile = good.clone();
        hostile[8..12].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(matches!(
            ProvenanceJournal::unpack(&hostile),
            Err(MerkleError::JournalCorrupt)
        ));
    }

    #[test]
    fn unpack_rejects_non_monotonic_records() {
        // Hand-craft a journal whose second record has a lower version.
        let mut j = ProvenanceJournal::new();
        j.append(rec(2, "a")).unwrap();
        let mut bytes = j.pack();
        // Append a second record with version 1 by re-packing manually: easiest
        // is to build two-record bytes from two single-record packs.
        let mut j2 = ProvenanceJournal::new();
        j2.append(rec(1, "b")).unwrap();
        let second = j2.pack();
        // Splice: bump count to 2 and append the second record body.
        bytes[8..12].copy_from_slice(&2u32.to_be_bytes());
        bytes.extend_from_slice(&second[MERKLE_JOURNAL_HEADER_SIZE..]);

        assert!(matches!(
            ProvenanceJournal::unpack(&bytes),
            Err(MerkleError::JournalCorrupt)
        ));
    }

    #[test]
    fn unpack_never_panics_on_arbitrary_bytes() {
        // Deterministic byte sweep standing in for the fuzz_merkle_journal
        // cargo-fuzz target so the never-panic property is exercised under
        // `cargo test` in CI (same pattern as version_wal's sweep). A simple
        // xorshift PRNG generates random and semi-structured inputs; the test
        // passing (i.e. not aborting) is the assertion.
        let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for _ in 0..10_000 {
            let len = (next() % 200) as usize;
            let mut bytes = Vec::with_capacity(len);
            for _ in 0..len {
                bytes.push((next() & 0xff) as u8);
            }

            // Half the time, plant a valid header (magic, format version,
            // zeroed reserved bytes) so the sweep reaches past the header
            // into record parsing, and sometimes a hostile record count on
            // top. The other half leaves the header bytes random, covering
            // the bad-magic / bad-version / nonzero-reserved rejections.
            if len >= MERKLE_JOURNAL_HEADER_SIZE && next() & 1 == 0 {
                bytes[0..4].copy_from_slice(&MERKLE_JOURNAL_MAGIC);
                bytes[4] = MERKLE_JOURNAL_VERSION;
                bytes[5..8].copy_from_slice(&[0, 0, 0]);
                if next() & 1 == 0 {
                    let planted = (1u64 << (next() % 33)) as u32;
                    bytes[8..12].copy_from_slice(&planted.to_be_bytes());
                }
            }

            // Must return Ok/Err, never panic; a parsed journal must
            // round-trip through its canonical encoding.
            if let Ok(journal) = ProvenanceJournal::unpack(&bytes) {
                let reparsed = ProvenanceJournal::unpack(&journal.pack()).unwrap();
                assert_eq!(reparsed.records(), journal.records());
            }
        }
    }

    #[test]
    fn unpack_rejects_trailing_bytes() {
        let mut j = ProvenanceJournal::new();
        j.append(rec(1, "snap-1")).unwrap();
        let mut bytes = j.pack();
        bytes.push(0xFF);
        assert!(matches!(
            ProvenanceJournal::unpack(&bytes),
            Err(MerkleError::JournalCorrupt)
        ));
    }
}
