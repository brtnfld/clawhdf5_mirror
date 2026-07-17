//! Write-side helpers for VDS (Virtual Dataset Source) mapping serialization.
//!
//! [`serialize_vds_mappings`] produces the byte blob stored in a global heap
//! object and referenced from a Data Layout v4 class=3 (Virtual) message.
//! Its output is byte-compatible with what [`crate::data_layout::parse_vds_mappings`]
//! can parse back.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::data_layout::VdsMapping;

/// Serialize a slice of [`VdsMapping`]s into the global-heap object byte format.
///
/// # Layout
///
/// ```text
/// version(1) · nused(length_size, LE) · entry[nused]
/// ```
///
/// Each entry:
/// - **version 0** (at least one external source file): null-terminated source
///   file name, then null-terminated source dataset name, then source selection
///   bytes (self-describing), then virtual selection bytes (self-describing).
/// - **version 1** (all same-file): a single `0x04` marker byte in place of the
///   file name, then null-terminated source dataset name, then the two
///   self-describing selection blobs.
///
/// The selections are written as-is from [`VdsMapping::source_selection`] and
/// [`VdsMapping::virtual_selection`]; the caller is responsible for ensuring
/// they are valid serialized `H5S` selections that [`crate::selection::Selection::decode_serialized`]
/// can consume.
///
/// `length_size` must be 2, 4, or 8; any other value falls back to 8.
pub fn serialize_vds_mappings(mappings: &[VdsMapping], length_size: u8) -> Vec<u8> {
    let mut buf = Vec::new();

    // Block version 0 = at least one external (non-same-file) source;
    // block version 1 = all sources are in the same file (source_file == ".").
    let all_same_file = mappings
        .iter()
        .all(|m| m.source_file.is_empty() || m.source_file == ".");
    let version: u8 = if all_same_file { 1 } else { 0 };
    buf.push(version);

    // nused: number of mappings, encoded as little-endian `length_size` bytes.
    write_length(&mut buf, mappings.len() as u64, length_size);

    for m in mappings {
        if version == 0 {
            // External file: write the file name as a null-terminated string.
            buf.extend_from_slice(m.source_file.as_bytes());
            buf.push(0u8);
        } else {
            // Same-file: the marker byte that `parse_vds_mappings` recognises as
            // the same-file sentinel (0x04).
            buf.push(0x04u8);
        }

        // Source dataset path: null-terminated string.
        buf.extend_from_slice(m.source_dataset.as_bytes());
        buf.push(0u8);

        // Source selection: raw self-describing bytes (no separate length prefix).
        buf.extend_from_slice(&m.source_selection);

        // Virtual selection: raw self-describing bytes (no separate length prefix).
        buf.extend_from_slice(&m.virtual_selection);
    }

    buf
}

/// Encode `val` as a little-endian integer of `size` bytes and push it into
/// `buf`.  Supported sizes: 2, 4, 8.  Any other value falls back to 8 bytes.
pub(crate) fn write_length(buf: &mut Vec<u8>, val: u64, size: u8) {
    match size {
        2 => buf.extend_from_slice(&(val as u16).to_le_bytes()),
        4 => buf.extend_from_slice(&(val as u32).to_le_bytes()),
        _ => buf.extend_from_slice(&val.to_le_bytes()),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_layout::parse_vds_mappings;

    /// A minimal, valid serialized H5S ALL selection (type=3, 16 bytes).
    ///
    /// Layout: type(4 LE) + version(4 LE) + reserved(4) + length(4) = 16 bytes.
    /// `decode_serialized` consumes exactly 16 bytes for ALL/NONE.
    fn all_sel() -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&3u32.to_le_bytes()); // type = H5S_SEL_ALL (3)
        v.extend_from_slice(&1u32.to_le_bytes()); // version = 1
        v.extend_from_slice(&[0u8; 4]); // reserved
        v.extend_from_slice(&[0u8; 4]); // length field (unused for ALL)
        v
    }

    #[test]
    fn roundtrip_same_file_two_mappings() {
        let sel = all_sel();
        let mappings = vec![
            VdsMapping {
                source_file: ".".into(),
                source_dataset: "/src_a".into(),
                source_selection: sel.clone(),
                virtual_selection: sel.clone(),
            },
            VdsMapping {
                source_file: ".".into(),
                source_dataset: "/src_b".into(),
                source_selection: sel.clone(),
                virtual_selection: sel.clone(),
            },
        ];
        let bytes = serialize_vds_mappings(&mappings, 8);
        // Block version must be 1 (same-file).
        assert_eq!(bytes[0], 1u8);
        let parsed = parse_vds_mappings(&bytes, 8).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].source_file, ".");
        assert_eq!(parsed[0].source_dataset, "/src_a");
        assert_eq!(parsed[1].source_file, ".");
        assert_eq!(parsed[1].source_dataset, "/src_b");
    }

    #[test]
    fn roundtrip_external_file_mapping() {
        let sel = all_sel();
        let mappings = vec![VdsMapping {
            source_file: "source.h5".into(),
            source_dataset: "/data".into(),
            source_selection: sel.clone(),
            virtual_selection: sel.clone(),
        }];
        let bytes = serialize_vds_mappings(&mappings, 8);
        // Block version must be 0 (external file present).
        assert_eq!(bytes[0], 0u8);
        let parsed = parse_vds_mappings(&bytes, 8).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].source_file, "source.h5");
        assert_eq!(parsed[0].source_dataset, "/data");
        assert_eq!(
            parsed[0].source_selection, sel,
            "source selection bytes must survive round-trip"
        );
        assert_eq!(
            parsed[0].virtual_selection, sel,
            "virtual selection bytes must survive round-trip"
        );
    }

    #[test]
    fn empty_mappings_roundtrip() {
        // Empty slice: version 1 (vacuously all same-file), nused=0.
        let bytes = serialize_vds_mappings(&[], 8);
        let parsed = parse_vds_mappings(&bytes, 8).unwrap();
        assert!(parsed.is_empty());
    }

    #[test]
    fn roundtrip_empty_source_file_treated_as_same_file() {
        // An empty source_file string is also treated as same-file (version 1).
        let sel = all_sel();
        let mappings = vec![VdsMapping {
            source_file: String::new(),
            source_dataset: "/ds".into(),
            source_selection: sel.clone(),
            virtual_selection: sel.clone(),
        }];
        let bytes = serialize_vds_mappings(&mappings, 8);
        assert_eq!(bytes[0], 1u8);
        let parsed = parse_vds_mappings(&bytes, 8).unwrap();
        assert_eq!(parsed.len(), 1);
        // parse_vds_mappings turns the 0x04 marker into "."
        assert_eq!(parsed[0].source_file, ".");
    }

    #[test]
    fn roundtrip_length_size_4() {
        let sel = all_sel();
        let mappings = vec![VdsMapping {
            source_file: ".".into(),
            source_dataset: "/x".into(),
            source_selection: sel.clone(),
            virtual_selection: sel.clone(),
        }];
        let bytes = serialize_vds_mappings(&mappings, 4);
        let parsed = parse_vds_mappings(&bytes, 4).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].source_dataset, "/x");
    }
}
