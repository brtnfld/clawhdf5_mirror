//! Regression tests for the 2026-10 red-team finding that object-header
//! continuation parsing could re-visit a block. A v1 depth limit of 32 did
//! not bound the work: blocks that each hold two continuation messages
//! pointing at the next level form a DAG whose path count doubles per level.
//! Before the fix a 1,232-byte file with 22 such levels parsed 8,388,607
//! messages in about a second, and each further level doubled time and
//! memory. Every continuation block must now be parsed at most once.

use clawhdf5_format::error::FormatError;
use clawhdf5_format::object_header::ObjectHeader;

const CONTINUATION: u16 = 0x10;

fn cont_msg(target: u64, len: u64) -> Vec<u8> {
    let mut m = Vec::new();
    m.extend_from_slice(&CONTINUATION.to_le_bytes());
    m.extend_from_slice(&16u16.to_le_bytes());
    m.extend_from_slice(&[0, 0, 0, 0]);
    m.extend_from_slice(&target.to_le_bytes());
    m.extend_from_slice(&len.to_le_bytes());
    m
}

/// A v1 header whose single message continues into a chain of `levels`
/// 48-byte blocks; each block holds `fanout` continuation messages pointing
/// at the next one (24 bytes each, so fanout <= 2), and the last is empty.
fn v1_chain(levels: usize, fanout: usize) -> Vec<u8> {
    let block = |k: usize| 64 + 48 * k;
    let mut f = vec![0u8; block(levels + 1) + 64];
    f[0] = 1;
    f[2..4].copy_from_slice(&1u16.to_le_bytes());
    f[4..8].copy_from_slice(&1u32.to_le_bytes());
    f[8..12].copy_from_slice(&24u32.to_le_bytes());
    f[16..40].copy_from_slice(&cont_msg(block(0) as u64, 48));
    for k in 0..levels {
        let next = block(k + 1) as u64;
        for j in 0..fanout {
            let at = block(k) + 24 * j;
            f[at..at + 24].copy_from_slice(&cont_msg(next, 48));
        }
    }
    f
}

#[test]
fn v1_continuation_dag_is_rejected_instead_of_exploding() {
    let f = v1_chain(22, 2);
    let t = std::time::Instant::now();
    let err = ObjectHeader::parse(&f, 0, 8, 8).unwrap_err();
    assert!(matches!(err, FormatError::NestingDepthExceeded), "got {err:?}");
    assert!(t.elapsed() < std::time::Duration::from_millis(100), "took {:?}", t.elapsed());
}

#[test]
fn v1_linear_continuation_chain_still_parses() {
    // One continuation per block is how real files are laid out.
    let f = v1_chain(10, 1);
    let hdr = ObjectHeader::parse(&f, 0, 8, 8).expect("a linear chain is valid");
    // The header's own message plus one per block.
    assert_eq!(hdr.messages.len(), 11);
}
