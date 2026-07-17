//! Test-dataset fixtures for the attack harness.
//!
//! P2.4 calls for the NOAA and EQSIM datasets. This repo does not commit real
//! satellite/HPC data (large binary files don't belong in source control), so
//! the harness works two ways for *each* dataset:
//!
//! - **Real dataset**: point it at a real, pre-downloaded file via `--file
//!   <path>` (NOAA GOES-18 ABI L1b, see `docs/NOAA_DATA.md`) or `--eqsim-file
//!   <path>` (an EQSIM HDF5 output file). The harness reads the dataset's
//!   *actual* on-disk HDF5 chunk layout (address + compressed size per chunk,
//!   via `clawhdf5_format::chunked_read`), so tampering attacks flip real
//!   bytes inside a real, filtered (shuffle+deflate) HDF5 chunk.
//! - **Synthetic fallback** (default, no args): a small in-process dataset
//!   with a representative chunked shape for each of the two datasets, so
//!   `cargo run` alone reproduces the full attack matrix on both datasets
//!   with no network access, no manual download step, and nothing large
//!   committed to the repo. NOAA's synthetic shape mimics a GOES-18 tile
//!   (many small chunks); EQSIM's mimics an HPC checkpoint/restart write
//!   (fewer, larger chunks) — see `synthetic_noaa_dataset`/
//!   `synthetic_eqsim_dataset`. Labels always say `(synthetic)` when this
//!   path is taken, so the committed `attack-results/matrix.csv` never
//!   implies real data was used unless it genuinely was.
//!
//! Either way, the harness treats the loaded bytes as an opaque byte buffer
//! plus a chunk address/size table — attacks tamper it with raw
//! `std::fs`/slice writes, never through any ClawHDF5 write API.

use clawhdf5_format::chunked_read::{ChunkInfo, collect_chunk_info};
use clawhdf5_format::data_layout::DataLayout;
use clawhdf5_format::extensible_array::{ExtensibleArrayHeader, read_extensible_array_chunks};
use clawhdf5_format::fixed_array::{FixedArrayHeader, read_fixed_array_chunks};
use clawhdf5_format::group_v2;
use clawhdf5_format::message_type::MessageType;
use clawhdf5_format::object_header::ObjectHeader;
use clawhdf5_format::signature;
use clawhdf5_format::superblock::Superblock;

/// A dataset the harness runs attacks against: raw file bytes plus the byte
/// range of each physical chunk within them.
pub struct HarnessDataset {
    /// Human-readable dataset label for the results CSV, e.g.
    /// `NOAA-GOES18`, `NOAA-GOES18 (synthetic)`, `EQSIM`, or
    /// `EQSIM (synthetic)`. Always carries the `(synthetic)` suffix when the
    /// bytes are the in-process fallback rather than a real downloaded file,
    /// so the label itself discloses whether real data was used.
    pub label: &'static str,
    /// The complete file bytes.
    pub bytes: Vec<u8>,
    /// `(start, end)` byte ranges of each chunk, in chunk-index order.
    pub chunk_ranges: Vec<(usize, usize)>,
}

impl HarnessDataset {
    /// Number of chunks.
    pub fn chunk_count(&self) -> usize {
        self.chunk_ranges.len()
    }

    /// Slice out chunk `idx`'s bytes from `bytes` (or a caller-supplied,
    /// possibly-tampered, same-length byte buffer).
    pub fn chunk<'a>(&self, bytes: &'a [u8], idx: usize) -> &'a [u8] {
        let (start, end) = self.chunk_ranges[idx];
        &bytes[start..end]
    }

    /// All chunks as a `Vec<&[u8]>`, in order — the shape `MerkleTree::from_chunks`
    /// and `Dataset::from_owned` expect.
    pub fn all_chunks<'a>(&self, bytes: &'a [u8]) -> Vec<&'a [u8]> {
        (0..self.chunk_count())
            .map(|i| self.chunk(bytes, i))
            .collect()
    }
}

/// Load a real HDF5 file (NOAA GOES-18 ABI L1b, an EQSIM output file, or any
/// other chunked HDF5/NetCDF4 file) and extract the named dataset's (e.g.
/// `"Rad"` for GOES-18) real, on-disk chunk layout.
///
/// `label` is attached to the returned [`HarnessDataset`] verbatim for the
/// results CSV — callers pass a real-data label (e.g. `"NOAA-GOES18"` or
/// `"EQSIM"`), never a `(synthetic)`-suffixed one, since this function only
/// ever returns real, on-disk bytes.
///
/// Mirrors `clawhdf5/examples/goes18_chunk_test.rs`'s chunk-index dispatch so
/// it handles whichever chunk-index type (B-tree v1, single chunk, fixed
/// array, extensible array) the specific downloaded file happens to use.
///
/// # Errors
///
/// Returns a boxed error if the file isn't valid HDF5, the dataset is absent,
/// or its layout isn't chunked.
pub fn load_real_dataset(
    path: &std::path::Path,
    dataset_name: &str,
    label: &'static str,
) -> Result<HarnessDataset, Box<dyn std::error::Error>> {
    // The dataset's true element-space shape (needed by the fixed/extensible
    // array chunk-index readers below) lives in the Dataspace message; reuse
    // the high-level reader rather than re-parsing it by hand.
    let shape: Vec<u64> = clawhdf5::File::open(path)?.dataset(dataset_name)?.shape()?;

    let bytes = std::fs::read(path)?;
    let sig_offset = signature::find_signature(&bytes)?;
    let superblock = Superblock::parse(&bytes, sig_offset)?;

    let addr = group_v2::resolve_path_any(&bytes, &superblock, dataset_name)?;
    let header = ObjectHeader::parse(
        &bytes,
        addr as usize,
        superblock.offset_size,
        superblock.length_size,
    )?;

    let layout_msg = header
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::DataLayout)
        .ok_or("dataset has no DataLayout message")?;
    let layout = DataLayout::parse(
        &layout_msg.data,
        superblock.offset_size,
        superblock.length_size,
    )?;

    let chunks: Vec<ChunkInfo> = match &layout {
        DataLayout::Chunked {
            chunk_dimensions,
            btree_address,
            version,
            chunk_index_type,
            ..
        } => {
            let addr = btree_address.ok_or("chunked layout has no chunk-index address")?;
            // `chunk_dimensions` is `[spatial dims..., element size]`, so a
            // chunked layout needs at least one entry. A file with a
            // DataLayout message claiming `dimensionality = 0` parses to an
            // empty `chunk_dimensions` (the byte is read unchecked in
            // `DataLayout::parse`); reject it here with a proper error
            // instead of clamping the rank and slicing out of bounds.
            if chunk_dimensions.is_empty() {
                return Err(
                    "chunked layout has empty chunk_dimensions (dimensionality = 0)".into(),
                );
            }
            let rank = chunk_dimensions.len() - 1;
            // The `is_empty()` guard above already guarantees `last()` is
            // `Some`; no fallback value is ever actually used here.
            let elem_size = *chunk_dimensions.last().unwrap() as usize;
            let spatial_dims: Vec<u32> = chunk_dimensions[..rank].to_vec();

            match (*version, *chunk_index_type) {
                (3, _) => collect_chunk_info(
                    &bytes,
                    addr,
                    rank + 1,
                    superblock.offset_size,
                    superblock.length_size,
                )?,
                (4, Some(3)) => {
                    let fa_header = FixedArrayHeader::parse(
                        &bytes,
                        addr as usize,
                        superblock.offset_size,
                        superblock.length_size,
                    )?;
                    read_fixed_array_chunks(
                        &bytes,
                        &fa_header,
                        &shape,
                        &spatial_dims,
                        elem_size as u32,
                        superblock.offset_size,
                        superblock.length_size,
                    )?
                }
                (4, Some(4)) => {
                    let ea_header = ExtensibleArrayHeader::parse(
                        &bytes,
                        addr as usize,
                        superblock.offset_size,
                        superblock.length_size,
                    )?;
                    read_extensible_array_chunks(
                        &bytes,
                        &ea_header,
                        &shape,
                        &spatial_dims,
                        elem_size as u32,
                        superblock.offset_size,
                        superblock.length_size,
                    )?
                }
                (4, Some(1)) => {
                    // Compute the single chunk's byte size in u64 throughout
                    // (element count * element size), only narrowing to u32
                    // at the end with a checked conversion -- the naive
                    // "narrow the product, then multiply" order used in
                    // `goes18_chunk_test.rs` silently truncates for a chunk
                    // over 4 GiB.
                    let elements: u64 = spatial_dims.iter().map(|&d| u64::from(d)).product();
                    let total_bytes = elements
                        .checked_mul(elem_size as u64)
                        .ok_or("single chunk size overflows u64")?;
                    let chunk_size = u32::try_from(total_bytes)
                        .map_err(|_| "single chunk size exceeds 4 GiB (u32::MAX bytes)")?;
                    vec![ChunkInfo {
                        chunk_size,
                        filter_mask: 0,
                        offsets: vec![0; rank],
                        address: addr,
                    }]
                }
                (v, t) => {
                    return Err(format!("unsupported chunk index: version={v}, type={t:?}").into());
                }
            }
        }
        _ => return Err("dataset is not chunked".into()),
    };

    // `c.address`/`c.chunk_size` come straight from the file's chunk index
    // (B-tree v1 / fixed array / extensible array): `collect_chunk_info` and
    // friends bounds-check the *index structure* they parse, but not that
    // the resulting (address, chunk_size) values actually fall within the
    // file. A chunk index that (falsely) claims a chunk lives beyond EOF
    // must not become an out-of-bounds range that later panics when sliced
    // in `HarnessDataset::chunk` -- validate against `bytes.len()` here,
    // with checked addition against a huge `chunk_size` overflowing `usize`.
    let chunk_ranges = chunks
        .iter()
        .map(|c| {
            let start = c.address as usize;
            let end = start
                .checked_add(c.chunk_size as usize)
                .ok_or("chunk range overflows usize (address + chunk_size)")?;
            if end > bytes.len() {
                return Err(format!(
                    "chunk range {start}..{end} exceeds file length {} bytes",
                    bytes.len()
                ));
            }
            Ok((start, end))
        })
        .collect::<Result<Vec<(usize, usize)>, String>>()?;

    Ok(HarnessDataset {
        label,
        bytes,
        chunk_ranges,
    })
}

/// Build a small synthetic dataset with `num_chunks` chunks of `chunk_size`
/// bytes each, requiring no network access, no download, and nothing large
/// committed to the repo.
///
/// Each chunk gets distinct, deterministic pseudo-random bytes so a tampered
/// chunk is unambiguously distinguishable from its neighbors.
///
/// # Panics
///
/// Panics if `chunk_size == 0` (there would be no byte to stamp with the
/// chunk's index). Both call sites below hardcode a nonzero size; this is a
/// documented precondition rather than a silently-wrong dataset.
fn synthetic_dataset(num_chunks: usize, chunk_size: usize, label: &'static str) -> HarnessDataset {
    assert!(
        chunk_size > 0,
        "synthetic_dataset: chunk_size must be nonzero"
    );
    let mut bytes = Vec::with_capacity(num_chunks * chunk_size);
    let mut chunk_ranges = Vec::with_capacity(num_chunks);
    // Small xorshift PRNG for deterministic, repo-reproducible content.
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut next_byte = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state & 0xff) as u8
    };
    for i in 0..num_chunks {
        let start = bytes.len();
        for _ in 0..chunk_size {
            bytes.push(next_byte());
        }
        // Make each chunk's index recoverable from its own bytes, matching
        // the way real satellite/simulation chunks carry distinguishable
        // content.
        bytes[start] = i as u8;
        chunk_ranges.push((start, start + chunk_size));
    }
    HarnessDataset {
        label,
        bytes,
        chunk_ranges,
    }
}

/// Synthetic stand-in for the NOAA GOES-18 dataset: many small chunks, the
/// same "satellite imagery tile" shape as a real GOES-18 `Rad` chunk
/// (typically 226×226, comparable byte size to the 512-byte chunks here).
#[must_use]
pub fn synthetic_noaa_dataset() -> HarnessDataset {
    synthetic_dataset(16, 512, "NOAA-GOES18 (synthetic)")
}

/// Synthetic stand-in for the EQSIM earthquake-simulation dataset: fewer,
/// larger chunks, representative of an HPC checkpoint/restart write (one
/// large contiguous block per rank/timestep) rather than NOAA's many small
/// imagery tiles — see S2-D2-Yr2 §7.4 ("Datasets"): EQSIM is "representative
/// of HPC checkpoint/restart patterns."
#[must_use]
pub fn synthetic_eqsim_dataset() -> HarnessDataset {
    synthetic_dataset(4, 4096, "EQSIM (synthetic)")
}
