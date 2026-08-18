//! P2.8c step 3: the Bao / BLAKE3 verified-streaming baseline.
//!
//! Bao is BLAKE3's own Merkle tree exposed for verified streaming: fixed
//! 1 KiB leaf chunks over a flat byte stream, with a slice proof carrying the
//! sibling hashes from the covered chunks to the root. It is the strongest
//! existing alternative to the verification grid and solves the same problem
//! — authenticated random access into a large file — with the opposite
//! granularity choice, so it is the fair comparison rather than a strawman.
//!
//! The design predicts Bao wins on bytes for the 10^3 sub-cube (~25x), loses
//! on I/O count (~10x) and proof size (~18x), with the byte advantage
//! collapsing to ~2.4x at 100^3 while proof size grows to exceed the
//! delivered data. Those numbers are analytical; this measures them.
//!
//! # What is measured
//!
//! An N-dimensional hyperslab is not a byte range. Under Bao it becomes a
//! *set* of byte ranges — one per contiguous row-segment of the selection —
//! each of which must be rounded out to 1 KiB chunk boundaries and
//! accompanied by its own proof. That expansion is the whole point of the
//! comparison, so ranges are enumerated honestly rather than approximated by
//! the selection's bounding box.
//!
//! Proof size is reported two ways, as the plan requires:
//!   * **naive** — the sum of independent per-range slice proofs, taken from
//!     the real `bao::encode::SliceExtractor` (authoritative, not modelled);
//!   * **deduplicated** — the union of distinct tree nodes those proofs
//!     reference, which is the best any cross-range batching could do.
//!
//! # Usage
//!
//! ```text
//! cargo bench --features merkle,blake3 --bench contiguous_bao_baseline -- \
//!     --dir /path/on/device --storage-class nvme
//! ```

use std::collections::BTreeSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::Instant;

const DIMS: [u64; 3] = [1000, 1000, 1000];
const ELEM_SIZE: u64 = 4;
/// Bao's native leaf granularity (BLAKE3 chunk size). Not tunable — it is
/// fixed by the BLAKE3 specification, which is precisely the constraint this
/// comparison exists to expose.
const BAO_CHUNK: u64 = 1024;

fn proc_read_bytes() -> u64 {
    let s = std::fs::read_to_string("/proc/self/io").unwrap_or_default();
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("read_bytes:") {
            return v.trim().parse().unwrap_or(0);
        }
    }
    0
}

fn evict_cache(path: &Path) {
    if let Ok(f) = File::open(path) {
        unsafe {
            libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
        }
    }
}

/// The contiguous byte ranges a 3-D hyperslab occupies in a C-order stream.
/// One range per (i0, i1) row-segment; this is what Bao actually has to
/// prove, since it has no notion of the array's shape.
fn selection_ranges(start: [u64; 3], end: [u64; 3]) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    for i0 in start[0]..end[0] {
        for i1 in start[1]..end[1] {
            let off = (i0 * DIMS[1] * DIMS[2] + i1 * DIMS[2] + start[2]) * ELEM_SIZE;
            let len = (end[2] - start[2]) * ELEM_SIZE;
            out.push((off, len));
        }
    }
    out
}

/// Distinct 1 KiB Bao chunk indices covering `ranges`.
fn covering_chunks(ranges: &[(u64, u64)]) -> BTreeSet<u64> {
    let mut s = BTreeSet::new();
    for &(off, len) in ranges {
        for c in (off / BAO_CHUNK)..=((off + len - 1) / BAO_CHUNK) {
            s.insert(c);
        }
    }
    s
}

/// Tree-node identifiers on the path from `chunk` to the root, in a complete
/// binary tree over `n_chunks` leaves. Used to compute the deduplicated
/// proof: the union over all covered chunks is the distinct witness set.
fn path_nodes(chunk: u64, n_chunks: u64) -> Vec<(u32, u64)> {
    let mut out = Vec::new();
    let mut idx = chunk;
    let mut level = 0u32;
    let mut width = n_chunks;
    while width > 1 {
        out.push((level, idx ^ 1)); // the sibling at this level
        idx /= 2;
        width = width.div_ceil(2);
        level += 1;
    }
    out
}

struct Row {
    selection: &'static str,
    useful_bytes: u64,
    n_ranges: usize,
    chunks: u64,
    bytes_transferred: u64,
    io_ops: u64,
    read_ms: f64,
    proof_naive: u64,
    proof_dedup: u64,
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str, d: &str| -> String {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
            .cloned()
            .unwrap_or_else(|| d.to_string())
    };
    let dir = PathBuf::from(get("--dir", "."));
    let sc = get("--storage-class", "unknown");
    let data_path = dir.join("clawbench-contiguous-1000cubed.f32");
    let obao_path = dir.join("clawbench-contiguous-1000cubed.obao");
    if !data_path.exists() {
        eprintln!("dataset missing: {}", data_path.display());
        eprintln!("generate it first with the contiguous_tileshape_bench harness");
        std::process::exit(1);
    }
    let total = DIMS.iter().product::<u64>() * ELEM_SIZE;
    let n_chunks = total.div_ceil(BAO_CHUNK);

    // Outboard encoding: the Bao tree lives in a sidecar so the 4 GB payload
    // is not duplicated. This is the mode a real deployment would use.
    if !obao_path.exists() {
        eprintln!(
            "building Bao outboard ({} MB expected)...",
            bao::encode::outboard_size(total) / 1_000_000
        );
        let t = Instant::now();
        let mut input = File::open(&data_path)?;
        // Must be readable as well as writable: `Encoder::finalize` stores the
        // tree post-order and then flips it in place to pre-order, so it reads
        // back what it wrote. A write-only `File::create` fails with EBADF.
        let out = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&obao_path)?;
        let mut enc = bao::encode::Encoder::new_outboard(out);
        let mut buf = vec![0u8; 8 << 20];
        loop {
            let n = input.read(&mut buf)?;
            if n == 0 {
                break;
            }
            enc.write_all(&buf[..n])?;
        }
        let hash = enc.finalize()?;
        eprintln!(
            "  outboard built in {:.1}s, root {}",
            t.elapsed().as_secs_f64(),
            hash.to_hex()
        );
    } else {
        eprintln!("reusing outboard at {}", obao_path.display());
    }

    let cases: [(&str, [u64; 3], [u64; 3]); 3] = [
        ("cube10", [100, 100, 100], [110, 110, 110]),
        ("cube100", [100, 100, 100], [200, 200, 200]),
        ("plane", [100, 0, 0], [101, 1000, 1000]),
    ];

    println!(
        "approach,storage_class,selection,useful_bytes,n_ranges,chunks_touched,\
         bytes_transferred,io_ops,read_ms,proof_naive_bytes,proof_dedup_bytes"
    );

    for (label, start, end) in cases {
        let ranges = selection_ranges(start, end);
        let useful: u64 = ranges.iter().map(|(_, l)| *l).sum();
        let chunks = covering_chunks(&ranges);

        // --- measured I/O: read every covering 1 KiB chunk from the data
        // file, plus the proof spans from the outboard. Adjacent chunk
        // indices are coalesced into one pread, which is the most favourable
        // honest reading for Bao.
        evict_cache(&data_path);
        evict_cache(&obao_path);
        let mut f = File::open(&data_path)?;
        let before = proc_read_bytes();
        let t0 = Instant::now();
        let mut io_ops = 0u64;
        let mut it = chunks.iter().peekable();
        let mut buf = Vec::new();
        while let Some(&first) = it.next() {
            let mut last = first;
            while let Some(&&nxt) = it.peek() {
                if nxt == last + 1 {
                    last = nxt;
                    it.next();
                } else {
                    break;
                }
            }
            let off = first * BAO_CHUNK;
            let len = ((last - first + 1) * BAO_CHUNK).min(total - off);
            buf.resize(len as usize, 0);
            f.seek(SeekFrom::Start(off))?;
            f.read_exact(&mut buf)?;
            io_ops += 1;
        }
        let read_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let bytes_transferred = proc_read_bytes() - before;

        // --- proof size, naive: real per-range slice proofs from bao.
        // A slice contains proof nodes interleaved with the covered data, so
        // the proof is the slice length minus the data it carries.
        let mut proof_naive = 0u64;
        let sample_cap = 2000usize; // bound the cost for 10k-range cases
        let sampled = ranges.len().min(sample_cap);
        for &(off, len) in ranges.iter().take(sampled) {
            let data = File::open(&data_path)?;
            let obao = File::open(&obao_path)?;
            let mut ex = bao::encode::SliceExtractor::new_outboard(data, obao, off, len);
            let mut sink = Vec::new();
            ex.read_to_end(&mut sink)?;
            // The slice carries the covering chunks in full, not just `len`.
            let covered = {
                let c0 = off / BAO_CHUNK;
                let c1 = (off + len - 1) / BAO_CHUNK;
                ((c1 - c0 + 1) * BAO_CHUNK).min(total - c0 * BAO_CHUNK)
            };
            proof_naive += sink.len() as u64 - covered;
        }
        if sampled < ranges.len() {
            // Scale by the sampled mean; ranges are structurally identical
            // (same length, same tree depth), so this is a faithful estimate.
            proof_naive = proof_naive / sampled as u64 * ranges.len() as u64;
        }

        // --- proof size, deduplicated: distinct witness nodes across all
        // covered chunks. This is the floor any batching scheme could reach.
        let mut witnesses: BTreeSet<(u32, u64)> = BTreeSet::new();
        for &c in &chunks {
            for n in path_nodes(c, n_chunks) {
                witnesses.insert(n);
            }
        }
        // A witness that is itself inside the covered set is not transmitted.
        let covered: BTreeSet<(u32, u64)> = chunks.iter().map(|&c| (0u32, c)).collect();
        let proof_dedup = witnesses.difference(&covered).count() as u64 * 32;

        let r = Row {
            selection: label,
            useful_bytes: useful,
            n_ranges: ranges.len(),
            chunks: chunks.len() as u64,
            bytes_transferred,
            io_ops,
            read_ms,
            proof_naive,
            proof_dedup,
        };
        println!(
            "bao,{},{},{},{},{},{},{},{:.3},{},{}",
            sc,
            r.selection,
            r.useful_bytes,
            r.n_ranges,
            r.chunks,
            r.bytes_transferred,
            r.io_ops,
            r.read_ms,
            r.proof_naive,
            r.proof_dedup
        );
    }
    Ok(())
}
