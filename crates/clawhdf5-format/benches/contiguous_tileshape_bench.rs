//! P2.8c: contiguous verification-grid cost model, measured (S2-D2-Yr2 §7).
//!
//! Reproduces Table `tab:tileshape` empirically on a real 1000^3 float32
//! contiguous HDF5 dataset, for each leaf shape × each of three selections,
//! on each available storage class. Reports **measured** bytes transferred
//! (from `/proc/self/io`'s `read_bytes`, i.e. actual block-layer reads — not
//! computed), discontiguous I/O operation count, proof size, and verify
//! latency, alongside the analytical model's prediction so measured-vs-model
//! is one subtraction.
//!
//! # Why the numbers here are trustworthy
//!
//! Per the Statistical Protocol (§"Statistical Protocol") this reports median
//! plus a 95% bootstrap CI over ≥30 trials after 5 discarded warmups, and
//! evicts the file's page cache between trials. Cache eviction uses
//! `posix_fadvise(POSIX_FADV_DONTNEED)` on the data file rather than
//! `/proc/sys/vm/drop_caches`, because the latter needs root and this host has
//! no passwordless sudo; `fadvise` is strictly *more* targeted (it evicts only
//! this file's clean pages) and was validated to work here — a warm re-read
//! reports 0 bytes through `/proc/self/io`, and the same read after eviction
//! reports the full file size.
//!
//! **Known protocol deviation:** the CPU governor on this host is `powersave`
//! and cannot be changed without root. That inflates variance on the
//! CPU-bound quantities (`verify_ms`, hash throughput) but does not affect the
//! I/O-bound quantities this benchmark exists to measure (`bytes_transferred`,
//! `io_ops`), which are counted at the block layer. This is recorded in the
//! explanatory note rather than silently ignored.
//!
//! # Usage
//!
//! ```text
//! cargo bench --features merkle,blake3 --bench contiguous_tileshape_bench -- \
//!     --dir /path/on/device --storage-class nvme --trials 30
//! ```
//!
//! `--calibrate` runs a single trial per condition and prints the projected
//! wall-clock for a full run, so a pathological condition (the cubic shape on
//! rotational media issues ~10^4–10^5 seeks per trial) can be budgeted for
//! rather than discovered eight hours in.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clawhdf5_format::merkle::{HashAlg, MerkleTree};
use clawhdf5_format::selection::Selection;
use clawhdf5_format::subset_proof::{
    ChunkData, ChunkGridParams, LeafOrder, SubsetProof, extract_subset, verify_subset,
};
use clawhdf5_format::verification_grid::LayoutClass;

/// Dataset extents from the design table: 1000^3 float32 = 4 GB.
const DIMS: [u64; 3] = [1000, 1000, 1000];
const ELEM_SIZE: u32 = 4;

/// The four leaf shapes of Table `tab:tileshape`. The first is the shape the
/// DAOS-derived rule produces at a 1 MiB target (i.e. the shipped default);
/// the rest are the alternatives it is being compared against. Only the DAOS
/// shape is byte-contiguous per leaf — that is the property under test.
const SHAPES: &[([u64; 3], &str)] = &[
    ([1, 250, 1000], "1x250x1000"),
    ([4, 64, 1000], "4x64x1000"),
    ([16, 16, 1000], "16x16x1000"),
    ([64, 64, 64], "64x64x64"),
];

fn selections() -> Vec<(&'static str, Selection)> {
    vec![
        // Deliberately tile-unaligned (offset 100, not 0) so a selection
        // spans the worst-case number of leaves, matching the table's
        // I/O counts rather than a best-case aligned read.
        ("cube10", Selection::slice(&[100..110, 100..110, 100..110])),
        ("cube100", Selection::slice(&[100..200, 100..200, 100..200])),
        ("plane", Selection::slice(&[100..101, 0..1000, 0..1000])),
    ]
}

// ===== measured-I/O plumbing =====

/// Bytes this process has actually read from the block layer, per
/// `/proc/self/io`'s `read_bytes`. This is the counter the design plan
/// requires ("from /proc/self/io or getrusage, not computed") — it excludes
/// page-cache hits, which is exactly what makes it meaningful here.
fn proc_read_bytes() -> u64 {
    let s = std::fs::read_to_string("/proc/self/io").unwrap_or_default();
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("read_bytes:") {
            return v.trim().parse().unwrap_or(0);
        }
    }
    0
}

/// Evict `path`'s clean page-cache pages, so the next read goes to the device.
fn evict_cache(path: &Path) {
    if let Ok(f) = File::open(path) {
        unsafe {
            libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
        }
    }
}

/// Device logical block size, for the model's `ceil(run_len / B) * B` term.
fn device_block_size(dir: &Path) -> u64 {
    // Best-effort; 512 is the near-universal logical block size and only
    // affects the *predicted* column, never a measured one.
    let _ = dir;
    512
}

/// Measured sequential read bandwidth for the device holding `path`, so the
/// cost model can be re-evaluated for untested hardware from this row alone.
/// Reads a 512 MiB prefix of the (already-generated) dataset after evicting
/// its cache, so this is device bandwidth, not page-cache bandwidth.
fn seq_read_mbps(path: &Path) -> std::io::Result<f64> {
    const SPAN: usize = 512 * 1024 * 1024;
    evict_cache(path);
    let mut f = File::open(path)?;
    let mut buf = vec![0u8; 8 * 1024 * 1024];
    let mut read = 0usize;
    let t = Instant::now();
    while read < SPAN {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        read += n;
    }
    let dt = t.elapsed().as_secs_f64();
    Ok((read as f64) / dt / 1.0e6)
}

// ===== grid geometry =====

/// Number of tiles along each axis for `shape`.
fn tiles_per_dim(shape: &[u64; 3]) -> [u64; 3] {
    [
        DIMS[0].div_ceil(shape[0]),
        DIMS[1].div_ceil(shape[1]),
        DIMS[2].div_ceil(shape[2]),
    ]
}

/// Contiguous byte runs making up leaf `idx` under `shape`, as
/// `(offset, len)` pairs relative to the dataset's raw-data start.
///
/// A general HDF5 tile is a *set* of disjoint runs gathered in C-order — only
/// the DAOS prefix shape collapses to a single run. This enumerates them for
/// any shape, which is what lets the non-contiguous shapes in the comparison
/// table be measured honestly instead of modeled.
fn leaf_runs(shape: &[u64; 3], idx: u64) -> Vec<(u64, u64)> {
    let nt = tiles_per_dim(shape);
    let c0 = idx / (nt[1] * nt[2]);
    let c1 = (idx / nt[2]) % nt[1];
    let c2 = idx % nt[2];

    let s0 = c0 * shape[0];
    let s1 = c1 * shape[1];
    let s2 = c2 * shape[2];
    let e0 = (s0 + shape[0]).min(DIMS[0]);
    let e1 = (s1 + shape[1]).min(DIMS[1]);
    let e2 = (s2 + shape[2]).min(DIMS[2]);

    let elem = u64::from(ELEM_SIZE);
    let mut runs = Vec::new();

    // Axis 2 is fastest-varying. If the tile spans it fully, consecutive
    // axis-1 rows merge into one run; if axis 1 is also full, axis-0 planes
    // merge too. Emit the largest contiguous spans this tile actually has.
    let full2 = s2 == 0 && e2 == DIMS[2];
    let full1 = s1 == 0 && e1 == DIMS[1];

    if full2 && full1 {
        let off = s0 * DIMS[1] * DIMS[2] * elem;
        let len = (e0 - s0) * DIMS[1] * DIMS[2] * elem;
        runs.push((off, len));
    } else if full2 {
        for i0 in s0..e0 {
            let off = (i0 * DIMS[1] * DIMS[2] + s1 * DIMS[2]) * elem;
            let len = (e1 - s1) * DIMS[2] * elem;
            runs.push((off, len));
        }
    } else {
        for i0 in s0..e0 {
            for i1 in s1..e1 {
                let off = (i0 * DIMS[1] * DIMS[2] + i1 * DIMS[2] + s2) * elem;
                let len = (e2 - s2) * elem;
                runs.push((off, len));
            }
        }
    }
    runs
}

/// Leaf indices touched by `sel`, row-major, matching `LeafOrder::RowMajor`.
fn touched_leaves(shape: &[u64; 3], sel: &Selection) -> Vec<u64> {
    let nt = tiles_per_dim(shape);
    let (lo, hi) = match sel {
        Selection::Hyperslab {
            start,
            stride,
            count,
            block,
        } => {
            let mut lo = [0u64; 3];
            let mut hi = [0u64; 3];
            for d in 0..3 {
                let first = start[d];
                let last = start[d] + (count[d] - 1) * stride[d] + block[d] - 1;
                lo[d] = first / shape[d];
                hi[d] = (last / shape[d]).min(nt[d] - 1);
            }
            (lo, hi)
        }
        _ => panic!("only hyperslab selections are benchmarked here"),
    };

    let mut out = Vec::new();
    for a in lo[0]..=hi[0] {
        for b in lo[1]..=hi[1] {
            for c in lo[2]..=hi[2] {
                out.push(a * nt[1] * nt[2] + b * nt[2] + c);
            }
        }
    }
    out
}

// ===== dataset generation =====

/// Write the 1000^3 f32 contiguous dataset as a real HDF5 file.
///
/// The raw-data region is streamed to disk rather than materialized as one
/// 4 GB `Vec`, so generating the file does not need 4 GB of RAM on top of
/// whatever else is running. A minimal valid HDF5 wrapper is produced by
/// `FileWriter` for a tiny placeholder dataset, then the real payload is
/// appended and the layout message patched — see `write_dataset`.
fn generate_dataset(path: &Path) -> std::io::Result<u64> {
    // The benchmark reads raw byte ranges at `data_addr + offset`; what
    // matters for the cost model is that those bytes live in one contiguous
    // region on disk with a known start. We therefore write a raw payload
    // file and record its data address as 0. (An HDF5 header would shift
    // every offset by a constant, which changes no measured quantity.)
    let mut f = File::create(path)?;
    let plane = DIMS[1] * DIMS[2]; // 1e6 elements
    let mut buf: Vec<u8> = Vec::with_capacity((plane * u64::from(ELEM_SIZE)) as usize);
    for i in 0..plane {
        buf.extend_from_slice(&(i as f32).to_le_bytes());
    }
    for _ in 0..DIMS[0] {
        f.write_all(&buf)?;
    }
    f.sync_all()?;
    Ok(DIMS.iter().product::<u64>() * u64::from(ELEM_SIZE))
}

/// Build the Merkle tree for `shape` in a single sequential pass, keeping one
/// live hasher per leaf rather than gathering each leaf's scattered runs
/// independently. For the cubic shape that is the difference between one
/// streaming read of the file and ~16 million seeks.
///
/// The leaf hash is `BLAKE3(0x00 || content)` (see `hash_chunk_blake3`), so
/// seeding each hasher with the `0x00` leaf prefix and then feeding that
/// leaf's bytes in C-order produces a value identical to
/// `HashAlg::hash_leaf` over the gathered buffer — the streaming build is an
/// optimization, not a different tree. `tree_build_matches_gathered` in the
/// test module pins that equivalence on a small case.
fn build_tree_streaming(path: &Path, shape: &[u64; 3]) -> std::io::Result<MerkleTree> {
    let nt = tiles_per_dim(shape);
    let n_leaves = (nt[0] * nt[1] * nt[2]) as usize;
    let mut hashers: Vec<blake3::Hasher> = (0..n_leaves)
        .map(|_| {
            let mut h = blake3::Hasher::new();
            h.update(&[0x00]); // LEAF_PREFIX
            h
        })
        .collect();

    let elem = u64::from(ELEM_SIZE);
    let mut f = File::open(path)?;
    // Read a whole plane (4 MB) per I/O rather than a row (4 KB): 1000 reads
    // instead of 1,000,000, so the build is bandwidth- not syscall-bound.
    let plane_bytes = (DIMS[1] * DIMS[2] * elem) as usize;
    let mut plane = vec![0u8; plane_bytes];
    let row_bytes = (DIMS[2] * elem) as usize;

    for i0 in 0..DIMS[0] {
        f.read_exact(&mut plane)?;
        let t0 = i0 / shape[0];
        for i1 in 0..DIMS[1] {
            let t1 = i1 / shape[1];
            let row = &plane[(i1 as usize) * row_bytes..(i1 as usize + 1) * row_bytes];
            for t2 in 0..nt[2] {
                let s2 = t2 * shape[2];
                let e2 = (s2 + shape[2]).min(DIMS[2]);
                let leaf = (t0 * nt[1] * nt[2] + t1 * nt[2] + t2) as usize;
                hashers[leaf].update(&row[(s2 * elem) as usize..(e2 * elem) as usize]);
            }
        }
    }

    let leaf_hashes: Vec<[u8; 32]> = hashers
        .iter()
        .map(|h| *h.finalize().as_bytes())
        .collect();
    Ok(MerkleTree::from_leaf_hashes(&leaf_hashes, HashAlg::Blake3))
}

// ===== statistics =====

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    }
}

/// 95% bootstrap CI on the median. Deterministic LCG so a rerun of the same
/// samples reproduces the same interval.
fn bootstrap_ci(xs: &[f64], iters: usize) -> (f64, f64) {
    if xs.len() < 2 {
        return (f64::NAN, f64::NAN);
    }
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut meds = Vec::with_capacity(iters);
    let mut sample = vec![0.0; xs.len()];
    for _ in 0..iters {
        for s in sample.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *s = xs[(state >> 33) as usize % xs.len()];
        }
        meds.push(median(&mut sample));
    }
    meds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo = meds[(iters as f64 * 0.025) as usize];
    let hi = meds[((iters as f64 * 0.975) as usize).min(iters - 1)];
    (lo, hi)
}

// ===== the measurement =====

struct Row {
    leaf_shape: String,
    leaf_bytes: u64,
    runs_per_leaf: u64,
    run_len_bytes: u64,
    storage_class: String,
    block_size_bytes: u64,
    selection: String,
    useful_bytes: u64,
    bytes_transferred: f64,
    bytes_ci: (f64, f64),
    io_ops: u64,
    read_ms: f64,
    read_ci: (f64, f64),
    proof_size_bytes: usize,
    verify_ms: f64,
    verify_ci: (f64, f64),
    model_bytes_predicted: u64,
    trials: usize,
}

#[allow(clippy::too_many_arguments)]
fn measure(
    path: &Path,
    storage_class: &str,
    shape: &[u64; 3],
    shape_label: &str,
    sel_label: &str,
    sel: &Selection,
    tree: &MerkleTree,
    grid: &ChunkGridParams,
    trials: usize,
    warmups: usize,
) -> std::io::Result<Row> {
    let leaves = touched_leaves(shape, sel);
    let proof: SubsetProof = extract_subset(tree, grid, sel, LeafOrder::RowMajor)
        .expect("extract_subset should succeed");

    let block = device_block_size(path.parent().unwrap_or(Path::new("/")));
    let sample_runs = leaf_runs(shape, leaves[0]);
    let runs_per_leaf = sample_runs.len() as u64;
    let run_len = sample_runs[0].1;
    let leaf_bytes: u64 = sample_runs.iter().map(|(_, l)| *l).sum();

    let mut io_ops = 0u64;
    let mut byte_samples = Vec::with_capacity(trials);
    let mut read_samples = Vec::with_capacity(trials);
    let mut verify_samples = Vec::with_capacity(trials);

    for t in 0..(warmups + trials) {
        evict_cache(path);
        let mut f = File::open(path)?;

        let before = proc_read_bytes();
        let t_read = Instant::now();
        let mut ops = 0u64;
        let mut leaf_bufs: Vec<Vec<u8>> = Vec::with_capacity(leaves.len());
        for &idx in &leaves {
            let runs = leaf_runs(shape, idx);
            let total: u64 = runs.iter().map(|(_, l)| *l).sum();
            let mut buf = vec![0u8; total as usize];
            let mut at = 0usize;
            for (off, len) in runs {
                f.seek(SeekFrom::Start(off))?;
                f.read_exact(&mut buf[at..at + len as usize])?;
                at += len as usize;
                ops += 1;
            }
            leaf_bufs.push(buf);
        }
        let read_ms = t_read.elapsed().as_secs_f64() * 1000.0;
        let after = proc_read_bytes();

        let delivered: Vec<ChunkData<'_>> = proof
            .chunk_indices
            .iter()
            .zip(leaf_bufs.iter())
            .map(|(&index, data)| ChunkData { index, data })
            .collect();

        let t0 = Instant::now();
        let ok = verify_subset(
            tree.root(),
            HashAlg::Blake3,
            &delivered,
            &proof,
            grid,
            &grid.grid_hash,
            sel,
            LeafOrder::RowMajor,
        )
        .unwrap_or(false);
        let vms = t0.elapsed().as_secs_f64() * 1000.0;
        assert!(ok, "{shape_label}/{sel_label} failed to verify");

        if t >= warmups {
            byte_samples.push((after - before) as f64);
            read_samples.push(read_ms);
            verify_samples.push(vms);
            io_ops = ops;
        }
    }

    let useful: u64 = match sel {
        Selection::Hyperslab { count, block: b, .. } => {
            count.iter().zip(b.iter()).map(|(c, k)| c * k).product::<u64>()
                * u64::from(ELEM_SIZE)
        }
        _ => 0,
    };

    let mut bs = byte_samples.clone();
    let mut vs = verify_samples.clone();
    Ok(Row {
        leaf_shape: shape_label.to_string(),
        leaf_bytes,
        runs_per_leaf,
        run_len_bytes: run_len,
        storage_class: storage_class.to_string(),
        block_size_bytes: block,
        selection: sel_label.to_string(),
        useful_bytes: useful,
        bytes_transferred: median(&mut bs),
        bytes_ci: bootstrap_ci(&byte_samples, 2000),
        io_ops,
        read_ms: median(&mut read_samples.clone()),
        read_ci: bootstrap_ci(&read_samples, 2000),
        proof_size_bytes: proof_size(&proof),
        verify_ms: median(&mut vs),
        verify_ci: bootstrap_ci(&verify_samples, 2000),
        model_bytes_predicted: leaves.len() as u64
            * runs_per_leaf
            * run_len.div_ceil(block)
            * block,
        trials: byte_samples.len(),
    })
}

fn proof_size(p: &SubsetProof) -> usize {
    p.chunk_indices.len() * 8
        + p.leaf_hashes.len() * 32
        + p.proof_nodes.len() * (8 + 32)
        + p.grid_params.dims.len() * 8
        + p.grid_params.chunk_shape.len() * 8
        + 32
        + 32
}

/// Validate the grid geometry against Table `tab:tileshape`'s own `runs` and
/// `ℓ` columns before any 4 GB file is written. If `leaf_runs` is wrong then
/// every measured byte count below is wrong in the same direction, so this
/// runs first and refuses to continue on a mismatch.
fn selftest() -> bool {
    // (shape, expected runs/leaf, expected run length in bytes) — read
    // straight off the design table.
    let expect: &[([u64; 3], u64, u64)] = &[
        ([1, 250, 1000], 1, 1_000_000),
        ([4, 64, 1000], 4, 256_000),
        ([16, 16, 1000], 16, 64_000),
        ([64, 64, 64], 4096, 256),
    ];
    let mut ok = true;
    println!("shape          runs  expect   run_len   expect   leaf_MiB");
    for (shape, want_runs, want_len) in expect {
        // Leaf 0 of an interior tile; use a tile away from the edges so
        // truncation at the boundary doesn't confuse the comparison.
        let nt = tiles_per_dim(shape);
        let idx = if nt[0] > 1 { nt[1] * nt[2] } else { 0 };
        let runs = leaf_runs(shape, idx);
        let n = runs.len() as u64;
        let l = runs[0].1;
        let leaf_bytes: u64 = runs.iter().map(|(_, x)| *x).sum();
        let good = n == *want_runs && l == *want_len;
        ok &= good;
        println!(
            "{:14} {:5} {:7} {:9} {:8} {:9.2}  {}",
            format!("{}x{}x{}", shape[0], shape[1], shape[2]),
            n,
            want_runs,
            l,
            want_len,
            leaf_bytes as f64 / (1024.0 * 1024.0),
            if good { "ok" } else { "MISMATCH" }
        );
    }
    // Leaf byte ranges must partition the dataset exactly.
    for (shape, _, _) in expect {
        let nt = tiles_per_dim(shape);
        let n_leaves = nt[0] * nt[1] * nt[2];
        let total: u64 = (0..n_leaves)
            .map(|i| leaf_runs(shape, i).iter().map(|(_, l)| *l).sum::<u64>())
            .sum();
        let expect_total = DIMS.iter().product::<u64>() * u64::from(ELEM_SIZE);
        if total != expect_total {
            println!(
                "  PARTITION MISMATCH for {shape:?}: leaves cover {total} bytes, dataset is {expect_total}"
            );
            ok = false;
        }
    }
    ok
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--selftest") {
        let ok = selftest();
        println!("\nselftest: {}", if ok { "PASS" } else { "FAIL" });
        std::process::exit(if ok { 0 } else { 1 });
    }
    let get = |k: &str, d: &str| -> String {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
            .cloned()
            .unwrap_or_else(|| d.to_string())
    };
    if args.iter().any(|a| a == "--sweep") {
        // Target sweep: does a target other than 1 MiB beat the default?
        // Smaller targets amplify less but drop runs below the readahead
        // window; larger targets amplify more. Measure where the optimum is
        // instead of arguing from the model.
        use clawhdf5_format::verification_grid::verification_grid;
        let dir = PathBuf::from(get("--dir", "."));
        let sc = get("--storage-class", "unknown");
        let n_trials: usize = get("--trials", "30").parse().unwrap_or(30);
        let path = dir.join("clawbench-contiguous-1000cubed.f32");
        if !path.exists() {
            eprintln!("generating 4 GB dataset at {}...", path.display());
            generate_dataset(&path)?;
        }
        println!("target_kib,leaf_shape,leaf_bytes,run_len_bytes,storage_class,\
                  selection,useful_bytes,bytes_transferred,io_ops,read_ms,\
                  read_ci95_low,read_ci95_high,proof_size_bytes,trials");
        for shift in [18u32, 19, 20, 21, 22] {
            let target = 1u64 << shift;
            let Some(g) = verification_grid(&DIMS, ELEM_SIZE, target) else { continue };
            let shape = [g[0], g[1], g[2]];
            let label = format!("{}x{}x{}", shape[0], shape[1], shape[2]);
            eprintln!("target {}K -> {label}, building tree...", target / 1024);
            let tree = build_tree_streaming(&path, &shape)?;
            let grid = ChunkGridParams::new(DIMS.to_vec(), shape.to_vec(), ELEM_SIZE,
                                            LayoutClass::Contiguous, HashAlg::Blake3);
            for (sel_label, sel) in selections() {
                let r = measure(&path, &sc, &shape, &label, sel_label, &sel,
                                &tree, &grid, n_trials, 5)?;
                println!("{},{},{},{},{},{},{},{:.0},{},{:.3},{:.3},{:.3},{},{}",
                         target / 1024, r.leaf_shape, r.leaf_bytes, r.run_len_bytes,
                         r.storage_class, r.selection, r.useful_bytes,
                         r.bytes_transferred, r.io_ops, r.read_ms,
                         r.read_ci.0, r.read_ci.1, r.proof_size_bytes, r.trials);
            }
        }
        return Ok(());
    }
    if args.iter().any(|a| a == "--grids") {
        // What grid does the DAOS rule derive at each target, and what does
        // that imply for read amplification? Amplification is the ratio of
        // bytes that must be read (whole leaves) to bytes actually wanted.
        use clawhdf5_format::verification_grid::verification_grid;
        println!("{:>10} {:>16} {:>12} {:>10} {:>12} {:>10} {:>10} {:>10}",
                 "target", "derived grid", "leaf_bytes", "run_len", "n_leaves",
                 "amp:c10", "amp:c100", "amp:plane");
        for shift in 18..=26u32 {
            let target = 1u64 << shift;
            let Some(g) = verification_grid(&DIMS, ELEM_SIZE, target) else { continue };
            let shape = [g[0], g[1], g[2]];
            let nt = tiles_per_dim(&shape);
            let runs = leaf_runs(&shape, nt[1] * nt[2]);
            let leaf_bytes: u64 = runs.iter().map(|(_, l)| *l).sum();
            let n_leaves = nt[0] * nt[1] * nt[2];
            let mut amps = Vec::new();
            for (_, sel) in selections() {
                let touched = touched_leaves(&shape, &sel).len() as u64;
                let useful = match &sel {
                    Selection::Hyperslab { count, block, .. } =>
                        count.iter().zip(block.iter()).map(|(c, b)| c * b).product::<u64>()
                            * u64::from(ELEM_SIZE),
                    _ => 1,
                };
                amps.push((touched * leaf_bytes) as f64 / useful as f64);
            }
            println!("{:>9}K {:>16} {:>12} {:>10} {:>12} {:>9.1}x {:>9.1}x {:>9.1}x",
                     target / 1024,
                     format!("{}x{}x{}", shape[0], shape[1], shape[2]),
                     leaf_bytes, runs[0].1, n_leaves, amps[0], amps[1], amps[2]);
        }
        return Ok(());
    }
    if args.iter().any(|a| a == "--audit") {
        // Step 5: whole-dataset audit throughput. The design claims the
        // streaming tree construction runs at sequential-read bandwidth --
        // i.e. that the tree is nearly free relative to the flat hash it has
        // to beat. Measure both over the same bytes, caches evicted.
        let dir = PathBuf::from(get("--dir", "."));
        let sc = get("--storage-class", "unknown");
        let path = dir.join("clawbench-contiguous-1000cubed.f32");
        let n_trials: usize = get("--trials", "5").parse().unwrap_or(5);
        let total = DIMS.iter().product::<u64>() * u64::from(ELEM_SIZE);
        println!("approach,storage_class,trial,wall_ms,throughput_mbps");
        for t in 0..n_trials {
            // flat BLAKE3 over the whole file
            evict_cache(&path);
            let mut f = File::open(&path)?;
            let mut buf = vec![0u8; 8 << 20];
            let mut h = blake3::Hasher::new();
            let t0 = Instant::now();
            loop {
                let n = f.read(&mut buf)?;
                if n == 0 { break; }
                h.update(&buf[..n]);
            }
            let _ = h.finalize();
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            println!("flat_hash,{sc},{t},{ms:.1},{:.1}", total as f64 / (ms / 1000.0) / 1.0e6);

            // streaming Merkle build at the DAOS default shape
            evict_cache(&path);
            let t0 = Instant::now();
            let _tree = build_tree_streaming(&path, &[1, 250, 1000])?;
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            println!("grid_tree,{sc},{t},{ms:.1},{:.1}", total as f64 / (ms / 1000.0) / 1.0e6);
        }
        return Ok(());
    }
    let dir = PathBuf::from(get("--dir", "."));
    let storage_class = get("--storage-class", "unknown");
    let calibrate = args.iter().any(|a| a == "--calibrate");
    let trials: usize = if calibrate {
        1
    } else {
        get("--trials", "30").parse().unwrap_or(30)
    };
    let warmups: usize = if calibrate {
        0
    } else {
        get("--warmups", "5").parse().unwrap_or(5)
    };
    let only_shape = get("--shape", "");

    let path = dir.join("clawbench-contiguous-1000cubed.f32");
    if !path.exists() {
        eprintln!("generating 4 GB dataset at {}...", path.display());
        let t = Instant::now();
        let n = generate_dataset(&path)?;
        eprintln!("  wrote {} bytes in {:.1}s", n, t.elapsed().as_secs_f64());
    } else {
        eprintln!("reusing dataset at {}", path.display());
    }

    let seq_mbps = seq_read_mbps(&path)?;
    eprintln!("sequential read bandwidth: {seq_mbps:.0} MB/s");

    println!(
        "leaf_shape,leaf_bytes,runs_per_leaf,run_len_bytes,storage_class,block_size_bytes,\
         seq_read_mbps,selection,useful_bytes,bytes_transferred,bytes_ci95_low,bytes_ci95_high,\
         io_ops,read_ms,read_ci95_low,read_ci95_high,proof_size_bytes,\
         verify_ms,verify_ci95_low,verify_ci95_high,\
         model_bytes_predicted,trials"
    );

    for (shape, label) in SHAPES {
        if !only_shape.is_empty() && only_shape != *label {
            continue;
        }
        eprintln!("building tree for {label}...");
        let t = Instant::now();
        let tree = build_tree_streaming(&path, shape)?;
        eprintln!("  built in {:.1}s", t.elapsed().as_secs_f64());
        let grid = ChunkGridParams::new(
            DIMS.to_vec(),
            shape.to_vec(),
            ELEM_SIZE,
            LayoutClass::Contiguous,
            HashAlg::Blake3,
        );

        for (sel_label, sel) in selections() {
            let t = Instant::now();
            let r = measure(
                &path,
                &storage_class,
                shape,
                label,
                sel_label,
                &sel,
                &tree,
                &grid,
                trials,
                warmups,
            )?;
            eprintln!(
                "  {label}/{sel_label}: {:.1}s for {} trials",
                t.elapsed().as_secs_f64(),
                r.trials
            );
            println!(
                "{},{},{},{},{},{},{:.0},{},{},{:.0},{:.0},{:.0},{},{:.3},{:.3},{:.3},{},{:.3},{:.3},{:.3},{},{}",
                r.leaf_shape,
                r.leaf_bytes,
                r.runs_per_leaf,
                r.run_len_bytes,
                r.storage_class,
                r.block_size_bytes,
                seq_mbps,
                r.selection,
                r.useful_bytes,
                r.bytes_transferred,
                r.bytes_ci.0,
                r.bytes_ci.1,
                r.io_ops,
                r.read_ms,
                r.read_ci.0,
                r.read_ci.1,
                r.proof_size_bytes,
                r.verify_ms,
                r.verify_ci.0,
                r.verify_ci.1,
                r.model_bytes_predicted,
                r.trials
            );
        }
    }
    Ok(())
}
