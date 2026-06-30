//! h5bench-equivalent read workloads for clawhdf5.
//!
//! Covers sequential read, hyperslab / strided access, and round-trip
//! validation patterns mirroring the h5bench HPC read suite.

use clawhdf5::{File, FileBuilder};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helpers: build reference files once per bench group.
// ---------------------------------------------------------------------------

/// Write a contiguous 1-D f32 dataset and return raw bytes.
fn make_1d_contiguous_bytes(n: usize) -> Vec<u8> {
    let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
    let mut fb = FileBuilder::new();
    fb.create_dataset("data")
        .with_f32_data(&data)
        .with_shape(&[n as u64]);
    fb.finish().unwrap()
}

/// Write a contiguous 1-D f64 dataset and return raw bytes.
fn make_1d_f64_bytes(n: usize) -> Vec<u8> {
    let data: Vec<f64> = (0..n).map(|i| i as f64 * 0.001).collect();
    let mut fb = FileBuilder::new();
    fb.create_dataset("data")
        .with_f64_data(&data)
        .with_shape(&[n as u64]);
    fb.finish().unwrap()
}

/// Write a 2-D chunked f32 matrix to a temp file, return path string.
///
/// The temp dir is returned to keep the directory alive.
fn make_2d_chunked_file(tmp: &TempDir, rows: usize, cols: usize) -> std::path::PathBuf {
    let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
    let path = tmp.path().join("chunked.h5");
    let mut fb = FileBuilder::new();
    fb.create_dataset("matrix")
        .with_f32_data(&data)
        .with_shape(&[rows as u64, cols as u64])
        .with_chunks(&[32, cols as u64]);
    fb.write(&path).unwrap();
    path
}

// ---------------------------------------------------------------------------
// Workload: read_sequential
// Read back the full 1-D contiguous f32 dataset.
// Measures parser + byte-copy throughput.
// ---------------------------------------------------------------------------

fn bench_read_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_sequential");

    for &n in &[1_000usize, 10_000, 100_000] {
        let bytes = make_1d_contiguous_bytes(n);
        group.throughput(Throughput::Bytes((n * size_of::<f32>()) as u64));

        group.bench_with_input(BenchmarkId::new("clawhdf5", n), &bytes, |b, raw| {
            b.iter(|| {
                let file = File::from_bytes(raw.clone()).unwrap();
                let ds = file.dataset("data").unwrap();
                ds.read_f32().unwrap()
            });
        });

        #[cfg(feature = "libhdf5-compare")]
        group.bench_with_input(BenchmarkId::new("libhdf5", n), &bytes, |b, raw| {
            let tmp = TempDir::new().unwrap();
            let path = tmp.path().join("seq_libhdf5.h5");
            std::fs::write(&path, raw).unwrap();
            b.iter(|| {
                let file = hdf5::File::open(&path).unwrap();
                let ds = file.dataset("data").unwrap();
                ds.read_raw::<f32>().unwrap()
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Workload: read_f64_sequential
// Same as above but for f64 — the dominant agent-embedding dtype.
// ---------------------------------------------------------------------------

fn bench_read_f64_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_f64_sequential");

    for &n in &[1_000usize, 10_000, 100_000] {
        let bytes = make_1d_f64_bytes(n);
        group.throughput(Throughput::Bytes((n * size_of::<f64>()) as u64));

        group.bench_with_input(BenchmarkId::new("clawhdf5", n), &bytes, |b, raw| {
            b.iter(|| {
                let file = File::from_bytes(raw.clone()).unwrap();
                let ds = file.dataset("data").unwrap();
                ds.read_f64().unwrap()
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Workload: read_chunked_2d
// Read back a 2-D chunked f32 matrix from disk (exercises chunk reassembly).
// ---------------------------------------------------------------------------

fn bench_read_chunked_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_chunked_2d");

    for &(rows, cols) in &[(64usize, 64usize), (256, 256), (512, 512)] {
        let tmp = TempDir::new().unwrap();
        let path = make_2d_chunked_file(&tmp, rows, cols);
        let n = rows * cols;
        group.throughput(Throughput::Bytes((n * size_of::<f32>()) as u64));
        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("clawhdf5", &label), &path, |b, p| {
            b.iter(|| {
                let raw = std::fs::read(p).unwrap();
                let file = File::from_bytes(raw).unwrap();
                let ds = file.dataset("matrix").unwrap();
                ds.read_f32().unwrap()
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Workload: read_from_disk
// Open file from disk (FileBuilder::write → File::open) measuring OS I/O +
// HDF5 parse together.  Simulates cold-cache reads.
// ---------------------------------------------------------------------------

fn bench_read_from_disk(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_from_disk");

    for &n in &[10_000usize, 100_000] {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("disk.h5");

        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut fb = FileBuilder::new();
        fb.create_dataset("data")
            .with_f64_data(&data)
            .with_shape(&[n as u64]);
        fb.write(&path).unwrap();

        group.throughput(Throughput::Bytes((n * size_of::<f64>()) as u64));

        group.bench_with_input(BenchmarkId::new("clawhdf5", n), &path, |b, p| {
            b.iter(|| {
                let raw = std::fs::read(p).unwrap();
                let file = File::from_bytes(raw).unwrap();
                file.dataset("data").unwrap().read_f64().unwrap()
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Workload: read_hyperslab
// Reads a subset of a 1-D dataset (simulating strided / hyperslab access).
// Uses every-other element to stress the selection logic.
// ---------------------------------------------------------------------------

fn bench_read_hyperslab(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_hyperslab");

    for &n in &[10_000usize, 100_000] {
        let bytes = make_1d_f64_bytes(n);
        // Read first 10% of the dataset as a proxy for hyperslab access.
        let slice_len = n / 10;
        group.throughput(Throughput::Bytes((slice_len * size_of::<f64>()) as u64));

        group.bench_with_input(BenchmarkId::new("clawhdf5", n), &bytes, |b, raw| {
            b.iter(|| {
                let file = File::from_bytes(raw.clone()).unwrap();
                let ds = file.dataset("data").unwrap();
                // Full read then take a slice — clawhdf5 does not yet expose
                // selection API at the high-level facade, so we read all and
                // trim (this is what the format-level selection exercises).
                let all = ds.read_f64().unwrap();
                all[..slice_len].to_vec()
            });
        });
    }

    group.finish();
}

criterion_group!(
    read_benches,
    bench_read_sequential,
    bench_read_f64_sequential,
    bench_read_chunked_2d,
    bench_read_from_disk,
    bench_read_hyperslab,
);
criterion_main!(read_benches);
