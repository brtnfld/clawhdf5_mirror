//! h5bench-equivalent write workloads for clawhdf5.
//!
//! Mirrors the sequential and chunked write patterns from the h5bench HPC
//! benchmark suite but implemented in pure Rust using Criterion for statistical
//! rigor.  The `libhdf5-compare` feature adds matching benchmarks via the `hdf5`
//! crate (requires a system libhdf5 install).

use clawhdf5::{AttrValue, FileBuilder};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Workload: write_1d_contiguous
// Write N × f32 as a single contiguous 1-D dataset.
// Measures raw serialization + HDF5 superblock / object-header overhead.
// ---------------------------------------------------------------------------

fn bench_write_1d_contiguous(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_1d_contiguous");

    for &n in &[1_000usize, 10_000, 100_000] {
        let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
        group.throughput(Throughput::Bytes((n * size_of::<f32>()) as u64));

        group.bench_with_input(BenchmarkId::new("clawhdf5", n), &data, |b, d| {
            let tmp = TempDir::new().unwrap();
            let path = tmp.path().join("write_1d_contiguous.h5");
            b.iter(|| {
                let mut fb = FileBuilder::new();
                fb.create_dataset("data")
                    .with_f32_data(d)
                    .with_shape(&[n as u64]);
                fb.write(&path).unwrap();
            });
        });

        #[cfg(feature = "libhdf5-compare")]
        group.bench_with_input(BenchmarkId::new("libhdf5", n), &data, |b, d| {
            let tmp = TempDir::new().unwrap();
            let path = tmp.path().join("write_1d_libhdf5.h5");
            b.iter(|| {
                let file = hdf5::File::create(&path).unwrap();
                let ds = file
                    .new_dataset::<f32>()
                    .shape([d.len()])
                    .create("data")
                    .unwrap();
                ds.write(d.as_slice()).unwrap();
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Workload: write_2d_chunked
// Write an M × N f32 matrix as a chunked 2-D dataset with deflate (level 6).
// Measures chunked layout creation + compression pipeline throughput.
// ---------------------------------------------------------------------------

fn bench_write_2d_chunked(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_2d_chunked");

    // (rows, cols, chunk_rows, chunk_cols)
    let configs: &[(usize, usize, u64, u64)] = &[
        (32, 32, 8, 32),
        (128, 128, 32, 128),
        (512, 512, 64, 512),
    ];

    for &(rows, cols, cr, cc) in configs {
        let n = rows * cols;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let label = format!("{rows}x{cols}");
        group.throughput(Throughput::Bytes((n * size_of::<f32>()) as u64));

        group.bench_with_input(BenchmarkId::new("clawhdf5", &label), &data, |b, d| {
            let tmp = TempDir::new().unwrap();
            let path = tmp.path().join("write_2d_chunked.h5");
            b.iter(|| {
                let mut fb = FileBuilder::new();
                fb.create_dataset("matrix")
                    .with_f32_data(d)
                    .with_shape(&[rows as u64, cols as u64])
                    .with_chunks(&[cr, cc])
                    .with_deflate(6);
                fb.write(&path).unwrap();
            });
        });

        #[cfg(feature = "libhdf5-compare")]
        group.bench_with_input(BenchmarkId::new("libhdf5", &label), &data, |b, d| {
            let tmp = TempDir::new().unwrap();
            let path = tmp.path().join("write_2d_libhdf5.h5");
            b.iter(|| {
                let file = hdf5::File::create(&path).unwrap();
                let ds = file
                    .new_dataset::<f32>()
                    .shape([rows, cols])
                    .chunk([cr as usize, cc as usize])
                    .deflate(6)
                    .create("matrix")
                    .unwrap();
                ds.write_raw(d.as_slice()).unwrap();
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Workload: write_2d_chunked_zstd
// Same matrix sizes as write_2d_chunked but uses Zstd level 3.
// Zstd level 3 typically encodes 500+ MiB/s vs deflate's ~300 MiB/s at the
// same or better compression ratio (arXiv 2604.06221, ROOT I/O 2019).
// ---------------------------------------------------------------------------

fn bench_write_2d_chunked_zstd(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_2d_chunked_zstd");

    let configs: &[(usize, usize, u64, u64)] = &[
        (32, 32, 8, 32),
        (128, 128, 32, 128),
        (512, 512, 64, 512),
    ];

    for &(rows, cols, cr, cc) in configs {
        let n = rows * cols;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let label = format!("{rows}x{cols}");
        group.throughput(Throughput::Bytes((n * size_of::<f32>()) as u64));

        group.bench_with_input(BenchmarkId::new("clawhdf5/zstd-3", &label), &data, |b, d| {
            let tmp = TempDir::new().unwrap();
            let path = tmp.path().join("write_2d_chunked_zstd.h5");
            b.iter(|| {
                let mut fb = FileBuilder::new();
                fb.create_dataset("matrix")
                    .with_f32_data(d)
                    .with_shape(&[rows as u64, cols as u64])
                    .with_chunks(&[cr, cc])
                    .with_zstd(3);
                fb.write(&path).unwrap();
            });
        });

        group.bench_with_input(
            BenchmarkId::new("clawhdf5/deflate-6", &label),
            &data,
            |b, d| {
                let tmp = TempDir::new().unwrap();
                let path = tmp.path().join("write_2d_chunked_deflate.h5");
                b.iter(|| {
                    let mut fb = FileBuilder::new();
                    fb.create_dataset("matrix")
                        .with_f32_data(d)
                        .with_shape(&[rows as u64, cols as u64])
                        .with_chunks(&[cr, cc])
                        .with_deflate(6);
                    fb.write(&path).unwrap();
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Workload: write_2d_chunked_pcodec
// Same matrix sizes as write_2d_chunked but uses Pcodec (arXiv:2502.06112).
// Pcodec achieves 30–94% better compression ratio than Zstd for f32/f64 at
// 1–5 GiB/s decompression speed via a quantile-based numerical codec.
// ---------------------------------------------------------------------------

fn bench_write_2d_chunked_pcodec(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_2d_chunked_pcodec");

    let configs: &[(usize, usize, u64, u64)] = &[
        (32, 32, 8, 32),
        (128, 128, 32, 128),
        (512, 512, 64, 512),
    ];

    for &(rows, cols, cr, cc) in configs {
        let n = rows * cols;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let label = format!("{rows}x{cols}");
        group.throughput(Throughput::Bytes((n * size_of::<f32>()) as u64));

        group.bench_with_input(
            BenchmarkId::new("clawhdf5/pcodec", &label),
            &data,
            |b, d| {
                let tmp = TempDir::new().unwrap();
                let path = tmp.path().join("write_2d_chunked_pcodec.h5");
                b.iter(|| {
                    let mut fb = FileBuilder::new();
                    fb.create_dataset("matrix")
                        .with_f32_data(d)
                        .with_shape(&[rows as u64, cols as u64])
                        .with_chunks(&[cr, cc])
                        .with_pcodec();
                    fb.write(&path).unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("clawhdf5/zstd-3", &label),
            &data,
            |b, d| {
                let tmp = TempDir::new().unwrap();
                let path = tmp.path().join("write_2d_chunked_zstd.h5");
                b.iter(|| {
                    let mut fb = FileBuilder::new();
                    fb.create_dataset("matrix")
                        .with_f32_data(d)
                        .with_shape(&[rows as u64, cols as u64])
                        .with_chunks(&[cr, cc])
                        .with_zstd(3);
                    fb.write(&path).unwrap();
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Workload: write_f64_batch
// Write batches of f64 elements — simulates the clawhdf5-agent embedding
// write path (one f64 vector per memory entry).
// ---------------------------------------------------------------------------

fn bench_write_f64_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_f64_batch");

    for &n in &[128usize, 512, 1_024] {
        let data: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        group.throughput(Throughput::Bytes((n * size_of::<f64>()) as u64));

        group.bench_with_input(BenchmarkId::new("clawhdf5", n), &data, |b, d| {
            let tmp = TempDir::new().unwrap();
            let path = tmp.path().join("write_f64_batch.h5");
            b.iter(|| {
                let mut fb = FileBuilder::new();
                fb.create_dataset("embedding")
                    .with_f64_data(d)
                    .with_shape(&[n as u64]);
                fb.write(&path).unwrap();
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Workload: write_multi_dataset
// Write K independent f32 datasets into one file — stresses the object-header
// + link-storage path (compact → dense transition at >8 datasets).
// ---------------------------------------------------------------------------

fn bench_write_multi_dataset(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_multi_dataset");

    for &k in &[4usize, 16, 64] {
        let rows = 100usize;
        let data: Vec<f32> = (0..rows).map(|i| i as f32).collect();
        group.throughput(Throughput::Elements(k as u64));

        group.bench_with_input(BenchmarkId::new("clawhdf5", k), &data, |b, d| {
            let tmp = TempDir::new().unwrap();
            let path = tmp.path().join("write_multi.h5");
            b.iter(|| {
                let mut fb = FileBuilder::new();
                for i in 0..k {
                    fb.create_dataset(&format!("ds_{i:04}"))
                        .with_f32_data(d)
                        .with_shape(&[rows as u64]);
                }
                fb.write(&path).unwrap();
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Workload: write_with_attrs
// Write a dataset with K attributes — exercises attribute message allocation.
// ---------------------------------------------------------------------------

fn bench_write_with_attrs(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_with_attrs");

    for &k in &[4usize, 16, 64] {
        group.throughput(Throughput::Elements(k as u64));

        group.bench_with_input(BenchmarkId::new("clawhdf5", k), &k, |b, &k| {
            let tmp = TempDir::new().unwrap();
            let path = tmp.path().join("write_attrs.h5");
            b.iter(|| {
                let mut fb = FileBuilder::new();
                let ds = fb
                    .create_dataset("data")
                    .with_f64_data(&[1.0, 2.0, 3.0])
                    .with_shape(&[3]);
                for i in 0..k {
                    ds.set_attr(&format!("attr_{i}"), AttrValue::I64(i as i64));
                }
                fb.write(&path).unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(
    write_benches,
    bench_write_1d_contiguous,
    bench_write_2d_chunked,
    bench_write_2d_chunked_zstd,
    bench_write_2d_chunked_pcodec,
    bench_write_f64_batch,
    bench_write_multi_dataset,
    bench_write_with_attrs,
);
criterion_main!(write_benches);
