//! h5bench-equivalent metadata workloads for clawhdf5.
//!
//! Measures attribute creation/read throughput and group traversal latency —
//! the workloads that h5bench's `metadata` mode targets against libhdf5.

use clawhdf5::{AttrValue, File, FileBuilder};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Workload: metadata_attrs_write
// Create K attributes on a single dataset.
// Exercises attribute message allocation and compact → dense header transition.
// ---------------------------------------------------------------------------

fn bench_metadata_attrs_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("metadata_attrs_write");

    for &k in &[4usize, 16, 64, 128] {
        group.throughput(Throughput::Elements(k as u64));

        group.bench_with_input(BenchmarkId::new("clawhdf5", k), &k, |b, &k| {
            let tmp = TempDir::new().unwrap();
            let path = tmp.path().join("attrs_write.h5");
            b.iter(|| {
                let mut fb = FileBuilder::new();
                let ds = fb
                    .create_dataset("data")
                    .with_f64_data(&[1.0, 2.0, 3.0])
                    .with_shape(&[3]);
                for i in 0..k {
                    ds.set_attr(&format!("attr_{i:04}"), AttrValue::I64(i as i64));
                }
                fb.write(&path).unwrap();
            });
        });

        #[cfg(feature = "libhdf5-compare")]
        group.bench_with_input(BenchmarkId::new("libhdf5", k), &k, |b, &k| {
            let tmp = TempDir::new().unwrap();
            let path = tmp.path().join("attrs_libhdf5.h5");
            b.iter(|| {
                let file = hdf5::File::create(&path).unwrap();
                let ds = file
                    .new_dataset::<f64>()
                    .shape([3])
                    .create("data")
                    .unwrap();
                ds.write(&[1.0f64, 2.0, 3.0]).unwrap();
                for i in 0..k {
                    ds.new_attr::<i64>()
                        .create(&format!("attr_{i:04}"))
                        .unwrap()
                        .write_scalar(&(i as i64))
                        .unwrap();
                }
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Workload: metadata_attrs_read
// Open a pre-built file and read all K attributes back.
// ---------------------------------------------------------------------------

fn bench_metadata_attrs_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("metadata_attrs_read");

    for &k in &[4usize, 16, 64, 128] {
        // Build the reference file in memory.
        let bytes = {
            let mut fb = FileBuilder::new();
            let ds = fb
                .create_dataset("data")
                .with_f64_data(&[1.0, 2.0, 3.0])
                .with_shape(&[3]);
            for i in 0..k {
                ds.set_attr(&format!("attr_{i:04}"), AttrValue::I64(i as i64));
            }
            fb.finish().unwrap()
        };

        group.throughput(Throughput::Elements(k as u64));

        group.bench_with_input(BenchmarkId::new("clawhdf5", k), &bytes, |b, raw| {
            b.iter(|| {
                let file = File::from_bytes(raw.clone()).unwrap();
                let ds = file.dataset("data").unwrap();
                ds.attrs().unwrap()
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Workload: metadata_groups_create
// Create K top-level groups (no datasets inside).
// Measures link-storage allocation: compact → dense B-tree transition.
// ---------------------------------------------------------------------------

fn bench_metadata_groups_create(c: &mut Criterion) {
    let mut group = c.benchmark_group("metadata_groups_create");

    for &k in &[4usize, 16, 32, 64] {
        group.throughput(Throughput::Elements(k as u64));

        group.bench_with_input(BenchmarkId::new("clawhdf5", k), &k, |b, &k| {
            let tmp = TempDir::new().unwrap();
            let path = tmp.path().join("groups_create.h5");
            b.iter(|| {
                let mut fb = FileBuilder::new();
                for i in 0..k {
                    let mut g = fb.create_group(&format!("group_{i:04}"));
                    // Minimal dataset inside each group to make it non-trivial.
                    g.create_dataset("x").with_f64_data(&[0.0]);
                    let finished = g.finish();
                    fb.add_group(finished);
                }
                fb.write(&path).unwrap();
            });
        });

        #[cfg(feature = "libhdf5-compare")]
        group.bench_with_input(BenchmarkId::new("libhdf5", k), &k, |b, &k| {
            let tmp = TempDir::new().unwrap();
            let path = tmp.path().join("groups_libhdf5.h5");
            b.iter(|| {
                let file = hdf5::File::create(&path).unwrap();
                for i in 0..k {
                    let g = file.create_group(&format!("group_{i:04}")).unwrap();
                    g.new_dataset::<f64>()
                        .shape([1])
                        .create("x")
                        .unwrap()
                        .write_scalar(&0.0f64)
                        .unwrap();
                }
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Workload: metadata_groups_traverse
// Open a pre-built file with K groups and traverse (list) the root group.
// ---------------------------------------------------------------------------

fn bench_metadata_groups_traverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("metadata_groups_traverse");

    for &k in &[4usize, 16, 32, 64] {
        // Pre-build.
        let bytes = {
            let mut fb = FileBuilder::new();
            for i in 0..k {
                let mut g = fb.create_group(&format!("group_{i:04}"));
                g.create_dataset("x").with_f64_data(&[0.0]);
                let finished = g.finish();
                fb.add_group(finished);
            }
            fb.finish().unwrap()
        };

        group.throughput(Throughput::Elements(k as u64));

        group.bench_with_input(BenchmarkId::new("clawhdf5", k), &bytes, |b, raw| {
            b.iter(|| {
                let file = File::from_bytes(raw.clone()).unwrap();
                let root = file.root();
                root.groups().unwrap()
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Workload: metadata_roundtrip_string_attrs
// Write and read back K variable-length string attributes.
// String attrs require a dedicated VL heap entry — distinct from numeric ones.
// ---------------------------------------------------------------------------

fn bench_metadata_string_attrs(c: &mut Criterion) {
    let mut group = c.benchmark_group("metadata_string_attrs");

    for &k in &[4usize, 16, 32] {
        group.throughput(Throughput::Elements(k as u64));

        group.bench_with_input(BenchmarkId::new("clawhdf5", k), &k, |b, &k| {
            b.iter(|| {
                let mut fb = FileBuilder::new();
                let ds = fb
                    .create_dataset("data")
                    .with_f64_data(&[1.0])
                    .with_shape(&[1]);
                for i in 0..k {
                    ds.set_attr(
                        &format!("label_{i:04}"),
                        AttrValue::String(format!("value-{i}-some-longer-string-payload")),
                    );
                }
                let bytes = fb.finish().unwrap();

                // Immediately read back to exercise both directions.
                let file = File::from_bytes(bytes).unwrap();
                let ds_r = file.dataset("data").unwrap();
                ds_r.attrs().unwrap()
            });
        });
    }

    group.finish();
}

criterion_group!(
    meta_benches,
    bench_metadata_attrs_write,
    bench_metadata_attrs_read,
    bench_metadata_groups_create,
    bench_metadata_groups_traverse,
    bench_metadata_string_attrs,
);
criterion_main!(meta_benches);
