//! Microbenchmark comparing SHA-256, BLAKE3, and KangarooTwelve (K12)
//! hash algorithms at various chunk sizes.
//!
//! Run with:
//!   cargo bench --features merkle -p clawhdf5-format --bench hash_bench
//!
//! To save results with system info:
//!   ./benches/run_hash_bench.sh

use clawhdf5_format::merkle::{hash_chunk, HashAlg};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

/// Chunk sizes to benchmark: 64 KB, 256 KB, 1 MB
const CHUNK_SIZES: &[(usize, &str)] = &[
    (64 * 1024, "64KB"),
    (256 * 1024, "256KB"),
    (1024 * 1024, "1MB"),
];

/// Generate reproducible test data of the given size.
fn make_chunk(size: usize) -> Vec<u8> {
    // Use a simple pattern that's reproducible but not trivially compressible
    (0..size).map(|i| (i ^ (i >> 8) ^ (i >> 16)) as u8).collect()
}

fn bench_hash_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_chunk");

    for &(size, label) in CHUNK_SIZES {
        let data = make_chunk(size);
        group.throughput(Throughput::Bytes(size as u64));

        // SHA-256
        group.bench_with_input(
            BenchmarkId::new("SHA-256", label),
            &data,
            |b, data| b.iter(|| hash_chunk(data, HashAlg::Sha256)),
        );

        // BLAKE3
        group.bench_with_input(
            BenchmarkId::new("BLAKE3", label),
            &data,
            |b, data| b.iter(|| hash_chunk(data, HashAlg::Blake3)),
        );

        // KangarooTwelve (K12)
        group.bench_with_input(
            BenchmarkId::new("K12", label),
            &data,
            |b, data| b.iter(|| hash_chunk(data, HashAlg::K12)),
        );
    }

    group.finish();
}

/// Benchmark raw hash throughput without domain separation prefix.
/// This measures the pure hash algorithm performance.
fn bench_raw_hash_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("raw_hash");

    for &(size, label) in CHUNK_SIZES {
        let data = make_chunk(size);
        group.throughput(Throughput::Bytes(size as u64));

        // SHA-256 raw
        group.bench_with_input(
            BenchmarkId::new("SHA-256", label),
            &data,
            |b, data| {
                use sha2::{Digest, Sha256};
                b.iter(|| {
                    let hash: [u8; 32] = Sha256::digest(data).into();
                    hash
                })
            },
        );

        // BLAKE3 raw
        group.bench_with_input(
            BenchmarkId::new("BLAKE3", label),
            &data,
            |b, data| {
                b.iter(|| {
                    let hash: [u8; 32] = blake3::hash(data).into();
                    hash
                })
            },
        );

        // K12 raw
        group.bench_with_input(
            BenchmarkId::new("K12", label),
            &data,
            |b, data| {
                use k12::digest::{ExtendableOutput, Update};
                use k12::KangarooTwelve;
                b.iter(|| {
                    let mut hasher = KangarooTwelve::default();
                    hasher.update(data);
                    let mut output = [0u8; 32];
                    hasher.finalize_xof_into(&mut output);
                    output
                })
            },
        );
    }

    group.finish();
}

/// Benchmark incremental hashing (simulates streaming large chunks).
fn bench_incremental_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("incremental_hash");

    // Use 1 MB data split into 64 KB blocks
    let total_size = 1024 * 1024;
    let block_size = 64 * 1024;
    let data = make_chunk(total_size);
    let blocks: Vec<&[u8]> = data.chunks(block_size).collect();

    group.throughput(Throughput::Bytes(total_size as u64));

    // SHA-256 incremental
    group.bench_function("SHA-256/1MB_in_64KB_blocks", |b| {
        use sha2::{Digest, Sha256};
        b.iter(|| {
            let mut hasher = Sha256::new();
            for block in &blocks {
                hasher.update(block);
            }
            let hash: [u8; 32] = hasher.finalize().into();
            hash
        })
    });

    // BLAKE3 incremental
    group.bench_function("BLAKE3/1MB_in_64KB_blocks", |b| {
        b.iter(|| {
            let mut hasher = blake3::Hasher::new();
            for block in &blocks {
                hasher.update(block);
            }
            let hash: [u8; 32] = hasher.finalize().into();
            hash
        })
    });

    // K12 incremental
    group.bench_function("K12/1MB_in_64KB_blocks", |b| {
        use k12::digest::{ExtendableOutput, Update};
        use k12::KangarooTwelve;
        b.iter(|| {
            let mut hasher = KangarooTwelve::default();
            for block in &blocks {
                hasher.update(block);
            }
            let mut output = [0u8; 32];
            hasher.finalize_xof_into(&mut output);
            output
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_hash_algorithms,
    bench_raw_hash_throughput,
    bench_incremental_hash,
);
criterion_main!(benches);
