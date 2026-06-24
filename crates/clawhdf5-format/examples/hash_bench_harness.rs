//! P1.2 hash-algorithm benchmark harness (SHA-256 vs BLAKE3 vs K12).
//!
//! `benches/hash_bench.rs` + `run_hash_bench.sh`/`parse_hash_bench.py` already
//! exist and work, but they rely on Criterion's own adaptive-sampling
//! statistics, and the parsed CSV's column names/units (`throughput_mibs`,
//! base-1024) don't match the spec's literal "Artifacts:" requirement
//! (`throughput_mbs`). This harness instead implements the exact Statistical
//! Protocol (S2-D2 spec, p.52) already used by the P1.2b baseline harness —
//! 5 discarded warmup trials, 30 measured trials, median + 95% bootstrap
//! confidence interval — and writes precisely the required columns.
//!
//! Run with:
//!   cargo run -p clawhdf5-format --release --example hash_bench_harness --features merkle
//!
//! Output: benches/results/hash-bench-$(hostname).csv

use clawhdf5_format::merkle::{HashAlg, hash_chunk};

use std::fs;
use std::time::Instant;

const CHUNK_SIZES_KB: &[usize] = &[64, 256, 1024];

/// Discarded warmup iterations run (and not recorded) before the measured
/// ones, per the Statistical Protocol (S2-D2 spec, p.52: "minimum 30 trials
/// after 5 discarded warmups").
const WARMUP_TRIALS: usize = 5;

/// Measured trials per (alg, chunk size) cell, per the same protocol.
const TRIALS: usize = 30;

/// Bootstrap resamples used to compute the 95% CI on the median.
const BOOTSTRAP_ITERATIONS: usize = 2000;

fn csv_field(s: &str) -> String {
    if s.contains(',') || s.contains('"') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn cpu_model() -> String {
    fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn hostname() -> String {
    std::process::Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn ram_gb() -> f64 {
    fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|l| l.starts_with("MemTotal:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|s| s.parse::<f64>().ok())
        })
        .map(|kb| kb / 1024.0 / 1024.0)
        .unwrap_or(f64::NAN)
}

fn now_utc_iso() -> String {
    std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// x86_64/aarch64 -> the spec's "x86"/"arm" platform labels.
fn platform() -> &'static str {
    match std::env::consts::ARCH {
        "x86_64" | "x86" => "x86",
        "aarch64" | "arm" => "arm",
        other => other,
    }
}

/// Minimal xorshift64* PRNG so bootstrap resampling needs no extra
/// dependency; not cryptographic, just a deterministic resampler.
/// (Duplicated from `clawhdf5/examples/baselines_bench.rs` rather than
/// shared, since both are standalone example binaries.)
struct Xorshift64(u64);

impl Xorshift64 {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn next_index(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

fn median(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    let mut s = samples.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = (p * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Median plus 95% bootstrap confidence interval, per the Statistical
/// Protocol (S2-D2 spec, p.52). `seed` only needs to differ across call
/// sites to avoid correlated resampling; it is not a security boundary.
fn bootstrap_median_ci(samples: &[f64], seed: u64) -> (f64, f64, f64) {
    if samples.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    if samples.len() == 1 {
        return (samples[0], samples[0], samples[0]);
    }
    let mut rng = Xorshift64(seed | 1);
    let n = samples.len();
    let mut medians: Vec<f64> = Vec::with_capacity(BOOTSTRAP_ITERATIONS);
    for _ in 0..BOOTSTRAP_ITERATIONS {
        let resample: Vec<f64> = (0..n).map(|_| samples[rng.next_index(n)]).collect();
        medians.push(median(&resample));
    }
    medians.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (
        median(samples),
        percentile(&medians, 0.025),
        percentile(&medians, 0.975),
    )
}

/// Deterministic, non-trivially-compressible chunk data — same pattern as
/// `hash_bench.rs::make_chunk`.
fn make_chunk(size: usize) -> Vec<u8> {
    (0..size)
        .map(|i| (i ^ (i >> 8) ^ (i >> 16)) as u8)
        .collect()
}

struct Row {
    alg: &'static str,
    chunk_size_kb: usize,
    throughput_mbs: f64,
    ci95_low: f64,
    ci95_high: f64,
    /// All 30 measured per-trial throughputs, kept alongside the summary
    /// row so callers can also report raw spread (min/max/IQR) — not part
    /// of the spec-mandated CSV columns, written to a separate sidecar file.
    raw_samples: Vec<f64>,
}

fn bench_alg_size(alg: HashAlg, alg_name: &'static str, chunk_kb: usize, seed: u64) -> Row {
    let data = make_chunk(chunk_kb * 1024);

    for _ in 0..WARMUP_TRIALS {
        let _ = hash_chunk(&data, alg);
    }

    let mut throughputs_mbs = Vec::with_capacity(TRIALS);
    for _ in 0..TRIALS {
        let start = Instant::now();
        let _ = hash_chunk(&data, alg);
        let elapsed_ns = start.elapsed().as_nanos() as f64;
        // MB/s, decimal (1 MB = 1_000_000 bytes), matching the spec's
        // "throughput_mbs" column name literally rather than MiB/s.
        let mbs = (data.len() as f64) * 1000.0 / elapsed_ns;
        throughputs_mbs.push(mbs);
    }

    let (med, lo, hi) = bootstrap_median_ci(&throughputs_mbs, seed);
    Row {
        alg: alg_name,
        chunk_size_kb: chunk_kb,
        throughput_mbs: med,
        ci95_low: lo,
        ci95_high: hi,
        raw_samples: throughputs_mbs,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== P1.2 hash benchmark harness (SHA-256 / BLAKE3 / K12) ===");
    println!("warmup={WARMUP_TRIALS} trials={TRIALS}");

    let host = hostname();
    let cpu = cpu_model();
    let plat = platform();

    let algs: &[(HashAlg, &'static str)] = &[
        (HashAlg::Sha256, "sha256"),
        (HashAlg::Blake3, "blake3"),
        (HashAlg::K12, "k12"),
    ];

    let mut rows = Vec::new();
    let mut seed: u64 = 0x5EED_0001;
    for &(alg, name) in algs {
        for &kb in CHUNK_SIZES_KB {
            println!("--- {name} / {kb} KB ---");
            let row = bench_alg_size(alg, name, kb, seed);
            seed = seed.wrapping_add(1);
            println!(
                "  {:.1} MB/s [95% CI {:.1}, {:.1}]",
                row.throughput_mbs, row.ci95_low, row.ci95_high
            );
            rows.push(row);
        }
    }

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let results_dir = format!("{manifest_dir}/benches/results");
    fs::create_dir_all(&results_dir)?;
    let out_path = format!("{results_dir}/hash-bench-{host}.csv");

    let mut csv = format!(
        "# hostname={host} cpu_model=\"{cpu}\" ram_gb={:.1} date={}\n",
        ram_gb(),
        now_utc_iso()
    );
    csv.push_str(
        "alg,chunk_size_kb,throughput_mbs,ci95_low,ci95_high,platform,hostname,cpu_model\n",
    );
    for r in &rows {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{}\n",
            csv_field(r.alg),
            r.chunk_size_kb,
            r.throughput_mbs,
            r.ci95_low,
            r.ci95_high,
            csv_field(plat),
            csv_field(&host),
            csv_field(&cpu),
        ));
    }
    fs::write(&out_path, csv)?;

    let raw_out_path = format!("{results_dir}/hash-bench-raw-trials-{host}.csv");
    let mut raw_csv = format!(
        "# hostname={host} cpu_model=\"{cpu}\" ram_gb={:.1} date={}\n",
        ram_gb(),
        now_utc_iso()
    );
    raw_csv.push_str("alg,chunk_size_kb,trial,throughput_mbs\n");
    for r in &rows {
        for (i, sample) in r.raw_samples.iter().enumerate() {
            raw_csv.push_str(&format!(
                "{},{},{},{}\n",
                csv_field(r.alg),
                r.chunk_size_kb,
                i,
                sample,
            ));
        }
    }
    fs::write(&raw_out_path, raw_csv)?;

    println!();
    println!("Wrote {} rows to {out_path}", rows.len());
    println!("Wrote raw per-trial samples to {raw_out_path}");
    println!(
        "Note: benches/results/ is gitignored — `git add -f` these files if you intend to commit them as P1.2 artifacts."
    );

    Ok(())
}
