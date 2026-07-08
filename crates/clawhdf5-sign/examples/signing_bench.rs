//! P2.1 step 5: Signing benchmark harness for RQ7.
//!
//! Measures sign and verify latency for hybrid Ed25519 + ML-DSA-65 signatures
//! across different dataset sizes (10^4 to 10^7 chunks).
//!
//! Usage:
//!   cargo run --example signing_bench --release --features "ed25519,mldsa"
//!
//! To collect the "x86_noavx" row on a machine whose CPU supports AVX-512
//! (runtime `is_x86_feature_detected!` would otherwise mislabel the run),
//! disable AVX-512 codegen and force the platform label explicitly:
//!
//!   RUSTFLAGS="-C target-feature=-avx512f,-avx512bw,-avx512cd,-avx512dq,-avx512vl" \
//!     CLAWHDF5_BENCH_PLATFORM=x86_noavx \
//!     cargo run --example signing_bench --release --features "ed25519,mldsa"
//!
//! On real ARM hardware, no platform override is needed — `detect_platform`
//! reports "arm" automatically. On ephemeral CI runners, set
//! CLAWHDF5_BENCH_HOSTNAME to a stable name so results don't fragment across
//! a new throwaway-hostname CSV on every run:
//!
//!   CLAWHDF5_BENCH_HOSTNAME=github-actions-arm64 \
//!     cargo run --example signing_bench --release --features "ed25519,mldsa"
//!
//! Output: benches/results/signing-$(hostname).csv (merged by platform column;
//! re-running for one platform label refreshes only that label's rows).

use clawhdf5_sign::{HashAlg, canonical_payload, sign_root, verify_sig};
use clawhdf5_sign::{SigningKey, mldsa::MlDsaSigningKey};
use std::io::Write as IoWrite;
use std::time::Instant;

const TRIALS: usize = 30;
const WARMUP_TRIALS: usize = 5;
const N_DATASETS: &[usize] = &[10_000, 100_000, 1_000_000, 10_000_000];

// ─────────────────────────────────────────────────────────────────────────────
// System info helpers
// ─────────────────────────────────────────────────────────────────────────────

fn hostname() -> String {
    // Hosted CI runners get a random ephemeral hostname per run, which would
    // otherwise fragment one platform's results across many throwaway
    // filenames — allow pinning a stable name (e.g. "github-actions-arm64").
    if let Ok(h) = std::env::var("CLAWHDF5_BENCH_HOSTNAME") {
        return h;
    }
    std::process::Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn cpu_model() -> String {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .unwrap_or_default()
            .trim()
            .to_string()
    }
    #[cfg(target_os = "linux")]
    {
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
        let field = |key: &str| -> Option<String> {
            cpuinfo
                .lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split(':').nth(1))
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        };
        // x86 /proc/cpuinfo has "model name"; some ARM boards report "Model".
        field("model name").or_else(|| field("Model")).unwrap_or_else(|| {
            // Cloud ARM64 (e.g. GitHub-hosted aarch64 runners, Neoverse cores)
            // typically lacks a friendly name field — fall back to raw codes.
            let implementer = field("CPU implementer").unwrap_or_else(|| "?".to_string());
            let part = field("CPU part").unwrap_or_else(|| "?".to_string());
            format!("aarch64 implementer={implementer} part={part}")
        })
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        "unknown".into()
    }
}

fn detect_platform() -> String {
    // Runtime CPUID detection can't tell "compiled without AVX-512" apart
    // from "CPU lacks AVX-512" — allow an explicit override for the former.
    if let Ok(p) = std::env::var("CLAWHDF5_BENCH_PLATFORM") {
        return p;
    }
    #[cfg(target_arch = "x86_64")]
    {
        // Check for AVX-512 support
        if is_x86_feature_detected!("avx512f") {
            "x86_avx512".to_string()
        } else {
            "x86_noavx".to_string()
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        "arm".to_string()
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        "unknown".to_string()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CSV output
// ─────────────────────────────────────────────────────────────────────────────

struct BenchmarkRow {
    scheme: &'static str,
    platform: String,
    n_datasets: usize,
    sign_ms: f64,
    verify_ms: f64,
    sig_size_bytes: usize,
    pubkey_size_bytes: usize,
    header_overhead_bytes: usize,
    trial: usize,
}

fn write_header(w: &mut impl IoWrite, hostname: &str) -> std::io::Result<()> {
    let cpu = cpu_model();
    let date = {
        std::process::Command::new("date")
            .arg("+%Y-%m-%dT%H:%M:%SZ")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .unwrap_or_default()
            .trim()
            .to_string()
    };
    writeln!(w, "# hostname={hostname} cpu_model=\"{cpu}\" date={date}")?;
    writeln!(
        w,
        "scheme,platform,n_datasets,sign_ms,verify_ms,sig_size_bytes,pubkey_size_bytes,\
         header_overhead_bytes,trial"
    )
}

fn write_row(w: &mut impl IoWrite, r: &BenchmarkRow) -> std::io::Result<()> {
    writeln!(
        w,
        "{},{},{},{:.3},{:.3},{},{},{},{}",
        r.scheme,
        r.platform,
        r.n_datasets,
        r.sign_ms,
        r.verify_ms,
        r.sig_size_bytes,
        r.pubkey_size_bytes,
        r.header_overhead_bytes,
        r.trial
    )
}

/// Merge this run's rows into `csv_path`, replacing any existing rows for
/// `platform` (the run being (re-)collected) while preserving rows for every
/// other platform already recorded in the file.
///
/// Returns the total number of data rows in the merged file.
fn merge_csv(
    csv_path: &std::path::Path,
    host: &str,
    platform: &str,
    new_rows: &[BenchmarkRow],
) -> std::io::Result<usize> {
    let mut kept_lines: Vec<String> = Vec::new();
    if csv_path.exists() {
        let existing = std::fs::read_to_string(csv_path)?;
        for line in existing.lines() {
            if line.starts_with('#') || line.starts_with("scheme,") {
                continue; // comment/header regenerated below
            }
            let row_platform = line.split(',').nth(1).unwrap_or("");
            if row_platform != platform {
                kept_lines.push(line.to_string());
            }
        }
    }

    let mut f = std::fs::File::create(csv_path)?;
    write_header(&mut f, host)?;
    for line in &kept_lines {
        writeln!(f, "{line}")?;
    }
    for r in new_rows {
        write_row(&mut f, r)?;
    }
    Ok(kept_lines.len() + new_rows.len())
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark harness
// ─────────────────────────────────────────────────────────────────────────────

fn run_hybrid_benchmark(n_datasets: usize, platform: &str, rows: &mut Vec<BenchmarkRow>) {
    println!("  Benchmarking hybrid_mldsa65 for n_datasets={n_datasets}");

    // Generate keys once
    let ed_key = SigningKey::generate();
    let ml_key = MlDsaSigningKey::generate();
    let ed_pub = ed_key.verifying_key();
    let ml_pub = ml_key.verifying_key();

    // Create canonical payload (simulating one dataset)
    let root = [0x42; 32];
    let companion_hash = [0x43; 32];
    let version = 1;
    let timestamp = 1704067200;
    let payload = canonical_payload(&root, &companion_hash, version, timestamp, HashAlg::Blake3);

    // Measure signature size (constant per scheme)
    let sample_sig = sign_root(&payload, &ed_key, &ml_key).unwrap();
    let sig_size_bytes = sample_sig.to_bytes().len();
    let pubkey_size_bytes = 32 + 1952; // Ed25519 (32) + ML-DSA-65 (1952)
    let header_overhead_bytes = sig_size_bytes + pubkey_size_bytes;

    // Warmup
    for _ in 0..WARMUP_TRIALS {
        let sig = sign_root(&payload, &ed_key, &ml_key).unwrap();
        let _ = verify_sig(&payload, &sig, &ed_pub, &ml_pub);
    }

    // Measure signing
    for trial in 0..TRIALS {
        let start = Instant::now();
        let sig = sign_root(&payload, &ed_key, &ml_key).unwrap();
        let sign_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Measure verification
        let start = Instant::now();
        verify_sig(&payload, &sig, &ed_pub, &ml_pub).unwrap();
        let verify_ms = start.elapsed().as_secs_f64() * 1000.0;

        rows.push(BenchmarkRow {
            scheme: "hybrid_mldsa65",
            platform: platform.to_string(),
            n_datasets,
            sign_ms,
            verify_ms,
            sig_size_bytes,
            pubkey_size_bytes,
            header_overhead_bytes,
            trial: trial + 1,
        });
    }
}

fn run_ed25519_benchmark(n_datasets: usize, platform: &str, rows: &mut Vec<BenchmarkRow>) {
    println!("  Benchmarking ed25519 for n_datasets={n_datasets}");

    // Generate keys once
    let ed_key = SigningKey::generate();
    let ed_pub = ed_key.verifying_key();

    // Create canonical payload
    let root = [0x42; 32];
    let companion_hash = [0x43; 32];
    let version = 1;
    let timestamp = 1704067200;
    let payload = canonical_payload(&root, &companion_hash, version, timestamp, HashAlg::Blake3);

    // Measure signature size
    let sample_sig = ed_key.sign_payload(&payload);
    let sig_size_bytes = sample_sig.len();
    let pubkey_size_bytes = 32;
    let header_overhead_bytes = sig_size_bytes + pubkey_size_bytes;

    // Warmup
    for _ in 0..WARMUP_TRIALS {
        let sig = ed_key.sign_payload(&payload);
        let _ = ed_pub.verify_payload(&payload, &sig);
    }

    // Measure signing and verification
    for trial in 0..TRIALS {
        let start = Instant::now();
        let sig = ed_key.sign_payload(&payload);
        let sign_ms = start.elapsed().as_secs_f64() * 1000.0;

        let start = Instant::now();
        ed_pub.verify_payload(&payload, &sig).unwrap();
        let verify_ms = start.elapsed().as_secs_f64() * 1000.0;

        rows.push(BenchmarkRow {
            scheme: "ed25519",
            platform: platform.to_string(),
            n_datasets,
            sign_ms,
            verify_ms,
            sig_size_bytes,
            pubkey_size_bytes,
            header_overhead_bytes,
            trial: trial + 1,
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let host = hostname();
    let platform = detect_platform();

    println!("P2.1 step 5: Signing benchmarks (RQ7)");
    println!("  hostname  : {host}");
    println!("  cpu       : {}", cpu_model());
    println!("  platform  : {platform}");
    println!("  trials    : {TRIALS}");
    println!();

    let mut rows: Vec<BenchmarkRow> = Vec::new();

    for &n in N_DATASETS {
        println!("Benchmarking n_datasets={n}");
        run_ed25519_benchmark(n, &platform, &mut rows);
        run_hybrid_benchmark(n, &platform, &mut rows);
    }

    // Ensure output directory exists
    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/clawhdf5-format/benches/results");
    std::fs::create_dir_all(&out_dir)?;

    let csv_path = out_dir.join(format!("signing-{host}.csv"));
    println!("\nMerging {} platform={platform} row(s) into {}", rows.len(), csv_path.display());

    let total = merge_csv(&csv_path, &host, &platform, &rows)?;

    println!("Done — {} new rows, {total} total rows in file.", rows.len());
    Ok(())
}
