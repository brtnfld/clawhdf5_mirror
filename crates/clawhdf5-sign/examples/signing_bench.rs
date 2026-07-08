//! P2.1 step 5: Signing benchmark harness for RQ7.
//!
//! Measures sign and verify latency for hybrid Ed25519 + ML-DSA-65 signatures
//! across different dataset sizes (10^4 to 10^7 chunks).
//!
//! Usage:
//!   cargo run --example signing_bench --release --features "ed25519,mldsa"
//!
//! Output: benches/results/signing-$(hostname).csv

use clawhdf5_sign::{HashAlg, canonical_payload, sign_root, verify_sig};
use clawhdf5_sign::{HybridSignature, SigningKey, mldsa::MlDsaSigningKey};
use std::io::Write as IoWrite;
use std::time::Instant;

const TRIALS: usize = 30;
const WARMUP_TRIALS: usize = 5;
const N_DATASETS: &[usize] = &[10_000, 100_000, 1_000_000, 10_000_000];

// ─────────────────────────────────────────────────────────────────────────────
// System info helpers
// ─────────────────────────────────────────────────────────────────────────────

fn hostname() -> String {
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
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("model name"))
                    .and_then(|l| l.split(':').nth(1))
                    .map(|v| v.trim().to_string())
            })
            .unwrap_or_default()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        "unknown".into()
    }
}

fn detect_platform() -> String {
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
    println!("\nWriting {}", csv_path.display());

    let mut f = std::fs::File::create(&csv_path)?;
    write_header(&mut f, &host)?;
    for r in &rows {
        write_row(&mut f, r)?;
    }

    println!("Done — {} rows written.", rows.len());
    Ok(())
}
