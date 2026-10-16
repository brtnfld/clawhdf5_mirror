//! Single-chunk verification against a cached root vs. a fresh signed root,
//! and in-memory update vs. update plus hybrid re-signing, on one machine.
//!
//! Table I of the CCGrid paper times single-chunk verification against a root
//! the verifier already trusts and updates that neither re-sign nor persist,
//! while the hybrid root-signature latency was measured separately. This
//! harness puts the four on the same machine, in one process, on the same
//! chunk sets, so they can be added:
//!
//! - `verify_cached`  — `verify_proof_standalone` against a trusted root
//!   (Table I's `verify_chunk`).
//! - `verify_fresh`   — verify the hybrid Ed25519 + ML-DSA-65 signature over the
//!   canonical payload binding the root, then the inclusion proof: what a
//!   reader pays the first time it sees a root.
//! - `update`         — hash the new chunk and recompute its path (Table I).
//! - `update_resign`  — `update`, then build the new canonical payload and
//!   produce a fresh hybrid signature over it. Persistence is not timed.
//! - `sig_verify` / `sig_sign` — the hybrid operations alone.
//!
//! Chunk sets match Table I's synthetic datasets (3.33 GiB at 64 KiB,
//! 256 KiB and 1 MiB chunks), generated in memory from a fixed seed; hashing
//! cost does not depend on content. Single-threaded timed regions; 30
//! measured trials after 5 discarded warm-ups; medians with 95% bootstrap
//! CIs (2000 resamples).
//!
//! Usage:
//!   cargo run -p clawhdf5-sign --release --features "ed25519,mldsa" \
//!     --example root_freshness_bench
//!
//! Output: crates/clawhdf5-format/benches/results/root-freshness-$(hostname).csv

use clawhdf5_format::merkle::MerkleTree;
use clawhdf5_sign::{HashAlg, LeafFormat, canonical_payload, sign_root, verify_sig};
use clawhdf5_sign::{SigningKey, mldsa::MlDsaSigningKey};
use std::hint::black_box;
use std::io::Write as IoWrite;
use std::time::Instant;

const TRIALS: usize = 30;
const WARMUP_TRIALS: usize = 5;
const BOOTSTRAP: usize = 2000;
const SEED: u64 = 0x5D2D_2027;

/// (chunk bytes, chunk count): Table I's three synthetic datasets.
const DATASETS: &[(usize, usize)] = &[(64 << 10, 54_608), (256 << 10, 13_652), (1 << 20, 3_413)];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

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
    std::fs::read_to_string("/proc/cpuinfo")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with("model name"))
        .and_then(|l| l.split(':').nth(1))
        .map(|v| v.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

/// Discard `WARMUP_TRIALS` iterations, then return `TRIALS` timings in ns.
fn time_trials<F: FnMut()>(mut f: F) -> Vec<f64> {
    for _ in 0..WARMUP_TRIALS {
        f();
    }
    (0..TRIALS)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_nanos() as f64
        })
        .collect()
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
}

/// Percentile bootstrap 95% CI of the median.
fn bootstrap_ci(v: &[f64], rng: &mut Rng) -> (f64, f64) {
    let mut meds: Vec<f64> = (0..BOOTSTRAP)
        .map(|_| {
            let sample: Vec<f64> = (0..v.len()).map(|_| v[(rng.next() % v.len() as u64) as usize]).collect();
            median(&sample)
        })
        .collect();
    meds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (meds[BOOTSTRAP * 25 / 1000], meds[BOOTSTRAP * 975 / 1000 - 1])
}

fn alg_name(alg: HashAlg) -> &'static str {
    match alg {
        HashAlg::Sha256 => "sha256",
        HashAlg::Blake3 => "blake3",
        _ => "other",
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let host = hostname();
    let cpu = cpu_model();
    let out_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../clawhdf5-format/benches/results");
    std::fs::create_dir_all(out_dir)?;
    let out_path = format!("{out_dir}/root-freshness-{host}.csv");
    let mut out = std::fs::File::create(&out_path)?;
    writeln!(
        out,
        "# hostname={host} cpu_model=\"{cpu}\" trials={TRIALS} warmup={WARMUP_TRIALS} bootstrap={BOOTSTRAP} seed={SEED:#x}"
    )?;
    writeln!(out, "chunk_bytes,n_chunks,alg,op,median_us,ci95_low_us,ci95_high_us,trial,us")?;

    let ed_key = SigningKey::generate();
    let ml_key = MlDsaSigningKey::generate();
    let ed_pub = ed_key.verifying_key();
    let ml_pub = ml_key.verifying_key();
    let mut boot_rng = Rng(SEED ^ 0xB007);

    let mut emit = |out: &mut std::fs::File, cb: usize, n: usize, alg: &str, op: &str, ns: Vec<f64>| -> std::io::Result<()> {
        let us: Vec<f64> = ns.iter().map(|x| x / 1e3).collect();
        let m = median(&us);
        let (lo, hi) = bootstrap_ci(&us, &mut boot_rng);
        println!("  {:>7} KiB {alg:<6} {op:<14} median {m:>10.2} us  [{lo:.2}, {hi:.2}]", cb >> 10);
        for (i, v) in us.iter().enumerate() {
            writeln!(out, "{cb},{n},{alg},{op},{m:.3},{lo:.3},{hi:.3},{},{v:.3}", i + 1)?;
        }
        Ok(())
    };

    println!("host {host}, cpu {cpu}");
    for &(cb, n) in DATASETS {
        let mut rng = Rng(SEED ^ cb as u64);
        let chunks: Vec<Vec<u8>> = (0..n)
            .map(|_| {
                let mut c = vec![0u8; cb];
                for w in c.chunks_exact_mut(8) {
                    w.copy_from_slice(&rng.next().to_le_bytes());
                }
                c
            })
            .collect();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let mid = n / 2;
        let new_mid: Vec<u8> = chunks[mid].iter().map(|b| b ^ 0xA5).collect();

        for alg in [HashAlg::Sha256, HashAlg::Blake3] {
            let a = alg_name(alg);
            let tree = MerkleTree::from_chunks(&refs, alg);
            let root = *tree.root();
            let companion = [0x43u8; 32];
            let grid = [0x44u8; 32];
            let payload = canonical_payload(&root, &companion, 1, 1_760_000_000, alg, LeafFormat::Content, &grid);
            let sig = sign_root(&payload, &ed_key, &ml_key)?;
            let proof = tree.proof(mid).expect("mid in range");

            let ns = time_trials(|| {
                assert!(MerkleTree::verify_proof_standalone(&root, mid, black_box(&chunks[mid]), &proof));
            });
            emit(&mut out, cb, n, a, "verify_cached", ns)?;

            let ns = time_trials(|| {
                verify_sig(black_box(&payload), &sig, &ed_pub, &ml_pub).expect("signature");
                assert!(MerkleTree::verify_proof_standalone(&root, mid, black_box(&chunks[mid]), &proof));
            });
            emit(&mut out, cb, n, a, "verify_fresh", ns)?;

            let mut t = tree.clone();
            let ns = time_trials(|| {
                let h = alg.hash_leaf(black_box(&new_mid));
                t.update_leaf(mid, h).unwrap();
                black_box(t.root());
            });
            emit(&mut out, cb, n, a, "update", ns)?;

            let mut t = tree.clone();
            let mut version = 1u64;
            let ns = time_trials(|| {
                let h = alg.hash_leaf(black_box(&new_mid));
                t.update_leaf(mid, h).unwrap();
                version += 1;
                let p = canonical_payload(t.root(), &companion, version, 1_760_000_000, alg, LeafFormat::Content, &grid);
                black_box(sign_root(&p, &ed_key, &ml_key).expect("sign"));
            });
            emit(&mut out, cb, n, a, "update_resign", ns)?;
        }
    }

    // The hybrid operations alone, once (independent of chunk size).
    let payload = canonical_payload(&[0x42; 32], &[0x43; 32], 1, 1_760_000_000, HashAlg::Blake3, LeafFormat::Content, &[0x44; 32]);
    let sig = sign_root(&payload, &ed_key, &ml_key)?;
    let ns = time_trials(|| verify_sig(black_box(&payload), &sig, &ed_pub, &ml_pub).expect("signature"));
    emit(&mut out, 0, 0, "-", "sig_verify", ns)?;
    let ns = time_trials(|| {
        black_box(sign_root(black_box(&payload), &ed_key, &ml_key).expect("sign"));
    });
    emit(&mut out, 0, 0, "-", "sig_sign", ns)?;

    println!("wrote {out_path}");
    Ok(())
}
