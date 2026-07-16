//! White-box red-team attack harness (S2-D2-Yr2 §7, task P2.4).
//!
//! Each attack opens a test HDF5-derived dataset, performs one targeted
//! modification using raw file/byte I/O (bypassing every ClawHDF5 write
//! path), then calls the verification API and asserts the expected result.
//!
//! ```text
//! cargo run -p clawhdf5-attack-harness --release
//! cargo run -p clawhdf5-attack-harness --release -- --file /path/to/goes18_sample.nc
//! ```
//!
//! With no `--file`, the harness builds a small synthetic dataset in-process
//! (same "many small chunks" shape as a real GOES-18 tile) so the whole suite
//! is reproducible from a single `cargo run` with no network access or
//! manual download step. Pointing `--file` at a real, pre-downloaded GOES-18
//! ABI L1b file (see `docs/NOAA_DATA.md`) runs the same attacks against that
//! dataset's actual on-disk HDF5 chunk layout instead.
//!
//! Writes `attack-results/matrix.csv` (relative to the crate root), the P2.4
//! artifact: `threat_class, attack_id, dataset, detected, verifier_fn,
//! latency_ms, root_cause`.

mod attacks;
mod fixture;
mod report;

use std::path::PathBuf;

use fixture::HarnessDataset;
use report::AttackResult;

/// Parses `--file <path>` from the command line.
///
/// Returns `Ok(None)` when `--file` wasn't passed at all (the normal
/// synthetic-fallback case), and `Err` when `--file` was passed with no
/// following value -- a user typo that should be reported, not silently
/// treated the same as "no --file given".
fn parse_file_arg() -> Result<Option<PathBuf>, String> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--file" {
            return args
                .next()
                .map(PathBuf::from)
                .map(Some)
                .ok_or_else(|| "--file requires a path argument".to_string());
        }
    }
    Ok(None)
}

fn load_dataset() -> HarnessDataset {
    let file_arg = parse_file_arg().unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    if let Some(path) = file_arg {
        match fixture::load_real_dataset(&path, "Rad") {
            Ok(ds) => {
                println!(
                    "Loaded real dataset: {} ({} chunks, {} bytes)",
                    path.display(),
                    ds.chunk_count(),
                    ds.bytes.len()
                );
                return ds;
            }
            Err(e) => {
                eprintln!(
                    "warning: failed to load real dataset at {} ({e}); falling back to synthetic",
                    path.display()
                );
            }
        }
    }
    let ds = fixture::synthetic_dataset(16, 512);
    println!(
        "Using synthetic dataset ({} chunks, {} bytes) -- pass --file <path> for a real GOES-18 run",
        ds.chunk_count(),
        ds.bytes.len()
    );
    ds
}

fn main() {
    println!("=== ClawHDF5 Attack Harness (S2-D2-Yr2 P2.4) ===\n");

    let ds = load_dataset();
    // Every attack indexes into the dataset's chunks (e.g. `chunk_count() / 2`,
    // `chunk_count() - 1`) on the assumption there's at least one. A dataset
    // with zero populated/allocated chunks is a real possibility for a real
    // HDF5 file (e.g. an entirely fill-valued dataset never written to), not
    // just a hypothetical -- catch it here with a clear message instead of
    // letting the first attack panic on an out-of-bounds index.
    if ds.chunk_count() == 0 {
        eprintln!(
            "error: dataset '{}' has zero chunks; the attack suite needs at least one chunk to tamper with",
            ds.label
        );
        std::process::exit(1);
    }
    println!();

    let results: Vec<AttackResult> = vec![
        // Dataset attacks (real GOES-18 chunks or synthetic fallback).
        attacks::t1a_chunk_data_tamper(&ds),
        attacks::t1b_companion_node_tamper(&ds),
        attacks::t2_single_bit_corruption(&ds),
        attacks::t6a_root_attribute_stripped(&ds),
        attacks::subset_a_omitted_chunk(&ds),
        attacks::subset_b_substituted_chunk(&ds),
        attacks::subset_c_wrong_coverage(&ds),
        // Mechanism attacks (purpose-built minimal fixtures).
        attacks::t3_provenance_forgery(),
        attacks::t4a_whole_file_rollback(),
        attacks::t4b_selective_chunk_rollback(),
        attacks::t5_post_quantum_forgery(),
        attacks::t6b_algorithm_downgrade(),
        attacks::t7_verification_dos(),
        attacks::t8_structural_leakage(),
    ];

    report::print_table(&results);

    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("attack-results");
    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        eprintln!("warning: could not create {}: {e}", out_dir.display());
    }
    let csv_path = out_dir.join("matrix.csv");
    match std::fs::write(&csv_path, report::to_csv(&results)) {
        Ok(()) => println!("\nWrote {}", csv_path.display()),
        Err(e) => eprintln!("warning: could not write {}: {e}", csv_path.display()),
    }

    let undetected = results.iter().filter(|r| !r.detected).count();
    let undocumented = results
        .iter()
        .filter(|r| !r.detected && r.root_cause.is_none())
        .count();
    if undocumented > 0 {
        eprintln!(
            "\nERROR: {undocumented} undetected attack(s) have no documented root_cause \
             (P2.4 requires either a fix or a documented limitation)."
        );
        std::process::exit(1);
    }
    println!(
        "\n{} attack(s) detected, {} undetected (each with a documented root_cause).",
        results.len() - undetected,
        undetected
    );
}
