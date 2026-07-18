//! White-box red-team attack harness (S2-D2-Yr2 §7, task P2.4).
//!
//! Each attack opens a test HDF5-derived dataset, performs one targeted
//! modification using raw file/byte I/O (bypassing every ClawHDF5 write
//! path), then calls the verification API and asserts the expected result.
//!
//! ```text
//! cargo run -p clawhdf5-attack-harness --release
//! cargo run -p clawhdf5-attack-harness --release -- --file /path/to/goes18_sample.nc
//! cargo run -p clawhdf5-attack-harness --release -- \
//!     --eqsim-file /path/to/eqsim_output.h5 --eqsim-dataset <dataset-name>
//! ```
//!
//! P2.4 requires the full attack suite on **both** the NOAA and EQSIM
//! datasets. With no `--file`/`--eqsim-file`, the harness builds small
//! synthetic datasets in-process (representative chunked shapes for each --
//! see `fixture::synthetic_noaa_dataset`/`synthetic_eqsim_dataset`) so the
//! whole suite, on both datasets, is reproducible from a single bare
//! `cargo run` with no network access, no manual download step, and nothing
//! large committed to the repo. Pointing `--file`/`--eqsim-file` at a real,
//! pre-downloaded file runs the same attacks against that dataset's actual
//! on-disk HDF5 chunk layout instead; the two flags are independent, so
//! either, both, or neither may be given. See `docs/NOAA_DATA.md` for where
//! to obtain a real GOES-18 file (`--file` defaults its dataset name to
//! `"Rad"`), and `docs/EQSIM_DATA.md` for EQSIM -- `--eqsim-file` has no
//! assumed dataset name and requires `--eqsim-dataset <name>` alongside it.
//!
//! Writes `attack-results/matrix.csv` (relative to the crate root), the P2.4
//! artifact: `threat_class, attack_id, dataset, detected, verifier_fn,
//! latency_ms, root_cause`. The mechanism attacks (T1d, T1e, T3, T4a, T4b, T5,
//! T6b, T7, T8) target a specific primitive rather than "a chunk in a dataset"
//! (see `attacks.rs`'s module doc) and so run once, independent of dataset,
//! reported with `dataset = n/a`.

mod attacks;
mod fixture;
mod report;

use std::path::PathBuf;

use fixture::HarnessDataset;
use report::AttackResult;

/// Parses `--<flag> <path>` from the command line.
///
/// Returns `Ok(None)` when `flag` wasn't passed at all (the normal
/// synthetic-fallback case), and `Err` when it was passed with no following
/// value -- a user typo that should be reported, not silently treated the
/// same as "flag not given".
fn parse_path_arg(flag: &str) -> Result<Option<PathBuf>, String> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == flag {
            return args
                .next()
                .map(PathBuf::from)
                .map(Some)
                .ok_or_else(|| format!("{flag} requires a path argument"));
        }
    }
    Ok(None)
}

/// Load the NOAA dataset: a real GOES-18 file if `--file <path>` was given
/// and loads successfully, otherwise the synthetic fallback.
fn load_noaa_dataset() -> HarnessDataset {
    let file_arg = parse_path_arg("--file").unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    if let Some(path) = file_arg {
        match fixture::load_real_dataset(&path, "Rad", "NOAA-GOES18") {
            Ok(ds) => {
                println!(
                    "Loaded real NOAA dataset: {} ({} chunks, {} bytes)",
                    path.display(),
                    ds.chunk_count(),
                    ds.bytes.len()
                );
                return ds;
            }
            Err(e) => {
                eprintln!(
                    "warning: failed to load real NOAA dataset at {} ({e}); falling back to synthetic",
                    path.display()
                );
            }
        }
    }
    let ds = fixture::synthetic_noaa_dataset();
    println!(
        "Using synthetic NOAA-shaped dataset ({} chunks, {} bytes) -- pass --file <path> for a real GOES-18 run",
        ds.chunk_count(),
        ds.bytes.len()
    );
    ds
}

/// Load the EQSIM dataset: a real EQSIM output file if `--eqsim-file <path>`
/// was given and loads successfully, otherwise the synthetic fallback.
///
/// Unlike NOAA's `"Rad"`, EQSIM output files have no single conventional
/// dataset/variable name this harness can assume -- see `docs/EQSIM_DATA.md`
/// for how to find it in a given file (e.g. `h5dump -n`). `--eqsim-file`
/// therefore *requires* `--eqsim-dataset <name>` alongside it; guessing a
/// name and silently falling back to synthetic on failure would make a
/// broken `--eqsim-file` invocation look like a successful real-data run
/// (the CSV's dataset column would still just say `EQSIM (synthetic)`,
/// hiding that the real file was never actually read).
fn load_eqsim_dataset() -> HarnessDataset {
    let file_arg = parse_path_arg("--eqsim-file").unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    if let Some(path) = file_arg {
        let dataset_name = parse_path_arg("--eqsim-dataset")
            .unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            })
            .unwrap_or_else(|| {
                eprintln!(
                    "error: --eqsim-file requires --eqsim-dataset <name> alongside it -- \
                     EQSIM output files have no single conventional dataset name to assume \
                     (see docs/EQSIM_DATA.md, e.g. `h5dump -n {}` to find one)",
                    path.display()
                );
                std::process::exit(1);
            });
        let dataset_name = dataset_name.to_string_lossy();
        match fixture::load_real_dataset(&path, &dataset_name, "EQSIM") {
            Ok(ds) => {
                println!(
                    "Loaded real EQSIM dataset: {} [{}] ({} chunks, {} bytes)",
                    path.display(),
                    dataset_name,
                    ds.chunk_count(),
                    ds.bytes.len()
                );
                return ds;
            }
            Err(e) => {
                eprintln!(
                    "error: failed to load real EQSIM dataset at {} [{}] ({e})",
                    path.display(),
                    dataset_name
                );
                std::process::exit(1);
            }
        }
    }
    let ds = fixture::synthetic_eqsim_dataset();
    println!(
        "Using synthetic EQSIM-shaped dataset ({} chunks, {} bytes) -- pass --eqsim-file <path> --eqsim-dataset <name> for a real EQSIM run",
        ds.chunk_count(),
        ds.bytes.len()
    );
    ds
}

/// Every attack indexes into the dataset's chunks (e.g. `chunk_count() / 2`,
/// `chunk_count() - 1`) on the assumption there's at least one. A dataset
/// with zero populated/allocated chunks is a real possibility for a real
/// HDF5 file (e.g. an entirely fill-valued dataset never written to), not
/// just a hypothetical -- catch it here with a clear message instead of
/// letting the first attack panic on an out-of-bounds index.
fn require_nonempty(ds: &HarnessDataset) {
    if ds.chunk_count() == 0 {
        eprintln!(
            "error: dataset '{}' has zero chunks; the attack suite needs at least one chunk to tamper with",
            ds.label
        );
        std::process::exit(1);
    }
}

/// Run the dataset-driven attacks (T1a, T1c, T1b, T2a, T2b, T6a,
/// subset-a/b/c) against one dataset, producing one CSV row per attack for
/// that dataset.
fn run_dataset_attacks(ds: &HarnessDataset) -> Vec<AttackResult> {
    vec![
        attacks::t1a_chunk_data_tamper(ds),
        attacks::t1c_compressed_payload_tamper(ds),
        attacks::t1b_companion_node_tamper(ds),
        attacks::t2a_single_bit_corruption(ds),
        attacks::t2b_burst_error_corruption(ds),
        attacks::t6a_root_attribute_stripped(ds),
        attacks::subset_a_omitted_chunk(ds),
        attacks::subset_b_substituted_chunk(ds),
        attacks::subset_c_wrong_coverage(ds),
        attacks::subset_d_malformed_proof_lengths(ds),
    ]
}

fn main() {
    println!("=== ClawHDF5 Attack Harness (S2-D2-Yr2 P2.4) ===\n");

    let noaa_ds = load_noaa_dataset();
    let eqsim_ds = load_eqsim_dataset();
    require_nonempty(&noaa_ds);
    require_nonempty(&eqsim_ds);
    println!();

    let mut results: Vec<AttackResult> = Vec::new();
    // Dataset attacks, run once per required dataset (P2.4: "repeat the full
    // harness on the EQSIM dataset").
    results.extend(run_dataset_attacks(&noaa_ds));
    results.extend(run_dataset_attacks(&eqsim_ds));
    // Mechanism attacks (purpose-built minimal fixtures, independent of
    // dataset -- see the module doc above).
    results.push(attacks::t1d_directed_companion_forgery());
    results.push(attacks::t1e_second_preimage_node_as_leaf());
    results.push(attacks::t3_provenance_forgery());
    results.push(attacks::t4a_whole_file_rollback());
    results.push(attacks::t4b_selective_chunk_rollback());
    results.push(attacks::t5_post_quantum_forgery());
    results.push(attacks::t6b_algorithm_downgrade());
    results.push(attacks::t7_verification_dos());
    results.push(attacks::t8_structural_leakage());

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
