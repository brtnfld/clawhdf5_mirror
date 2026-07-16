//! Result reporting: the P2.4 `attack-results/matrix.csv` artifact plus a
//! human-readable stdout table.

use std::fmt::Write as _;
use std::time::Duration;

/// The outcome of one attack, matching the P2.4 CSV schema exactly:
/// `threat_class, attack_id, dataset, detected, verifier_fn, latency_ms, root_cause`.
pub struct AttackResult {
    /// Threat class from S2-D2-Yr2 §`sec:threat` (`T1`..`T8`).
    pub threat_class: &'static str,
    /// Specific attack variant (e.g. `T1a`, `T1b`).
    pub attack_id: &'static str,
    /// Which test dataset this attack ran against (`NOAA-GOES18` or `synthetic`).
    pub dataset: &'static str,
    /// Whether the verification API rejected the tampered state.
    pub detected: bool,
    /// The verification function/API that caught it (or would need to).
    pub verifier_fn: &'static str,
    /// Wall-clock time to run the targeted modification + verification call.
    pub latency: Duration,
    /// Required when `detected` is `false`: why, and what would close the gap.
    pub root_cause: Option<&'static str>,
}

impl AttackResult {
    #[must_use]
    pub fn latency_ms(&self) -> f64 {
        self.latency.as_secs_f64() * 1000.0
    }
}

/// Quote a CSV field per RFC 4180 if it contains a comma, quote, or newline
/// (several `root_cause` values do), doubling any embedded quotes.
fn csv_field(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

/// Render the CSV artifact required by P2.4 (`attack-results/matrix.csv`).
#[must_use]
pub fn to_csv(results: &[AttackResult]) -> String {
    let mut out = String::new();
    out.push_str("threat_class,attack_id,dataset,detected,verifier_fn,latency_ms,root_cause\n");
    for r in results {
        let _ = writeln!(
            out,
            "{},{},{},{},{},{:.4},{}",
            csv_field(r.threat_class),
            csv_field(r.attack_id),
            csv_field(r.dataset),
            if r.detected { "yes" } else { "no" },
            csv_field(r.verifier_fn),
            r.latency_ms(),
            csv_field(r.root_cause.unwrap_or("")),
        );
    }
    out
}

/// Print a human-readable results table to stdout.
pub fn print_table(results: &[AttackResult]) {
    println!(
        "{:<4} {:<8} {:<12} {:<9} {:<28} {:>12}  root_cause",
        "T#", "attack", "dataset", "detected", "verifier_fn", "latency_ms"
    );
    println!("{}", "-".repeat(110));
    for r in results {
        println!(
            "{:<4} {:<8} {:<12} {:<9} {:<28} {:>12.4}  {}",
            r.threat_class,
            r.attack_id,
            r.dataset,
            if r.detected { "yes" } else { "no" },
            r.verifier_fn,
            r.latency_ms(),
            r.root_cause.unwrap_or(""),
        );
    }
    println!("{}", "-".repeat(110));
    // Plain counts only -- whether every undetected attack actually has a
    // documented root_cause isn't known yet at this point (main() checks that
    // afterward, and prints its own, accurate summary once it has); asserting
    // it here unconditionally would be a claim this function hasn't verified.
    let undetected = results.iter().filter(|r| !r.detected).count();
    println!(
        "{}/{} attacks detected ({undetected} undetected)",
        results.len() - undetected,
        results.len(),
    );
}
