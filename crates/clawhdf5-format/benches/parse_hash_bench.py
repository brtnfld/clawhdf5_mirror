#!/usr/bin/env python3
"""
Parse hash benchmark results and convert to CSV format.

Usage:
    ./parse_hash_bench.py [input.txt]

If no input file is specified, reads from the most recent results file.
Output is written to the same directory as input with .csv extension.
"""

import re
import csv
import sys
import platform
import subprocess
from pathlib import Path


def get_cpu_model():
    """Get CPU model string."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":")[1].strip()
    except FileNotFoundError:
        pass

    # macOS fallback
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass

    return "Unknown"


def get_platform():
    """Get platform architecture (x86/arm)."""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64", "i386", "i686"):
        return "x86"
    elif machine in ("arm64", "aarch64", "arm"):
        return "arm"
    return machine


def get_hostname():
    """Get system hostname."""
    return platform.node()


def parse_throughput(text):
    """Parse throughput value and convert to MiB/s.

    Criterion outputs in GiB/s, we convert to MiB/s (1 GiB = 1024 MiB).
    """
    # Match patterns like "2.2168 GiB/s" or "5.5041 GiB/s"
    match = re.search(r"([\d.]+)\s*GiB/s", text)
    if match:
        gib_per_sec = float(match.group(1))
        # Convert GiB/s to MiB/s (1 GiB = 1024 MiB)
        return gib_per_sec * 1024
    return None


def parse_benchmark_results(filepath):
    """Parse criterion benchmark output and extract results.

    Returns list of dicts with benchmark data.
    """
    results = []

    with open(filepath, "r") as f:
        content = f.read()

    # Extract CPU model from header if present
    cpu_match = re.search(r"CPU:\s*(.+)", content)
    cpu_model = cpu_match.group(1).strip() if cpu_match else get_cpu_model()

    # Extract hostname from header if present
    host_match = re.search(r"Hostname:\s*(\S+)", content)
    hostname = host_match.group(1).strip() if host_match else get_hostname()

    plat = get_platform()

    # Pattern to match benchmark results
    # Example: "hash_chunk/SHA-256/64KB time:   [27.192 µs 27.343 µs 27.533 µs]"
    #          "                        thrpt:  [2.2168 GiB/s 2.2322 GiB/s 2.2446 GiB/s]"

    # Find all hash_chunk benchmark results (the primary benchmark group)
    # Matches both "64KB" and "1MB" formats
    pattern = re.compile(
        r"hash_chunk/(\w+[-\d]*)/(\d+)(KB|MB)\s+"
        r"time:\s*\[[\d.]+ [µu]s\s+([\d.]+) [µu]s\s+[\d.]+ [µu]s\]\s*"
        r"thrpt:\s*\[([\d.]+) GiB/s\s+([\d.]+) GiB/s\s+([\d.]+) GiB/s\]",
        re.MULTILINE | re.DOTALL
    )

    for match in pattern.finditer(content):
        alg = match.group(1).lower()
        size_val = int(match.group(2))
        size_unit = match.group(3)

        # Convert to KB
        if size_unit == "MB":
            chunk_kb = size_val * 1024
        else:
            chunk_kb = size_val

        # Throughput values: [low, median, high] in GiB/s
        thrpt_low = float(match.group(5)) * 1024   # Convert to MiB/s
        thrpt_med = float(match.group(6)) * 1024
        thrpt_high = float(match.group(7)) * 1024

        results.append({
            "alg": alg,
            "chunk_size_kb": chunk_kb,
            "throughput_mibs": round(thrpt_med, 2),
            "ci95_low": round(thrpt_low, 2),
            "ci95_high": round(thrpt_high, 2),
            "platform": plat,
            "hostname": hostname,
            "cpu_model": cpu_model,
        })

    return results


def write_csv(results, output_path):
    """Write results to CSV file."""
    fieldnames = [
        "alg",
        "chunk_size_kb",
        "throughput_mibs",
        "ci95_low",
        "ci95_high",
        "platform",
        "hostname",
        "cpu_model",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"CSV written to: {output_path}")


def find_latest_results():
    """Find the most recent results file."""
    results_dir = Path(__file__).parent / "results"
    txt_files = list(results_dir.glob("hash-bench-*.txt"))
    if not txt_files:
        return None
    return max(txt_files, key=lambda p: p.stat().st_mtime)


def main():
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
    else:
        input_path = find_latest_results()
        if not input_path:
            print("No results file found. Run the benchmark first:")
            print("  ./benches/run_hash_bench.sh")
            sys.exit(1)

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    print(f"Parsing: {input_path}")

    results = parse_benchmark_results(input_path)

    if not results:
        print("No benchmark results found in file.")
        print("Make sure the file contains criterion benchmark output.")
        sys.exit(1)

    print(f"Found {len(results)} benchmark results")

    # Output CSV to same directory with .csv extension
    output_path = input_path.with_suffix(".csv")
    write_csv(results, output_path)

    # Print summary table
    print("\nSummary:")
    print(f"{'Algorithm':<10} {'Chunk':<8} {'Throughput (MiB/s)':<20} {'95% CI'}")
    print("-" * 60)
    for r in results:
        chunk = f"{r['chunk_size_kb']} KB"
        thrpt = f"{r['throughput_mibs']:.1f}"
        ci = f"[{r['ci95_low']:.1f}, {r['ci95_high']:.1f}]"
        print(f"{r['alg']:<10} {chunk:<8} {thrpt:<20} {ci}")


if __name__ == "__main__":
    main()
