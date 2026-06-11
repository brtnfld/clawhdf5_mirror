#!/bin/bash
# Run hash algorithm benchmark and save results with system info.
#
# Usage: ./benches/run_hash_bench.sh
#
# Output: benches/results/hash-bench-$(hostname).txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
HOSTNAME="$(hostname)"
OUTPUT_FILE="${RESULTS_DIR}/hash-bench-${HOSTNAME}.txt"
TIMESTAMP="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

# Ensure results directory exists
mkdir -p "${RESULTS_DIR}"

# Get CPU model
get_cpu_model() {
    if [[ -f /proc/cpuinfo ]]; then
        grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//'
    elif command -v sysctl &>/dev/null; then
        sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown"
    else
        echo "Unknown"
    fi
}

# Get RAM size
get_ram_size() {
    if [[ -f /proc/meminfo ]]; then
        local kb
        kb=$(grep -m1 'MemTotal' /proc/meminfo | awk '{print $2}')
        echo "$((kb / 1024 / 1024)) GB"
    elif command -v sysctl &>/dev/null; then
        local bytes
        bytes=$(sysctl -n hw.memsize 2>/dev/null || echo "0")
        echo "$((bytes / 1024 / 1024 / 1024)) GB"
    else
        echo "Unknown"
    fi
}

# Get OS info
get_os_info() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        echo "${PRETTY_NAME:-${NAME} ${VERSION}}"
    elif command -v sw_vers &>/dev/null; then
        echo "macOS $(sw_vers -productVersion)"
    else
        uname -sr
    fi
}

# Get Rust version
get_rust_version() {
    rustc --version 2>/dev/null || echo "Unknown"
}

CPU_MODEL="$(get_cpu_model)"
RAM_SIZE="$(get_ram_size)"
OS_INFO="$(get_os_info)"
RUST_VERSION="$(get_rust_version)"

# Write header
cat > "${OUTPUT_FILE}" << EOF
================================================================================
Hash Algorithm Benchmark Results
================================================================================
Timestamp:    ${TIMESTAMP}
Hostname:     ${HOSTNAME}
CPU:          ${CPU_MODEL}
RAM:          ${RAM_SIZE}
OS:           ${OS_INFO}
Rust:         ${RUST_VERSION}
================================================================================

Benchmarking SHA-256, BLAKE3, and KangarooTwelve (K12)
Chunk sizes: 64 KB, 256 KB, 1 MB

================================================================================
RESULTS
================================================================================

EOF

echo "Running hash benchmark..."
echo "Results will be saved to: ${OUTPUT_FILE}"
echo ""

# Change to crate directory and run benchmark
cd "${SCRIPT_DIR}/.."

# Run benchmark and append to output file
# Use --noplot to skip HTML generation (faster)
cargo bench --features merkle --bench hash_bench 2>&1 | tee -a "${OUTPUT_FILE}"

echo ""
echo "================================================================================
" >> "${OUTPUT_FILE}"
echo "Benchmark complete. Results saved to: ${OUTPUT_FILE}"

# Generate CSV from results
if command -v python3 &>/dev/null; then
    echo ""
    echo "Generating CSV..."
    python3 "${SCRIPT_DIR}/parse_hash_bench.py" "${OUTPUT_FILE}"
else
    echo "Python3 not found, skipping CSV generation."
    echo "Run manually: ./benches/parse_hash_bench.py ${OUTPUT_FILE}"
fi
