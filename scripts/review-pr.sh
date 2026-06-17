#!/usr/bin/env bash
# review-pr.sh — assemble PR review context for clawhdf5.
#
# Emits ONE document: the expert review prompt (docs/PR_REVIEW_PROMPT.md) +
# the PR's merge-base diff + grep risk-signals over added lines. Hand it to a
# reviewer (human or LLM); see docs/PR_REVIEW_PROMPT.md "How to use this".
#
# Usage:
#   ./scripts/review-pr.sh [BASE_REF]                 # default BASE_REF=main, to stdout
#   ./scripts/review-pr.sh main --out /tmp/review.md  # write to a file
#   ./scripts/review-pr.sh main --out /tmp/r.md --copy  # ...and copy to clipboard
#   git diff main...HEAD | ./scripts/review-pr.sh -   # review an arbitrary piped diff
#
# Then, e.g.:  in Claude Code ask "Review this PR using /tmp/review.md", or
#              ./scripts/review-pr.sh main | claude -p "Perform this review"
#
# Flags:
#   BASE_REF      branch/ref to diff against (positional; default: main)
#   --out FILE    write to FILE instead of stdout
#   --copy        with --out, also copy to clipboard (wl-copy/xclip)
#   -             read the diff from stdin instead of computing it from git
#
# Note: risk-signal line numbers are positions within the diff, not source lines.
#
# Exit codes:
#   0 — context assembled
#   1 — not a git repo / bad base ref / no changes

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROMPT="$ROOT/docs/PR_REVIEW_PROMPT.md"

BASE_REF="main"
OUT=""
COPY=0
STDIN_DIFF=0

for arg in "$@"; do
    case "$arg" in
        --copy) COPY=1 ;;
        --out) OUT="__NEXT__" ;;
        -) STDIN_DIFF=1 ;;
        *)
            if [[ "$OUT" == "__NEXT__" ]]; then OUT="$arg"; else BASE_REF="$arg"; fi
            ;;
    esac
done

if ! git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    echo "error: not a git repository: $ROOT" >&2
    exit 1
fi

# Collect diff + file list.
if [[ "$STDIN_DIFF" == "1" ]]; then
    DIFF="$(cat)"
    FILES="$(printf '%s\n' "$DIFF" | grep '^+++ b/' | sed 's|^+++ b/||')"
    RANGE="(piped diff)"
else
    if ! git -C "$ROOT" rev-parse --verify "$BASE_REF" >/dev/null 2>&1; then
        echo "error: base ref not found: $BASE_REF" >&2
        exit 1
    fi
    MERGE_BASE="$(git -C "$ROOT" merge-base "$BASE_REF" HEAD)"
    RANGE="${BASE_REF}...HEAD (merge-base ${MERGE_BASE:0:9})"
    DIFF="$(git -C "$ROOT" diff "$MERGE_BASE"...HEAD)"
    FILES="$(git -C "$ROOT" diff --name-only "$MERGE_BASE"...HEAD)"
fi

if [[ -z "${DIFF// }" ]]; then
    echo "error: no changes to review against $BASE_REF" >&2
    exit 1
fi

# Risk signals: heuristics that map to the review prompt's priorities.
risk_grep() { printf '%s\n' "$DIFF" | grep -E '^\+' | grep -nE "$1" || true; }

UNSAFE="$(risk_grep '\bunsafe\b')"
PANICS="$(risk_grep '\.unwrap\(\)|\.expect\(|panic!|unreachable!|todo!|unimplemented!')"
RAWSLICE="$(risk_grep 'from_raw_parts|Box::from_raw|Box::into_raw|transmute|\bas \*')"
UNCHECKED_IDX="$(risk_grep 'as u8|as u16|as u32|as i32|as usize')"
FFI="$(printf '%s\n' "$FILES" | grep -E 'clawhdf5-(android|napi|py)/' || true)"
PARSERS="$(printf '%s\n' "$FILES" | grep -E 'clawhdf5-format/src/' || true)"
FUZZ="$(printf '%s\n' "$FILES" | grep -E 'fuzz/' || true)"
WAL_HNSW="$(printf '%s\n' "$FILES" | grep -E 'wal\.rs|clawhdf5-ann/' || true)"

section() { [[ -n "${2// }" ]] && { echo "### $1"; echo '```'; printf '%s\n' "$2"; echo '```'; echo; }; }

# Assemble.
{
    cat "$PROMPT" 2>/dev/null || echo "warning: $PROMPT missing" >&2
    echo
    echo "---"
    echo
    echo "## PR under review"
    echo
    echo "- **Range:** $RANGE"
    echo "- **Changed files:**"
    echo '```'
    printf '%s\n' "$FILES"
    echo '```'
    echo
    echo "## Automated risk signals (verify, don't trust)"
    echo
    echo "These are grep heuristics over added lines, mapped to prompt priorities."
    echo "Absence here is not a clean bill of health; presence is a place to look."
    echo
    section "Added \`unsafe\` (→ priority 2)" "$UNSAFE"
    section "Added panic-capable calls — unwrap/expect/panic! (→ priority 1)" "$PANICS"
    section "Raw pointer / transmute / Box raw (→ priority 2,3)" "$RAWSLICE"
    section "Integer casts — check truncation on untrusted paths (→ priority 1)" "$UNCHECKED_IDX"
    section "Touched FFI crates (→ priority 3)" "$FFI"
    section "Touched format parsers — need fuzz + interop coverage (→ priority 1,5)" "$PARSERS"
    section "Fuzz targets touched" "$FUZZ"
    section "Touched WAL / HNSW (→ priority 4)" "$WAL_HNSW"
    echo "## Full diff"
    echo
    echo '```diff'
    printf '%s\n' "$DIFF"
    echo '```'
} > "${OUT:-/dev/stdout}"

if [[ -n "$OUT" ]]; then
    echo "wrote review context → $OUT" >&2
    if [[ "$COPY" == "1" ]]; then
        if command -v wl-copy >/dev/null 2>&1; then
            wl-copy < "$OUT" && echo "copied to clipboard (wl-copy)" >&2
        elif command -v xclip >/dev/null 2>&1; then
            xclip -selection clipboard < "$OUT" && echo "copied to clipboard (xclip)" >&2
        else
            echo "note: --copy ignored, no wl-copy/xclip found" >&2
        fi
    fi
elif [[ "$COPY" == "1" ]]; then
    echo "note: --copy requires --out <file>" >&2
fi
