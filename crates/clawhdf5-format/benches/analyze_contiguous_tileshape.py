#!/usr/bin/env python3
"""Compare measured P2.8c tile-shape results against Table `tab:tileshape`.

Reads one or more `contiguous-tileshape*.csv` files (one per storage class)
and prints the measured-vs-published comparison plus the derived findings the
explanatory note has to justify: whether the *ordering* the analytical table
predicts survives measurement, and where the `ceil(l/B)*B` block-tax model
breaks down.

Usage:
    ./analyze_contiguous_tileshape.py results/contiguous-tileshape.csv
"""
import csv
import sys
from collections import defaultdict

MiB = 1024.0 * 1024.0

# Table tab:tileshape, as published: shape -> selection -> (MiB, IOs)
PUBLISHED = {
    "1x250x1000": {"cube10": (9.6, 10), "cube100": (95.7, 100), "plane": (3.8, 4)},
    "4x64x1000": {"cube10": (3.0, 12), "cube100": (73.8, 300), "plane": (15.8, 64)},
    "16x16x1000": {"cube10": (1.0, 16), "cube100": (49.0, 784), "plane": (63.0, 1008)},
    "64x64x64": {
        "cube10": (16.0, 4096),
        "cube100": (432.0, 110592),
        "plane": (4096.0, 1048576),
    },
}
SHAPE_ORDER = ["1x250x1000", "4x64x1000", "16x16x1000", "64x64x64"]
SEL_ORDER = ["cube10", "cube100", "plane"]


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                if not r.get("leaf_shape"):
                    continue
                rows.append(r)
    return rows


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    rows = load(sys.argv[1:])
    by = {}
    classes = []
    for r in rows:
        sc = r["storage_class"]
        if sc not in classes:
            classes.append(sc)
        by[(sc, r["leaf_shape"], r["selection"])] = r

    for sc in classes:
        seq = next(
            (r["seq_read_mbps"] for r in rows if r["storage_class"] == sc and r["seq_read_mbps"]),
            "?",
        )
        print(f"\n{'='*78}\nstorage class: {sc}   (sequential read {seq} MB/s)\n{'='*78}")
        print(
            f"{'shape':<13} {'selection':<9} {'measured':>10} {'published':>10} "
            f"{'ratio':>7} {'IOs':>8} {'pub IOs':>8} {'model MiB':>10} {'meas/model':>10}"
        )
        for shape in SHAPE_ORDER:
            for sel in SEL_ORDER:
                r = by.get((sc, shape, sel))
                if not r:
                    continue
                meas = float(r["bytes_transferred"]) / MiB
                pub, pub_io = PUBLISHED[shape][sel]
                model = float(r["model_bytes_predicted"]) / MiB
                print(
                    f"{shape:<13} {sel:<9} {meas:>10.2f} {pub:>10.1f} "
                    f"{meas/pub:>7.2f}x {int(r['io_ops']):>8} {pub_io:>8} "
                    f"{model:>10.2f} {meas/model if model else float('nan'):>9.2f}x"
                )

        # Does the published ordering (which shape wins per selection) survive?
        print("\n  ordering check (lowest bytes transferred wins):")
        for sel in SEL_ORDER:
            got = sorted(
                (
                    (float(by[(sc, s, sel)]["bytes_transferred"]), s)
                    for s in SHAPE_ORDER
                    if (sc, s, sel) in by
                )
            )
            want = sorted((PUBLISHED[s][sel][0], s) for s in SHAPE_ORDER)
            got_o = [s for _, s in got]
            want_o = [s for _, s in want]
            mark = "MATCHES" if got_o == want_o else "DIFFERS"
            print(f"    {sel:<9} measured: {' < '.join(got_o)}")
            print(f"    {'':<9} published: {' < '.join(want_o)}   [{mark}]")

        # Where does the block-tax model break?
        print("\n  model fidelity (measured / model, by run length):")
        seen = {}
        for shape in SHAPE_ORDER:
            for sel in SEL_ORDER:
                r = by.get((sc, shape, sel))
                if not r:
                    continue
                rl = int(r["run_len_bytes"])
                model = float(r["model_bytes_predicted"])
                if model:
                    seen.setdefault(rl, []).append(
                        float(r["bytes_transferred"]) / model
                    )
        for rl in sorted(seen, reverse=True):
            v = seen[rl]
            print(
                f"    run_len {rl:>9} B: mean over-read {sum(v)/len(v):>6.2f}x "
                f"(n={len(v)})"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
