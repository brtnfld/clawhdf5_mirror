# Figure for phase1-explanatory-note-COG.md, rendered as a vector PDF for
# direct \includegraphics use in LaTeX.
#
# Data points are the per-cell median + 95% bootstrap CI (2000 resamples,
# same protocol as hash_bench_harness.rs/baselines_bench.rs) computed from
# phase1-COG.csv's 30 raw trials per cell. Inlined here as a datablock
# rather than re-deriving them in gnuplot — the committed CSV remains the
# source of truth; this script only renders already-verified numbers.
#
# Run with: gnuplot phase1-plot-COG.gp

set terminal pdfcairo size 6in,4.2in enhanced font 'Helvetica,11'
set output 'phase1-plot-COG.pdf'

set title "P1.6 Phase 1 benchmark -- median wall time vs. chunk count\n{/*0.8 COG, AMD Ryzen 9 9950X3D, synthetic 10GB file (3 chunk sizes) + NOAA sample, median \\& 95% bootstrap CI, 30 trials/cell}"
set xlabel "n\\_chunks"
set ylabel "median wall\\_time\\_ms"
set logscale x
set logscale y
set grid xtics ytics lt 0 lc rgb '#dddddd'
set key outside right top box
set xtics (240,3413,13652,54608)
set format y "10^{%T}"
set format x "%.0f"

# n_chunks  verify_dataset(med,lo,hi)  full_rebuild(med,lo,hi)  flat_verify(med,lo,hi)  verify_chunk(med,lo,hi)  extend_merkle(med,lo,hi)  update_merkle(med,lo,hi)
$DATA << EOD
240    2.4140 2.4080 2.4200    2.4020 2.3990 2.4080    3.7750 3.7740 3.7765    0.0100 0.0100 0.0100    0.0010 0.0010 0.0010    0.0010 0.0010 0.0010
3413   375.1660 374.7775 375.4915    377.3505 376.5590 377.9260    1278.5655 1277.6835 1279.0550    0.0830 0.0830 0.0830    0.0040 0.0030 0.0040    0.0030 0.0030 0.0030
13652  431.9795 431.5420 432.9100    432.2635 431.2660 433.9665    1278.6455 1277.9480 1283.7580    0.0240 0.0240 0.0240    0.0150 0.0150 0.0160    0.0150 0.0150 0.0150
54608  636.9245 635.9070 637.1540    633.9815 633.1950 634.5320    1278.9965 1278.2015 1280.3505    0.0100 0.0100 0.0100    0.3765 0.3700 0.4040    0.3685 0.3665 0.3700
EOD

set label 1 "verify\\_chunk falls as N grows:\nO(chunk\\_size), no proof-path term" \
    at 3413,0.083 offset 1,0.6 font 'Helvetica,9' tc rgb '#2ca02c'

set label 2 "extend/update\\_merkle jump\nwhen tree.clone() spills\nL2 -> L3 (Anomaly 1)" \
    at 54608,0.3765 offset -13,0.9 font 'Helvetica,9' tc rgb '#ff7f0e'

set label 3 "flat\\_verify: flat baseline\n(fixed total bytes)" \
    at 13652,1278.6 offset -10,1.0 font 'Helvetica,9' tc rgb '#d62728'

plot $DATA using 1:2:3:4  with errorlines lw 2 pt 7  ps 1.3 lc rgb '#1f77b4' title 'verify\_dataset', \
     $DATA using 1:5:6:7  with errorlines lw 2 pt 5  ps 1.1 lc rgb '#9467bd' title 'full\_rebuild', \
     $DATA using 1:8:9:10 with errorlines lw 2 pt 9  ps 1.3 lc rgb '#d62728' title 'flat\_verify (baseline)', \
     $DATA using 1:11:12:13 with errorlines lw 2 pt 13 ps 1.3 lc rgb '#2ca02c' title 'verify\_chunk', \
     $DATA using 1:14:15:16 with errorlines lw 2 pt 11 ps 1.3 lc rgb '#ff7f0e' title 'extend\_merkle', \
     $DATA using 1:17:18:19 with errorlines lw 2 pt 4  ps 1.1 lc rgb '#8c564b' title 'update\_merkle'
