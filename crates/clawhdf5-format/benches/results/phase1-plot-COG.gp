# Figure for phase1-explanatory-note-COG.md.
#
# Data points are the per-cell medians already reported and verified in
# that note's "Expected trends" table (computed from phase1-COG.csv's 30
# raw trials per cell). Inlined here as a datablock rather than re-deriving
# medians in gnuplot, since the source-of-truth numbers are the committed
# CSV and the note's own table — this script only renders them.
#
# Run with: gnuplot phase1-plot-COG.gp

set terminal pngcairo size 1100,750 enhanced font 'Helvetica,11'
set output 'phase1-plot-COG.png'

set title "P1.6 Phase 1 benchmark — median wall time vs. chunk count\n{/*0.8 COG, AMD Ryzen 9 9950X3D, synthetic 10GB file (3 chunk sizes) + NOAA sample, 30 trials/cell}"
set xlabel "n\\_chunks (log scale)"
set ylabel "median wall\\_time\\_ms (log scale)"
set logscale x
set logscale y
set grid xtics ytics lt 0 lc rgb '#dddddd'
set key outside right top box
set xtics (240,3413,13652,54608)
set format y "10^{%T}"
set format x "%.0f"

$DATA << EOD
# n_chunks  verify_dataset  full_rebuild  flat_verify  verify_chunk  extend_merkle  update_merkle
240        2.414           2.402         3.775        0.0100        0.0010         0.0010
3413       375.17          377.35        1278.566     0.0830        0.0040         0.0030
13652      431.98          432.26        1278.646     0.0240        0.0150         0.0150
54608      636.92          633.98        1278.997     0.0100        0.3765         0.3685
EOD

set label 1 "verify\\_chunk falls as N grows:\nO(chunk\\_size), no proof-path term" \
    at 3413,0.083 offset 1,0.6 font 'Helvetica,9' tc rgb '#2ca02c'

set label 2 "extend/update\\_merkle jump\nwhen tree.clone() spills\nL2 -> L3 (Anomaly 1)" \
    at 54608,0.3765 offset -13,0.9 font 'Helvetica,9' tc rgb '#ff7f0e'

set label 3 "flat\\_verify: flat baseline\n(fixed total bytes)" \
    at 13652,1278.6 offset -10,1.0 font 'Helvetica,9' tc rgb '#d62728'

plot $DATA using 1:2 with linespoints lw 2 pt 7  ps 1.3 lc rgb '#1f77b4' title 'verify\_dataset', \
     $DATA using 1:3 with linespoints lw 2 pt 5  ps 1.1 lc rgb '#9467bd' title 'full\_rebuild', \
     $DATA using 1:4 with linespoints lw 2 pt 9  ps 1.3 lc rgb '#d62728' title 'flat\_verify (baseline)', \
     $DATA using 1:5 with linespoints lw 2 pt 13 ps 1.3 lc rgb '#2ca02c' title 'verify\_chunk', \
     $DATA using 1:6 with linespoints lw 2 pt 11 ps 1.3 lc rgb '#ff7f0e' title 'extend\_merkle', \
     $DATA using 1:7 with linespoints lw 2 pt 4  ps 1.1 lc rgb '#8c564b' title 'update\_merkle'
