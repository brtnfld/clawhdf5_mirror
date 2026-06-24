# Figure for phase1-explanatory-note-COG.md, rendered as a vector PDF for
# direct \includegraphics use in LaTeX. The four scenarios whose 30-trial
# spread is tight enough to read as a single median+CI line:
# verify_dataset, full_rebuild, flat_verify, verify_chunk.
#
# Summary statistics (median + 95% bootstrap CI) are computed from
# phase1-COG.csv's 30 raw trials per cell and inlined below as a datablock
# -- the CSV remains the source of truth; this script only renders
# already-verified numbers.
#
# Run with: gnuplot phase1-plot-lines-COG.gp

set terminal pdfcairo size 6in,4in enhanced font 'Helvetica,10'
set output 'phase1-plot-lines-COG.pdf'

set title "Bulk verification and baseline scenarios"
set xlabel "Chunk Count" font 'Helvetica-Bold,10'
set ylabel "Median Wall Time (ms)" font 'Helvetica-Bold,10'
set logscale x
set logscale y
set yrange [0.005:3000]
set grid xtics ytics lt 0 lc rgb '#dddddd'
set key inside top left box font 'Helvetica,8'
set xtics (240,3413,13652,54608)
set format y "10^{%T}"
set format x "%.0f"

# n_chunks  verify_dataset(med,lo,hi)  full_rebuild(med,lo,hi)  flat_verify(med,lo,hi)  verify_chunk(med,lo,hi)
$DATA << EOD
240    2.4140 2.4080 2.4200    2.4020 2.3990 2.4080    3.7750 3.7740 3.7765    0.0100 0.0100 0.0100
3413   375.1660 374.7775 375.4915    377.3505 376.5590 377.9260    1278.5655 1277.6835 1279.0550    0.0830 0.0830 0.0830
13652  431.9795 431.5420 432.9100    432.2635 431.2660 433.9665    1278.6455 1277.9480 1283.7580    0.0240 0.0240 0.0240
54608  636.9245 635.9070 637.1540    633.9815 633.1950 634.5320    1278.9965 1278.2015 1280.3505    0.0100 0.0100 0.0100
EOD

plot $DATA using 1:2:3:4  with errorlines lw 1 pt 7  ps 0.4 lc rgb '#1f77b4' title 'verify\_dataset', \
     $DATA using 1:5:6:7  with errorlines lw 1 pt 5  ps 0.4 lc rgb '#9467bd' title 'full\_rebuild', \
     $DATA using 1:8:9:10 with errorlines lw 1 pt 9  ps 0.4 lc rgb '#d62728' title 'flat\_verify (baseline)', \
     $DATA using 1:11:12:13 with errorlines lw 1 pt 13 ps 0.4 lc rgb '#2ca02c' title 'verify\_chunk'
