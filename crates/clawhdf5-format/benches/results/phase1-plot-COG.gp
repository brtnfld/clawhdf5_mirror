# Figure for phase1-explanatory-note-COG.md, rendered as a vector PDF for
# direct \includegraphics use in LaTeX. Two stacked panels:
#   top    - the four scenarios that don't show trial-to-trial spread
#            worth plotting (median + 95% bootstrap CI, line per scenario).
#   bottom - extend_merkle/update_merkle, the two scenarios whose raw
#            30-trial spread includes real cold-start outliers (see the
#            note's "no discarded warmups" caveat) -- drawn as Tukey
#            box-and-whisker plots with outliers beyond 1.5*IQR shown as
#            individual points, rather than collapsed to a median line.
#
# All summary statistics (median/CI for the top panel; quartiles/whiskers/
# outliers for the bottom panel) are computed from phase1-COG.csv's 30 raw
# trials per cell and inlined as datablocks -- the CSV remains the source
# of truth; this script only renders already-verified numbers.
#
# Run with: gnuplot phase1-plot-COG.gp

set terminal pdfcairo size 6in,7in enhanced font 'Helvetica,10'
set output 'phase1-plot-COG.pdf'

set multiplot layout 2,1

# ---------------------------------------------------------------------------
# Top panel: median + 95% CI, line per scenario
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Bottom panel: extend_merkle / update_merkle, Tukey box-and-whisker + outliers
# ---------------------------------------------------------------------------
unset logscale x
set title "extend\\_merkle / update\\_merkle: per-cell distribution (30 trials)"
set xlabel "Chunk Count" font 'Helvetica-Bold,10'
set ylabel "Wall Time (ms)" font 'Helvetica-Bold,10'
set xtics ("240" 1, "3413" 2, "13652" 3, "54608" 4)
set xrange [0.5:4.5]
set yrange [0.0007:1.5]
set boxwidth 0.22
set key inside top left box font 'Helvetica,8'

# pos  q1       whisklo  whiskhi  q3       median   (extend_merkle, offset -0.15)
$EXT << EOD
0.85   0.00100  0.00100  0.00100  0.00100  0.00100
1.85   0.00300  0.00300  0.00500  0.00400  0.00400
2.85   0.01500  0.01500  0.01900  0.01775  0.01500
3.85   0.36900  0.36400  0.48600  0.42675  0.37650
EOD

# pos  q1       whisklo  whiskhi  q3       median   (update_merkle, offset +0.15)
$UPD << EOD
1.15   0.00100  0.00100  0.00100  0.00100  0.00100
2.15   0.00300  0.00300  0.00300  0.00300  0.00300
3.15   0.01500  0.01500  0.01500  0.01500  0.01500
4.15   0.36600  0.36400  0.37500  0.37000  0.36850
EOD

# pos  value  (every trial beyond its cell's 1.5*IQR fence, both scenarios)
$OUTLIERS << EOD
0.85   0.003
1.85   0.017
2.85   0.022
2.85   0.025
2.85   0.029
2.85   0.037
2.85   0.056
3.85   0.529
3.85   0.535
3.85   0.965
2.15   0.004
2.15   0.004
3.15   0.016
4.15   0.383
EOD

set logscale y
plot $EXT using 1:2:3:4:5 with candlesticks lw 1 lc rgb '#ff7f0e' title 'extend\_merkle' whiskerbars 0.5, \
     $EXT using 1:6:6:6:6 with candlesticks lw 2 lc rgb '#ff7f0e' notitle, \
     $UPD using 1:2:3:4:5 with candlesticks lw 1 lc rgb '#8c564b' title 'update\_merkle' whiskerbars 0.5, \
     $UPD using 1:6:6:6:6 with candlesticks lw 2 lc rgb '#8c564b' notitle, \
     $OUTLIERS using 1:2 with points pt 6 ps 0.4 lc rgb '#000000' notitle

unset multiplot
