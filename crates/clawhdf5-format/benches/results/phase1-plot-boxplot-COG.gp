# Figure for phase1-explanatory-note-COG.md, rendered as a vector PDF for
# direct \includegraphics use in LaTeX. extend_merkle/update_merkle, the two
# scenarios whose raw 30-trial spread includes cold-start outliers, plotted
# as Tukey box-and-whisker plots (outliers beyond 1.5*IQR shown as open
# circles) rather than collapsed to a median line -- alongside
# update_leaf_inplace, the control scenario that isolates update_leaf's true
# O(log N) cost from the O(N) tree.clone() the other two pay every trial
# (see the note's Anomaly 1).
#
# Quartile/whisker/outlier statistics are computed from phase1-COG.csv's 30
# raw trials per cell (after the harness's 5 discarded warmup trials) and
# inlined below as datablocks -- the CSV remains the source of truth; this
# script only renders already-verified numbers.
#
# Run with: gnuplot phase1-plot-boxplot-COG.gp

set terminal pdfcairo size 6in,4in enhanced font 'Helvetica,10'
set output 'phase1-plot-boxplot-COG.pdf'

set title "extend\\_merkle / update\\_merkle / update\\_leaf\\_inplace: per-cell distribution (30 trials)"
set xlabel "Chunk Count" font 'Helvetica-Bold,10'
set ylabel "Wall Time (ms)" font 'Helvetica-Bold,10'
set xtics ("240" 1, "3413" 2, "13652" 3, "54608" 4)
set xrange [0.5:4.5]
set yrange [0.0007:1.5]
set boxwidth 0.16
set key inside top left box font 'Helvetica,8'
set logscale y

# pos  q1       whisklo  whiskhi  q3       median   (extend_merkle, offset -0.22)
$EXT << EOD
0.78   0.00100  0.00100  0.00100  0.00100  0.00100
1.78   0.00300  0.00300  0.00400  0.00400  0.00300
2.78   0.01500  0.01500  0.01500  0.01500  0.01500
3.78   0.35825  0.35300  0.37300  0.36475  0.36200
EOD

# pos  q1       whisklo  whiskhi  q3       median   (update_merkle, offset 0)
$UPD << EOD
1.00   0.00100  0.00100  0.00100  0.00100  0.00100
2.00   0.00300  0.00300  0.00300  0.00300  0.00300
3.00   0.01500  0.01500  0.01500  0.01500  0.01500
4.00   0.35725  0.35600  0.36600  0.36100  0.35900
EOD

# pos  q1       whisklo  whiskhi  q3       median   (update_leaf_inplace, offset +0.22)
$INPLACE << EOD
1.22   0.00100  0.00100  0.00100  0.00100  0.00100
2.22   0.00200  0.00200  0.00200  0.00200  0.00200
3.22   0.00200  0.00200  0.00200  0.00200  0.00200
4.22   0.00200  0.00200  0.00200  0.00200  0.00200
EOD

# pos  value  (every trial beyond its cell's 1.5*IQR fence)
$OUTLIERS << EOD
3.78   0.379
2.00   0.004
2.00   0.004
3.00   0.017
EOD

plot $EXT using 1:2:3:4:5 with candlesticks lw 1 lc rgb '#ff7f0e' title 'extend\_merkle' whiskerbars 0.5, \
     $EXT using 1:6:6:6:6 with candlesticks lw 2 lc rgb '#ff7f0e' notitle, \
     $UPD using 1:2:3:4:5 with candlesticks lw 1 lc rgb '#8c564b' title 'update\_merkle' whiskerbars 0.5, \
     $UPD using 1:6:6:6:6 with candlesticks lw 2 lc rgb '#8c564b' notitle, \
     $INPLACE using 1:2:3:4:5 with candlesticks lw 1 lc rgb '#17becf' title 'update\_leaf\_inplace' whiskerbars 0.5, \
     $INPLACE using 1:6:6:6:6 with candlesticks lw 2 lc rgb '#17becf' notitle, \
     $OUTLIERS using 1:2 with points pt 6 ps 0.4 lc rgb '#000000' notitle
