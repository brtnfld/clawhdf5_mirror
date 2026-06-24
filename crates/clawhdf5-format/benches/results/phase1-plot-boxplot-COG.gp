# Figure for phase1-explanatory-note-COG.md, rendered as a vector PDF for
# direct \includegraphics use in LaTeX. extend_merkle/update_merkle, the two
# scenarios whose raw 30-trial spread includes real cold-start outliers (see
# the note's "no discarded warmups" caveat) -- drawn as Tukey box-and-whisker
# plots with outliers beyond 1.5*IQR shown as individual points, rather than
# collapsed to a median line.
#
# Quartile/whisker/outlier statistics are computed from phase1-COG.csv's 30
# raw trials per cell and inlined below as datablocks -- the CSV remains the
# source of truth; this script only renders already-verified numbers.
#
# Run with: gnuplot phase1-plot-boxplot-COG.gp

set terminal pdfcairo size 6in,4in enhanced font 'Helvetica,10'
set output 'phase1-plot-boxplot-COG.pdf'

set title "extend\\_merkle / update\\_merkle: per-cell distribution (30 trials)"
set xlabel "Chunk Count" font 'Helvetica-Bold,10'
set ylabel "Wall Time (ms)" font 'Helvetica-Bold,10'
set xtics ("240" 1, "3413" 2, "13652" 3, "54608" 4)
set xrange [0.5:4.5]
set yrange [0.0007:1.5]
set boxwidth 0.22
set key inside top left box font 'Helvetica,8'
set logscale y

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

plot $EXT using 1:2:3:4:5 with candlesticks lw 1 lc rgb '#ff7f0e' title 'extend\_merkle' whiskerbars 0.5, \
     $EXT using 1:6:6:6:6 with candlesticks lw 2 lc rgb '#ff7f0e' notitle, \
     $UPD using 1:2:3:4:5 with candlesticks lw 1 lc rgb '#8c564b' title 'update\_merkle' whiskerbars 0.5, \
     $UPD using 1:6:6:6:6 with candlesticks lw 2 lc rgb '#8c564b' notitle, \
     $OUTLIERS using 1:2 with points pt 6 ps 0.4 lc rgb '#000000' notitle
