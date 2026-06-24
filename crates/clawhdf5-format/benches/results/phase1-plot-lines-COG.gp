# Figure for phase1-explanatory-note-COG.md, rendered as a vector PDF for
# direct \includegraphics use in LaTeX. The five scenarios whose 30-trial
# spread is tight enough to read as a single median+CI line:
# verify_dataset, full_rebuild, flat_verify, verify_chunk, verify_proof.
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

# n_chunks  verify_dataset(med,lo,hi)  full_rebuild(med,lo,hi)  flat_verify(med,lo,hi)  verify_chunk(med,lo,hi)  verify_proof(med,lo,hi)
$DATA << EOD
240    2.4060 2.4035 2.4085    2.4080 2.4055 2.4110    3.7795 3.7775 3.7850    0.0100 0.0100 0.0100    0.0110 0.0110 0.0110
3413   374.2805 373.9415 375.0580    376.6800 375.8680 378.2470    1276.9450 1276.3620 1277.3710    0.0830 0.0830 0.0830    0.0850 0.0850 0.0850
13652  430.4955 430.1515 431.1930    432.2650 431.1935 432.7755    1279.4920 1278.0505 1283.5500    0.0240 0.0240 0.0240    0.0260 0.0260 0.0260
54608  627.9145 627.4685 629.0370    628.2025 627.5045 629.0570    1281.9150 1278.1745 1283.6540    0.0100 0.0100 0.0100    0.0120 0.0120 0.0120
EOD

plot $DATA using 1:2:3:4   with errorlines lw 1 pt 7  ps 0.4 lc rgb '#1f77b4' title 'verify\_dataset', \
     $DATA using 1:5:6:7   with errorlines lw 1 pt 5  ps 0.4 lc rgb '#9467bd' title 'full\_rebuild', \
     $DATA using 1:8:9:10  with errorlines lw 1 pt 9  ps 0.4 lc rgb '#d62728' title 'flat\_verify (baseline)', \
     $DATA using 1:11:12:13 with errorlines lw 1 pt 13 ps 0.4 lc rgb '#2ca02c' title 'verify\_chunk', \
     $DATA using 1:14:15:16 with errorlines lw 1 pt 11 ps 0.4 lc rgb '#17becf' title 'verify\_proof'
