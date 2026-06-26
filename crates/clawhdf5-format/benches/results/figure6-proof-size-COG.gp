# Figure 6 (left panel) from S2-D2-Yr2, rendered as a vector PDF for direct
# \includegraphics use in LaTeX. Log-log line chart: subset-proof size |π| in
# bytes vs. number of selected chunks k, for three hyperslab shapes
# (contiguous, strided, random) plus the theoretical O(k·log N) bound
# (N=65,536; log2(N)=16), answering RQ6.
#
# The contiguous series falls well below the bound because adjacent proof
# paths share internal Merkle nodes — the deduplication benefit is largest
# when the selected chunks form a compact, spatially-coherent region.
# Strided/random selections yield less sharing and approach the bound more
# closely, though still come in under it.
#
# Summary statistics are drawn from subset-proof-size.csv (the source of
# truth) and inlined below as a datablock — the script renders
# already-verified numbers only.
#
# Run with: gnuplot figure6-proof-size-COG.gp

set terminal pdfcairo size 6in,4in enhanced font 'Helvetica,10'
set output 'figure6-proof-size-COG.pdf'

set title "Subset-proof size vs. selected-chunk count k (N = 65,536)"
set xlabel "k (selected chunks)" font 'Helvetica-Bold,10'
set ylabel "Proof size |π| (bytes)" font 'Helvetica-Bold,10'
set logscale x
set logscale y
set grid xtics ytics lt 0 lc rgb '#dddddd'
set key inside top left box font 'Helvetica,8'
set xtics (64, 256, 1024, 4096)
set format x "%.0f"
set format y "10^{%T}"
set xrange [40:6000]

# k  contiguous  strided  random  theoretical
$DATA << EOD
64     8080    33280    30600    43620
256    31040   112640   104040   174180
1024   123120  368640   336200   696420
4096   491680  1146880  1027760  2785380
EOD

plot $DATA using 1:2 with linespoints lw 2 pt 7  ps 0.6 lc rgb '#1f77b4' title 'contiguous', \
     $DATA using 1:3 with linespoints lw 2 pt 9  ps 0.6 lc rgb '#ff7f0e' title 'strided', \
     $DATA using 1:4 with linespoints lw 2 pt 5  ps 0.6 lc rgb '#d62728' title 'random', \
     $DATA using 1:5 with lines     lw 1 lt 2 dt 2 lc rgb '#555555' title 'O(k·log N) bound'
