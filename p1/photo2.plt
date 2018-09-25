reset
#as shown in Fig. 2. This can be achieved by settings the binstart to half the binwidth:
set yrange [0:24]
#set xrange [0:25]
set title 'Secs deviation between CONGRRUENT and INCONGRUENT' font 'arial,12'
set grid
set key off
set tics font 'arial,12'
unset xtics
set ylabel 'Seconds deviation' font 'Helvetica,12'

set style rect fc lt -1 fs solid 0.15 noborder
set obj 1 rect from 0,5.8 to 24,12
set obj 2 rect from 0,1.8 to 24,4

set term pngcairo transparent size 600,400
set o 'D:\self-development\data science\udacity\p1\photo2.png'
plot 'D:\self-development\data science\udacity\p1\stroopdata.txt' u ($2-$1) w lp pt 7

set term wxt
set o