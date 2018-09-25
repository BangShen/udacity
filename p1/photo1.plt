reset
#as shown in Fig. 2. This can be achieved by settings the binstart to half the binwidth:
set yrange [0:38]
#set xrange [0:25]
set title 'Secs consuming by CONGRRUENT and INCONGRUENT' font 'arial,12'
set grid
set key left Left box reverse font 'arial,12'
set tics font 'arial,12'
unset xtics
set label 1 'Seconds' rotate by 90 at graph -0.08,.5 font 'Helvetica,12'
set style histogram clustered gap 2
set style fill solid 0.8 noborder
set boxwidth 1.5 relative
set style data histograms

set term pngcairo transparent size 600,400
set o 'D:\self-development\data science\udacity\p1\photo1.png'
plot 'D:\self-development\data science\udacity\p1\stroopdata.txt' u 1 t 'congruent','' u 2 t 'Incongruent'

set term wxt
set o