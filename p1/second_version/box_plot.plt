reset
#as shown in Fig. 2. This can be achieved by settings the binstart to half the binwidth:
#set yrange [0:38]
#set xrange [0:25]
set macros
FONT ="font 'Helvetica,12'"
set key off

set style fill solid 0.25 border -1
set style boxplot outliers pointtype 7
set style data boxplot
set boxwidth  0.5
set pointsize 0.5 


set title 'Box plot of Congruent and Incongruent' @FONT
set xtics ("Congruent" 1, "Incongruent" 2) scale 0.0 @FONT
set xtics nomirror
set ytics nomirror

set grid
set ytics @FONT


set term pngcairo transparent size 600,400
set o 'D:\self-development\data science\udacity\p1\second_version\box_plot.png'
plot 'D:\self-development\data science\udacity\p1\second_version\box_plot.txt' u (1):1 w boxplot,\
'' u (2):2 w boxplot
set term wxt
set o