reset
#as shown in Fig. 2. This can be achieved by settings the binstart to half the binwidth:
#set yrange [0:38]
#set xrange [0:25]
set macros
FONT ="font 'Helvetica,12'"
set key @FONT at graph 0.9,0.3 reverse Left box

set title 'Scatter plot of Z_i and Z_{fi}' @FONT
set xlabel 'Z_i' @FONT
set ylabel 'Z_{fi}' @FONT
set grid
set tics @FONT


set term pngcairo transparent size 600,400
set o 'D:\self-development\data science\udacity\p1\scatter_plot.png'
plot 'D:\self-development\data science\udacity\p1\scatter_plot.txt' u 2:4 w p pt 7 ps 1.5 lc 2 title 'Data',\
'' u 2:(4,$4>1.5?$4:NaN) w p pt 7 ps 1.5 lc 7 t 'Outliers',\
'' u 2:(4,$4<-1.5?$4:NaN) w p pt 7 ps 1.5 lc 7 notitle
set term wxt
set o