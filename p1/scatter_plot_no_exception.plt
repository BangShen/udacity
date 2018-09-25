reset
#as shown in Fig. 2. This can be achieved by settings the binstart to half the binwidth:
#set yrange [0:38]
#set xrange [0:25]
set macros
FONT ="font 'Helvetica,12'"
set key at graph 0.5,.8 @FONT
set label 'R^2=0.967' @FONT at graph 0.2,.65

set title 'Scatter plot of Z_i and Z_{fi} without outliers' @FONT
set xlabel 'Z_i' @FONT
set ylabel 'Z_{fi}' @FONT
set grid
set tics @FONT
f(x)=a+b*x


set term pngcairo transparent size 600,400
set o 'D:\self-development\data science\udacity\p1\scatter_plot_without_outliers.png'


fit f(x) 'D:\self-development\data science\udacity\p1\scatter_plot_no_exception.txt' u 2:4 via a,b
ti = sprintf("%.5f+%.5fx",a,b)
plot 'D:\self-development\data science\udacity\p1\scatter_plot_no_exception.txt' u 2:4 w p pt 7 ps 1.5 lc 2 notitle,\
f(x) w l lw 2 t ti

set term wxt
set o