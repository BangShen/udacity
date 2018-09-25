reset
#as shown in Fig. 2. This can be achieved by settings the binstart to half the binwidth:
#set yrange [0:24]
#set xrange [0:25]
clear
reset
set key off
set border 3

# Add a vertical dotted line at x=0 to show centre (mean) of distribution.
set yzeroaxis

# Each bar is half the (visual) width of its x-range.
set boxwidth 0.05 absolute
set style fill solid 1.0 noborder

bin_width = 4
bin_number(x) = floor(x/bin_width)
rounded(x) = bin_width * ( bin_number(x) + 2 )

set title 'Secs deviation between CONGRRUENT and INCONGRUENT' font 'arial,12'
set grid
set key off
set tics font 'arial,12'
set ylabel 'Seconds deviation' font 'arial,12'
#set term pngcairo transparent size 600,400
#set o 'D:\self-development\data science\udacity\p1\photo3.png'
plot 'D:\self-development\data science\udacity\p1\stroopdata.txt' u (rounded($1)):(1) smooth frequency with boxes,\
'' u (rounded($2)):(2) smooth frequency with boxes
#set term wxt
#set o