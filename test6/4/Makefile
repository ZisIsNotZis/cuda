SHELL=bash

m.csv: m.ncu-rep
	ncu -i m.ncu-rep --csv --metrics gpu__time_duration.sum > m.csv

m.ncu-rep: m
	ncu --replay-mode application -f -o m m

t=$(shell seq 0 $N)
m: m.o $(foreach x,$t,$(foreach y,$t,$(foreach z,$t,$x_$y_$z.o)))
	nvcc *.o -o m

%.o: %.cu
	nvcc -arch native -extra-device-vectorization --use_fast_math -c $<

t=,
%.cu:
	echo -e '#include"m.h"\nvoid m$(subst _,,$(@:.cu=))(int4*_){m<$N,$(subst _,$t,$(@:.cu=))>(_);}'>$@

m.cu:
	{ for((x=0;x<=$N;x++));do for((y=0;y<=$N;y++));do for((z=0;z<=$N;z++));do echo "void m$$x$$y$$z(int4*_);";done;done;done;echo 'int main(){int4*_;cudaMalloc(&_,$N*$N*512);';for((x=0;x<=$N;x++));do for((y=0;y<=$N;y++));do for((z=0;z<=$N;z++));do echo "m$$x$$y$$z(_);";done;done;done;echo };}>m.cu
