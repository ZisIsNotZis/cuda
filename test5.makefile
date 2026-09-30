SHELL := bash
CU := $(wildcard $X$Y$Z*.cu)
O := $(CU:.cu=.o)

all: $O $X$Y$Z.o
	nvcc -o $X$Y$Z $^
	ncu --replay-mode application -fo $X$Y$Z $X$Y$Z
$X$Y$Z%.o: $X$Y$Z%.cu m.h
	nvcc -arch native -extra-device-vectorization --use_fast_math -c $<
$X$Y$Z.o: $X$Y$Z.cu
	nvcc -c $^

gen:
	>$X$Y$Z.cu
	for((x=1;x<=$X;x*=2));do for((y=1;y<=$Y;y*=2));do for((z=1;z<=$Z;z*=2));do [ -e $X$Y$Z$$x$$y$$z.cu ] || echo -e "#include\"m.h\"\nvoid m$X$Y$Z$$x$$y$$z(int4*_){m<$X,$Y,$Z,$$x,$$y,$$z>(_);}">$X$Y$Z$$x$$y$$z.cu;echo "void m$X$Y$Z$$x$$y$$z(int4*_);">>$X$Y$Z.cu;done;done;done
	echo 'int main(){int4*_;cudaMalloc(&_,max($X*max($Y,$Z),$Y*$Z)*512);'>>$X$Y$Z.cu
	for((x=1;x<=$X;x*=2));do for((y=1;y<=$Y;y*=2));do for((z=1;z<=$Z;z*=2));do echo "m$X$Y$Z$$x$$y$$z(_);">>$X$Y$Z.cu;done;done;done
	echo '}'>>$X$Y$Z.cu