all: pgm.o	hough

global:	houghGlobal.cu pgm.o
	nvcc houghGlobal.cu pgm.o -o houghGlobal

constant: houghConstant.cu pgm.o
	nvcc houghConstant.cu pgm.o -o houghConstant

shared: houghShared.cu pgm.o
	nvcc houghShared.cu pgm.o -o houghShared

pgm.o:	common/pgm.cpp
	g++ -c common/pgm.cpp -o ./pgm.o
