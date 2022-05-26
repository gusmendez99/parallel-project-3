all: pgm.o	hough

hough:	houghBase.cu pgm.o
	nvcc houghBase.cu pgm.o -o hough

pgm.o:	common/pgm.cpp
	g++ -c common/pgm.cpp -o ./pgm.o
