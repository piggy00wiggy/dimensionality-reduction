CC = clang++
MPICXX = mpicxx

SPECDEP=makefile
FLAG=/Users/Eric/include/eigen3/


serial: svd_serial.cc $(SPECDEP)
	$(CC) -I $(FLAG) svd_serial.cc -o svd_serial 

mpi: svd_mpi.cc $(SPECDEP)
	$(MPICXX) -I $(FLAG) svd_mpi.cc -o svd_mpi


