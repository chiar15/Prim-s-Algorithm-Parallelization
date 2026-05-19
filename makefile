.PHONY: all clean test compile0 compile1 compile2 compile3 test0 test1 test2 test3

all: directories compile0 compile1 compile2 compile3 

clean: 
	rm -rf ./Build/*
	rm -rf ./Data/*
	rm -rf ./Information/*
	rm -rf ./PrimResults/*
	rm -rf ./Plots
	rm -rf ./TimeResults
	rm -rf ./Tables

test: test0 test1 test2 test3 analysis

directories:
	python3 ./Source/directories.py

compile0:
	gcc -c -o ./Build/TMatrix.o ./Source/TMatrix.c -O0
	gcc -c -o ./Build/graphGenerator.o ./Source/graphGenerator.c -O0
	gcc -c -o ./Build/TMatrixOMP.o ./Source/TMatrix.c -fopenmp -O0
	gcc -c -o ./Build/graphGeneratorOMP.o ./Source/graphGenerator.c -fopenmp -O0
	nvcc -c -o ./Build/TMatrix.obj ./Source/TMatrix.c -Xcompiler -fopenmp -O0
	nvcc -c -o ./Build/graphGenerator.obj ./Source/graphGenerator.c -Xcompiler -fopenmp -O0

	gcc -c -o ./Build/Sequential_prim.o ./Source/Sequential_prim.c -O0
	mpicc -c -o ./Build/MPI_OMP_prim.o ./Source/MPI_OMP_prim.c -fopenmp -O0
	nvcc -c -o ./Build/CUDA_prim.obj ./Source/CUDA_prim.cu -Xcompiler -fopenmp -O0

	gcc -o ./Build/Sequential_prim0.exe ./Build/Sequential_prim.o ./Build/TMatrix.o ./Build/graphGenerator.o -O0
	mpicc -o ./Build/MPI_OMP_prim0.exe ./Build/MPI_OMP_prim.o ./Build/TMatrixOMP.o ./Build/graphGeneratorOMP.o -fopenmp -O0
	nvcc -o ./Build/CUDA_prim0.exe ./Build/CUDA_prim.obj ./Build/TMatrix.obj ./Build/graphGenerator.obj -Xcompiler -fopenmp -O0

compile1:
	gcc -c -o ./Build/TMatrix.o ./Source/TMatrix.c -O1
	gcc -c -o ./Build/graphGenerator.o ./Source/graphGenerator.c -O1
	gcc -c -o ./Build/TMatrixOMP.o ./Source/TMatrix.c -fopenmp -O1
	gcc -c -o ./Build/graphGeneratorOMP.o ./Source/graphGenerator.c -fopenmp -O1
	nvcc -c -o ./Build/TMatrix.obj ./Source/TMatrix.c -Xcompiler -fopenmp -O1
	nvcc -c -o ./Build/graphGenerator.obj ./Source/graphGenerator.c -Xcompiler -fopenmp -O1

	gcc -c -o ./Build/Sequential_prim.o ./Source/Sequential_prim.c -O1
	mpicc -c -o ./Build/MPI_OMP_prim.o ./Source/MPI_OMP_prim.c -fopenmp -O1
	nvcc -c -o ./Build/CUDA_prim.obj ./Source/CUDA_prim.cu -Xcompiler -fopenmp -O1

	gcc -o ./Build/Sequential_prim1.exe ./Build/Sequential_prim.o ./Build/TMatrix.o ./Build/graphGenerator.o -O1
	mpicc -o ./Build/MPI_OMP_prim1.exe ./Build/MPI_OMP_prim.o ./Build/TMatrixOMP.o ./Build/graphGeneratorOMP.o -fopenmp -O1
	nvcc -o ./Build/CUDA_prim1.exe ./Build/CUDA_prim.obj ./Build/TMatrix.obj ./Build/graphGenerator.obj -Xcompiler -fopenmp -O1

compile2:
	gcc -c -o ./Build/TMatrix.o ./Source/TMatrix.c -O2
	gcc -c -o ./Build/graphGenerator.o ./Source/graphGenerator.c -O2
	gcc -c -o ./Build/TMatrixOMP.o ./Source/TMatrix.c -fopenmp -O2
	gcc -c -o ./Build/graphGeneratorOMP.o ./Source/graphGenerator.c -fopenmp -O2
	nvcc -c -o ./Build/TMatrix.obj ./Source/TMatrix.c -Xcompiler -fopenmp -O2
	nvcc -c -o ./Build/graphGenerator.obj ./Source/graphGenerator.c -Xcompiler -fopenmp -O2

	gcc -c -o ./Build/Sequential_prim.o ./Source/Sequential_prim.c -O2
	mpicc -c -o ./Build/MPI_OMP_prim.o ./Source/MPI_OMP_prim.c -fopenmp -O2
	nvcc -c -o ./Build/CUDA_prim.obj ./Source/CUDA_prim.cu -Xcompiler -fopenmp -O2

	gcc -o ./Build/Sequential_prim2.exe ./Build/Sequential_prim.o ./Build/TMatrix.o ./Build/graphGenerator.o -O2
	mpicc -o ./Build/MPI_OMP_prim2.exe ./Build/MPI_OMP_prim.o ./Build/TMatrixOMP.o ./Build/graphGeneratorOMP.o -fopenmp -O2
	nvcc -o ./Build/CUDA_prim2.exe ./Build/CUDA_prim.obj ./Build/TMatrix.obj ./Build/graphGenerator.obj -Xcompiler -fopenmp -O2

compile3:
	gcc -c -o ./Build/TMatrix.o ./Source/TMatrix.c -O3
	gcc -c -o ./Build/graphGenerator.o ./Source/graphGenerator.c -O3
	gcc -c -o ./Build/TMatrixOMP.o ./Source/TMatrix.c -fopenmp -O3
	gcc -c -o ./Build/graphGeneratorOMP.o ./Source/graphGenerator.c -fopenmp -O3
	nvcc -c -o ./Build/TMatrix.obj ./Source/TMatrix.c -Xcompiler -fopenmp -O3
	nvcc -c -o ./Build/graphGenerator.obj ./Source/graphGenerator.c -Xcompiler -fopenmp -O3

	gcc -c -o ./Build/Sequential_prim.o ./Source/Sequential_prim.c -O3
	mpicc -c -o ./Build/MPI_OMP_prim.o ./Source/MPI_OMP_prim.c -fopenmp -O3
	nvcc -c -o ./Build/CUDA_prim.obj ./Source/CUDA_prim.cu -Xcompiler -fopenmp -O3

	gcc -o ./Build/Sequential_prim3.exe ./Build/Sequential_prim.o ./Build/TMatrix.o ./Build/graphGenerator.o -O3
	mpicc -o ./Build/MPI_OMP_prim3.exe ./Build/MPI_OMP_prim.o ./Build/TMatrixOMP.o ./Build/graphGeneratorOMP.o -fopenmp -O3
	nvcc -o ./Build/CUDA_prim3.exe ./Build/CUDA_prim.obj ./Build/TMatrix.obj ./Build/graphGenerator.obj -Xcompiler -fopenmp -O3

analysis:
	python3 ./Statistics/analize.py
	python3 ./Statistics/plotCreator.py
	python3 ./Statistics/tableCreator.py

test0:
	python3 ./Test/test0.py

test1:
	python3 ./Test/test1.py

test2:
	python3 ./Test/test2.py

test3:
	python3 ./Test/test3.py