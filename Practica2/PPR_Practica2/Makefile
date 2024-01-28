SRC = matriz_x_vector1.cpp
OBJ = matriz_x_vector1.o
EXE = matriz_x_vector1

SRC1 = matriz_x_vector2.cpp
OBJ1 = matriz_x_vector2.o
EXE1 = matriz_x_vector2

SRC2 = descomposicion_bidimensional.cpp
OBJ2 = descomposicion_bidimensional.o
EXE2 = descomposicion_bidimensional

$(EXE):$(SRC)
	mpiCC $(SRC) -o $(EXE)
	mpirun -np 4 $(EXE) 12

$(EXE1):$(SRC1)
	mpiCC $(SRC1) -o $(EXE1)
	mpirun -np 9 $(EXE1) 1800

$(EXE2):$(SRC2)
	mpiCC $(SRC2) -o $(EXE2)
	mpirun -np 9 $(EXE2) 1800

clean:
	rm $(EXE) $(EXE1) $(EXE2)
