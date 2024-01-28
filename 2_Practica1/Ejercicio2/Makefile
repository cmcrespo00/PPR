# CUDA code generation flags

GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60
GENCODE_SMXX    := -gencode arch=compute_70,code=compute_70
GENCODE_FLAGS   ?=  $(GENCODE_SM50) $(GENCODE_SM60) $(GENCODE_SMXX)

OBJ = operacion_vectorial
SRC = operacion_vectorial.cu
OBJ1 = operacion_vectorial_CPUimplementada
SRC1 = operacion_vectorial_CPUimplementada.cu

$(OBJ):$(SRC)
	nvcc -I./includes  -O3 -m64   $(GENCODE_FLAGS) $(SRC) -o $(OBJ)

$(OBJ1):$(SRC1)
	nvcc -I./includes  -O3 -m64   $(GENCODE_FLAGS) $(SRC1) -o $(OBJ1)

clean:
	rm $(OBJ) $(OBJ1)
