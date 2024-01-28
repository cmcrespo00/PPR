#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>

using namespace std;

const int M=32;
const int NBlocks=4;

/*const int M=64;
//const int NBlocks=32000;
//const int NBlocks=64000;
//const int NBlocks=320000;
const int NBlocks=1000000;*/

/*const int M=128;
//const int NBlocks=16000;
//const int NBlocks=32000;
//const int NBlocks=160000;
const int NBlocks=500000;*/

/*const int M=256;
//const int NBlocks=8000;
//const int NBlocks=16000;
//const int NBlocks=80000;
const int NBlocks=250000;*/

//***************************************************************************
//*************************** KERNEL PRIMER PUNTO ***************************
//***************************************************************************
__global__ void kernel_calcularC(float *A, float *B, float *C, const int N) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  for (int j = (blockIdx.x * blockDim.x); j < ((blockIdx.x * blockDim.x) + blockDim.x); j++) {
    if((i+2) * (max(A[j], B[j])) != 0)
      C[i] += (i*B[j] - A[j]*A[j]) / ((i+2) * (max(A[j], B[j])));
  }
}

//******************* CON VARIABLES DE MEMORIA COMPARTIDA *******************
__global__ void kernel_calcularC_mc(float *A, float *B, float *C, const int N) {
  extern __shared__ float sdata[];
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  float *s_A = &sdata[0],
        *s_B = &sdata[blockDim.x];

  *(s_A+tid) = A[i];
  *(s_B+tid) = B[i];
  __syncthreads();

  for (int j = 0; j < blockDim.x; j++) {
    if(((i+2) * (max((*(s_A+j)), (*(s_B+j))))) != 0)
      C[i] += (i*(*(s_B+j)) - (*(s_A+j))*(*(s_A+j))) / ((i+2) * (max((*(s_A+j)), (*(s_B+j)))));

    __syncthreads();
  }
}

//***************************************************************************
//************************** KERNEL SEGUNDO PUNTO ***************************
//***************************************************************************
__global__ void kernel_calcularMaximo(float * V_in, float * V_out, const int N) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int i_seg =  gridDim.x * blockDim.x + i;

  float segundo_valor;

  sdata[tid] = ((i < N) ? V_in[i] : -100000000.0f);
  segundo_valor = ((i_seg < N) ? V_in[i] : -100000000.0f);

  sdata[tid] = max(sdata[tid], segundo_valor);
  __syncthreads();

  for(int s = blockDim.x/2; s > 0; s >>= 1){
    if (tid < s)
      sdata[tid]=max(sdata[tid],sdata[tid+s]);
    __syncthreads();
  }

  if (tid == 0)
    V_out[blockIdx.x] = sdata[0];
}

//***************************************************************************
//*************************** KERNEL TERCER PUNTO ***************************
//***************************************************************************
__global__ void kernel_calcularMediaBloque(float * V_in, float * V_out, const int N) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  sdata[tid] = ((i < N) ? V_in[i] : 0.0f);

  __syncthreads();

  for (int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];

    __syncthreads();
  }

  if (tid == 0)
    V_out[blockIdx.x] = sdata[0] / blockDim.x;
}

//***************************************************************************
int main(int argc, char* argv[]) {
  /* Vectores del host */
  float *A, *B;
  /* Vectores memoria compartida*/
  float *A_d, *B_d;

  /*Tamaño de los vectores*/
  int N = M * NBlocks;

  /* Reservar memoria de los vectores en el host */
  A = (float*) malloc(N*sizeof(float));
  B = (float*) malloc(N*sizeof(float));

  /* Reservar memoria de los vectores en memoria compartida */
  cudaMalloc ((void **) &A_d, sizeof(float)*N);
  cudaMalloc ((void **) &B_d, sizeof(float)*N);

  /* Inicializar vectores */
  for (int i = 0; i < N; i++){
    A[i] = 1.5 * ((1 + ((5 * i) % 7)) / (1 + (i % 5)));
    B[i] = 2 * ((2 + (i % 5)) / (1 + (i % 7)));
  }

  /* Copiar los datos de los vectores en el host a los
      vectores en memoria compatida */
  cudaMemcpy(A_d, A, sizeof(float)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, sizeof(float)*N, cudaMemcpyHostToDevice);

  /* Vectores resultado */
  float *C_CPU, *C, *C_mc, *D_CPU, *D;    // En el host
  float *C_d, *C_mc_d, *D_d;             // En memoria compartida

  float *vmax;
  float *vmax_d;

  /* Reservar memoria de los vectores en el host */
  C_CPU = (float*) malloc(N*sizeof(float));
  C = (float*) malloc(N*sizeof(float));
  C_mc = (float*) malloc(N*sizeof(float));
  D_CPU = (float*) malloc(NBlocks*sizeof(float));
  D = (float*) malloc(NBlocks*sizeof(float));

  vmax = (float*) malloc(NBlocks*sizeof(float));

  /* Reservar memoria de los vectores en memoria compartida */
  cudaMalloc ((void **) &C_d, sizeof(float)*N);
  cudaMalloc ((void **) &C_mc_d, sizeof(float)*N);
  cudaMalloc ((void **) &D_d, sizeof(float)*NBlocks);

  cudaMalloc ((void **) &vmax_d, sizeof(float)*NBlocks);

  for (int i = 0; i < N; i++){
    C[i] = 0.0;
  }

  /* Copiar los datos de los vectores en el host a los
      vectores en memoria compatida */
  cudaMemcpy(C_d, C, sizeof(float)*N, cudaMemcpyHostToDevice);

  /* Variables de tiempo */
  double time, Tcpu_ej1, Tcpu_ej2, Tcpu_ej3,
               Tgpu_ej1, Tgpu_ej1_mc, Tgpu_ej2, Tgpu_ej3,
               Tcpu_total, Tgpu_total, Tgpu_total_mc;

  //***************************************************************************
  //********************* CALCULO PRIMER PUNTO EN LA CPU **********************
  //***************************************************************************
  Tcpu_total = clock();
  time = clock();

  for (int k=0; k<NBlocks;k++){
    int istart = k * M;
    int iend  =istart + M;
    for (int i = istart; i < iend; i++){
      C[i] = 0.0;
      for (int j = istart; j < iend; j++)
        C[i] += fabs((i* B[j]-A[j]*A[j])/((i+2)*max(A[j],B[j])));
    }
  }

  Tcpu_ej1 = (clock() - time) / CLOCKS_PER_SEC;

  //***************************************************************************
  //********************* CALCULO SEGUNDO PUNTO EN LA CPU *********************
  //***************************************************************************
  time = clock();

  float mx = C[0];
  for (int i = 0; i < N; i++){
    mx = max(C[i], mx);
  }

  Tcpu_ej2 = (clock() - time) / CLOCKS_PER_SEC;

  //***************************************************************************
  //********************* CALCULO TERCER PUNTO EN LA CPU **********************
  //***************************************************************************
  time = clock();

  for (int k = 0; k < NBlocks; k++){
    int istart = k * M;
    int iend = istart + M;
    D[k] = 0.0;
    for (int i = istart; i < iend; i++){
      D[k] += C[i];
    }
    D[k] /= M;
  }

  Tcpu_ej3 = (clock() - time) / CLOCKS_PER_SEC;
  Tcpu_total = (clock() - Tcpu_total) / CLOCKS_PER_SEC;

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  dim3 threadsPerBlock(M, 1);
  dim3 numBlocks(NBlocks, 1);

  int smemSize = threadsPerBlock.x*sizeof(float);

  //***************************************************************************
  //********************* CALCULO PRIMER PUNTO EN LA GPU **********************
  //***************************************************************************
  Tgpu_total = clock();
  time = clock();

  /*Llamada al kernel*/
  kernel_calcularC<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, N);
  cudaDeviceSynchronize();

  Tgpu_ej1 = (clock() - time) / CLOCKS_PER_SEC;

  /* Copiar los datos de la memoria compartida a la memeoria del host */
  cudaMemcpy(C, C_d, N*sizeof(float),cudaMemcpyDeviceToHost);

  //***************************************************************************
  //********************* CALCULO SEGUNDO PUNTO EN LA GPU *********************
  //***************************************************************************
  time = clock();

  /*Llamada al kernel*/
  kernel_calcularMaximo<<<numBlocks, threadsPerBlock, smemSize>>>(C_d, vmax_d, N);

  Tgpu_ej2 = (clock() - time) / CLOCKS_PER_SEC;

  /* Copiar los datos de la memoria compartida a la memeoria del host */
  cudaMemcpy(vmax, vmax_d, numBlocks.x*sizeof(float),cudaMemcpyDeviceToHost);

  float maximo_GPU = -10000000.0f;
  for (int i=0; i<numBlocks.x; i++){
    maximo_GPU = max(maximo_GPU,vmax[i]);
  }

  //***************************************************************************
  //********************* CALCULO TERCER PUNTO EN LA GPU **********************
  //***************************************************************************
  time = clock();

  /*Llamada al kernel*/
  kernel_calcularMediaBloque<<<numBlocks, threadsPerBlock, smemSize>>>(C_d, D_d, N);

  Tgpu_ej3 = (clock() - time) / CLOCKS_PER_SEC;

  /* Copiar los datos de la memoria compartida a la memeoria del host */
  cudaMemcpy(D, D_d, numBlocks.x*sizeof(float),cudaMemcpyDeviceToHost);

  Tgpu_total = (clock() - Tgpu_total) / CLOCKS_PER_SEC;

  //········································································//
  //········································································//
  //········································································//

  //***************************************************************************
  //********************* CALCULO PRIMER PUNTO EN LA GPU **********************
  //***************************************************************************
  /* VARIABLES MEMORIA COMPARTIDA */
  Tgpu_total_mc = clock();
  time = clock();

  /*Llamada al kernel*/
  kernel_calcularC_mc<<<numBlocks, threadsPerBlock, smemSize>>>(A_d, B_d, C_mc_d, N);
  cudaDeviceSynchronize();

  Tgpu_ej1_mc = (clock() - time) / CLOCKS_PER_SEC;

  /* Copiar los datos de la memoria compartida a la memeoria del host */
  cudaMemcpy(C_mc, C_mc_d, N*sizeof(float),cudaMemcpyDeviceToHost);

  //***************************************************************************
  //********************* CALCULO SEGUNDO PUNTO EN LA GPU *********************
  //***************************************************************************
  time = clock();

  /*Llamada al kernel*/
  kernel_calcularMaximo<<<numBlocks, threadsPerBlock, smemSize>>>(C_d, vmax_d, N);

  Tgpu_ej2 = (clock() - time) / CLOCKS_PER_SEC;

  /* Copiar los datos de la memoria compartida a la memeoria del host */
  cudaMemcpy(vmax, vmax_d, numBlocks.x*sizeof(float),cudaMemcpyDeviceToHost);

  maximo_GPU = -10000000.0f;
  for (int i=0; i<numBlocks.x; i++){
    maximo_GPU = max(maximo_GPU,vmax[i]);
  }

  //***************************************************************************
  //********************* CALCULO TERCER PUNTO EN LA GPU **********************
  //***************************************************************************
  time = clock();

  /*Llamada al kernel*/
  kernel_calcularMediaBloque<<<numBlocks, threadsPerBlock, smemSize>>>(C_d, D_d, N);

  Tgpu_ej3 = (clock() - time) / CLOCKS_PER_SEC;

  /* Copiar los datos de la memoria compartida a la memeoria del host */
  cudaMemcpy(D, D_d, numBlocks.x*sizeof(float),cudaMemcpyDeviceToHost);

  Tgpu_total_mc = (clock() - Tgpu_total_mc) / CLOCKS_PER_SEC;

  //········································································//
  //········································································//
  //········································································//

  //Comprobación de valores
  /*for (int i = 0; i < N; i++) {
    cout << "C[" << i << "] = " << C[i] << "   ";
    cout << "C_mc[" << i << "] = " << C_mc[i] << "   ";
    cout << "C_CPU[" << i << "] = " << C_CPU[i] << endl;
  }

  cout << endl << "Maximo vector C GPU = " << maximo_GPU << endl;
  cout << "Maximo vector C CPU = " << mx << endl << endl;

  for (int i = 0; i < NBlocks; i++) {
    cout << "D[" << i << "] = " << D[i] << "   ";
    cout << "D_CPU[" << i << "] = " << D_CPU[i] << endl;
  }*/

  /* Mostrar tiempos */
  cout<<"....................................................." <<endl;
  cout << "Tiempo primer ejercicio CPU = " << Tcpu_ej1 << endl;
  cout << "Tiempo segundo ejercicio CPU = " << Tcpu_ej2 << endl;
  cout << "Tiempo tercer ejercicio CPU = " << Tcpu_ej3 << endl;
  cout<<"....................................................."<<endl<<endl;

  cout<<"....................................................." <<endl;
  cout << "Tiempo primer ejercicio GPU = " << Tgpu_ej1 << endl;
  cout << "Tiempo primer ejercicio GPU_mc = " << Tgpu_ej1_mc << endl;
  cout << "Tiempo segundo ejercicio GPU = " << Tgpu_ej2 << endl;
  cout << "Tiempo tercer ejercicio GPU = " << Tgpu_ej3 << endl;
  cout<<"....................................................."<<endl<<endl;

  cout<<"....................................................." <<endl;
  cout << "Tiempo total CPU = " << Tcpu_total << endl;
  cout << "Tiempo total GPU = " << Tgpu_total << endl;
  cout << "Tiempo total GPU_mc = " << Tgpu_total_mc << endl;
  cout<<"....................................................."<<endl<<endl;

  /* Liberar memoria */
  free(A); free(B);
  free(C_CPU); free(C); free(C_mc); free(D_CPU); free(D);
  cudaFree(A_d); cudaFree(B_d);
  cudaFree(C_d); cudaFree(C_mc_d); cudaFree(D_d);
}
