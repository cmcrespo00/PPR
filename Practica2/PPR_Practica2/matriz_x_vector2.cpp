/*
============================================================================
Name        : matriz_x_vector2.cpp
Author      : Jose Miguel Mantas Ruiz
Copyright   : GNU Open Souce and Free license
Description : Tutorial 5. Multiplicacion de Matrix por Vector.
============================================================================
*/

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

using namespace std;

int main(int argc, char * argv[]) {

  int numeroProcesadores, id_Proceso;

  float *A, // Matriz global a multiplicar
        *x, // Vector a multiplicar
        *y, // Vector resultado
        *local_A,  // Matriz local de cada proceso
        *local_y;  // POrción local del resultado en  cada proceso


  double tInicio, // Tiempo en el que comienza la ejecucion
         Tpar, Tseq;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numeroProcesadores);
  MPI_Comm_rank(MPI_COMM_WORLD, &id_Proceso);

  int n;

  if (argc <= 1) {// si no se pasa el size de la matriz, se coge n=10
    if (id_Proceso==0)
    cout << "The dimension N of the matrix is missing (N x N matrix)"<< endl;
    MPI_Finalize();
    return 0;
  } else
    n = atoi(argv[1]);

  x = new float[n]; //reservamos espacio para el vector x (n floats).

  // Proceso 0 genera matriz A y vector x
  if (id_Proceso==0)  {
    A = new float[n*n];//reservamos espacio para la matriz (n x n floats)
    y = new float[n];//reservamos espacio para el vector resultado final y (n floats)

    // Rellena la matriz y el vector
    for (int i = 0; i < n; i++) {
      x[i] = (float) (1.5*(1+(5*(i))%3)/(1+(i)%5));
      for (int j = 0; j < n; j++) {
        A[i*n+j] = (float) (1.5*(1+(5*(i+j))%3)/(1+(i+j)%5));
      }
    }
  }

  // Cada proceso reserva espacio para su porción de A y para el vector x
  const int local_A_size=n*n/numeroProcesadores;
  const int local_y_size=n/numeroProcesadores;
  local_A = new float[local_A_size];//reservamos espacio para la matriz (n x n floats)
  local_y = new  float[local_y_size]; //reservamos espacio para el vector y (n/num_procs floats).

  // Repartimos una bloque de filas de A a cada proceso
  MPI_Scatter(A, // Matriz que vamos a compartir
              local_A_size, // Numero de filas a entregar
              MPI_FLOAT, // Tipo de dato a enviar
              local_A, // Vector en el que almacenar los datos
              local_A_size, // Numero de filas a recibir
              MPI_FLOAT, // Tipo de dato a recibir
              0, // Proceso raiz que envia los datos
              MPI_COMM_WORLD); // Comunicador utilizado (En este caso, el global)

  // Difundimos el vector x entre todas los procesos
  MPI_Bcast(x, // Dato a compartir
            n, // Numero de elementos que se van a enviar y recibir
            MPI_FLOAT, // Tipo de dato que se compartira
            0, // Proceso raiz que envia los datos
            MPI_COMM_WORLD); // Comunicador utilizado (En este caso, el global)


  // Hacemos una barrera para asegurar que todas los procesos comiencen la ejecucion
  // a la vez, para tener mejor control del tiempo empleado
  MPI_Barrier(MPI_COMM_WORLD);

  // Inicio de medicion de tiempo
  tInicio = MPI_Wtime();

  for (int i = 0; i < local_y_size; i++) {
    local_y[i]=0.0;
    for (int j = 0; j < n; j++) {
      local_y[i] += local_A[i*n+j] * x[j];
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  // fin de medicion de tiempo
  Tpar = MPI_Wtime()-tInicio;

  // Recogemos los datos de la multiplicacion, por cada proceso sera un escalar
  // y se recoge en un vector, Gather se asegura de que la recolecci�n se haga
  // en el mismo orden en el que se hace el Scatter, con lo que cada escalar
  // acaba en su posicion correspondiente del vector.
  MPI_Gather(local_y, // Dato que envia cada proceso
             local_y_size, // Numero de elementos que se envian
             MPI_FLOAT, // Tipo del dato que se envia
             y, // Vector en el que se recolectan los datos
             local_y_size, // Numero de datos que se esperan recibir por cada proceso
             MPI_FLOAT, // Tipo del dato que se recibira
             0, // proceso que va a recibir los datos
             MPI_COMM_WORLD); // Canal de comunicacion (Comunicador Global)

  // Terminamos la ejecucion de los procesos, despues de esto solo existira
  // el proceso 0
  // Ojo! Esto no significa que los demas procesos no ejecuten el resto
  // de codigo despues de "Finalize", es conveniente asegurarnos con una
  // condicion si vamos a ejecutar mas codigo (Por ejemplo, con "if(rank==0)".
  MPI_Finalize();

  if (id_Proceso == 0) {
    float * comprueba = new float [n];
    //Calculamos la multiplicacion secuencial para
    //despues comprobar que es correcta la solucion.

    tInicio = MPI_Wtime();
    for (int i = 0; i < n; i++) {
      comprueba[i] = 0;
      for (int j = 0; j < n; j++) {
        comprueba[i] += A[i*n+j] * x[j];
      }
    }
    Tseq = MPI_Wtime()-tInicio;


    int errores = 0;
    for (unsigned int i = 0; i < n; i++) {
      //cout << "\t" << y[i] << "\t|\t" << comprueba[i] << endl;
      if (comprueba[i] != y[i])
        errores++;
    }
    cout << ".......Obtained and expected result can be seen above......." << endl;

    delete [] y;
    delete [] comprueba;
    delete [] A;

    if (errores) {
      cout << "Found " << errores << " Errors!!!" << endl;
    } else {
      cout << "No Errors!" << endl<<endl;
    }
    cout << "...Parallel time (without initial distribution and final gathering)= " << Tpar << " seconds." << endl<<endl;
    cout << "...Sequential time= " << Tseq << " seconds." << endl<<endl;
  }


  delete [] local_A;
  delete [] local_y;
  delete [] x;

  return 0;
}
