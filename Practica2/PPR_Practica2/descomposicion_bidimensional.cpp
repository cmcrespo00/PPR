/*
============================================================================
Name        : descomposicion_bidimensional.cpp
Author      : Cristina María Crespo Arco
============================================================================
*/

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <mpi.h>

using namespace std;

int main(int argc, char * argv[]) {
  int numeroProcesadores,
      id_Proceso;

  float *A, // Matriz global a multiplicar
        *x, // Vector a multiplicar
        *y, // Vector resultado
        *local_A,  // Matriz local de cada proceso
        *local_x,  // Vector local de cada proceso
        *local_y,  // Porción local del resultado en  cada proceso
        *l_y;

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

  int numero_filas_columnas = ceil(sqrt(numeroProcesadores));
  int elementos_fila_columna = ceil(n/(sqrt(numeroProcesadores)));
  x = new float[n]; //reservamos espacio para el vector x (n floats).
  y = new float[n];//reservamos espacio para el vector resultado final y (n floats)

  float *buf_envio = new float[n*n];

  // Proceso 0 genera matriz A y vector x
  if (id_Proceso==0)  {
    A = new float[n*n];//reservamos espacio para la matriz (n x n floats)

    // Rellena la matriz y el vector
    for (int i = 0; i < n; i++) {
      x[i] = (float) (1.5*(1+(5*(i))%3)/(1+(i)%5));
      for (int j = 0; j < n; j++) {
        A[i*n+j] = (float) (1.5*(1+(5*(i+j))%3)/(1+(i+j)%5));
      }
    }

    /*Defino el tipo de bloque cuadrado*/
    MPI_Datatype MPI_BLOQUE;
    MPI_Type_vector(elementos_fila_columna,
                    elementos_fila_columna,
                    n,
                    MPI_FLOAT,
                    &MPI_BLOQUE);

    /*Creo el nuevo tipo*/
    MPI_Type_commit(&MPI_BLOQUE);

    /*Empaqueta bloque a bloque en el buffer de envío*/
    for (int i = 0, posicion = 0; i < numeroProcesadores; i++) {
      /*Calculo la posicion de comienzo de cada submatriz*/
      int fila_P = i / numero_filas_columnas;
      int columna_P = i % numero_filas_columnas;
      int comienzo = (columna_P*elementos_fila_columna) + (fila_P*elementos_fila_columna*elementos_fila_columna*numero_filas_columnas);
      MPI_Pack(&A[comienzo],
               1,
               MPI_BLOQUE,
               buf_envio,
               sizeof(float)*n*n,
               &posicion,
               MPI_COMM_WORLD);
    }

    /*Libero el tipo bloque*/
    MPI_Type_free(&MPI_BLOQUE);
  }

  // Cada proceso reserva espacio para su porción de A y para el vector x
  const int local_A_size = (elementos_fila_columna)*(elementos_fila_columna);
  const int local_x_size = elementos_fila_columna;
  const int local_y_size = elementos_fila_columna;
  local_A = new float[local_A_size];//reservamos espacio para la matriz
  local_x = new float[local_x_size]; //reservamos espacio para el vector x.
  local_y = new float[local_y_size]; //reservamos espacio para el vector y.
  l_y = new float[local_y_size]; //reservamos espacio para el vector y.

  for (int i = 0; i < local_x_size; i++) {
    local_x[i] = 0;
  }

  // Repartimos una bloque de filas de A a cada proceso
  MPI_Scatter(buf_envio, // Matriz que vamos a compartir
              sizeof(float)*local_A_size, // Numero de filas a entregar
              MPI_PACKED, // Tipo de dato a enviar
              local_A, // Vector en el que almacenar los datos
              local_A_size, // Numero de filas a recibir
              MPI_FLOAT, // Tipo de dato a recibir
              0, // Proceso raiz que envia los datos
              MPI_COMM_WORLD); // Comunicador utilizado (En este caso, el global)

  // Repartimos y difundimos el vector x a cada proceso
  MPI_Comm comm_diagonal; // Comunicador para los elementos de la diagonal.
  int diagonal = 0;

  for (int i = 0; (i < numero_filas_columnas) && (diagonal == 0); i++) {
    if((i * sqrt(numeroProcesadores) + i) == id_Proceso)
      diagonal = 1;
  }

  // creamos un nuevo cominicador para los elementos en la diagonal
  MPI_Comm_split(MPI_COMM_WORLD, // a partir del comunicador global.
                 diagonal, // lo de la diagonal entraran en el comunicador
                 id_Proceso, // indica el orden de asignacion de rango dentro de los nuevos comunicadores
                 &comm_diagonal); // Referencia al nuevo comunicador creado.

  // Repartimos el vector x entre la diagonal
  MPI_Scatter(x, // Matriz que vamos a compartir
              local_x_size, // Numero de filas a entregar
              MPI_FLOAT, // Tipo de dato a enviar
              local_x, // Vector en el que almacenar los datos
              local_x_size, // Numero de filas a recibir
              MPI_FLOAT, // Tipo de dato a recibir
              0, // Proceso raiz que envia los datos
              comm_diagonal); // Comunicador utilizado, el de la diagonal

  // Repartimos el vector x entre todas las columnas
  MPI_Comm comm_columnas; // Comunicador para los elementos de la columnas.

  int columna = floor(id_Proceso / numero_filas_columnas);

  // creamos un nuevo cominicador para los elementos en la diagonal
  MPI_Comm_split(MPI_COMM_WORLD, // a partir del comunicador global.
                 columna, // lo de la diagonal entraran en el comunicador
                 id_Proceso, // indica el orden de asignacion de rango dentro de los nuevos comunicadores
                 &comm_columnas); // Referencia al nuevo comunicador creado.

  MPI_Bcast(local_x, // Dato a compartir
            local_x_size, // Numero de elementos que se van a enviar y recibir
            MPI_FLOAT, // Tipo de dato que se compartira
            columna, // Proceso raiz que envia los datos
            comm_columnas); // Comunicador utilizado

  // Hacemos una barrera para asegurar que todas los procesos comiencen la ejecucion
  // a la vez, para tener mejor control del tiempo empleado
  MPI_Barrier(MPI_COMM_WORLD);

  // Inicio de medicion de tiempo
  tInicio = MPI_Wtime();

  for (int i = 0; i < local_y_size; i++) {
    local_y[i] = 0.0;
    for (int j = 0; j < elementos_fila_columna; j++) {
      local_y[i] += local_A[i*(elementos_fila_columna)+j] * local_x[i];
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  // fin de medicion de tiempo
  Tpar = MPI_Wtime()-tInicio;

  for (int i = 0; i < local_y_size; i++) {
    l_y[i] = 0;
  }

  MPI_Reduce (local_y, // Dato que envia cada proceso
              l_y, //Dato que recibe el proceso raiz
              local_y_size, // Numero de elementos que se envian-reciben
              MPI_FLOAT, // Tipo del dato que se envia-recibe
              MPI_SUM, //Operacion que se va a realizar
              columna, // proceso que va a recibir los datos
              comm_columnas); // Canal de comunicacion

  // Recogemos los datos de la multiplicacion, por cada proceso sera un escalar
  // y se recoge en un vector, Gather se asegura de que la recolecci�n se haga
  // en el mismo orden en el que se hace el Scatter, con lo que cada escalar
  // acaba en su posicion correspondiente del vector.
  MPI_Gather(l_y, // Dato que envia cada proceso
             local_y_size, // Numero de elementos que se envian
             MPI_FLOAT, // Tipo del dato que se envia
             y, // Vector en el que se recolectan los datos
             local_y_size, // Numero de datos que se esperan recibir por cada proceso
             MPI_FLOAT, // Tipo del dato que se recibira
             0, // proceso que va a recibir los datos
             comm_diagonal); // Canal de comunicacion

  // Terminamos la ejecucion de los procesos, despues de esto solo existira
  // el proceso 0
  // Ojo! Esto no significa que los demas procesos no ejecuten el resto
  // de codigo despues de "Finalize", es conveniente asegurarnos con una
  // condicion si vamos a ejecutar mas codigo (Por ejemplo, con "if(rank==0)".
  MPI_Comm_free(&comm_diagonal);
  MPI_Comm_free(&comm_columnas);
  MPI_Finalize();

  if (id_Proceso == 0) {
    float * comprueba = new float [n];
    //Calculamos la multiplicacion secuencial para
    //despues comprobar que es correcta la solucion.

    tInicio = MPI_Wtime();
    for (int i = 0; i < n; i++) {
      comprueba[i] = 0;
      for (int j = 0; j < n; j++) {
        comprueba[i] += A[i*n+j] * x[i];
      }
    }
    Tseq = MPI_Wtime()-tInicio;


    int errores = 0;
    for (unsigned int i = 0; i < n; i++) {
      cout << "\t" << y[i] << "\t|\t" << comprueba[i] << endl;
      if (comprueba[i] != y[i])
        errores++;
    }
    cout << ".......Obtained and expected result can be seen above......." << endl;

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

  delete [] buf_envio;
  delete [] local_A;
  delete [] local_x;
  delete [] local_y;
  delete [] l_y;
  delete [] x;
  delete [] y;

  return 0;
}
