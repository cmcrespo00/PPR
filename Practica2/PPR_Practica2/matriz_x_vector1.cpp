/*
 ============================================================================
 Name        : matriz_x_vector1.cpp
 Author      : Jose Miguel Mantas Ruiz
 Copyright   : GNU Open Souce and Free license
 Description : Tutorial 5. Multiplicacion de Matrix por Vector.
    Multiplica un vector por una matriz (b).
 ============================================================================
 */

#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

int main(int argc, char * argv[]) {
  float *A, // Matriz a multiplicar
        *x, // Vector que vamos a multiplicar
        *comprueba; // Guarda el resultado final (calculado secuencialmente), su valor

  int n;

  if (argc <= 1) {// si no se pasa el size de la matriz, se coge n=10
    cout << "Square Matrix dimension? (Default: 10)"<< endl;
    n = 10;
  } else
    n = atoi(argv[1]);

  A = new float[n*n]; //reservamos espacio para la matriz (n x n floats)
  x = new float[n];   //reservamos espacio para el vector (n floats).

  //Rellena la matriz y el vector
  for (int i = 0; i < n; i++) {
    x[i] = (float) (1.5*(1+(5*(i))%3)/(1+(i)%5));
    for (int j = 0; j < n; j++) {
      A[i*n+j] = (float) (1.5*(1+(5*(i+j))%3)/(1+(i+j)%5));
    }
  }

  // Muestra A y x
  cout << "The matrix and vector are " << endl<<endl;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (j == 0) cout << "[";
        cout << A[i*n+j];
      if (j == n - 1)
        cout << "]";
      else
        cout << "  ";
    }
    cout << "\t  [" << x[i] << "]" << endl;
  }
  cout << "\n";

  comprueba = new float [n];
  //Calculamos la multiplicacion secuencial para
  //despues comprobar que es correcta la solucion.
  for (int i = 0; i < n; i++) {
    comprueba[i] = 0;
    for (int j = 0; j < n; j++) {
      comprueba[i] += A[i*n+j] * x[j];
    }
  }

  cout << "The result is:" << endl << endl;
  for (int i = 0; i < n; i++) {
    cout << comprueba[i] << endl;
  }

  delete [] comprueba;
  delete [] A;
  delete [] x;

  return 0;
}
