/* ******************************************************************** */
/*               Algoritmo Branch-And-Bound Secuencial                  */
/* ******************************************************************** */
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include "libbb.h"

#define color_carga 0
#define color_cota  1

using namespace std;

//Variables globales necesarias para las dos funciones
unsigned int NCIUDADES;
int size,
    id_proceso;
bool token_presente;  // Indica si el proceso posee el token

MPI_Comm comunicadorCarga;	// Para la distribuci�n de la carga
MPI_Comm comunicadorCota;	// Para la difusi�n de una nueva cota superior detectada

int main (int argc, char **argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &id_proceso);

  MPI_Comm_split(MPI_COMM_WORLD, color_carga, id_proceso, &comunicadorCarga);
  MPI_Comm_split(MPI_COMM_WORLD, color_cota, id_proceso, &comunicadorCota);

  if(id_proceso == 0){
    if (argc != 3) {
  		cerr << "La sintaxis es: mpirun -np <numero_procesos> bbpar <tamaño> <archivo>" << endl;
  		exit(1);
  	}
  }

  NCIUDADES = atoi(argv[1]);

	int** tsp0 = reservarMatrizCuadrada(NCIUDADES);
	tNodo	nodo,                  // nodo a explorar
			  lnodo,                 // hijo izquierdo
			  rnodo,                 // hijo derecho
			  solucion;              // mejor solucion
	bool  fin,                   // condicion de fin
		    nueva_U;               // hay nuevo valor de c.s.
	int   U,                     // valor de c.s.
	      iteraciones = 0,       // numero de iteraciones de cada proceso
        total_iteraciones = 0; // numero de iteraciones total
	tPila pila;                  // pila de nodos a explorar

	U = INFINITO;           // inicializa cota superior
	InicNodo (&nodo);       // inicializa estructura nodo

  if(id_proceso == 0){
    LeerMatriz(argv[2], tsp0);
    token_presente = true;
  }

  MPI_Bcast(&tsp0[0][0],
            NCIUDADES * NCIUDADES,
            MPI_INT,
            0,
            MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);

	double t = MPI_Wtime();

	if(id_proceso != 0){
		Equilibrado_Carga(pila, fin, solucion);
		if(!fin){
			pila.pop(nodo);
		}
	}
	fin = Inconsistente(tsp0);

  while(!fin){   // ciclo del Branch&Bound
    Ramifica(&nodo, &lnodo, &rnodo, tsp0);
    nueva_U = false;
    if(Solucion(&rnodo)){
      if(rnodo.ci() < U){
        U = rnodo.ci(); //Actualiza cota sup
        nueva_U = true;
        CopiaNodo(&rnodo, &solucion);
      }
    }else{  // Si no es un nodo hoja
      if(rnodo.ci() < U)
        pila.push(rnodo);
    }

    if(Solucion(&lnodo)){
      if(lnodo.ci() < U){
        U = lnodo.ci(); //Actualiza cota sup
        nueva_U = true;
        CopiaNodo(&lnodo, &solucion);
      }
    }else{  // Si no es un nodo hoja
      if(lnodo.ci() < U)
        pila.push(lnodo);
    }

    bool hay_nueva_cota_superior = Difusion_Cota_Superior(U);

    if(hay_nueva_cota_superior)
      pila.acotar(U);

    Equilibrado_Carga(pila, fin, solucion);
    if(!fin)
      pila.pop(nodo);

    iteraciones++;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  t = MPI::Wtime() - t;

  std::cout << "Proceso[" << id_proceso << "] --> Numero iteraciones = " << iteraciones << '\n';
	MPI_Reduce(&iteraciones,
             &total_iteraciones,
             1,
             MPI_INT,
             MPI_SUM,
             0,
             MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

	if(id_proceso == 0){
    printf ("Solucion: \n");
  	EscribeNodo(&solucion);
    cout << "Tiempo gastado= " << t << endl;
  	cout << "Numero de iteraciones = " << total_iteraciones << endl << endl;
	}

	liberarMatriz(tsp0);
	return 0;
}
