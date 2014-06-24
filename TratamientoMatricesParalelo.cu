//Includes CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//Includes C++
#include <iostream>

//Includes C
#include <stdio.h>
#include <time.h>

/*
 * Este define controla el numero de threads por bloque que se utiliza
 * 
 * Cuanto mas se utilicen mas rapido y com mayor paralelismo se realizara la ejecucion
 * 
 * En el caso de una GPU GeForce GT 720M con 96 nucleos CUDA el valor optimo corresponderia a 96, que indicaria
 * que estamos haciendo uso de todos los nucleos CUDA
 */

#define BLOCK_SIZE 96

using namespace std;

/*
 * A la hora de crear metodos en CUDA existen 3 prefijos diferentes:
 *		__global__  ->  Indica que el metodo se ejecuta en la GPU pero se le invoca desde la GPU
 *		__device__  ->  Indica que el metodo se invoca y se ejecuta en la GPU
 *		__host__    ->  Indica que el metodo se invoca y se ejecuta en la CPU (es la opcion por defecto)
 */
 
/*
 * El kernel que se encarga de realiazar las multiplicaciones con la GPU
 Este metodo implementa el comportamiento SOLO para un bloque de procesamiento
 *
 * Nota: las matrices de pasan en formato vector o array para que sea mas sencillo realizar las operaciones,
 *		se le pasa el tamaño de la matriz para que se tengan en cuenta las filas y las columnas de
 *		la matriz
 */
__global__ void Multiplica_Matrices_GM(float *C, float *A, float *B, int nfil, int ncol)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int index = idy * ncol + idx;
	if(idy < nfil && idx < ncol){
		float sum = 0.0f;
		for (int i = 0; i < ncol; i++)
			sum += A[idy*ncol+i] * B[i*ncol+idx];
		C[index] = sum;
	}
}

/*
 * El kernel que se encarga de realiazar las sumas con la GPU
 * Este metodo implementa el comportamiento SOLO para un bloque de procesamiento
 *
 * Nota: las matrices de pasan en formato vector o array para que sea mas sencillo realizar las operaciones,
 *		se le pasa el tamaño de la matriz para que se tengan en cuenta las filas y las columnas de
 *		la matriz
 */
__global__ void Suma_Matrices_GM(float *C, float *A, float *B, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < N)
		C[idx] = A[idx] + B[idx];
}

/*
 * Funcion que crea dos matrices de forma aleatoria, las suma y las multiplica e indica 
 * el tiempo que tarda en realizar las operaciones
 */
__host__ void ejecutarPruebas(int dimension)
{
	// PARTE DE LA INICIALIZACION DE LAS VARIABLES Y LAS MATRICES
	
	clock_t inicio, fin; // Variables para calcular el tiempo
	float *A_h, *B_h, *C_h;
	float *A_d, *B_d, *C_d;
	int nfil = dimension;
	int ncol = dimension;
	int N = nfil * ncol;
	size_t size = N * sizeof(float);

	// Se reserva memoria de la CPU para las matrices
	A_h = (float*) malloc(size);
	B_h = (float*) malloc(size);
	C_h = (float*) malloc(size);

	/*
	 * Para simplificar la implementacion se crea un vector que se llenara con numeros aleatorios, solo que el vector tiene
	 * el tamaño de la suma de todas las filas y se puede tratar como una matriz
	 */
	for(int i = 0; i < nfil; i++){
		for(int j = 0; j < ncol; j++){
			A_h[i*ncol+j] = ((float) rand()) / ((float)(rand()+1));
			B_h[i*ncol+j] = ((float) rand()) / ((float)(rand()+1));
		}
	}

	// PARTE DE LAS SUMAS

	inicio = clock();
	printf("\tSuma de matrices de %dx%d elementos\n", nfil, ncol);

	// Se reserva memoria de la GPU para las matrices
	cudaMalloc((void**) &A_d, size);
	cudaMalloc((void**) &B_d, size);
	cudaMalloc((void**) &C_d, size);

	// Se copian las matrices de la CPU a la GPU
	cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

	// Calculos para indicar el numero de bloques y therads a utilizar al llamar al kernel
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 n_blocks(((int) (((float) ncol) / ((float) block_size.x)) + 1), ((int) (((float) nfil) / ((float) block_size.y)) + 1));

	/*
	 * Llamada al kernel que ejecutara la operacion de suma
	 *
	 * Entre "<<<" y ">>>" se definen el numero de bloques utilizados para ja ejecucion y los threads que utilizara
	 * cada bloque basados en los calculos anteriores
	 */
	Suma_Matrices_GM<<<n_blocks, block_size>>>(C_d, A_d, B_d, N);

	// Se copia el resultado de la GPU a la CPU
	cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

	fin = clock();
	cout << "\t\tTiempo transcurrido: " << (fin-inicio)/(double)CLOCKS_PER_SEC << " segundos\n\n" << endl;
	
	// PARTE DE LAS MULTIPLICACIONES

	inicio = clock();
	printf("\tMultiplicacion de matrices de %dx%d elementos\n", nfil, ncol);

	// Se reserva memoria de la GPU para las matrices
	cudaMalloc((void**) &A_d, size);
	cudaMalloc((void**) &B_d, size);
	cudaMalloc((void**) &C_d, size);

	// Se copian las matrices de la CPU a la GPU
	cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

	/*
	 * Llamada al kernel que ejecutara la operacion de multiplicacion
	 *
	 * Entre "<<<" y ">>>" se definen el numero de bloques utilizados para ja ejecucion y los threads que utilizara
	 * cada bloque basados en los calculos anteriores
	 */
	Multiplica_Matrices_GM<<<n_blocks, block_size>>>(C_d, A_d, B_d, nfil, ncol);

	// Se copia el resultado de la GPU a la CPU
	cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

	fin = clock();
	cout << "\t\tTiempo transcurrido: " << (fin-inicio)/(double)CLOCKS_PER_SEC << " segundos\n\n" << endl;

	// PARTE DE LA LIBERACION DE RECURSOS

	free(A_h);
	free(B_h);
	free(C_h);

	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
}

/*
 * Funcion principal que realiza sumas y multiplicaciones con matrices de diferentes tamaños
 */
__host__ int main()
{
	/*
	 *	srand() sirve para cambiar la semilla (numero) que se utiliza para generar los numeros
	 *	aleatorios. Al pasarle como parametro una llamada a la funcion time(), que obtiene el
	 *	tiempo actual del sistema, en cada ejecucion cada semilla (y en consecuencia los numeros
	 *	aleatorios generados) seran diferentes siempre
	 */
	srand(time(NULL));

	clock_t inicioGeneral, finGeneral; // Variables para calcular el tiempo
	/*
	 * La llamada al metodo clock() devuelve el instante de tiempo actual, en formato numerico double
	 *
	 * A lo largo de las pruebas es llama al metodo clock() para obtener el tiempo en el que se inica una operacion y
	 * el tiempo en el que termina una operacion, para asi calculando la diferencia poder hallar el tiempo invertido en
	 * realizar la operacion
	 */
	inicioGeneral = clock();
	
	printf("Inicio de las pruebas con 10 elementos\n\n");
	ejecutarPruebas(10);
	printf("Inicio de las pruebas con 100 elementos\n\n");
	ejecutarPruebas(100);
	printf("Inicio de las pruebas con 300 elementos\n\n");
	ejecutarPruebas(300);
	printf("Inicio de las pruebas con 500 elementos\n\n");
	ejecutarPruebas(500);
	printf("Inicio de las pruebas con 700 elementos\n\n");
	ejecutarPruebas(700);
	printf("Inicio de las pruebas con 1000 elementos\n\n");
	ejecutarPruebas(1000);
	printf("Inicio de las pruebas con 3000 elementos\n\n");
	ejecutarPruebas(3000);

	finGeneral = clock();

	cout << "FIN DE LAS PRUEBAS" << endl; cout << endl;

	cout << "Tiempo transcurrido en la ejecucion de las pruebas: " << endl;
	cout << (finGeneral-inicioGeneral)/(double)CLOCKS_PER_SEC << " segundos\n\n" << endl;

	printf("Pulsa una tecla para salir\n");
	getchar();
	return 0;
}
