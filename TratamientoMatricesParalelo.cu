//Includes CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//Includes C++
#include <iostream>

//Includes C
#include <stdio.h>
#include <time.h>

#define BLOCK_SIZE 4

using namespace std;

/*
 * El kernel que se encarga de realiazar las multiplicaciones con la GPU
 *
 * Nota: las matrices de pasan en formato vector o array para que sea mas sencillo realizar las operaciones,
 *		aunque se le pasa el tamaño de la matriz para que se tengan en cuenta las filas y las columnas de
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
 *
 * Nota: las matrices de pasan en formato vector o array para que sea mas sencillo realizar las operaciones,
 *		aunque se le pasa el tamaño de la matriz para que se tengan en cuenta las filas y las columnas de
 *		la matriz
 */
__global__ void Suma_Matrices_GM(float *C, float *A, float *B, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < N)
		C[idx] = A[idx] + B[idx];
}

/*
 * Funcion que cre ados matrices de forma aleatoria, las suma y las multiplica e indica 
 * el tiempo que tarda en realizar las operaciones
 */
void ejecutarPruebas(int dimension)
{
	// PARTE DE LA INICIALIZACION DE LAS VARIABLES Y LAS MATRICES
	
	clock_t inicio, fin; 
	float *A_h, *B_h, *C_h;
	float *A_d, *B_d, *C_d;
	int nfil = dimension;
	int ncol = dimension;
	int N = nfil * ncol;
	size_t size = N * sizeof(float);

	A_h = (float*) malloc(size);
	B_h = (float*) malloc(size);
	C_h = (float*) malloc(size);

	/*
	 * Para simplificar la implementacion se crea un vector que se llenara con numeros aleatorios, solo que el vector tiene
	 * el tamaño de la suma de todas las filasy se puede tratar como una matriz
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

	cudaMalloc((void**) &A_d, size);
	cudaMalloc((void**) &B_d, size);
	cudaMalloc((void**) &C_d, size);

	cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

	//Calculos relacionados a los bloques para la ejecucion en la GPU
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 n_blocks(((int) (((float) ncol) / ((float) block_size.x)) + 1), ((int) (((float) nfil) / ((float) block_size.y)) + 1));

	Suma_Matrices_GM<<<n_blocks, block_size>>>(C_d, A_d, B_d, N);

	cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

	fin = clock();
	cout << "\t\tTiempo transcurrido: " << (fin-inicio)/(double)CLOCKS_PER_SEC << " segundos\n\n" << endl;
	
	// PARTE DE LAS MULTIPLICACIONES

	inicio = clock();
	printf("\tMultiplicacion de matrices de %dx%d elementos\n", nfil, ncol);

	cudaMalloc((void**) &A_d, size);
	cudaMalloc((void**) &B_d, size);
	cudaMalloc((void**) &C_d, size);

	cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

	Multiplica_Matrices_GM<<<n_blocks, block_size>>>(C_d, A_d, B_d, nfil, ncol);

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
int main()
{
	clock_t inicioGeneral, finGeneral;
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

	finGeneral = clock();

	cout << "FIN DE LAS PRUEBAS" << endl; cout << endl;

	cout << "Tiempo transcurrido en la ejecucion de las pruebas: " << endl;
	cout << (finGeneral-inicioGeneral)/(double)CLOCKS_PER_SEC << " segundos\n\n" << endl;

	printf("Pulsa una tecla para salir\n");
	getchar();
	return 0;
}
