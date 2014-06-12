
#include "stdafx.h"

#include <iostream> 
#include <time.h>

using namespace std;

void sumasYProductosMatrices();
float** rellenarMatriz(int tamano);
void sumarMatrices(float **matriz1, float **matriz2, int tamano);
void multiplicarMatrices(float **matriz1, float **matriz2, int tamano);
void imprimirMatriz(float **matriz, int tamano);


int _tmain(int argc, _TCHAR* argv[])
{
	sumasYProductosMatrices();
	getchar();
	return 0;
}

void sumasYProductosMatrices()
{
	clock_t inicio, fin, inicioGeneral, finGeneral;
	/*
		srand() sirve para cambiar la semilla (numero) que se utiliza para generar los numeros
		aleatorios. Al pasarle como parametro una llamada a la funcion time(), que obtiene el
		tiempo actual del sistema, en cada ejecucion cada semilla (y en consecuencia los numeros
		aleatorios generados) seran diferentes
	*/
	srand(time(NULL));
	
	const int tam1 = 10;
	const int tam2 = 100;
	const int tam3 = 300;
	const int tam4 = 500;
	const int tam5 = 700;
	const int tam6 = 1000;
	const int tam7 = 3000;
	
	cout << "INICIO DE LA CARGA DE MATRICES" << endl;
	inicio = clock();
	float **matriz1a = rellenarMatriz(tam1);
	float **matriz1b = rellenarMatriz(tam1);
	float **matriz2a = rellenarMatriz(tam2);
	float **matriz2b = rellenarMatriz(tam2);
	float **matriz3a = rellenarMatriz(tam3);
	float **matriz3b = rellenarMatriz(tam3);
	float **matriz4a = rellenarMatriz(tam4);
	float **matriz4b = rellenarMatriz(tam4);
	float **matriz5a = rellenarMatriz(tam5);
	float **matriz5b = rellenarMatriz(tam5);
	float **matriz6a = rellenarMatriz(tam6);
	float **matriz6b = rellenarMatriz(tam6);
	float **matriz7a = rellenarMatriz(tam7);
	float **matriz7b = rellenarMatriz(tam7);
	fin = clock();
	cout << "Tiempo transcurrido: " << (fin-inicio)/(double)CLOCKS_PER_SEC << " segundos\n\n" << endl << endl << endl;

	cout << "INICIO DE LAS PRUEBAS DE SUMAS Y PRODUCTOS DE MATRICES" << endl; 
	cout << endl;

	inicioGeneral = clock();

			// PRUEBAS 10x10

			cout << endl;
			cout << "PRUEBAS CON MATRICES DE 10x10 ELEMENTOS" << endl;
			cout << endl; cout << endl;

			inicio = clock();
			cout << "\tSuma de matrices de 10x10 elementos" << endl;
			sumarMatrices(matriz1a, matriz1b, tam1);
			fin = clock();
			cout << "\t\tTiempo transcurrido: " << (fin-inicio)/(double)CLOCKS_PER_SEC << " segundos\n\n" << endl;

			inicio = clock();
			cout << "\tProducto de matrices de 10x10 elementos" << endl;
			multiplicarMatrices(matriz1a, matriz1b, tam1);
			fin = clock();
			cout << "\t\tTiempo transcurrido: " << (fin-inicio)/(double)CLOCKS_PER_SEC << " segundos\n\n" << endl;

			// PRUEBAS 100x100

			cout << endl;
			cout << "PRUEBAS CON MATRICES DE 100x100 ELEMENTOS" << endl;
			cout << endl; cout << endl;

			inicio = clock();
			cout << "\tSuma de matrices de 100x100 elementos" << endl;
			sumarMatrices(matriz2a, matriz2b, tam2);
			fin = clock();
			cout << "\t\tTiempo transcurrido: " << (fin-inicio)/(double)CLOCKS_PER_SEC << " segundos\n\n" << endl;

			inicio = clock();
			cout << "\tProducto de matrices de 100x100 elementos" << endl;
			multiplicarMatrices(matriz2a, matriz2b, tam2);
			fin = clock();
			cout << "\t\tTiempo transcurrido: " << (fin-inicio)/(double)CLOCKS_PER_SEC << " segundos\n\n" << endl;

			// PRUEBAS 300x300

			cout << endl;
			cout << "PRUEBAS CON MATRICES DE 300x300 ELEMENTOS" << endl;
			cout << endl; cout << endl;

			inicio = clock();
			cout << "\tSuma de matrices de 300x300 elementos" << endl;
			sumarMatrices(matriz3a, matriz3b, tam3);
			fin = clock();
			cout << "\t\tTiempo transcurrido: " << (fin-inicio)/(double)CLOCKS_PER_SEC << " segundos\n\n" << endl;

			inicio = clock();
			cout << "\tProducto de matrices de 300x300 elementos" << endl;
			multiplicarMatrices(matriz3a, matriz3b, tam3);
			fin = clock();
			cout << "\t\tTiempo transcurrido: " << (fin-inicio)/(double)CLOCKS_PER_SEC << " segundos\n\n" << endl;

			// PRUEBAS 500x500

			cout << endl;
			cout << "PRUEBAS CON MATRICES DE 500x500 ELEMENTOS" << endl;
			cout << endl; cout << endl;

			inicio = clock();
			cout << "\tSuma de matrices de 500x500 elementos" << endl;
			sumarMatrices(matriz4a, matriz4b, tam4);
			fin = clock();
			cout << "\t\tTiempo transcurrido: " << (fin-inicio)/(double)CLOCKS_PER_SEC << " segundos\n\n" << endl;

			cout << "\tProducto de matrices de 500x500 elementos" << endl;
			multiplicarMatrices(matriz4a, matriz4b, tam4);
			fin = clock();
			cout << "\t\tTiempo transcurrido: " << (fin-inicio)/(double)CLOCKS_PER_SEC << " segundos\n\n" << endl;

			// PRUEBAS 700x700

			cout << endl;
			cout << "PRUEBAS CON MATRICES DE 700x700 ELEMENTOS" << endl;
			cout << endl; cout << endl;

			inicio = clock();
			cout << "\tSuma de matrices de 700x700 elementos" << endl;
			sumarMatrices(matriz5a, matriz5b, tam5);
			fin = clock();
			cout << "\t\tTiempo transcurrido: " << (fin-inicio)/(double)CLOCKS_PER_SEC << " segundos\n\n" << endl;

			inicio = clock();
			cout << "\tProducto de matrices de 700x700 elementos" << endl;
			multiplicarMatrices(matriz5a, matriz5b, tam5);
			fin = clock();
			cout << "\t\tTiempo transcurrido: " << (fin-inicio)/(double)CLOCKS_PER_SEC << " segundos\n\n" << endl;

			// PRUEBAS 1000x1000

			cout << endl;
			cout << "PRUEBAS CON MATRICES DE 1000x1000 ELEMENTOS" << endl;
			cout << endl; cout << endl;

			inicio = clock();
			cout << "\tSuma de matrices de 1000x1000 elementos" << endl;
			sumarMatrices(matriz6a, matriz6b, tam6);
			fin = clock();
			cout << "\t\tTiempo transcurrido: " << (fin-inicio)/(double)CLOCKS_PER_SEC << " segundos\n\n" << endl;

			inicio = clock();
			cout << "\tProducto de matrices de 1000x1000 elementos" << endl;
			multiplicarMatrices(matriz6a, matriz6b, tam6);
			fin = clock();
			cout << "\t\tTiempo transcurrido: " << (fin-inicio)/(double)CLOCKS_PER_SEC << " segundos\n\n" << endl;

			
	finGeneral = clock();

	cout << "FIN DE LAS PRUEBAS" << endl; cout << endl;

	cout << "Tiempo transcurrido en la ejecucion de las pruebas: " << endl;
	cout << (finGeneral-inicioGeneral)/(double)CLOCKS_PER_SEC << " segundos\n\n" << endl;

}

float** rellenarMatriz(int tamano)
{	
	/*
		Para asignar la memoria necesaria a la matriz se utiliza primero un malloc que
		reserva memoria para toda la matriz y luego se recorre cada fila para inicializar
		cada array de float

		Primero se reserva memoria para un array de arrays y despues para cada posicion de ese
		array (que contiene un array de float) se reserva un tamano de memoria que es la
		multiplicacion entreel numero de elementos y el tamaño que tiene el tipo de dato

		Para ello se utiliza el parametro tamano
	*/
	float **matriz = (float**) malloc(tamano * sizeof(float*));

	for(int k = 0; k < tamano; k++)
		matriz[k] = (float*) malloc(tamano * sizeof(float));

	for (int i = 0; i < tamano; ++i) 
		for (int j = 0; j < tamano; ++j) 
			/*
				Como la funcion rand() genera un numero entero, para conseguir un numero decimal
				se divide un numero aleatorio entre otro numero aleatorio para generar decimales,
				sumandole 1 al divisor para que o pueda ser 0 y no falle en ejecucion

				Tambien hay que hacer castings a tipo float a ambos operadores por separado para que la
				division entre ellos devuelva float (dividir un entero entre un entero devuelve un
				entero aunque el resultado real no sea un entero)
			*/
			matriz[i][j] = ((float) rand()) / ((float)(rand()+1));

	/* Se puede llamar a un metodo imprimirMatriz para comprobar que realmente se estan 
		introduciendo valores numericos en la matriz */

	//imprimirMatriz(matriz, tamano);
	return matriz;
}

void sumarMatrices(float **matriz1, float **matriz2, int tamano)
{	
	/*
		Para asignar la memoria necesaria a la matriz se utiliza primero un malloc que
		reserva memoria para toda la matriz y luego se recorre cada fila para inicializar
		cada array de float

		Primero se reserva memoria para un array de arrays y despues para cada posicion de ese
		array (que contiene un array de float) se reserva un tamano de memoria que es la
		multiplicacion entreel numero de elementos y el tamaño que tiene el tipo de dato

		Para ello se utiliza el parametro tamano
	*/
	float **matrizFinal = (float**) malloc(tamano * sizeof(float*));

	for(int k = 0; k < tamano; k++)
		matrizFinal[k] = (float*) malloc(tamano * sizeof(float));

	for (int i = 0; i < tamano; i++)
		for (int j = 0; j < tamano; j++)
			matrizFinal[i][j] = matriz1[i][j] + matriz2[i][j];

}

void multiplicarMatrices(float **matriz1, float **matriz2, int tamano)
{	
	/*
		Para asignar la memoria necesaria a la matriz se utiliza primero un malloc que
		reserva memoria para toda la matriz y luego se recorre cada fila para inicializar
		cada array de float

		Primero se reserva memoria para un array de arrays y despues para cada posicion de ese
		array (que contiene un array de float) se reserva un tamano de memoria que es la
		multiplicacion entreel numero de elementos y el tamaño que tiene el tipo de dato

		Para ello se utiliza el parametro tamano
	*/
	float **matrizFinal = (float**) malloc(tamano * sizeof(float*));

	for(int m = 0; m < tamano; m++)
		matrizFinal[m] = (float*) malloc(tamano * sizeof(float));

	for(int l = 0; l < tamano; l++)
		for(int n = 0; n < tamano; n++)
			matrizFinal[l][n] = (float) 0;

	for (int i = 0; i < tamano; i++)
		for (int j = 0; j < tamano; j++)
			for (int k = 0; k < tamano; k++)
				matrizFinal[i][j] += matriz1[i][k] * matriz2[k][j];
}

void imprimirMatriz(float **matriz, int tamano)
{
	for(int i = 0; i < tamano; i++) {
		for(int j = 0; j < tamano; j++) {
			cout << matriz[i][j] << " ";
		}
		cout << endl;
	}
}
