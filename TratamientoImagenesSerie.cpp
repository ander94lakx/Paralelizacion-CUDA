#include "stdafx.h"

// Includes OpenCV
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp" 

// Includes C++
#include <iostream>

// Includes C
#include <stdio.h> 
#include <time.h>

// Indica los namespaces ques se estan usando para que la sintaxis quede mas limpia
using namespace std;
using namespace cv;

void binarizacion();
void binarizacion(int p);
void histograma();

const char* PATH_IMAGEN = "C:/lena_std.tif"; // Constante global que indica la ruta de la imagen
const bool DEBUG = true;

int main()
{
	int opc;
	while (true)
	{
		cout << "Introduce el modo que quieres ejecutar" << endl;
		cout << "\t 1 -> Binarizacion" << endl;
		cout << "\t 2 -> Histograma de frecuencias" << endl;
		cout << "\t 0 -> Salir" << endl;
		cin >> opc;
		if(opc == 1) {
			cout << "Introduce el valor umbral para la binarizacion: " << endl;
			int valor;
			cin >> valor;
			binarizacion(valor);
		}
		else if(opc == 2)
			histograma();
		else if(opc == 0)
			return 0;
		cout << endl << endl;
	}
}

// Metodo sobrecargado para no necesitar una variable
void binarizacion()
{
	binarizacion((int) 255/2);
}

/*
* Calcula y muestra por pantalla la imagen binarizada y la original
*
* Nota: la variable threshold indica a partir de que nivel se va a determinar si es blanco o negro
*/
void binarizacion(int p)
{
	clock_t inicio, fin, inicioBin, finBin;
	inicio = clock();

	IplImage* src; // Imagen de color base  
	IplImage* gray; // Contendrá la imagen convertida en escala de grises 
	IplImage* grayThresh; // Imagen binaria conseguida a partir de la imagen en escala de grises 

	int threshold;
	if(p > 0 && p <255)
		threshold = p; // Definimos el valor umbral 
	else
		threshold = 160; // Definimos el valor umbral
	int maxValue = 255; // Definimos el valor máximo 
	int thresholdType = CV_THRESH_BINARY; // Definimos el tipo de binarización 

	src = cvLoadImage(PATH_IMAGEN, CV_LOAD_IMAGE_COLOR); // Cargamos la imagen de color 
	gray = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_8U, 1); 
		// La imagen de intensidad tendrá la misma configuración que la fuente pero con un solo canal 
	cvCvtColor(src, gray, CV_BGR2GRAY); // Pasamos la imagen de color a escala de grises 
	grayThresh = cvCloneImage( gray ); // Copiamos la imagen en escala de grises
	
	cvNamedWindow("Imagen a color original", 1 ); 
	cvShowImage("Imagen a color original", src ); // Representamos la imagen de color 

	inicioBin = clock();

	cvThreshold(gray, grayThresh, threshold, maxValue, thresholdType); // Binarizamos la imagen en escala de grises 

	finBin = clock();
	cout << "\tTiempo transcurrido ESPECIFICAMENTE en la operacion de binarizacion: " 
		<< (finBin-inicioBin)/(double)CLOCKS_PER_SEC << " segundos\n\n" << endl;

	cvNamedWindow("Imagen a escala de grises binarizada", 1 );
	cvShowImage("Imagen a escala de grises binarizada", grayThresh ); // Representamosla imagen a escala de grises binarizada 
	
	fin = clock();
	cout << "\tTiempo TOTAL transcurrido en binarizar: " 
		<< (fin-inicio)/(double)CLOCKS_PER_SEC << " segundos\n\n" << endl;

	cvWaitKey(0); // Pulsamos una tecla para terminar 
	
	// Destruimos las ventanas y eliminamos las imagenes
	cvDestroyAllWindows();
	cvReleaseImage( &src );
	cvReleaseImage( &gray );
	cvReleaseImage( &grayThresh );
	
}

/*
* Calcula y muestra por pantalla el histograma de la imagen usando para ello la GPU
*/
void histograma()
{
	clock_t inicio, fin, inicioCalcHist, finCalcHist;
	inicio = clock();

	// Se carga la imagen 
	Mat src = imread(PATH_IMAGEN, 1 );
	if(!src.data) return;

	// Se separa las imagenes en los 3 colores ( R,G,B )
	vector<Mat> bgr_planes;
	split( src, bgr_planes );

	// Se stablece el contador 
	int histSize = 256;

	// Se establecen los rangos (para B,G,R)
	float range[] = { 0, 256 } ;
	const float* histRange = { range };

	bool uniform = true;
	bool accumulate = false;

	cv::Mat b_hist, g_hist, r_hist;

	inicioCalcHist = clock();

	//Se calculan los histogramas
	calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

	finCalcHist = clock();
	cout << "\tTiempo transcurrido ESPECIFICAMENTE en la operacion del calculo del histograma: " 
		<< (finCalcHist-inicioCalcHist)/(double)CLOCKS_PER_SEC << " segundos\n\n" << endl;

	if (DEBUG){
		//Salida de los histogramas
		cout << "B = "<< endl << " "  << b_hist << endl << endl;
		cout << "G = "<< endl << " "  << g_hist << endl << endl;
		cout << "R = "<< endl << " "  << r_hist << endl << endl; 
	}

	// Se dibuja el histograma 
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound( ((double) hist_w)/((double)histSize) );

	cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );

	cv::normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	cv::normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	cv::normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

	if (DEBUG)
	{
		// Salida de las matrices normalizadas
		cout << "B(norm) = "<< endl << " "  << b_hist << endl << endl;
		cout << "G(norm) = "<< endl << " "  << g_hist << endl << endl;
		cout << "R(norm) = "<< endl << " "  << r_hist << endl << endl; 
	}

	// Se dibuja para cada canal de RGB 
	for(int i = 1; i < histSize; i++ )
	{
		cv::line( histImage, 
			Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ),
			Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ), 
			Scalar( 255, 0, 0), 2,8, 0 );
		cv::line( histImage, 
			Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ),
			Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
			Scalar( 0, 255, 0), 2, 8,0 );
		cv::line( histImage, 
			Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ),
			Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
			Scalar( 0, 0, 255), 2, 8, 0 );
	}

	//Se muestra el resultado
	cv::imshow("Resultado histograma", histImage );
	cv::imshow("Imagen", src );

	fin = clock();
	cout << "\tTiempo TOTAL transcurrido en calculary mostrar el histograma: " 
		<< (fin-inicio)/(double)CLOCKS_PER_SEC << " segundos\n\n" << endl;

	cvWaitKey(0); // Pulsamos una tecla para terminar
	cvDestroyAllWindows(); // Destruimos las ventanas
	
	// Liberamos de la memoria las imagenes
	src.release(); 
	histImage.release();
}
