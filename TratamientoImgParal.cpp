#include "stdafx.h"

#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

//includes histograma
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp" 

#include <iostream> 
#include <stdio.h> 

#include <time.h>

// Indica los namespaces ques se estan usando para que la sintaxis quede mas limpia
using namespace std;
using namespace cv;
using namespace cv::gpu;

void binarizacionParalelo();
void binarizacionParalelo(int p);
void histogramaParalelo();
void info();

const char* PATH_IMAGEN = "C:/lena_std.tif"; // Constante global que indica la ruta de la imagen
const char* PATH_IMAGEN_8BIT = "C:/lena512_8bit.jpg"; // Imagen de 8 bits (color)
const bool DEBUG = true;

int main()
{
	while (true)
	{
		int opc;
		cout << "Introduce el modo que quieres ejecutar" << endl;
		cout << "\t 1 -> Binarizacion" << endl;
		cout << "\t 2 -> Histograma de frecuencias" << endl;
		cout << "\t 3 -> Informacion CUDA + OpenCV" << endl;
		cout << "\t 0 -> Salir" << endl;
		cin >> opc;
		if(opc == 1) {
			cout << "Introduce el valor umbral para la binarizacion: " << endl;
			int valor;
			cin >> valor;
			binarizacionParalelo(valor);
		}
		else if(opc == 2)
			histogramaParalelo();
		else if(opc == 3)
			info();
		else if(opc == 0)
			return 0;
		cout << endl << endl; 
	}
}

// Metodo sobrecargado para no necesitar una variable
void binarizacionParalelo()
{
	binarizacionParalelo((int) 255/2);
}

/*
 * Calcula y muestra por pantalla la imagen binarizada y la original, usando para binarizarla la GPU
 *
 * Nota: la variable threshold indica a partir de que nivel se va a determinar si es blanco o negro
 */
void binarizacionParalelo(int p)
{
	clock_t inicio, fin;
	inicio = clock();

	IplImage* src; // Imagen de color base 
	IplImage* colorThresh; // Contendrá la imagen de color binarizada 
	IplImage* gray; // Contendrá la imagen convertida en escala de grises 
	IplImage* grayThresh; // Imagen binaria conseguida a partir de la imagen en escala de grises 

	int threshold;
	if(p > 0 && p <255)
		threshold = p; // Definimos el valor umbral 
	else
		threshold = 160; // Definimos el valor umbral
	int maxValue = 255; // Definimos el valor máximo 
	int thresholdType = CV_THRESH_BINARY; // Definimos el tipo de binarización 
	src = cvLoadImage(PATH_IMAGEN, 1); // Cargamos imagen de color 
	colorThresh = cvCloneImage( src ); // Copiamos esa imagen de color 
	gray = cvCreateImage( cvSize(src->width, src->height), IPL_DEPTH_8U, 1 ); 
		// La imagen de intensidad tendrá la misma configuración que la fuente pero con un solo canal 
	cvCvtColor( src, gray, CV_BGR2GRAY ); // Pasamos la imagen de color a escala de grises 
	grayThresh = cvCloneImage( gray ); // Copiamos la imagen en escala de grises
	
	cvNamedWindow("src", 1 ); 
	cvShowImage("src", src ); // Representamos la imagen de color 

	cvNamedWindow("gray", 1 ); 
	cvShowImage("gray", gray ); // Representamos la imagen de intensidad

	cvThreshold(src, colorThresh, threshold, maxValue, thresholdType); // Binarizamos la imagen de color 
	cvNamedWindow("colorThresh", 1 );
	cvShowImage("colorThresh", colorThresh ); // Representamos la imagen de color binarizada 

	cvThreshold(gray, grayThresh, threshold, maxValue, thresholdType); // Binarizamos la imagen en escala de grises 
	cvNamedWindow("grayThresh", 1 );
	cvShowImage("grayThresh", grayThresh ); // Representamosla imagen de intensidad binarizada 
	
	fin = clock();
	cout << "\t\tTiempo transcurrido en binarizar: " 
		<< (fin-inicio)/(double)CLOCKS_PER_SEC << " segundos\n\n" << endl;

	cvWaitKey(0); // Pulsamos una tecla para terminar 
	
	// Destruimos las ventanas y eliminamos las imagenes
	cvDestroyAllWindows();
	cvReleaseImage( &src );
	cvReleaseImage( &colorThresh );
	cvReleaseImage( &gray );
	cvReleaseImage( &grayThresh );
}

/*
 * Calcula y muestra por pantalla el histograma de la imagen usando para ello la GPU
 * 
 * Nota: Solo utiliza la GPU para realizar los calculos necesarios para hallar los niveles de color,
 * por lo que la GPU NO se utiliza para dibujar la imagen y mostrarla
 */
void histogramaParalelo()
{
	clock_t inicio, fin;
	inicio = clock();

	// Se cargar imagen 
	Mat src = imread(PATH_IMAGEN, 1 );
	if(!src.data) return;

	// Se separa las imagenes en los 3 colores ( R,G,B )
	vector<Mat> bgr_planes;
	split( src, bgr_planes );

	// Se guardan en variables separadas de tipo GpuMat que carga el contenido en la GPU
	GpuMat b_src(bgr_planes[0]);
	GpuMat g_src(bgr_planes[1]);
	GpuMat r_src(bgr_planes[2]);

	GpuMat b_hist_gpu, g_hist_gpu, r_hist_gpu, b1, b2, b3;
	cv::gpu::Stream s1 = Stream::Null();
	cv::gpu::Stream s2 = Stream::Null();
	cv::gpu::Stream s3 = Stream::Null();

	// Se calculan los histogramas
	cv::gpu::calcHist(b_src, b_hist_gpu, b1, s1);
	cv::gpu::calcHist(g_src, g_hist_gpu, b2, s2);
	cv::gpu::calcHist(r_src, r_hist_gpu, b3, s3);

	//Se vuelven a comvertir en Matrices normales tras calcular sus histogramas
	Mat b_hist_temp(b_hist_gpu);
	Mat g_hist_temp(g_hist_gpu);
	Mat r_hist_temp(r_hist_gpu);
	Mat b_hist, g_hist, r_hist;

	if (DEBUG) {
		// Salida de los histogramas
		cout << "B = "<< endl << " "  << b_hist_temp << endl << endl;
		cout << "G = "<< endl << " "  << g_hist_temp << endl << endl;
		cout << "R = "<< endl << " "  << r_hist_temp << endl << endl; 
	}

	//Se trasponen las matrices para que se pueda dibuhar el histograma correctamente
	cv::transpose(b_hist_temp, b_hist);
	cv::transpose(g_hist_temp, g_hist);
	cv::transpose(r_hist_temp, r_hist);

	if (DEBUG) {
		// Salida de las matrices traspuestas del histograma
		cout << "B(t) = "<< endl << " "  << b_hist << endl << endl;
		cout << "G(t) = "<< endl << " "  << g_hist << endl << endl;
		cout << "R(t) = "<< endl << " "  << r_hist << endl << endl; 
	}

	// Se dibuja el histograma 
	int histSize = 256;
	int hist_w = 512; 
	int hist_h = 400;
	int bin_w = cvRound( ((double) hist_w)/((double)histSize) );

	cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );
	
	cv::normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	cv::normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	cv::normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

	if (DEBUG) {
		// Salida de los histogramas normalizados
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

	// Se muestra el resultado
	cv::imshow("Resultado histograma", histImage );
	cv::imshow("Imagen", src );

	fin = clock();
	cout << "\t\tTiempo transcurrido en calcular el histograma: " 
		<< (fin-inicio)/(double)CLOCKS_PER_SEC << " segundos\n\n" << endl;

	cvWaitKey(0); // Pulsamos una tecla para terminar
	cvDestroyAllWindows(); // Destruimos las ventanas
	
	// Liberamos de la memoria las imagenes
	src.release(); 
	histImage.release();
}
/*
 * Muestra la info de la GPU y prueba el procesamiento de imagenes con la GPU y OpenCV
 * En caso de haber varias GPU utiliza la que el sistema use por defecto
 */
void info()
{
	DeviceInfo info = gpu::DeviceInfo();
	int minVersion = info.minorVersion();
	int maxVersion = info.majorVersion();
	size_t totalMemory = info.totalMemory();
	string name = info.name();
	int numProcs = info.multiProcessorCount();
	int id = info.deviceID();

	cout << "Informacion de la GPU:" << endl;
	cout << "\tNombre: " + name << endl;
	cout << "\tID: " + to_string(id) << endl;
	cout << "\tVersion minima: " + to_string(minVersion) << endl;
	cout << "\tVersion maxima: " + to_string(maxVersion) << endl;
	cout << "\tMemoria total: " + to_string(totalMemory/1024/1024) + " MB" << endl;
	cout << "\tNumero de procesadores: " + to_string(numProcs) << endl << endl;
	cout << "Prueba de tratamiento de imagen con la GPU y openCV:" << endl << endl;
	
	// Prueba
	Mat src = imread("C:/lena_std.tif", 0);
	gpu::GpuMat d_src(src);
	gpu::GpuMat d_dst;
	gpu::bilateralFilter(d_src, d_dst, -1, 50, 7);
	gpu::Canny(d_dst, d_dst, 50, 150, 3);
	Mat dst(d_dst);
	imshow("Original",src);
	imshow("Canny GPU",dst);

	cvWaitKey(0);
	cvDestroyAllWindows();
}
