#include "stdafx.h"

// Includes OpenCV
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
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
using namespace cv::gpu;

// Declaraciones de las funciones
void binarizacionParalelo(int p);
void histogramaParalelo();
void info();

const char* PATH_IMAGEN = "C:/lena_std.tif"; // Constante global que indica la ruta de la imagen
const bool DEBUG = false; // Constante que se usa para depurar el programa y mostrarr info extra por la salida estandar

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

/*
 * Calcula y muestra por pantalla la imagen binarizada y la original
 *
 * Nota: la variable threshold indica a partir de que nivel se va a determinar si es blanco o negro
 *			Si es mayor que ese numero -> 255 = Negro
 *			Si es menor que ese numero ->   0 = Blanco
 */
void binarizacionParalelo(int p)
{
	clock_t inicio, fin, inicioBin, finBin; // Variables para calcular el tiempo
	inicio = clock();

	GpuMat gray; // Contendrá la imagen convertida en escala de grises 
	GpuMat grayThresh; // Imagen binaria conseguida a partir de la imagen en escala de grises 

	int threshold; // Definimos el valor umbral
	if(p > 0 && p <255)
		threshold = p; 
	else
		threshold = 160; 
	int maxValue = 255; // Definimos el valor máximo 
	int thresholdType = CV_THRESH_BINARY; // Definimos el tipo de binarización 

	// Se carga la imagen
	Mat src = imread(PATH_IMAGEN, CV_LOAD_IMAGE_UNCHANGED);
	Mat srcGray = imread(PATH_IMAGEN, CV_LOAD_IMAGE_GRAYSCALE);

	// Se copia a la GPU
	gray.upload(srcGray);

	 // Representamos la imagen de color 
	cvNamedWindow("Imagen a color original", 1 ); 
	imshow("Imagen a color original", src );

	inicioBin = clock();

	// Binarizamos la imagen en escala de grises
	gpu::threshold(gray, grayThresh, threshold, maxValue, thresholdType);

	// Copiamos la imagen binarizada desde la GPU a la CPU
	Mat grayBin(grayThresh);

	// Representamosla imagen de intensidad binarizada 
	cvNamedWindow("Imagen binarizada", 1 );
	imshow("Imagen binarizada", grayBin );

	finBin = clock();
	cout << "\t\tTiempo transcurrido ESPECIFICAMENTE en la operacion de binarizacion de la imagen: " 
		<< (finBin-inicioBin)/(double)CLOCKS_PER_SEC << " segundos\n\n" << endl;
	
	fin = clock();
	cout << "\t\tTiempo TOTAL transcurrido en binarizar: " 
		<< (fin-inicio)/(double)CLOCKS_PER_SEC << " segundos\n\n" << endl;

	cvWaitKey(0); // Pulsamos una tecla para terminar 
	
	// Destruimos las ventanas y eliminamos las imagenes
	cvDestroyAllWindows();
	src.release();
	srcGray.release();
	gray.release();
	grayThresh.release();
	grayBin.release();
}

/*
 * Calcula y muestra por pantalla el histograma de la imagen usando para ello la GPU
 * 
 * Nota: Solo utiliza la GPU para realizar los calculos necesarios para hallar los niveles de color,
 * por lo que la GPU NO se utiliza para dibujar la imagen y mostrarla
 */
void histogramaParalelo()
{
	clock_t inicio, fin, inicioCalcHist, finCalcHist; // Variables para calcular el tiempo
	inicio = clock();

	// Se cargar imagen 
	Mat src = imread(PATH_IMAGEN, 1 );
	if(!src.data) return;

	// Se separa las imagenes en los 3 colores (R,G,B)
	vector<Mat> bgr_planes;
	split( src, bgr_planes );

	// Se guardan en variables separadas de tipo GpuMat que carga el contenido en la GPU
	GpuMat b_src(bgr_planes[0]);
	GpuMat g_src(bgr_planes[1]);
	GpuMat r_src(bgr_planes[2]);

	// Se inicializan buffers y streams para realizar el calculo del histograma de manera optima (OPCIONAL)
	GpuMat b_hist_gpu, g_hist_gpu, r_hist_gpu, b1, b2, b3;
	gpu::Stream s1 = Stream::Null();
	gpu::Stream s2 = Stream::Null();
	gpu::Stream s3 = Stream::Null();

	inicioCalcHist = clock();

	// Se calculan los histogramas
	gpu::calcHist(b_src, b_hist_gpu, b1, s1);
	gpu::calcHist(g_src, g_hist_gpu, b2, s2);
	gpu::calcHist(r_src, r_hist_gpu, b3, s3);

	finCalcHist = clock();
	cout << "\t\tTiempo transcurrido ESPECIFICAMENTE en la operacion del calculo del histograma: " 
		<< (finCalcHist-inicioCalcHist)/(double)CLOCKS_PER_SEC << " segundos\n\n" << endl;

	//Se vuelven a comvertir en Matrices normales tras calcular sus histogramas
	Mat b_hist_temp(b_hist_gpu);
	Mat g_hist_temp(g_hist_gpu);
	Mat r_hist_temp(r_hist_gpu);
	Mat b_hist, g_hist, r_hist;

	if (DEBUG) {
		// Salida de los histogramas
		cout << "B = "<< endl << " "  << b_hist_temp << endl << endl;
		cout << b_hist_temp.cols << " " << b_hist_temp.rows << " " << b_hist_temp.size << " " << b_hist_temp.channels() << " " << b_hist_temp.dims << endl;
		cout << "G = "<< endl << " "  << g_hist_temp << endl << endl;
		cout << "R = "<< endl << " "  << r_hist_temp << endl << endl; 
	}

	//Se trasponen las matrices para que se pueda dibujar el histograma correctamente
	cv::transpose(b_hist_temp, b_hist);
	cv::transpose(g_hist_temp, g_hist);
	cv::transpose(r_hist_temp, r_hist);

	if (DEBUG) {
		// Salida de las matrices traspuestas del histograma
		cout << "B(t) = "<< endl << " "  << b_hist << endl << endl;
		cout << b_hist.cols << " " << b_hist.rows << " " << b_hist.size << " " << b_hist.channels() << " " << b_hist.dims << endl;
		cout << "G(t) = "<< endl << " "  << g_hist << endl << endl;
		cout << "R(t) = "<< endl << " "  << r_hist << endl << endl; 
	}

	// Se definen variables para la ventana que mostrara el histograma
	int histSize = 256;
	int hist_w = 512; 
	int hist_h = 400;
	int bin_w = cvRound( ((double) hist_w)/((double)histSize) );

	// Se crea la imagen base
	cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );
	
	// Se normalizan los valores de los histogramas
	cv::normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	cv::normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	cv::normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

	if (DEBUG) {
		// Salida de los histogramas normalizados
		cout << "B(norm) = "<< endl << " "  << b_hist << endl << endl;
		cout << "G(norm) = "<< endl << " "  << g_hist << endl << endl;
		cout << "R(norm) = "<< endl << " "  << r_hist << endl << endl; 
	}
	
	// Se dibuja una linea en funcion del histograma para cada canal de RGB 
	for(int i = 1; i < histSize; i++ )
	{
		// Color azul
		cv::line( histImage, 
			Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ),
			Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ), 
			Scalar( 255, 0, 0), 2,8, 0 );
		// Color verde
		cv::line( histImage, 
			Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ),
			Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
			Scalar( 0, 255, 0), 2, 8,0 );
		// Color rojo
		cv::line( histImage, 
			Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ),
			Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
			Scalar( 0, 0, 255), 2, 8, 0 );
	}

	//Se muestra el histograma dibujado junto a la imagen original
	imshow("Resultado histograma", histImage );
	imshow("Imagen", src );

	fin = clock();
	cout << "\t\tTiempo TOTAL transcurrido en calcular y mostrar el histograma: " 
		<< (fin-inicio)/(double)CLOCKS_PER_SEC << " segundos\n\n" << endl;

	cvWaitKey(0); // Pulsamos una tecla para terminar
	
	// Destruimos las ventanas y liberamos de la memoria las imagenes
	cvDestroyAllWindows();
	src.release(); 
	histImage.release();
}

/*
 * Muestra la info de la GPU y prueba el procesamiento de imagenes con la GPU y OpenCV
 * En caso de haber varias GPU utiliza la que el sistema use por defecto
 */
void info()
{
	cout << endl; gpu::printCudaDeviceInfo(0); cout << endl;
	cout << "\tEl hardware grafico es compatible con el modulo gpu de OpenCV: ";
	if(gpu::DeviceInfo().isCompatible()) cout << "SI" << endl; else cout << "NO" << endl;
	
	// Prueba para ver el funcionamiento de las librerias y el uso de la GPU poniendo un filtro a una imagen con la GPU
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
