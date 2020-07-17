#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <chrono>

using namespace cv;

/** @function main */
int main()
{
	// Declare variables
	Mat src, GaussianImage, AveragingImage, GreyImage, SobelImage, GaussianLaplaceImage;

	Point anchor;
	double delta;
	int ddepth;
	int kernel_size;
	int scale;
	int c;
	// Load an image
	src = imread("remastered-lena-512x512.tiff");
	if (!src.data)
	{
		return -1;
	}
	imshow("originalImage", src);
	// Create window
	// Initialize arguments for the filter
	anchor = Point(-1, 1);
	delta = 0;
	ddepth = 0;
	scale = 1;
	// Loop - Will filter the image with different kernel sizes each 0.5 seconds
	int ind = 0;
	
	// Gaussian Filter
	Mat gaussian(5, 5, CV_32F);
	gaussian.at<float>(0, 0) = 2.0f;
	gaussian.at<float>(1, 0) = 4.0f;
	gaussian.at<float>(2, 0) = 5.0f;
	gaussian.at<float>(3, 0) = 4.0f;
	gaussian.at<float>(4, 0) = 2.0f;

	gaussian.at<float>(0, 1) = 4.0f;
	gaussian.at<float>(1, 1) = 9.0f;
	gaussian.at<float>(2, 1) = 12.0f;
	gaussian.at<float>(3, 1) = 9.0f;
	gaussian.at<float>(4, 1) = 4.0f;
		
	gaussian.at<float>(0, 2) = 5.0f;
	gaussian.at<float>(1, 2) = 12.0f;
	gaussian.at<float>(2, 2) = 15.0f;
	gaussian.at<float>(3, 2) = 12.0f;
	gaussian.at<float>(4, 2) = 5.0f;

	gaussian.at<float>(0, 3) = 4.0f;
	gaussian.at<float>(1, 3) = 9.0f;
	gaussian.at<float>(2, 3) = 12.0f;
	gaussian.at<float>(3, 3) = 9.0f;
	gaussian.at<float>(4, 3) = 4.0f;

	gaussian.at<float>(0, 4) = 2.0f;
	gaussian.at<float>(1, 4) = 4.0f;
	gaussian.at<float>(2, 4) = 5.0f;
	gaussian.at<float>(3, 4) = 4.0f;
	gaussian.at<float>(4, 4) = 2.0f;

	auto gaussina_Start = std::chrono::high_resolution_clock::now();

	filter2D(src, GaussianImage, ddepth, gaussian / 159, anchor, delta, BORDER_DEFAULT);

	auto gaussina_End= std::chrono::high_resolution_clock::now();
	double time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(gaussina_End - gaussina_Start).count();
	time_taken *= 1e-9;
	printf("Time taken by program is(GaussianFilter): %f sec\n", time_taken);
	imshow("GaussianFilter", GaussianImage);
	//Averaging Filter
	Mat averaging(3, 3, CV_32F);;
	averaging.at<float>(0, 0) = 1.0f;
	averaging.at<float>(0, 1) = 1.0f;
	averaging.at<float>(0, 2) = 1.0f;
	averaging.at<float>(1, 0) = 1.0f;
	averaging.at<float>(1, 1) = 1.0f;
	averaging.at<float>(1, 2) = 1.0f;
	averaging.at<float>(2, 0) = 1.0f;
	averaging.at<float>(2, 1) = 1.0f;
	averaging.at<float>(2, 2) = 1.0f;
		
	auto averaging_Start = std::chrono::high_resolution_clock::now();

	filter2D(src, AveragingImage, ddepth, averaging / 9, anchor, delta, BORDER_DEFAULT);
	auto averaging_End = std::chrono::high_resolution_clock::now();
	double time_taken1= std::chrono::duration_cast<std::chrono::nanoseconds>(averaging_End - averaging_Start).count();
	time_taken1*= 1e-9;
	printf("Time taken by program is(AveragingFilter): %f sec\n", time_taken1);
	imshow("AveragingFilter", AveragingImage);

	//Sobel Filter
	auto sobel_Start = std::chrono::high_resolution_clock::now();
	cvtColor(GaussianImage, GreyImage, COLOR_RGB2GRAY);
	Mat Gx, Gy;
	Mat abs_Gx, abs_Gy;
	Sobel(GreyImage, Gx, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(Gx, abs_Gx);
	Sobel(GreyImage, Gy, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(Gy, abs_Gy);
	addWeighted(Gx, 0.5, Gy, 0.5, 0, SobelImage);
	auto sobel_End = std::chrono::high_resolution_clock::now();
	double time_taken2= std::chrono::duration_cast<std::chrono::nanoseconds>(sobel_End - sobel_Start).count();
	time_taken2*= 1e-9;
	printf("Time taken by program is(SobelFilter): %f sec\n", time_taken2);
	imshow("SobelFilter", SobelImage);

	//Gaussian Laplace Filter
	Mat gaussianLaplace(9, 9, CV_32F);
	gaussianLaplace.at<float>(0, 0) = 0.0f;
	gaussianLaplace.at<float>(1, 0) = 0.0f;
	gaussianLaplace.at<float>(2, 0) = 3.0f;
	gaussianLaplace.at<float>(3, 0) = 2.0f;
	gaussianLaplace.at<float>(4, 0) = 2.0f;
	gaussianLaplace.at<float>(5, 0) = 2.0f;
	gaussianLaplace.at<float>(6, 0) = 3.0f;
	gaussianLaplace.at<float>(7, 0) = 0.0f;
	gaussianLaplace.at<float>(8, 0) = 0.0f;

	gaussianLaplace.at<float>(0, 1) = 0.0f;
	gaussianLaplace.at<float>(1, 1) = 2.0f;
	gaussianLaplace.at<float>(2, 1) = 3.0f;
	gaussianLaplace.at<float>(3, 1) = 5.0f;
	gaussianLaplace.at<float>(4, 1) = 5.0f;
	gaussianLaplace.at<float>(5, 1) = 5.0f;
	gaussianLaplace.at<float>(6, 1) = 3.0f;
	gaussianLaplace.at<float>(7, 1) = 2.0f;
	gaussianLaplace.at<float>(8, 1) = 0.0f;

	gaussianLaplace.at<float>(0, 2) = 3.0f;
	gaussianLaplace.at<float>(1, 2) = 3.0f;
	gaussianLaplace.at<float>(2, 2) = 5.0f;
	gaussianLaplace.at<float>(3, 2) = 3.0f;
	gaussianLaplace.at<float>(4, 2) = 0.0f;
	gaussianLaplace.at<float>(5, 2) = 3.0f;
	gaussianLaplace.at<float>(6, 2) = 5.0f;
	gaussianLaplace.at<float>(7, 2) = 3.0f;
	gaussianLaplace.at<float>(8, 2) = 3.0f;

	gaussianLaplace.at<float>(0, 3) = 2.0f;
	gaussianLaplace.at<float>(1, 3) = 5.0f;
	gaussianLaplace.at<float>(2, 3) = 3.0f;
	gaussianLaplace.at<float>(3, 3) = -12.0f;
	gaussianLaplace.at<float>(4, 3) = -23.0f;
	gaussianLaplace.at<float>(5, 3) = -12.0f;
	gaussianLaplace.at<float>(6, 3) = 3.0f;
	gaussianLaplace.at<float>(7, 3) = 5.0f;
	gaussianLaplace.at<float>(8, 3) = 2.0f;

	gaussianLaplace.at<float>(0, 4) = 2.0f;
	gaussianLaplace.at<float>(1, 4) = 5.0f;
	gaussianLaplace.at<float>(2, 4) = 0.0f;
	gaussianLaplace.at<float>(3, 4) = -23.0f;
	gaussianLaplace.at<float>(4, 4) = -40.0f;
	gaussianLaplace.at<float>(5, 4) = -23.0f;
	gaussianLaplace.at<float>(6, 4) = 0.0f;
	gaussianLaplace.at<float>(7, 4) = 5.0f;
	gaussianLaplace.at<float>(8, 4) = 2.0f;

	gaussianLaplace.at<float>(0, 5) = 2.0f;
	gaussianLaplace.at<float>(1, 5) = 5.0f;
	gaussianLaplace.at<float>(2, 5) = 3.0f;
	gaussianLaplace.at<float>(3, 5) = -12.0f;
	gaussianLaplace.at<float>(4, 5) = -23.0f;
	gaussianLaplace.at<float>(5, 5) = -12.0f;
	gaussianLaplace.at<float>(6, 5) = 3.0f;
	gaussianLaplace.at<float>(7, 5) = 5.0f;
	gaussianLaplace.at<float>(8, 5) = 2.0f;

	gaussianLaplace.at<float>(0, 6) = 3.0f;
	gaussianLaplace.at<float>(1, 6) = 3.0f;
	gaussianLaplace.at<float>(2, 6) = 5.0f;
	gaussianLaplace.at<float>(3, 6) = 3.0f;
	gaussianLaplace.at<float>(4, 6) = 0.0f;
	gaussianLaplace.at<float>(5, 6) = 3.0f;
	gaussianLaplace.at<float>(6, 6) = 5.0f;
	gaussianLaplace.at<float>(7, 6) = 3.0f;
	gaussianLaplace.at<float>(8, 6) = 3.0f;

	gaussianLaplace.at<float>(0, 7) = 0.0f;
	gaussianLaplace.at<float>(1, 7) = 2.0f;
	gaussianLaplace.at<float>(2, 7) = 3.0f;
	gaussianLaplace.at<float>(3, 7) = 5.0f;
	gaussianLaplace.at<float>(4, 7) = 5.0f;
	gaussianLaplace.at<float>(5, 7) = 5.0f;
	gaussianLaplace.at<float>(6, 7) = 3.0f;
	gaussianLaplace.at<float>(7, 7) = 2.0f;
	gaussianLaplace.at<float>(8, 7) = 0.0f;

	gaussianLaplace.at<float>(0, 8) = 0.0f;
	gaussianLaplace.at<float>(1, 8) = 0.0f;
	gaussianLaplace.at<float>(2, 8) = 3.0f;
	gaussianLaplace.at<float>(3, 8) = 2.0f;
	gaussianLaplace.at<float>(4, 8) = 2.0f;
	gaussianLaplace.at<float>(5, 8) = 2.0f;
	gaussianLaplace.at<float>(6, 8) = 3.0f;
	gaussianLaplace.at<float>(7, 8) = 0.0f;
	gaussianLaplace.at<float>(8, 8) = 0.0f;

	auto gaussianLaplace_Start = std::chrono::high_resolution_clock::now();

	filter2D(src, GaussianLaplaceImage, ddepth, gaussianLaplace, anchor, delta, BORDER_DEFAULT);
	auto gaussianLaplace_End = std::chrono::high_resolution_clock::now();
	double time_taken3 = std::chrono::duration_cast<std::chrono::nanoseconds>(gaussianLaplace_End - gaussianLaplace_Start).count();
	time_taken3 *= 1e-9;
	printf("Time taken by program is(GaussianLaplaceFilter): %f sec\n", time_taken3);
	imshow("GaussianLaplaceFilter", GaussianLaplaceImage);
	waitKey(0);
	return 0;
}