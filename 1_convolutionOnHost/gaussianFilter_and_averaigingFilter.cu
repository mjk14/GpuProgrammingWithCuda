#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include<cmath>
#include <chrono>
using namespace cv;

void CPU_2D_Convolution_gaussian(unsigned char* inImg, unsigned char* outImg, float* filter, int numRows, int numCols, int numChans, int filterWidth) {
	float sum;
	int cornerRow, cornerCol;
	int filterRow, filterCol;

	for (int row = 0; row < numRows; row++) {
		for (int col = 0; col < numCols; col++) {

			cornerRow = row - filterWidth / 2;
			cornerCol = col - filterWidth / 2;

			// loop through the channels
			for (int c = 0; c < numChans; c++) {
				// reset accumulator
				sum = 0;

				// accumulate values inside filter
				for (int i = 0; i < filterWidth; i++) {
					for (int j = 0; j < filterWidth; j++) {
						// compute pixel coordinates inside filter
						filterRow = cornerRow + i;
						filterCol = cornerCol + j;

						// make sure we are within image boundaries
						if ((filterRow >= 0) && (filterRow <= numRows) && (filterCol >= 0) && (filterCol <= numCols)) {
							sum += inImg[(filterRow*numCols + filterCol)*numChans + c] * filter[i*filterWidth + j];
						}
					}
				}
				outImg[(row*numCols + col)*numChans + c] = (unsigned char)sum;
			}
		}
	}
}

void CPU_2D_Convolution_averaging(unsigned char* inImg, unsigned char* outImg, float* filter, int numRows, int numCols, int numChans, int filterWidth) {
	float sum;
	int cornerRow, cornerCol;
	int filterRow, filterCol;

	for (int row = 0; row < numRows; row++) {
		for (int col = 0; col < numCols; col++) {

			cornerRow = row - filterWidth / 2;
			cornerCol = col - filterWidth / 2;

			// loop through the channels
			for (int c = 0; c < numChans; c++) {
				// reset accumulator
				sum = 0;

				// accumulate values inside filter
				for (int i = 0; i < filterWidth; i++) {
					for (int j = 0; j < filterWidth; j++) {
						// compute pixel coordinates inside filter
						filterRow = cornerRow + i;
						filterCol = cornerCol + j;

						// make sure we are within image boundaries
						if ((filterRow >= 0) && (filterRow <= numRows) && (filterCol >= 0) && (filterCol <= numCols)) {
							sum += inImg[(filterRow*numCols + filterCol)*numChans + c] * filter[i*filterWidth + j];
						}
					}
				}
				sum /= filterWidth*filterWidth;
				outImg[(row*numCols + col)*numChans + c] = (unsigned char)sum;
			}
		}
	}
}
void Sobel_calculation(unsigned char* Gx, unsigned char* Gy, unsigned char* G, unsigned char* theta, int height, int width) {
	for (int i = 0; i < height*width; i++) {
		G[i] = (unsigned char)sqrt(pow(float(Gx[i]), 2) + pow(float(Gy[i]), 2));
		theta[i] = (unsigned char)atan2f(Gy[i], Gx[i]);
	}
}

int main() {
	//load image
	Mat inputImage = imread("remastered-lena-512x512.tiff", CV_LOAD_IMAGE_COLOR);
	if (inputImage.empty())
	{
		printf("!!! Failed imread(): image not found\n");
		exit(1);
	}
	int imgchannels = inputImage.channels();
	int imgWidth = inputImage.cols;
	int imgHeight = inputImage.rows;

	unsigned char* input = inputImage.data;
	unsigned char* output = nullptr;

	int imageSize = sizeof(unsigned char)*imgHeight*imgWidth*imgchannels;
	output = (unsigned char*)malloc(imageSize);

	std::cout << "List of filters:\n_________________\n(1)Gausssian filter\n(2)Averagin\n_________________\nselect the filter:";
	int in;
	std::cin >> in;

	switch (in)
	{
		//apply gaussian filter
	case 1: {
		float filter[25] = {
			0.0126f, 0.0252f, 0.0314f, 0.0252f, 0.0126f,
			0.0252f, 0.0566f, 0.0755f, 0.0566f, 0.0252f,
			0.0314f, 0.0755f, 0.0943f, 0.0755f, 0.0314f,
			0.0252f,0.0566f, 0.0755f, 0.0566f, 0.0252f,
			0.0126f, 0.0252f, 0.0314f, 0.0252f, 0.0126f,
		};
		auto start = std::chrono::high_resolution_clock::now();
		CPU_2D_Convolution_gaussian(input, output, filter, imgHeight, imgWidth, imgchannels, 5);
		auto end = std::chrono::high_resolution_clock::now();
		double time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
		time_taken *= 1e-9;
		printf("Time taken by program is : %f sec\n", time_taken);
		Mat outputImage(imgHeight, imgWidth, CV_8UC3, output);
		//display images
		imshow("GaussianFilter", outputImage);
		imshow("original image", inputImage);
		waitKey(0);
		break;
	}
			//apply averaging filter
	case 2: {
		float filter[9] = {
			1,1,1,
			1,1,1,
			1,1,1
		};
		auto start = std::chrono::high_resolution_clock::now();
		CPU_2D_Convolution_averaging(input, output, filter, imgHeight, imgWidth, imgchannels, 3);
		auto end = std::chrono::high_resolution_clock::now();
		double time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
		time_taken *= 1e-9;
		printf("Time taken by program is : %f sec\n", time_taken);
		Mat outputImage(imgHeight, imgWidth, CV_8UC3, output);
		//display images
		imshow("GaussianFilter", outputImage);
		imshow("original image", inputImage);
		waitKey(0);
		break;
	}
	}
	
	
	return 0;
}