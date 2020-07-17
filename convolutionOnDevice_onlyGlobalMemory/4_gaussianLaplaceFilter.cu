#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "opencv2/core/core.hpp"  
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include<stdlib.h>

using namespace cv;
using namespace std;

#define MASK_WIDTH 9
__global__ void Convolution_2D_globalMemory(unsigned char* imgInput, unsigned char* imgOutput, const float* mask, int height, int width, int channels) {

	int Row, Col, filterRow, filterCol;

	int rows = threadIdx.x + blockIdx.x * blockDim.x;
	int cols = threadIdx.y + blockIdx.y * blockDim.y;
	float sum = 0;

	Row = rows - MASK_WIDTH / 2;
	Col = cols - MASK_WIDTH / 2;
	for (int c = 0; c < channels; c++)
	{
		sum = 0;
		for (int i = 0; i < MASK_WIDTH; i++)
		{
			for (int j = 0; j < MASK_WIDTH; j++)
			{
				filterRow = Row + i;
				filterCol = Col + j;

				if ((filterRow >= 0) && (filterRow < height) && (filterCol >= 0) && (filterCol < width))
				{
					sum += imgInput[(filterRow * height + filterCol) * channels + c] * mask[i * MASK_WIDTH + j];
				}
				else { sum = 0; }
			}
		}
		imgOutput[(rows * width + cols) * channels + c] = (unsigned char)sum;
	}
}

__host__ int compute() {
	cudaDeviceProp propertise;
	cudaGetDeviceProperties(&propertise, 0);
	int blocksPerSM = propertise.maxThreadsPerMultiProcessor / propertise.maxThreadsPerBlock;
	int threadsPerSM = propertise.maxThreadsPerMultiProcessor;
	int dimension = 1;
	float result = 0;
	while (result <= 1.0) {
		dimension *= 2;
		result = (blocksPerSM*dimension*dimension) / threadsPerSM;
	}
	return dimension / 2;
}

int main() {
	Mat inputImage = imread("remastered-lena-512x512.tiff", CV_LOAD_IMAGE_COLOR);
	if (inputImage.empty())
	{
		printf("!!! Failed imread(): image not found\n");
		exit(1);
	}
	imshow("OriginalImage", inputImage);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	unsigned char* input = nullptr;
	unsigned char* output = nullptr;
	unsigned char* d_input = nullptr;
	unsigned char* d_output = nullptr;
	float* d_filter = nullptr;
	float filter[MASK_WIDTH * MASK_WIDTH] = {
		0,0,3,2,2,2,3,0,0,
		0,2,3,5,5,5,3,2,0,
		3,3,5,3,0,3,5,3,3,
		2,5,3,-12,-23,-12,3,5,2,
		2,5,0,-23,-40,-23,0,5,2,
		2,5,3,-12,-23,-12,3,5,2,
		3,3,5,3,0,3,5,3,3,
		0,2,3,5,5,5,3,2,0,
		0,0,3,2,2,2,3,0,0
	};

	int imgHeight = inputImage.rows;
	int imgWidth = inputImage.cols;
	int imgChannels = inputImage.channels();

	Mat outputImage(imgHeight, imgWidth, CV_8UC3);
	outputImage.data = inputImage.data;
	input = inputImage.data;

	int imageSize = sizeof(unsigned char) * imgHeight*imgWidth*imgChannels;
	output = (unsigned char*)malloc(imageSize);

	cudaMalloc((void**)&d_input, imageSize);
	cudaMalloc((void**)&d_output, imageSize);
	cudaMalloc((void**)&d_filter, MASK_WIDTH * MASK_WIDTH* sizeof(float));

	cudaMemcpy(d_input, input, imageSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_filter, filter, MASK_WIDTH * MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

	// compute optimeze kernel configuration
	int optimalFactor = compute();
	printf("optimal factor for your gpu is: %d \n", optimalFactor);
	dim3 DimBlock(optimalFactor, optimalFactor);
	dim3 DimGrid((imgWidth + DimBlock.x - 1) / DimBlock.x, (imgHeight + DimBlock.y - 1) / DimBlock.y);

	//kernel launch
	cudaEventRecord(start, 0);
	Convolution_2D_globalMemory << <DimGrid, DimBlock >> > (d_input, d_output, d_filter, imgHeight, imgWidth, imgChannels);
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);

	cudaMemcpy(outputImage.data, d_output, imageSize, cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);

	float miliSeconds;
	cudaEventElapsedTime(&miliSeconds, start, stop);
	printf("kernel execution time: %f ms", miliSeconds);
	imshow("GaussianConvolution", outputImage);
	waitKey(0);
	//clean up
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_filter);

	return 0;


}