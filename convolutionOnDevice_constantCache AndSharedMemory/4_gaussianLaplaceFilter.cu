#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "opencv2/core/core.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include <stdio.h>
#include <stdlib.h>
using namespace cv;
using namespace std;

#define MASK_WIDTH  9
#define TILE_WIDTH 32 
#define w (TILE_WIDTH + MASK_WIDTH - 1) 
__constant__ float d_filter[MASK_WIDTH * MASK_WIDTH];
__global__ void Convolution_2D_globalMemory_constantCache_sharedMemory(unsigned char* imgInput, unsigned char* imgOutput, int height, int width, int channels)
{
	__shared__ float N_ds[w][w];
	int k;
	for (k = 0; k < channels; k++)
	{
		int dest = threadIdx.y * TILE_WIDTH + threadIdx.x,
			destY = dest / w, destX = dest % w,
			srcY = blockIdx.y * TILE_WIDTH + destY - MASK_WIDTH / 2,
			srcX = blockIdx.x * TILE_WIDTH + destX - MASK_WIDTH / 2,
			src = (srcY * width + srcX) * channels + k;
		if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
			N_ds[destY][destX] = imgInput[src];
		else
			N_ds[destY][destX] = 0.0;

		for (int iter = 1; iter <= (w * w) / (TILE_WIDTH * TILE_WIDTH); iter++)
		{
			// Second batch loading
			dest = threadIdx.y * TILE_WIDTH + threadIdx.x + iter * (TILE_WIDTH * TILE_WIDTH);
			destY = dest / w, destX = dest % w;
			srcY = blockIdx.y * TILE_WIDTH + destY - MASK_WIDTH / 2;
			srcX = blockIdx.x * TILE_WIDTH + destX - MASK_WIDTH / 2;
			src = (srcY * width + srcX) * channels + k;
			if (destY < w && destX < w)
			{
				if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
					N_ds[destY][destX] = imgInput[src];
				else
					N_ds[destY][destX] = 0.0;
			}
		}
		__syncthreads();

		float sum = 0;
		int y, x;
		for (y = 0; y < MASK_WIDTH; y++)
			for (x = 0; x < MASK_WIDTH; x++)
				sum += N_ds[threadIdx.y + y][threadIdx.x + x] * d_filter[y * MASK_WIDTH + x];
		y = blockIdx.y * TILE_WIDTH + threadIdx.y;
		x = blockIdx.x * TILE_WIDTH + threadIdx.x;
		if (y < height && x < width)
			imgOutput[(y * width + x) * channels + k] = sum;
		__syncthreads();
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
	outputImage = inputImage;
	input = inputImage.data;

	int imageSize = sizeof(unsigned char) * imgHeight*imgWidth*imgChannels;
	output = (unsigned char*)malloc(imageSize);

	cudaMalloc((void**)&d_input, imageSize);
	cudaMalloc((void**)&d_output, imageSize);

	cudaMemcpy(d_input, input, imageSize, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_filter, filter, MASK_WIDTH * MASK_WIDTH * sizeof(float));

	// compute optimeze kernel configuration
	int optimalFactor = compute();
	printf("optimal factor for your gpu (TILE_WIDTH) is: %d \n", optimalFactor);
	dim3 DimBlock(optimalFactor, optimalFactor);
	dim3 DimGrid((imgWidth + DimBlock.x - 1) / DimBlock.x, (imgHeight + DimBlock.y - 1) / DimBlock.y);

	//kernel launch
	cudaEventRecord(start, 0);
	Convolution_2D_globalMemory_constantCache_sharedMemory << <DimGrid, DimBlock >> > (d_input, d_output, imgHeight, imgWidth, imgChannels);
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