#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "opencv2/core/core.hpp"  
#include "opencv2/highgui/highgui.hpp"   
#include <stdio.h>
using namespace std;
using namespace cv;


#define MASK_WIDTH 3
#define BLOCKSIZE 32
#define TILE_SIZE 16
#define TILE_WIDTH 16

const int FILTER_AREA = (2 * MASK_WIDTH + 1) * (2 * MASK_WIDTH + 1);
const int BLOCK_WIDTH = TILE_WIDTH + 2 * MASK_WIDTH;
__device__ float getSqrtf(float f2)
{
	return sqrtf(f2);
}
__device__ unsigned char clamp(int value1, int value2) {
	if (value1 < 0 && value2 < 0) {
		value1 = 0;
		value2 = 0;
	}
	else   if (value1 > 255 && value2 > 255) {
		value1 = 255;
		value2 = 255;
	}
	return  getSqrtf((value1 * value1) + (value2 * value2));
}

__global__ void sobel_globalMemory_constantCache_sharedMemory(unsigned char* imgInput, char* __restrict__ Mask, char* __restrict__ Mask1, unsigned char* imgOutput, int width, int height) {
	__shared__ float N_ds[TILE_SIZE + MASK_WIDTH - 1][TILE_SIZE + MASK_WIDTH - 1];
	int n = MASK_WIDTH / 2;
	int dest = threadIdx.y * TILE_SIZE + threadIdx.x, destY = dest / (TILE_SIZE + MASK_WIDTH - 1), destX = dest % (TILE_SIZE + MASK_WIDTH - 1),
		srcY = blockIdx.y * TILE_SIZE + destY - n, srcX = blockIdx.x * TILE_SIZE + destX - n,
		src = (srcY * width + srcX);
	if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
		N_ds[destY][destX] = imgInput[src];
	else
		N_ds[destY][destX] = 0;

	// Second batch loading
	dest = threadIdx.y * TILE_SIZE + threadIdx.x + TILE_SIZE * TILE_SIZE;
	destY = dest / (TILE_SIZE + MASK_WIDTH - 1), destX = dest % (TILE_SIZE + MASK_WIDTH - 1);
	srcY = blockIdx.y * TILE_SIZE + destY - n;
	srcX = blockIdx.x * TILE_SIZE + destX - n;
	src = (srcY * width + srcX);
	if (destY < TILE_SIZE + MASK_WIDTH - 1) {
		if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
			N_ds[destY][destX] = imgInput[src];
		else
			N_ds[destY][destX] = 0;
	}
	__syncthreads();

	int accum = 0;
	int accum1 = 0;
	int y, x;
	for (y = 0; y < MASK_WIDTH; y++)
		for (x = 0; x < MASK_WIDTH; x++)
			accum += N_ds[threadIdx.y + y][threadIdx.x + x] * Mask[y * MASK_WIDTH + x];
	accum1 += N_ds[threadIdx.y + y][threadIdx.x + x] * Mask1[y * MASK_WIDTH + x];
	y = blockIdx.y * TILE_SIZE + threadIdx.y;
	x = blockIdx.x * TILE_SIZE + threadIdx.x;
	if (y < height && x < width)
		imgOutput[(y * width + x)] = clamp(accum, accum1);
	__syncthreads();
}


__global__ void grayScale(unsigned char* imgInput, unsigned char* imgOutput, int Row, int Col) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if ((row < Col) && (col < Row)) {
		imgOutput[row * Row + col] = imgInput[(row * Row + col) * 3 + 2] * 0.299 + imgInput[(row * Row + col) * 3 + 1] * 0.587 + imgInput[(row * Row + col) * 3] * 0.114;
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
		result = (blocksPerSM * dimension * dimension) / threadsPerSM;
	}
	return dimension / 2;
}
void convolution(Mat image, unsigned char* In, unsigned char* sobelImage, char* Gx_mask, char* Gy_mask, int Mask_Width, int Row, int Col, int op) {

	int imgSize = sizeof(unsigned char) * Row * Col * image.channels();
	int imgSize_gray = sizeof(unsigned char) * Row * Col;
	int maskSize = sizeof(char) * (MASK_WIDTH * MASK_WIDTH);
	unsigned char* d_imageInput = nullptr, *d_imageOutput = nullptr, *d_sobelOutput = nullptr;
	char* d_Gx_mask = nullptr;
	char* d_Gy_mask = nullptr;


	cudaMalloc((void**)&d_imageInput, imgSize);
	cudaMalloc((void**)&d_imageOutput, imgSize_gray);
	cudaMalloc((void**)&d_Gx_mask, maskSize);
	cudaMalloc((void**)&d_Gy_mask, maskSize);
	cudaMalloc((void**)&d_sobelOutput, imgSize_gray);

	cudaMemcpy(d_imageInput, In, imgSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Gx_mask, Gx_mask, maskSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Gy_mask, Gy_mask, maskSize, cudaMemcpyHostToDevice);

	int optimalFactor = compute();
	printf("optimal factor for your gpu is: %d \n", optimalFactor);
	dim3 DimBlock(optimalFactor, optimalFactor);
	dim3 DimGrid((Row + DimBlock.x - 1) / DimBlock.x, (Col + DimBlock.y - 1) / DimBlock.y);


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	grayScale << <DimBlock, DimGrid >> > (d_imageInput, d_imageOutput, Row, Col);
	cudaDeviceSynchronize();

	sobel_globalMemory_constantCache_sharedMemory << <DimBlock, DimGrid >> > (d_imageOutput, d_Gx_mask, d_Gy_mask, d_sobelOutput, Row, Col);
	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float miliSecods;
	cudaEventElapsedTime(&miliSecods, start, stop);
	printf("kernel execution time: %f s", miliSecods);
	cudaMemcpy(sobelImage, d_sobelOutput, imgSize_gray, cudaMemcpyDeviceToHost);
	cudaFree(d_imageInput);
	cudaFree(d_imageOutput);
	cudaFree(d_Gx_mask);
	cudaFree(d_Gy_mask);
	cudaFree(d_sobelOutput);
}

int main() {
	Mat inputImage, result_image;
	inputImage = imread("remastered-lena-512x512.tiff", CV_LOAD_IMAGE_COLOR);
	if (inputImage.empty())
	{
		printf("!!! Failed imread(): image not found\n");
		exit(1);
	}
	imshow("originalImage", inputImage);
	char Gx_mask[] = { -1,0,1,-2,0,2,-1,0,1 };
	char Gy_mask[] = { 1,2,1,0,0,0,-1,-2,-1 };

	int imgHeight = inputImage.rows;
	int imgWidth = inputImage.cols;
	int imgChannels = inputImage.channels();
	unsigned char* In = (unsigned char*)malloc(sizeof(unsigned char) * imgHeight * imgWidth * imgChannels);
	unsigned char* sobelOutput = (unsigned char*)malloc(sizeof(unsigned char) * imgHeight * imgWidth);

	In = inputImage.data;

	convolution(inputImage, In, sobelOutput, Gx_mask, Gy_mask, MASK_WIDTH, imgHeight, imgWidth, 3);

	result_image.create(imgWidth, imgHeight, CV_8UC1);
	result_image.data = sobelOutput;

	imshow("sobelFilter", result_image);
	waitKey(0);
	return 0;
}