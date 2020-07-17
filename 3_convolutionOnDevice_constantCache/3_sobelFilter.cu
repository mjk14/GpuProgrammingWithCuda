#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "opencv2/core/core.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
using namespace cv;
using namespace std;
__constant__ float mask[256];
__global__ void convolve(unsigned char* imgInput, int width, int height, int paddingX, int paddingY, int kWidth, int kHeight, unsigned int offset, unsigned char* imgOutput)
{
	// Calculate our pixel's location
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	float sum = 0.0;
	int   pWidth = kWidth / 2;
	int   pHeight = kHeight / 2;

	//Solo ejecuta validos pixeles
	if (x >= pWidth + paddingX && y >= pHeight + paddingY && x < (blockDim.x * gridDim.x) - pWidth - paddingX &&
		y < (blockDim.y * gridDim.y) - pHeight - paddingY)
	{
		for (int j = -pHeight; j <= pHeight; j++)
		{
			for (int i = -pWidth; i <= pWidth; i++)
			{
				// Sample the weight for this location
				int ki = (i + pWidth);
				int kj = (j + pHeight);
				float w = mask[(kj * kWidth) + ki + offset];


				sum += w * float(imgInput[((y + j) * width) + (x + i)]);
			}
		}
	}
	imgOutput[(y * width) + x] = (unsigned char)sum;
}

__global__ void pythagoras(unsigned char* Gx, unsigned char* Gy, unsigned char* G, unsigned char* theta)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	float af = float(Gx[idx]);
	float bf = float(Gy[idx]);

	G[idx] = (unsigned char)sqrtf(af * af + bf * bf);
	theta[idx] = (unsigned char)atan2f(af, bf)*63.994;

}
__global__ void greyConvertor(uchar3* const imgInput, unsigned char* const imgOutput, int imgheight, int imgwidth) {
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx < imgwidth && idy < imgheight)
	{
		uchar3 rgb_Val = imgInput[idy * imgwidth + idx];
		imgOutput[idy * imgwidth + idx] = 0.299f * rgb_Val.x + 0.587f * rgb_Val.y + 0.114f * rgb_Val.z;
	}
}

__host__ unsigned char* createImageBuffer(unsigned int bytes, unsigned char** devicePtr)
{
	unsigned char* ptr = NULL;
	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped);
	cudaHostGetDevicePointer(devicePtr, ptr, 0);
	return ptr;
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

	Mat inputImage = imread("remastered-lena-512x512.tiff");
	if (inputImage.empty())
	{
		printf("!!! Failed imread(): image not found\n");
		exit(1);
	}
	imshow("originalimage", inputImage);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	const float gaussianFilter[25] = {
		2.f / 159.f,  4.f / 159.f,  5.f / 159.f,  4.f / 159.f, 2.f / 159.f,
		4.f / 159.f,  9.f / 159.f, 12.f / 159.f,  9.f / 159.f, 4.f / 159.f,
		5.f / 159.f, 12.f / 159.f, 15.f / 159.f, 12.f / 159.f, 5.f / 159.f,
		4.f / 159.f,  9.f / 159.f, 12.f / 159.f,  9.f / 159.f, 4.f / 159.f,
		2.f / 159.f,  4.f / 159.f,  5.f / 159.f,  4.f / 159.f, 2.f / 159.f,
	};


	const float sobleFilter_Gx[9] = {
		-1.f, 0.f, 1.f,
		-2.f, 0.f, 2.f,
		-1.f, 0.f, 1.f,
	};

	const float sobelFilter_Gy[9] = {
		1.f, 2.f, 1.f,
		0.f, 0.f, 0.f,
		-1.f, -2.f, -1.f,
	};
	cudaMemcpyToSymbol(mask, gaussianFilter, sizeof(gaussianFilter), 0);//The fourth cudaMemcpySymbol's parameter indicates offset in constant chache
	cudaMemcpyToSymbol(mask, sobleFilter_Gx, sizeof(sobleFilter_Gx), sizeof(gaussianFilter));
	cudaMemcpyToSymbol(mask, sobelFilter_Gy, sizeof(sobelFilter_Gy), sizeof(gaussianFilter) + sizeof(sobleFilter_Gx));

	const unsigned int gaussinaMaskOffset = 0;
	const unsigned int sobel_Gx_maskOffset = sizeof(gaussianFilter) / sizeof(float);
	const unsigned int sobel_Gy_maskOffset = (sizeof(sobleFilter_Gx) / sizeof(float)) + sobel_Gx_maskOffset;


	int imgHeight = inputImage.rows;
	int imgWidth = inputImage.cols;
	int imgChannels = inputImage.channels();

	cudaEventRecord(start);//start time-----------------------

	unsigned char* sourceData = nullptr, *blurredData = nullptr, *sobel_G_Data = nullptr, *sobel_theta_Data = nullptr;
	Mat source(inputImage.size(), CV_8U, createImageBuffer(imgWidth * imgHeight, &sourceData));
	Mat blurred(inputImage.size(), CV_8U, createImageBuffer(imgWidth * imgHeight, &blurredData));
	Mat sobel_G(inputImage.size(), CV_8U, createImageBuffer(imgWidth * imgHeight, &sobel_G_Data));
	Mat sobel_theta(inputImage.size(), CV_8U, createImageBuffer(imgWidth * imgHeight, &sobel_theta_Data));


	uchar3* d_input = nullptr;
	cudaMalloc((void**)&d_input, imgWidth * imgHeight*sizeof(uchar3));
	cudaMemcpy(d_input, inputImage.data, imgWidth * imgHeight * sizeof(uchar3), cudaMemcpyHostToDevice);

	unsigned char* d_Gx = nullptr;
	unsigned char* d_Gy = nullptr;
	cudaMalloc(&d_Gx, imgWidth * imgHeight);
	cudaMalloc(&d_Gy, imgWidth * imgHeight);

	// compute optimeze kernel configuration
	int optimalFactor = compute();
	printf("optimal factor for your gpu is: %d \n", optimalFactor);
	dim3 DimBlock(optimalFactor, optimalFactor);
	dim3 DimGrid((imgWidth + DimBlock.x - 1) / DimBlock.x, (imgHeight + DimBlock.y - 1) / DimBlock.y);
	//--------------------------------------

	dim3 pBlocks(inputImage.size().width * inputImage.size().height / 256);
	dim3 pThreads(256, 1);

	greyConvertor << <DimGrid, DimBlock >> > (d_input, sourceData, imgWidth, imgHeight);
	convolve << <DimGrid, DimBlock >> > (sourceData, imgWidth, imgHeight, 0, 0, 5, 5, gaussinaMaskOffset, blurredData);

	// sobel gradient convolutions (x&y padding is now 2 because there is a border of 2 around a 5x5 gaussian filtered image)
	convolve << <DimGrid, DimBlock >> > (blurredData, imgWidth, imgHeight, 2, 2, 3, 3, sobel_Gx_maskOffset, d_Gx);
	convolve << <DimGrid, DimBlock >> > (blurredData, imgWidth, imgHeight, 2, 2, 3, 3, sobel_Gy_maskOffset, d_Gy);
	pythagoras << <pBlocks, pThreads >> > (d_Gx, d_Gy, sobel_G_Data, sobel_theta_Data);
	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float miliSeconds;
	cudaEventElapsedTime(&miliSeconds, start, stop);
	printf("kernel execution time: %f ms", miliSeconds);
	imshow("GreyImage", source);
	imshow("gaussianFilterImage", blurred);
	imshow("sobelFilter_G", sobel_G);
	imshow("sobelFilter-theta", sobel_theta);
	waitKey(0);

	//cleanup
	cudaFreeHost(source.data);
	cudaFreeHost(blurred.data);
	cudaFreeHost(sobel_G.data);
	cudaFreeHost(sobel_theta.data);
	cudaFree(d_input);
	cudaFree(d_Gx);
	cudaFree(d_Gy);
	return 0;

}
