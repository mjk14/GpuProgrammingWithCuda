#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#define BLUR_SIZE 1
using namespace cv;

__global__ void Blurrig(unsigned char* output, unsigned char* input, int height, int width) {
	int Col = threadIdx.x + blockIdx.x * blockDim.x;
	int Row = threadIdx.y + blockIdx.y * blockDim.y;

	if (Col < width && Row < height) {
		int pixVal = 0;
		int pixels = 0;
		for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow)
		{
			for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol)
			{
				int curRow = Row + blurRow;
				int curCol = Col + blurCol;
				//verify we have a valid image pixel
				if (curRow > -1 && curRow<height && curCol>-1 && curCol < width) {
					pixVal += input[curRow * width + curCol];
					pixels++;//keep track of number of pixels in the avg
				}
			}
		}
		//write our new pixel value
		output[Row * width + Col] = (unsigned char)(pixVal / pixels);
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
	//load image
	Mat inputImage = imread("remastered-lena-512x512.tiff", IMREAD_GRAYSCALE);

	const uint h = inputImage.rows;//height
	const uint w = inputImage.cols;//width
	if (inputImage.empty())
	{
		printf("!!! Failed imread(): image not found\n");
		exit(1);
	}
	Mat outputImage(h, w, CV_8UC1);
	//device variables
	unsigned char* d_inputImage = nullptr;
	unsigned char* d_outputImage = nullptr;
	//size definition
	const int size = sizeof(unsigned char) * h * w;
	//device memory allocation
	cudaMalloc((void**)&d_inputImage, size);
	cudaMalloc((void**)&d_outputImage, size);
	//copy inputImage to d_inputImage (copy data form host to device)
	cudaMemcpy(d_inputImage, inputImage.data, size, cudaMemcpyHostToDevice);
	//compute optimeze kernel configuration
	int optimalFactor = compute();
	printf("optimal factor for your gpu is: %d \n", optimalFactor);
	dim3 DimBlock(optimalFactor, optimalFactor);
	dim3 DimGrid((w + DimBlock.x - 1) / DimBlock.x, (h + DimBlock.y - 1) / DimBlock.y);
	//kernel lauch
	Blurrig << <DimGrid, DimBlock >> > (d_outputImage, d_inputImage, h, w);
	cudaDeviceSynchronize();//for sureing form terminate execution time of kernel
	//copy d_outputImage in outputImage.data (copy data from device to host)
	cudaMemcpy(outputImage.data, d_outputImage, size, cudaMemcpyDeviceToHost);
	//display images
	imshow("grey scale image", outputImage);
	imshow("original image", inputImage);
	waitKey(0);
	//cleanup
	cudaFree(d_inputImage); d_inputImage = nullptr;
	cudaFree(d_outputImage); d_outputImage = nullptr;
	cudaDeviceReset();// use of Nsight
	return 0;
}