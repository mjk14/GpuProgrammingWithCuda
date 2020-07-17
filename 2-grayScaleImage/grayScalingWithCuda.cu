#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

__global__ void greyConvertor(unsigned char* output, uchar3 const* input, const uint height, const uint width) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {
		int grayOffset = y*width + x;
		unsigned char r = input[grayOffset].x;
		unsigned char g = input[grayOffset].y;
		unsigned char b = input[grayOffset].z;
		output[grayOffset] = 0.21f*r + 0.72f*g + 0.07f*b;
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
	Mat inputImage = imread("lena512color.tiff", CV_LOAD_IMAGE_COLOR);
	const uint h = inputImage.rows;//height
	const uint w = inputImage.cols;//width
	if (inputImage.empty())
	{
		printf("!!! Failed imread(): image not found\n");
		exit(1);
	}
	Mat outputImage(h, w, CV_8UC1);
	//device variables
	uchar3* d_inputImage = nullptr;
	unsigned char* d_outputImage = nullptr;
	//size definition
	const int size_i = sizeof(uchar3)*h*w;
	const int size_o = sizeof(unsigned char)*h*w;
	//device memory allocation
	cudaMalloc((void **)&d_inputImage, size_i);
	cudaMalloc((void **)&d_outputImage, size_o);
	//copy inputImage to d_inputImage (copy data form host to device)
	cudaMemcpy(d_inputImage, inputImage.data, size_i, cudaMemcpyHostToDevice);
	//compute optimal kernel configuration
	int optimalFactor = compute();
	printf("optimal factor for your gpu is: %d \n", optimalFactor);
	dim3 DimBlock(optimalFactor, optimalFactor);
	dim3 DimGrid((w + DimBlock.x - 1) / DimBlock.x, (h + DimBlock.y - 1) / DimBlock.y);
	//kernel lauch
	greyConvertor << <DimGrid, DimBlock >> >(d_outputImage, d_inputImage, h, w);
	cudaDeviceSynchronize();//for sureing form terminate execution time of kernel
							//copy d_outputImage in outputImage.data (copy data from device to host)
	cudaMemcpy(outputImage.data, d_outputImage, size_o, cudaMemcpyDeviceToHost);
	//display images
	imshow("grey scale image", outputImage);
	imshow("original image", inputImage);
	waitKey(0);
	//cleanup
	cudaFree(d_inputImage);d_inputImage = nullptr;
	cudaFree(d_outputImage);d_outputImage = nullptr;
	cudaDeviceReset();// use of Nsight
	return 0;
}