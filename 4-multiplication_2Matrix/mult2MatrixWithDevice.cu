#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define WIDTH 100

__global__ void mult2Matrix(float *M, float *N, float *P) {
	// Calculate the row index of the P element and M
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	// Calculate the column index of P and N
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	if ((Row < WIDTH) && (Col < WIDTH)) {
		float Pvalue = 0;
		// each thread computes one element of the block sub-matrix
		for (int k = 0; k < WIDTH; ++k) {
			Pvalue += M[Row*WIDTH + k] * N[k*WIDTH + Col];
		}
		P[Row*WIDTH + Col] = Pvalue;
	}
}
__host__ int compute() {
	cudaDeviceProp propertise;
	cudaGetDeviceProperties(&propertise, 0);
	int blocksPerSM = propertise.maxThreadsPerMultiProcessor / propertise.maxThreadsPerBlock;//maxblocksPerSM
	int threadsPerSM = propertise.maxThreadsPerMultiProcessor;//threadsPerSM
	int dimension = 1;
	float result = 0;
	while (result <= 1.0) {
		dimension *= 2;
		result = (blocksPerSM*dimension*dimension) / threadsPerSM;
	}
	return dimension/2;
}

__host__ void random_floats(float* a, int length)
{
	for (int i = 0; i<length; i++)
		a[i] = (float)(rand() % 9);
}
int main() {
	//host variables
	float *M = nullptr;
	float *N = nullptr;
	float *P = nullptr;
	//device variables
	float *d_M = nullptr;
	float *d_N = nullptr;
	float *d_P = nullptr;
	int size = WIDTH*WIDTH*sizeof(float);
	// Alloc space for device
	cudaMalloc((void **)&d_M, size);
	cudaMalloc((void **)&d_N, size);
	cudaMalloc((void **)&d_P, size);
	// Aloc space for host
	M = (float*)malloc(size); random_floats(M, WIDTH*WIDTH);
	N = (float*)malloc(size); random_floats(N, WIDTH*WIDTH);
	P = (float*)malloc(size);
	// Copy inputs to device
	cudaMemcpy(d_M, M, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_N, N, size, cudaMemcpyHostToDevice);
	//compute optimal kernel configuration
	int optimalFactor = compute();
	printf("optimalFactor: %d\n", optimalFactor);
	dim3 DimBlock(optimalFactor, optimalFactor);
	dim3 DimGrid((WIDTH + DimBlock.x - 1) / DimBlock.x, (WIDTH + DimBlock.y - 1) / DimBlock.y);
	mult2Matrix << <DimGrid, DimBlock >> >(d_M, d_N, d_P);
	cudaDeviceSynchronize();
	// Copy result back to host
	cudaMemcpy(P, d_P, size, cudaMemcpyDeviceToHost);
	//display output of matrix
	for (int i = 0; i<WIDTH; i++) {
		for (int j = 0; j<WIDTH; j++) {
			printf("   %.2f", M[i*WIDTH + j]);
		}
		printf("\n");
	}
	printf("-------------------------------------\n");
	for (int i = 0; i<WIDTH; i++) {
		for (int j = 0; j<WIDTH; j++) {
			printf("   %.2f", N[i*WIDTH + j]);
		}
		printf("\n");
	}
	printf("-------------------------------------\n");
	for (int i = 0; i<WIDTH; i++) {
		for (int j = 0; j<WIDTH; j++) {
			printf("   %.2f", P[i*WIDTH + j]);
		}
		printf("\n");
	}
	printf("-------------------------------------\n");
	//cleanup
	free(M); M = nullptr;
	free(N); N = nullptr;
	free(P); P = nullptr;
	cudaFree(d_M); d_M = nullptr;
	cudaFree(d_N); d_N = nullptr;
	cudaFree(d_P); d_P = nullptr;
	cudaDeviceReset();// use of Nsight
	return 0;
}

