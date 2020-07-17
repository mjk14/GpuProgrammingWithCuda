#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#define WIDTH 100
#define TILE_WIDTH 32

__global__ void mult2Matrix(float *M, float *N, float *P) {
	__shared__ int shared_m_tile[TILE_WIDTH][TILE_WIDTH];
	__shared__ int shared_n_tile[TILE_WIDTH][TILE_WIDTH];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	//check if thread directly maps to the dimensions of the resulting matrix
	if (row < WIDTH && col < WIDTH)
	{
		float result = 0;
		int k;
		int phase;
		//calculate P matrix indexes in phases. Each phase shares 
		//TILE_SIZE * TILE_SIZE data copied to the shared matrix M 
		//and matrix N.
		for (phase = 0; phase <= WIDTH / TILE_WIDTH; phase++)
		{
			shared_m_tile[ty][tx] = M[row * WIDTH + phase * TILE_WIDTH + tx];
			shared_n_tile[ty][tx] = N[(phase * TILE_WIDTH + ty) * WIDTH + col];
			__syncthreads();

			for (k = 0; k < TILE_WIDTH; k++)
			{
				if (k + (phase * TILE_WIDTH) < WIDTH)
				{
					result += (shared_m_tile[ty][k] * shared_n_tile[k][tx]);
				}
			}
			__syncthreads();
		}
		P[row * WIDTH + col] = result;
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
		result = blocksPerSM*dimension*dimension / threadsPerSM;
	}
	dimension /= 2;
	size_t sharedMemPerBlock = propertise.sharedMemPerBlock / 1024;
	size_t temp_sharedMemoryPerBlock = (2 * 4 * dimension*dimension) / 1024;
	while (temp_sharedMemoryPerBlock >= sharedMemPerBlock)
	{
		dimension--;
		temp_sharedMemoryPerBlock = 2 * 4 * dimension*dimension;
	}
	printf("amout shared memory used per Block: %.2fkB and size of TILE_WIDTH: %d\n", (float)(2 * 4 * dimension*dimension / 1024), dimension);
	return dimension;
}
__host__ void random_floats(float* a, int length)
{
	for (int i = 0; i<length; i++)
		a[i] = (float)(rand() % 9);
}
///////////////////////
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
	//compute optimeze kernel configuration
	compute();
	dim3 DimBlock(TILE_WIDTH, TILE_WIDTH);
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

