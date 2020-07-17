#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

int main() {
	int numberOfDevices;
	cudaGetDeviceCount(&numberOfDevices);
	printf("number of GPUs : %d", numberOfDevices);
	for (int i = 0; i<numberOfDevices; i++)
	{
		cudaDeviceProp propertise;
		cudaGetDeviceProperties(&propertise, i);
		printf("\n-------------------------------\n");
		printf("Device Number : %d\n", i);
		printf("Device name : %s\n", propertise.name);
		printf("Maximum number of threads per block : %d\n", propertise.maxThreadsPerBlock);
		printf("number of SMs  : %d\n", propertise.multiProcessorCount);
		printf("Maximum number of threads per SM : %d \n", propertise.maxThreadsPerMultiProcessor);
		printf("Shared memory available per block : %lu bytes\n", propertise.sharedMemPerBlock);
		printf("Shared memory available per SM : %zu bytes\n", propertise.sharedMemPerMultiprocessor);
		printf("Clock Frequency per processor : %.0f MHz (%0.2f GHz)\n", propertise.clockRate * 1e-3f, propertise.clockRate * 1e-6f);
		printf("Peak memory clock frequency : %.0f Mhz\n", propertise.memoryClockRate * 1e-3f);
		printf("Global memory bus width  : %d-bit\n", propertise.memoryBusWidth);
		printf("Maximum size of each dimension of a block (x,y,z): (%d, %d, %d) \n", propertise.maxThreadsDim[0], propertise.maxThreadsDim[1], propertise.maxThreadsDim[2]);
		printf("Maximum size of each dimension of a grid  (x,y,z): (%d, %d, %d)\n", propertise.maxGridSize[0], propertise.maxGridSize[1], propertise.maxGridSize[2]);
		printf("Warp size in threads  : %d\n", propertise.warpSize);
		printf("Global memory available on device :  %.0f MBytes (%llu bytes)\n",
			(float)propertise.totalGlobalMem / 1048576.0f, (unsigned long long) propertise.totalGlobalMem);
		printf("Constant memory available on device : %lu bytes\n", propertise.totalConstMem);
																							  
		printf("32-bit registers available per block  : %d\n", propertise.regsPerBlock);
		printf("32-bit registers available per multiprocessor : %d\n", propertise.regsPerMultiprocessor);
		printf("Compute capability (major) : %d ,(minor) :%d\n", propertise.major, propertise.minor);
	}
	return 0;
}
