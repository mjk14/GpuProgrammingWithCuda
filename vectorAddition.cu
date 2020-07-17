#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<chrono> 
#define SIZE 10000000
#define Ts_per_B 192

void random_ints(int* a,int N)
{
    for(int i=0;i<N;i++)
      a[i]=rand()%200;
}
__global__ void add(int *a, int *b, int *c) {
    int index=threadIdx.x+blockIdx.x*blockDim.x;
    if(index<SIZE)
    {
        c[index] = a[index] + b[index];
    }
}
int main() {
    auto start = std::chrono::high_resolution_clock::now();
int *a, *b, *c;// host copies of variables a, b & c
int *d_a, *d_b, *d_c; // device copies of a, b, c
int size = SIZE * sizeof(int);

// Alloc space for device
cudaMalloc((void **)&d_a,size);
cudaMalloc((void **)&d_b,size);
cudaMalloc((void **)&d_c,size);

//allocate memory to arrays and initalize elemets of a & b
a=(int*)malloc(size);random_ints(a,SIZE);
b=(int*)malloc(size);random_ints(b,SIZE);
c=(int*)malloc(size);


// Copy inputs to device
cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

// Launch add() kernel on GPU with SIZE blocks
add<<<(SIZE+Ts_per_B-1)/Ts_per_B,Ts_per_B>>>(d_a, d_b, d_c);

// Copy result back to host
cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

printf("tested by majid_kakavandi!\n");


// Cleanup
free(a); free(b); free(c);
cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

auto end = std::chrono::high_resolution_clock::now();
double time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
time_taken *= 1e-9;
printf("Time taken by program is : %f sec\n", time_taken);
return 0;
}


