#include <stdio.h>
#include <stdlib.h>
#define WIDTH 100
void random_floats(float* a, int length)
{
	for (int i = 0; i<length; i++)
		a[i] = (float)(rand() % 9);
}
int main() {
	float *M = nullptr;
	float *N = nullptr;
	float *P = nullptr;
	int size = WIDTH*WIDTH*sizeof(float);
	M = (float*)malloc(size); random_floats(M, WIDTH*WIDTH);
	N = (float*)malloc(size); random_floats(N, WIDTH*WIDTH);
	P = (float*)malloc(size);
	for (int i = 0; i<WIDTH; i++) {
		for (int j = 0; j<WIDTH; j++) {
			float Pvalue = 0;
			for (int k = 0; k < WIDTH; k++) {
				Pvalue += M[i*WIDTH + k] * N[k*WIDTH + j];
			}
			P[i*WIDTH + j] = Pvalue;
		}
	}
	for (int i = 0; i<WIDTH; i++) {
		for (int j = 0; j<WIDTH; j++) {
			printf("  %.2f", M[i*WIDTH + j]);
		}
		printf("\n");
	}
	printf("-------------------------------------\n");
	for (int i = 0; i<WIDTH; i++) {
		for (int j = 0; j<WIDTH; j++) {
			printf("  %.2f", N[i*WIDTH + j]);
		}
		printf("\n");
	}
	printf("-------------------------------------\n");
	for (int i = 0; i<WIDTH; i++) {
		for (int j = 0; j<WIDTH; j++) {
			printf("  %.2f", P[i*WIDTH + j]);
		}
		printf("\n");
	}
	printf("-------------------------------------\n");
	free(M); M = nullptr;
	free(N); N = nullptr;
	free(P); P = nullptr;
	return 0;
}
