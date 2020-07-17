#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#define BLUR_SIZE 1
using namespace cv;

int main() {
	//load image
	Mat inputImage = imread("cameraman.tiff", CV_LOAD_IMAGE_COLOR);
	unsigned char* input = inputImage.data;
	const uint h = inputImage.rows;//height
	const uint w = inputImage.cols;//width
	if (inputImage.empty())
	{
		printf("!!! Failed imread(): image not found\n");
		exit(1);
	}
	Mat outputImage(h, w, CV_8UC1);
	unsigned char* output = nullptr;
	//size definition
	const int size = sizeof(unsigned char)*h*w;
	output = (unsigned char*)malloc(size);
	int pixVal, pixels;
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			pixVal = 0;
			pixels = 0;
			for (int k = -BLUR_SIZE; k < BLUR_SIZE + 1; k++) {
				for (int t = -BLUR_SIZE; t < BLUR_SIZE + 1; t++) {
					int curRow = i + k;
					int curCol = j + t;
					if (curRow>-1 && curRow <w &&curCol>-1 && curCol<h) {
						pixVal += input[curRow*w + curCol];
						pixels++;
					}
				}
			}
			output[i*w + j] = (unsigned char)(pixVal / pixels);
		}
	}
	outputImage.data = output;
	//display images
	imshow("blur image", outputImage);
	imshow("original image", inputImage);
	waitKey(0);
	cudaDeviceReset();// use of Nsight
	return 0;
}