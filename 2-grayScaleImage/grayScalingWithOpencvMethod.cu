
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>

using namespace cv;
int main() {
	Mat image = imread("lena512color.tiff", CV_LOAD_IMAGE_GRAYSCALE);
	if (image.empty())
	{
		printf("!!! Failed imread(): image not found\n");
		exit(-1);
	}
	namedWindow("Display window", CV_WINDOW_AUTOSIZE);// Create a window for display
	imshow("Display window", image);
	waitKey(0);
	return 0;
}