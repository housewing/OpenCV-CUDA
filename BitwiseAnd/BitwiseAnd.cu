#include <opencv2\opencv.hpp>
#include <opencv2\cudaimgproc.hpp>
#include <iostream>

#include "bitwise.cuh"

using namespace cv;

int main(int argc, char** argv)
{
	Mat mask = imread("data/1.png", 0);
	Mat input = imread("data/white.png", 0);
	Mat output = Mat(mask.size(), CV_8UC1);

	Mat maskClone = imread("data/1.png", 0);
	Mat inputClone = imread("data/white.png", 0);
	Mat outputClone;

	double t = (double)getTickCount();
	bitwise_and(inputClone, maskClone, outputClone);
	t = ((double)getTickCount() - t) / getTickFrequency();
	std::cout << "OpenCV CPU Times passed in seconds: " << t << std::endl;

	t = (double)getTickCount();
	applyBitwiseCuda(input, mask, output);
	t = ((double)getTickCount() - t) / getTickFrequency();
	std::cout << "Full Own GPU Times passed in seconds: " << t << std::endl;

	

	imshow("mask", mask);
	imshow("input", input);
	imshow("out", output);

	waitKey();
	return 0;
}