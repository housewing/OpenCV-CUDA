#include <opencv2\opencv.hpp>
#include <iostream>

#include "oilpaint.cuh"

using namespace cv;

int main(int argc, char** argv)
{
	Mat src = imread("data/04.png");
	if (src.empty())
	{
		return -1;
	}

	Mat dst = Mat::zeros(src.size(), src.type());

	double t = (double)getTickCount();
	applyOilPaintCuda(src, dst, 5, 20);
	t = ((double)getTickCount() - t) / getTickFrequency();
	std::cout << "Full Times passed in seconds: " << t << std::endl;

	imshow("src", src);
	imshow("dst", dst);

	waitKey();
	return 0;
}