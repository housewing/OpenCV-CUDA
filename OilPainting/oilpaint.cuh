#ifndef APPLY_OILPAINT_CUH
#define APPLY_OILPAINT_CUH

#include<iostream>
#include<cstdio>
#include<cuda_runtime.h>

#include <opencv2\opencv.hpp>

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n", msg, file_name, line_number, cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

__global__ void applyOilPaintKernel(uchar3* const _input, uchar3* const _output, const int width, const int height, const int r, const int levels)
{
	int CountIntensity[20];
	int RedAverage[20];
	int GreenAverage[20];
	int BlueAverage[20];

	for (int i = 0; i < levels; i++)
	{
		CountIntensity[i] = 0;
		RedAverage[i] = 0;
		GreenAverage[i] = 0;
		BlueAverage[i] = 0;
	}

	//2D Index of current thread
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	//Only valid threads perform memory I/O
	if ((x < width) && (y < height))
	{
		for (int i = -r; i <= r; ++i)
		{
			int crtY = y + i; //clamp the neighbor pixel, prevent overflow
			if (crtY >= 0)
			{
				for (int j = -r; j <= r; ++j)
				{
					int crtX = x + j;
					if (crtX >= 0)
					{
						const uchar3 curPix = _input[crtY * width + crtX];
						int nCurIntensity = (((curPix.x + curPix.y + curPix.z) / 3.0) * levels) / 255.0;

						int tmp = nCurIntensity;
						CountIntensity[tmp]++;

						BlueAverage[tmp] = BlueAverage[tmp] + curPix.x;
						GreenAverage[tmp] = GreenAverage[tmp] + curPix.y;
						RedAverage[tmp] = RedAverage[tmp] + curPix.z;
					}
				}
			}
		}

		int nCurMax = -1;
		int nMaxIndex = -1;
		for (int nI = 0; nI < levels; nI++)
		{
			if (CountIntensity[nI] > nCurMax)
			{
				nCurMax = CountIntensity[nI];
				nMaxIndex = nI;
			}
		}

		if (nCurMax == 0) { nCurMax = 1; }
		double tmpB = BlueAverage[nMaxIndex] / nCurMax;
		double tmpG = GreenAverage[nMaxIndex] / nCurMax;
		double tmpR = RedAverage[nMaxIndex] / nCurMax;

		uchar3 color;
		color.x = tmpB;
		color.y = tmpG;
		color.z = tmpR;

		_output[y * width + x] = color;
	}
}

void applyOilPaintCuda(cv::Mat& _src, cv::Mat& _dst, int _radius, int _levels)
{
	const size_t numRows = _src.rows;
	const size_t numCols = _src.cols;
	const size_t numElemts = numRows * numCols;

	uchar3 *gpu_src, *gpu_dst;
	SAFE_CALL(cudaMalloc<uchar3>(&gpu_src, numElemts * sizeof(uchar3)), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<uchar3>(&gpu_dst, numElemts * sizeof(uchar3)), "CUDA Malloc Failed");

	SAFE_CALL(cudaMemcpy(gpu_src, _src.ptr<uchar3>(), numElemts * sizeof(uchar3), cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	//SAFE_CALL(cudaMemcpy(gpu_dst, _dst.ptr<uchar3>(), numElemts * sizeof(uchar3), cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	//// specify a reasonable grid and block sizes
	//const dim3 block(16, 16);
	//// calculate grid size to cover the whole image
	//const dim3 grid((numCols + block.x - 1) / block.x, (numRows + block.y - 1) / block.y);

	const dim3 block(16, 16, 1);
	const dim3 grid(ceil((float)numCols / block.x), ceil((float)numRows / block.y), 1);
	
	double t = (double)cv::getTickCount();
	applyOilPaintKernel << <grid, block >> >(gpu_src, gpu_dst, numCols, numRows, 5, 20);
	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	std::cout << "Times passed in seconds: " << t << std::endl;

	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	SAFE_CALL(cudaMemcpy(_dst.ptr(), gpu_dst, numElemts * sizeof(uchar3), cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	SAFE_CALL(cudaFree(gpu_src), "CUDA Free Failed");
	SAFE_CALL(cudaFree(gpu_dst), "CUDA Free Failed");
}

#endif