#ifndef APPLY_BITWISE_CUH
#define APPLY_BITWISE_CUH

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

__global__ void applyBitwiseKernel(unsigned char* const _input, unsigned char* const _mask, unsigned char* const _output, const int numRows, const int numCols)
{
	const long pointIndex = threadIdx.x + blockDim.x*blockIdx.x;

	if (pointIndex < numRows*numCols) { // this is necessary only if too many threads are started
		uchar const inputPoint = _input[pointIndex];
		uchar const maskPoint = _mask[pointIndex];

		_output[pointIndex] = inputPoint & maskPoint;
	}
}

void applyBitwiseCuda(cv::Mat& _input, cv::Mat& _mask, cv::Mat& _output)
{
	const size_t numRows = _input.rows;
	const size_t numCols = _input.cols;
	const size_t numElemts = numRows * numCols;

	uchar *gpu_input, *gpu_mask, *gpu_output;
	SAFE_CALL(cudaMalloc<uchar>(&gpu_input, numElemts * sizeof(uchar)), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<uchar>(&gpu_mask, numElemts * sizeof(uchar)), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<uchar>(&gpu_output, numElemts * sizeof(uchar)), "CUDA Malloc Failed");


	SAFE_CALL(cudaMemcpy(gpu_input, _input.ptr<uchar>(), numElemts * sizeof(uchar), cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(gpu_mask, _mask.ptr<uchar>(), numElemts * sizeof(uchar), cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	const int blockThreadSize = 512;
	const int numberOfBlocks = 1 + ((numRows*numCols - 1) / blockThreadSize); // a/b rounded up
	const dim3 blockSize(blockThreadSize, 1, 1);
	const dim3 gridSize(numberOfBlocks, 1, 1);

	double t = (double)cv::getTickCount();
	applyBitwiseKernel << <gridSize, blockSize >> >(gpu_input, gpu_mask, gpu_output, numRows, numCols);
	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	std::cout << "Times(applyBitwise) passed in seconds: " << t << std::endl;

	SAFE_CALL(cudaMemcpy(_output.ptr(), gpu_output, numElemts * sizeof(uchar), cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	SAFE_CALL(cudaFree(gpu_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(gpu_mask), "CUDA Free Failed");
	SAFE_CALL(cudaFree(gpu_output), "CUDA Free Failed");
}

#endif