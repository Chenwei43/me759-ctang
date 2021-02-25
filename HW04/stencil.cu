#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stencil.cuh"


__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R) {
	// Computes the convolution of image and mask, storing the result in output.
	// Each thread should compute _one_ element of output.
	// Shared memory should be allocated _dynamically_ only.
	//
	// image is an array of length n of managed memory.
	// mask is an array of length (2 * R + 1) of managed memory.
	// output is an array of length n of managed memory.
	//
	// Assumptions:
	// - 1D configuration
	// - blockDim.x >= 2 * R + 1
	//
	int maskSize = 2 * R + 1;
	int imageIdxL = blockIdx.x * blockDim.x - R;
	int imageIdxR = imageIdxL + n - 1;
	int imageStep = blockDim.x;

	int outputIdxL = blockIdx.x * blockDim.x;
	int outputStep = blockDim.x;

	// sharedArray contains
	// - The entire mask
	// - The elements of image needed to compute the elements of output corresponding to the threads in the given block
	// - The output image elements corresponding to the given block before it is written back to global memory
	extern __shared__ float sharedArray[];
	for (int i = imageIdxL, j = outputIdxL; i <= imageIdxR; i += imageStep, j+= outputStep) {
		// load tiles from global mem to shared mem
		int imageIdx = i + threadIdx.x;
		int outputIdx = j + threadIdx.x;
		if (outputIdx < n) {
			if (imageIdx >= 0 && imageIdx < (int)n) {
				sharedArray[threadIdx.x] = image[imageIdx];
			}
			else if (imageIdx < 0 || imageIdx >= (int)n) {
				sharedArray[threadIdx.x] = 0;
			}
			
			if (threadIdx.x < maskSize) {
				sharedArray[blockDim.x + threadIdx.x] = mask[threadIdx.x];
			}

			sharedArray[blockDim.x + maskSize + threadIdx.x] = 0.0;
					
			__syncthreads();

			// conv
			for (int k = 0; k < maskSize; k++) {
				sharedArray[blockDim.x + maskSize + threadIdx.x] += sharedArray[threadIdx.x + k] * sharedArray[blockDim.x + k];
			}
			__syncthreads();
			// move output to global mem
			output[outputIdx] = sharedArray[blockDim.x + maskSize + threadIdx.x];		
		}
	}			
}


__host__ void stencil(const float* image, const float* mask, float* output, unsigned int n, unsigned int R, unsigned int threads_per_block) {
	// Computes the convolution of image and mask, storing the result in output.
	// image is an array of length n of managed memory.
	// mask is an array of length (2 * R + 1) of managed memory.
	// output is an array of length n of managed memory.
	// Makes one call to stencil_kernel with threads_per_block threads per block.
	// The kernel call should be followed by a call to cudaDeviceSynchronize for timing purposes.
	//
	// Assumptions:
	// - threads_per_block >= 2 * R + 1
	int maskSize = 2 * R + 1;
	unsigned int numBlocks = (n + threads_per_block - 1) / threads_per_block;
	stencil_kernel<<<numBlocks, threads_per_block, (threads_per_block*2+maskSize) * sizeof(float)>>>(image, mask, output, n, R);
	cudaDeviceSynchronize();
}