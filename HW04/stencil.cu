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
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int index_padded = threadIdx.x + R;
	int outputIndex = blockDim.x + 2*R + maskSize + threadIdx.x;
	extern __shared__ float sharedArray[];

	if (index < n) {
		// sharedArray contains
		// - The entire mask
		// - The elements of image needed: KEY!! it's 2R + num_threads
		// - The output image elements corresponding to the given block before it is written back to global memory
		

		// load tiles from global mem to shared mem
		sharedArray[index_padded] = image[index];
		if (threadIdx.x < R) {
			if (int(index - R) < 0) {
				sharedArray[index_padded - R] = 1;
			}
			else
			{
				sharedArray[index_padded - R] = image[index - R];
			}
			if (index + blockDim.x > n - 1) {
				sharedArray[index_padded + blockDim.x] = 1;
			}
			else {
				sharedArray[index_padded + blockDim.x] = image[index + blockDim.x];
			}

		}

		if (threadIdx.x < maskSize) {
			sharedArray[blockDim.x + 2 * R + threadIdx.x] = mask[threadIdx.x];
		}

		sharedArray[outputIndex] = 0.0;
		__syncthreads();

		// conv		
		for (int k = -R; k < int(R+1); k++) {
			sharedArray[outputIndex] += sharedArray[index_padded + k] * sharedArray[blockDim.x + 3 * R + k];
		}
		//__syncthreads();


		// move output to global mem
		output[index] = sharedArray[outputIndex];
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

	// Shared mem: R-threads-R of image (padding of Radius needed at start and end), mask and output
	stencil_kernel<<<numBlocks, threads_per_block, (threads_per_block*2 + 2*R +maskSize) * sizeof(float)>>>(image, mask, output, n, R);
	cudaDeviceSynchronize();
}