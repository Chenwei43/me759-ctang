#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// NOTE that each test function below calls the template matmul_kernel<TYPE>;
// The template function must meet the following requirements.
//  - Computes the matrix product C = AB using the tiled method from Lecture 11
//  - A, B, and C are row-major representations of nxn matrices in managed memory
//  - n need not be a multiple of blockDim.x
//  - Expects 2D configuration as in the slides
//  - Uses only dynamically allocated shared memory
// Function Prototype:
// __global__ void matmul_kernel(const TYPE* A, const TYPE* B, TYPE* C, unsigned int n)


template <typename T>
__global__ void matmul_kernel(const T* A, const T* B, T* C, unsigned int n) {
	extern __shared__ float sharedArray[];

	// indices of entries in block
	int aBegin = n * blockDim.x * blockIdx.y;
	int aEnd = aBegin + n - 1;
	int aStep = blockDim.x;

	int bBegin = blockDim.y * blockIdx.x;
	int bStep = blockDim.y * n;

	float Csub = 0;
	
	// grab tiles As and Bs from global to shared mem
	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
		unsigned int index_shared = threadIdx.y * blockDim.x + threadIdx.x;
		sharedArray[index_shared] = A[a + n * threadIdx.y + threadIdx.x];
		sharedArray[blockDim.x* blockDim.x + index_shared] = B[b + n * threadIdx.y + threadIdx.x];
		__syncthreads();

		//mul
		for (int k = 0; k < blockDim.x; k++) {
			Csub += sharedArray[threadIdx.y * blockDim.x + k] * sharedArray[blockDim.x * blockDim.x + k * blockDim.x + threadIdx.x];
		}
		__syncthreads();
	}
	C[blockIdx.x*blockDim.x + blockIdx.y * blockDim.y*n + threadIdx.y*n + threadIdx.x] = Csub;

}

