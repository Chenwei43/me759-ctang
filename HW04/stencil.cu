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

	int maskSize = 2 * R + 1;
	int imageIdxL = blockIdx.x * blockDim.x;
    int imageIdxR =(blockIdx.x+1)  * blockDim.x;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int outputIndex = blockDim.x + maskSize + threadIdx.x;
    if (index < n){

        // sharedArray contains
        // - The entire mask
        // - The elements of image needed to compute the elements of output corresponding to the threads in the given block
        // - The output image elements corresponding to the given block before it is written back to global memory
        extern __shared__ float sharedArray[];
        
        // load tiles from global mem to shared mem
        sharedArray[threadIdx.x] = image[index];

        if (threadIdx.x < maskSize) {
            sharedArray[blockDim.x + threadIdx.x] = mask[threadIdx.x];
        }

        sharedArray[blockDim.x + maskSize + threadIdx.x] = 0.0;					
        __syncthreads();

        // conv
        for (int k = 0; k < 2*R+1; k++) {
            int imageIndex = index - R + k;   
            if (imageIndex >= imageIdxL && imageIndex < imageIdxR) {
                sharedArray[outputIndex] += sharedArray[threadIdx.x + k - R] * sharedArray[blockDim.x + k];
            }
            else
            {
                sharedArray[outputIndex] += sharedArray[blockDim.x + k];
            }              
            
        }
        __syncthreads();            
                
        // move output to global mem
        output[index] = sharedArray[outputIndex];
    }
	
}


__host__ void stencil(const float* image, const float* mask, float* output, unsigned int n, unsigned int R, unsigned int threads_per_block) {

	int maskSize = 2 * R + 1;
	unsigned int numBlocks = (n + threads_per_block - 1) / threads_per_block;
	stencil_kernel<<<numBlocks, threads_per_block, (threads_per_block * 2 + maskSize) * sizeof(float)>>>(image, mask, output, n, R);
	cudaDeviceSynchronize();
}