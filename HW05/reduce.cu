#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


// implements the 'first add during global load' version (Kernel 4) for the parallel reduction
// g_idata is the array to be reduced, and is available on the device.
// g_odata is the array that the reduced results will be written to, and is available on the device.
// expects a 1D configuration.
// uses only dynamically allocated shared memory.
__global__ void reduce_kernel(float* g_idata, float* g_odata, unsigned int n) {
    extern __shared__ float sharedArray[];


    // bring two entries a time
    unsigned int idx_pair = blockIdx.x * (blockDim.x*2) + threadIdx.x;

    sharedArray[threadIdx.x] = g_idata[idx_pair] + g_idata[idx_pair + blockDim.x];
    __syncthreads();

    // 
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedArray[threadIdx.x] += sharedArray[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        g_odata[blockIdx.x] = sharedArray[0];
    }
    
    
}

// the sum of all elements in the *input array should be written to the first element of the *input array.
// calls reduce_kernel repeatedly if needed. _No part_ of the sum should be computed on host.
// *input is an array of length N in device memory.
// *output is an array of length = (number of blocks needed for the first call of the reduce_kernel) in device memory.
// configures the kernel calls using threads_per_block threads per block.
// the function should end in a call to cudaDeviceSynchronize for timing purposes
__host__ void reduce(float** input, float** output, unsigned int N, unsigned int threads_per_block) {
    unsigned int n = N;
    //std::printf("%f, %f \n", (*input)[0], (*output)[0]);

    
    while(n > 1) {
        unsigned int numBlocks=1;
        if (threads_per_block < n){
            numBlocks = (n + threads_per_block - 1) / (2 * threads_per_block);

        }	
        
        reduce_kernel<<<numBlocks, threads_per_block, threads_per_block * sizeof(float)>>>(*input, *output, n);
        n = numBlocks;

        // overwrite output as new input
        float* temp;
        temp = *output;
        *output = *input;
        *input = temp;

    }
    
    cudaDeviceSynchronize();

}