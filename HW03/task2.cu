// compute a * x + y
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda.h>
#include <iostream>
#include <random>
#include <cstdio>

__global__ void mulAdd(int* dA, int a, int Nthreads) {
    dA[blockIdx.x * Nthreads + threadIdx.x] = blockIdx.x + a*threadIdx.x;    
}

int main() {
    const int numThreads = 8;
    const int numBlocks = 2;
    const int lenArray = 16;
    int hA[lenArray];

    // cuda arrays
    int* dA;
    cudaMalloc((void**)&dA, sizeof(int) * lenArray);  
    cudaMemset(dA, 0, sizeof(int) * lenArray);

    int my_seed = 137;
    std::mt19937 generator(my_seed);
    const int min = -100, max = 100;
    std::uniform_int_distribution<int> dist(min, max);
    int a = dist(generator);
    
    mulAdd<<<numBlocks, numThreads>>>(dA, a, numThreads);

    cudaMemcpy(&hA, dA, sizeof(int) * lenArray, cudaMemcpyDeviceToHost);
    for (unsigned int i = 0; i < lenArray; i++){
        std::cout << hA[i] << " ";
    }
    std::cout << "\n";

    cudaFree(dA);
    return 0;
}
