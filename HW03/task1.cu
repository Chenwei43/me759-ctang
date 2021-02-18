// computes the factorial of integers from 1 to 8
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cuda.h>

__global__ void factorial(int* b, int* a) {
    b[threadIdx.x]=1;
    for (unsigned int i = 0; i < a[threadIdx.x]; i++) {
	    b[threadIdx.x] *= (i+1);
    }
    std::printf("%d!=%d \n", a[threadIdx.x], b[threadIdx.x]);
}

int main() {
    const int numThreads = 8;
    const int lenArray = 8;
    int a_host[lenArray] = { 1,2,3,4,5,6,7,8 };

    // cuda arrays
    int *a, *b;
    cudaMalloc((void**)&a, sizeof(int) * lenArray);
    cudaMalloc((void**)&b, sizeof(int) * lenArray);
    cudaMemcpy(a, &a_host, sizeof(int) * lenArray, cudaMemcpyHostToDevice);
    cudaMemset(b, 0, sizeof(int) * lenArray);

    // factorial calc
    factorial<<<1, numThreads>>>(b, a);

    cudaDeviceSynchronize();

    cudaFree(a);
    cudaFree(b);
    return 0;
}
