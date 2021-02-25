#include <cstdio>
#include <cuda.h>
#include "cuda_runtime.h"
#include "matmul.cuh"

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n)
{
    float product = 0.0;
    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    int tx = index / n;
    int ty = index - tx * n;
    if (index < n * n) {
        for (int i = 0; i < n; i++) {
            float a = A[tx * n + i];
            float b = B[i * n + ty];
            product += a * b;
        }
        C[index] = product;
        //std::printf("idx = %d, c[i] = %f\n", index, C[index]);
    }

}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block) {

    unsigned int numBlocks = (n*n + threads_per_block - 1) / threads_per_block;
    matmul_kernel<<<numBlocks, threads_per_block >>> (A, B, C, n);
    cudaDeviceSynchronize();
}