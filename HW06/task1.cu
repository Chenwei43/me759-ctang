// matmul by cuBLAS
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include "mmul.h"

int main(int argc, char* argv[]) {
    unsigned int n = std::strtoul(argv[1], nullptr, 10);
    unsigned int n_tests = std::atoi(argv[2]);
    
    float* A, * B, * C;
    cudaMallocManaged(&A, sizeof(float) * n * n);
    cudaMallocManaged(&B, sizeof(float) * n * n);
    cudaMallocManaged(&C, sizeof(float) * n * n);

    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    //cuBLAS is col major order. ii is col index, although it doesn't really matter here
    for (unsigned int ii = 0; ii < n ; ii++) {
        for (unsigned int jj = 0; jj < n; jj++) {
            A[ii * n + jj] = dist(generator);
            B[ii * n + jj] = dist(generator);
            C[ii * n + jj] = dist(generator);
        }
    }

    float running_time = 0.0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    for (unsigned int trial = 0; trial < n_tests; trial++) {
        cudaEvent_t startEvent, stopEvent;
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
        cudaEventRecord(startEvent, 0);
        
        mmul(handle, A, B, C, n);

        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);

        running_time += elapsedTime;
    }
    running_time /= n_tests;
    std::cout << running_time << "\n";


    cublasDestroy(handle);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}