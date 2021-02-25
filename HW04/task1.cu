#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <random>

#include <cuda.h>
#include "cuda_runtime.h"
#include "matmul.cuh"

int main(int argc, char* argv[]) {
    unsigned int n;
    unsigned int threads_per_block;
    std::istringstream input_1(argv[1]);
    std::istringstream input_2(argv[2]);
    if (input_1 >> n && input_2>>threads_per_block && input_2.eof()) {        

        // arrays on unified memory
        float *A, *B, *C;
        cudaMallocManaged(&A, sizeof(float) * n * n);
        cudaMallocManaged(&B, sizeof(float) * n * n);
        cudaMallocManaged(&C, sizeof(float) * n * n);

        // init A and B 
        std::random_device entropy_source;
        std::mt19937 generator(entropy_source());
        std::uniform_real_distribution<float> dist(-1.0, 1.0);

        for (size_t i = 0; i < n*n; i++) {
            A[i] = dist(generator);
            B[i] = dist(generator);
        }


        // kernel call
        cudaEvent_t startEvent, stopEvent; 
        cudaEventCreate(&startEvent); 
        cudaEventCreate(&stopEvent);    
        cudaEventRecord(startEvent, 0);

        matmul(A, B, C, n, threads_per_block);
        
        cudaEventRecord(stopEvent, 0); 
        cudaEventSynchronize(stopEvent); 
        float elapsedTime; 
        cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);   
        cudaEventDestroy(startEvent); 
        cudaEventDestroy(stopEvent);     

        std::cout << C[n*n-1] << "\n" ;
        std::cout << elapsedTime << "\n" ;        

        cudaFree(A);
        cudaFree(B);
        cudaFree(C);

    }

    return 0;
}