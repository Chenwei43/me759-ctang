#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "scan.cuh"

int main(int argc, char* argv[]) {
    unsigned int n = std::strtoul(argv[1], nullptr, 10);
    unsigned int threads_per_block = std::atoi(argv[2]);

    float* input, * output;
    cudaMallocManaged(&input, sizeof(float) * n);
    cudaMallocManaged(&output, sizeof(float) * n);

    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);
    for (unsigned int ii = 0; ii < n; ii++) {
        input[ii] = dist(generator);
    }

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);

    scan(input, output, n, threads_per_block);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    /*
    // correctness checked 
    for (unsigned int i = 0; i < n; i++) {
        std::cout << input[i] << " " << output[i] << "\n";
    }
    */
    
    std::cout << output[n-1] << "\n";
    std::cout << elapsedTime << "\n";

    cudaFree(input);
    cudaFree(output);

    return 0;
}