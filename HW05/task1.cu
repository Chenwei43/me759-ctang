#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "reduce.cuh"

int main(int argc, char* argv[]) {
    unsigned int n;
    unsigned int threads_per_block;
    std::istringstream input_1(argv[1]);
    std::istringstream input_2(argv[2]);
    if (input_1 >> n && input_2>>threads_per_block && input_2.eof()) {
        //create arrays on host, pinned mem
        float *inputHost = new float[n];
        //cudaMallocHost(&inputHost, sizeof(float) * n);

        std::random_device entropy_source;
        std::mt19937 generator(entropy_source());
        std::uniform_real_distribution<float> dist(-1.0, 1.0);
        for (unsigned int i = 0; i < n; i++) {
            inputHost[i] = dist(generator);
            //inputHost[i] = 1.0;
        }

        // cuda arrays
        unsigned int numBlocks = 1;
        if (threads_per_block < n) {
            numBlocks = (n + threads_per_block - 1) / (2 * threads_per_block);
        }
        float *input, *output;
        cudaMalloc((void**)&input, sizeof(float ) * n);
        cudaMalloc((void**)&output, sizeof(float ) * numBlocks);	
        cudaMemcpy(input, inputHost, sizeof(float)*n, cudaMemcpyHostToDevice);
        cudaMemset(output, 0, sizeof(int) * numBlocks);
        /*float* test = new float[n];
        cudaMemcpy(test, input, sizeof(float*) * n, cudaMemcpyDeviceToHost);
        std::cout << test[0] << "\n";*/
        
        
        // kernel call
        cudaEvent_t startEvent, stopEvent;
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
        cudaEventRecord(startEvent, 0);

        reduce(&input, &output, n, threads_per_block);

        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);

        float result;
        cudaMemcpy(&result, input, sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << result << "\n";
        std::cout << elapsedTime << "\n";
        
        //delete[] test;

        cudaFree(output);
        cudaFree(input);
        
    }
    return 0;
}