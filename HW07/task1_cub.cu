// CUB reduce
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/random/linear_congruential_engine.h>

#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <random>

#define CUB_STDERR // print CUDA runtime errors to console
#include <stdio.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include "cub/util_debug.cuh"
using namespace cub;
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

int main(int argc, char* argv[]) {
    // std::ofstream timefile;
    // timefile.open("timingTask1_cub.txt");
    // for (int j = 10; j < 31; j++) {

    unsigned int n = std::strtoul(argv[1], nullptr, 10);

    // Set up host arrays
    float *h_in = new float[n];
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);
    for (unsigned int i = 0; i < n; i++) {
        h_in[i] = dist(generator);
    }
    
    // Set up device arrays
    float* d_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_in, sizeof(float) *n));
    // Initialize device input
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(float) * n, cudaMemcpyHostToDevice));
    // Setup device output array
    float* d_sum = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_sum, sizeof(float) * 1));
    // Request and allocate temporary storage
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, n));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Do the actual reduce operation
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);

    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, n);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);        
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);        
    
    float gpu_sum;
    CubDebugExit(cudaMemcpy(&gpu_sum, d_sum, sizeof(float) * 1, cudaMemcpyDeviceToHost));
    std::cout << gpu_sum << std::endl;
    std::cout << elapsedTime << std::endl;
    //timefile << n << " " << elapsedTime << "\n";

    /*
    // Check for correctness
    float  sum = 0.0;
    for (unsigned int i = 0; i < n; i++)
        sum += h_in[i];

    printf("\t%s\n", (gpu_sum == sum ? "Test passed." : "Test falied."));
    printf("\tSum is: %f\n", gpu_sum);
    printf("\tSum is: %f\n", sum);
    */

    // Cleanup
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_sum) CubDebugExit(g_allocator.DeviceFree(d_sum));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    delete []h_in;
    //}
    return 0;
}


/*
        
*/