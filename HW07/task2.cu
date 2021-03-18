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
#include "count.cuh"

__host__ static __inline__ float rand_pm1()
{
    const int RANGE = 500;
    return rand() % (RANGE+1);
}

int main(int argc, char* argv[]) {   

    unsigned int n = std::strtoul(argv[1], nullptr, 10);

    thrust::host_vector<int> h_in(n);
    thrust::generate(h_in.begin(), h_in.end(), rand_pm1);
    thrust::device_vector<int> d_in =  h_in;
    thrust::device_vector<int> values;
    thrust::device_vector<int> counts;

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);

    count(d_in, values, counts);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);        
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    std::cout << values[values.size()-1] << std::endl;
    std::cout << counts[values.size()-1] << std::endl;
    std::cout << elapsedTime << std::endl;    
    
    return 0;
}