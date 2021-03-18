// Thrust reduce
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

__host__ static __inline__ float rand_pm1()
{
    return (float)(rand()) / (float)(RAND_MAX/2) - 1.f;
}

int main(int argc, char* argv[]) {
    // std::ofstream timefile;
    // timefile.open("timingTask1_thrust.txt");
    // for (int j = 10; j < 31; j++) {

    unsigned int n = std::strtoul(argv[1], nullptr, 10);

    thrust::host_vector<float> h_vec(n);
    thrust::generate(h_vec.begin(), h_vec.end(), rand_pm1);
    
    thrust::device_vector<float> d_vec =  h_vec;
    float init = d_vec[0];

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);

    float result = thrust::reduce(d_vec.begin(), d_vec.end(), init);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    // timefile << n << " " << elapsedTime << "\n";
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    std::cout << result << std::endl;
    std::cout << elapsedTime << std::endl;
    // }

    return 0;
}