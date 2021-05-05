#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "getDCF.cuh"

__global__ void get_dcf(const float* kx, const float* ky, const float fr, const float fw, float* wx, float* wy)
{   
    // shared mem holds kx, ky
    extern  __shared__  float sharedArray[];    
    unsigned int idxFull = threadIdx.x + blockIdx.x * blockDim.x;
    float weightx = 0.f;
    float weighty = 0.f;

    // grab tiles As and Bs from global to shared mem
    sharedArray[threadIdx.x] = kx[idxFull];
    sharedArray[blockDim.x + threadIdx.x] = ky[idxFull];
    weightx = 1 / (1 + exp((abs(sharedArray[threadIdx.x]) - fr) / fw));
    weighty = 1 / (1 + exp((abs(sharedArray[blockDim.x + threadIdx.x]) - fr) / fw));
    wx[idxFull] = weightx;
    wy[idxFull] = weighty;

}