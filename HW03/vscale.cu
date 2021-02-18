#include <cstdio>
#include <cuda.h>
#include "cuda_runtime.h"
#include "vscale.cuh"

__global__ void vscale(const float *a, float *b, unsigned int n)
{   
    unsigned int whichEntry = threadIdx.x + blockIdx.x * blockDim.x;
    if (whichEntry < n){
        b[whichEntry] *= a[whichEntry];
        //std::printf("idx = %d, a = %f, b = %f, temp = %f\n", whichEntry, a[whichEntry], b[whichEntry], temp);
    }
}