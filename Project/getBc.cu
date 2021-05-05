#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "getBc.cuh"

__global__ void getBc_spiral(const thrust::device_vector<float>& gx, const thrust::device_vector<float>& gy,
    const thrust::device_vector<float>& x, const thrust::device_vector<float>& y, const thrust::device_vector<float>& z,
    const float dt, const float B0, const thrust::device_vector<float>& affine, thrust::device_vector<float>& phi_c)
{
    //unsigned int idxFull = threadIdx.x + blockIdx.x * blockDim.x;
    thrust::device_vector<float> gx2(gx.size());
    thrust::transform(gx.begin(), gx.end(), gx2.begin(), thrust::square<float>());
    thrust::device_vector<float> gy2(gx.size());
    thrust::transform(gy.begin(), gy.end(), gy2.begin(), thrust::square<float>());
    thrust::device_vector<float> gz2(gz.size());
    thrust::transform(gz.begin(), gz.end(), gz2.begin(), thrust::square<float>());
    thrust::device_vector<float> gxgz(gx.size());
    thrust::transform(gx.begin(), gx.end(), gx2.begin(), );
    thrust::device_vector<float> gygz(gx.size());
    thrust::transform(gx.begin(), gx.end(), gx2.begin(), );

    thrust::device_vector<float> x2(x.size());
    thrust::transform(x.begin(), x.end(), x2.begin(), thrust::square<float>());
    thrust::device_vector<float> y2(x.size());
    thrust::transform(y.begin(), y.end(), y2.begin(), thrust::square<float>());
    thrust::device_vector<float> z2(x.size());
    thrust::transform(z.begin(), z.end(), z2.begin(), thrust::square<float>());
    thrust::device_vector<float> xz(x.size());
    thrust::transform(x.begin(), x.end(), x2.begin(), );
    thrust::device_vector<float> yz(x.size());
    thrust::transform(x.begin(), x.end(), x2.begin(), );


    thrust::device_vector<float> Bc(pow(x.size(),3) * gx.size());


    thrust::reduce(Bc.begin(), Bc.end(), Bc[0]);

}