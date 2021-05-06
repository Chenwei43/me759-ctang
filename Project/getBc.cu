#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include "getBc.cuh"

void getBc_spiral(const thrust::device_vector<float>& gx, const thrust::device_vector<float>& gy,
    const thrust::device_vector<float>& x, const thrust::device_vector<float>& y, const thrust::device_vector<float>& z,
    const float dt, const float B0, const thrust::device_vector<float>& affine, thrust::device_vector<float>& phi_c)
{
//unsigned int idxFull = threadIdx.x + blockIdx.x * blockDim.x;
thrust::device_vector<float> gx2(gx.size());
thrust::transform(gx.begin(), gx.end(), gx2.begin(), thrust::square<float>());
thrust::device_vector<float> gy2(gx.size());
thrust::transform(gy.begin(), gy.end(), gy2.begin(), thrust::square<float>());
thrust::device_vector<float> g(gx.size());
thrust::transform(gx.begin(), gx.end(), gy.begin(), g, thrust::plus<float>());

thrust::device_vector<float> x2(x.size());
thrust::transform(x.begin(), x.end(), x2.begin(), thrust::square<float>());
thrust::device_vector<float> y2(x.size());
thrust::transform(y.begin(), y.end(), y2.begin(), thrust::square<float>());
thrust::device_vector<float> z2(x.size());
thrust::transform(z.begin(), z.end(), z2.begin(), thrust::square<float>());
thrust::device_vector<float> xz(x.size());
thrust::transform(x.begin(), x.end(), z.begin(), xz.begin(), thrust::multiplies<float>());
thrust::device_vector<float> yz(x.size());
thrust::transform(y.begin(), y.end(), z.begin(), yz.begin(), thrust::multiplies<float>());
thrust::device_vector<float> xy(x.size());
thrust::transform(x.begin(), x.end(), y.begin(), xy.begin(), thrust::multiplies<float>());

float f1 = .25 * (pow(affine[0] ,2) + pow(affine[3] ,2)) * (pow(affine[6] ,2) + pow(affine[7] ,2)) + pow(affine[6] ,2) * (pow(affine[1] ,2) + pow(affine[4] ,2)) - affine[6] * affine[7] * (affine[0] * affine[1] + affine[3] * affine[4]);
float f2 = .25 * (pow(affine[1] ,2) + pow(affine[4] ,2)) * (pow(affine[6] ,2) + pow(affine[7] ,2)) + pow(affine[7] ,2) * (pow(affine[0] ,2) + pow(affine[3] ,2)) - affine[6] * affine[7] * (affine[0] * affine[1] + affine[3] * affine[4]);
float f3 = .25 * (pow(affine[2], 2) + pow(affine[5], 2)) * (pow(affine[6], 2) + pow(affine[7], 2)) + pow(affine[8], 2) * (pow(affine[0], 2) + pow(affine[1], 2) + pow(affine[3], 2) + pow(affine[4], 2))
- affine[6] * affine[8] * (affine[0] * affine[2] + affine[3] * affine[5])
- affine[7] * affine[8] * (affine[1] * affine[2] + affine[4] * affine[5]);
float f4 = .5 * (affine[1] * affine[2] + affine[4] * affine[5]) * (pow(affine[6] ,2) - pow(affine[7] ,2)) 
+ affine[7] * affine[8] * (2 * pow(affine[0] ,2) + pow(affine[1] ,2) + 2 * pow(affine[3] ,2) + pow(affine[4] ,2))
- affine[6] * affine[7] * (affine[0] * affine[2] + affine[3] * affine[5]) 
- affine[6] * affine[8] * (affine[0] * affine[1] + affine[3] * affine[4]);
float f5 = .5 * (affine[0] * affine[2] + affine[3] * affine[5]) * (pow(affine[7] ,2) - pow(affine[6] ,2)) 
+ affine[6] * affine[8] * (2 * pow(affine[0] ,2) + pow(affine[1] ,2) + 2 * pow(affine[3] ,2) + pow(affine[4] ,2)) 
- affine[6] * affine[7] * (affine[1] * affine[2] + affine[4] * affine[5]) 
- affine[7] * affine[8] * (affine[0] * affine[1] + affine[3] * affine[4]);
float f6 = -.5 * (affine[0] * affine[1] + affine[3] * affine[4]) * (pow(affine[6],2)+pow(affine[7] ,2)) 
+ affine[6] * affine[7] * (pow(affine[0] ,2) + pow(affine[1] ,2) + pow(affine[3] ,2) + pow(affine[4] ,2));

cublasHandle_t handle;
cublasCreate(&handle);    
cublasSscal(handle, x.size(), &f1, thrust::raw_pointer_cast(&x2[0]), 1);
cublasSscal(handle, x.size(), &f2, thrust::raw_pointer_cast(&y2[0]), 1);
cublasSscal(handle, x.size(), &f3, thrust::raw_pointer_cast(&z2[0]), 1);
cublasSscal(handle, x.size(), &f4, thrust::raw_pointer_cast(&yz[0]), 1);
cublasSscal(handle, x.size(), &f5, thrust::raw_pointer_cast(&xz[0]), 1);
cublasSscal(handle, x.size(), &f6, thrust::raw_pointer_cast(&xy[0]), 1);

// spatial term, res*res
thrust::device_vector<float> spatial(x.size());
thrust::transform(x2.begin(), x2.end(), y2.begin(), spatial, thrust::plus<float>());
thrust::transform(spatial.begin(), spatial.end(), z2.begin(), spatial, thrust::plus<float>());
thrust::transform(spatial.begin(), spatial.end(), z2.begin(), spatial, thrust::plus<float>());
thrust::transform(spatial.begin(), spatial.end(), xz.begin(), spatial, thrust::plus<float>());
thrust::transform(spatial.begin(), spatial.end(), yz.begin(), spatial, thrust::plus<float>());
thrust::transform(spatial.begin(), spatial.end(), xy.begin(), spatial, thrust::plus<float>());

//Bc
float* Bc;
cudaMallocManaged(&Bc, sizeof(float) * x.size() * y.size() * gx.size());
float* ptrSpatial = thrust::raw_pointer_cast(&spatial[0]);

for (unsigned int ii = 0; ii < gx.size(); ii++) {
thrust::device_vector<float> grad(x.size()*y.size());
thrust::fill(grad.begin(), grad.end(), g[ii]);
thrust::transform(spatial.begin(), spatial.end(), grad.begin(), Bc+ii*(x.size() * y.size()), thrust::multiplies<float>());
}

// Phi(t) = gamma_bar* scan(Bc*dT)



}