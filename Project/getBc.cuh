#define GAMMA_BAR  42580000.0   // Hz/T
#define GAMMA 267520000.0       //#rad/(s*T)
#define PI 3.14159265359

#ifndef GETBC_CUH
#define GETBC_CUH

// spiral
__global__ void getBc_spiral(const thrust::device_vector<float>& gx, const thrust::device_vector<float>& gy,
    const thrust::device_vector<float>& x, const thrust::device_vector<float>& y, const thrust::device_vector<float>& z,
    const float dt, const float B0, const thrust::device_vector<float>& affine, thrust::device_vector<float>& phi_c);

#endif