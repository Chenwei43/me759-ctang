#include <vector>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/random/linear_congruential_engine.h>

int main()
{   
    //const unsigned int nproj = 9990;
    const float B0 = .3;
    const unsigned int Npe = 110;
    const unsigned int Nfreq = 110;
    const unsigned int Npts = Npe * Nfreq;
    const unsigned int threads_per_block = 1024;
    const unsigned int res = 256;
    const float fr = 0.9 * Nfreq / 2.0;
    const float fw = 0.1 * Nfreq / 2.0;

    // kspace coords - TODO switch to vds
    const float Tread = .3;     // in sec
    float* kx, * ky;
    cudaMallocManaged(&kx, sizeof(float) * Npts);
    cudaMallocManaged(&ky, sizeof(float) * Npts);
    float tres = Tread / (Npts);
    for (unsigned int i = 0; i < Npts; i++) {
        float tt = sqrt((0.0 + tres * i) / Tread);
        kx[i] = Nfreq / 2 * tt * cos(2 * PI * Nfreq / 2 * tt);
        ky[i] = - Nfreq / 2 * tt * sin(2 * PI * Nfreq / 2 * tt);
    }

    // gradient waveform
    float* gx, * gy;
    cudaMallocManaged(&gx, sizeof(float) * Npts);
    cudaMallocManaged(&gy, sizeof(float) * Npts);
    for (unsigned int i = 0; i < Npts; i++) {
        if (i != Npts - 1) {
            gx[i] = 1 / GAMMA_BAR * (kx[i] - kx[i + 1]) / tres;
            gy[i] = 1 / GAMMA_BAR * (ky[i] - ky[i + 1]) / tres;
        }
        else {
            gx[i] = 1 / GAMMA_BAR * (kx[i] - 0.f) / tres;
            gy[i] = 1 / GAMMA_BAR * (ky[i] - 0.f) / tres;
        }        
    }
    

    // dcf 
    float* wx, * wy;
    cudaMallocManaged(&gx, sizeof(float) * Npts);
    cudaMallocManaged(&gy, sizeof(float) * Npts);

    unsigned int numBlocks = (Npts + threads_per_block - 1) / threads_per_block;
    get_dcf<<<numBlocks, threads_per_block, 2 * threads_per_block * sizeof(float) >>> (kx, ky, fr, fw, wx, wy);
    cudaDeviceSynchronize();


    //logical coords
    float* x, * y, * z;
    cudaMallocManaged(&x, sizeof(float) * res * res);
    cudaMallocManaged(&y, sizeof(float) * res * res);
    cudaMallocManaged(&z, sizeof(float) * res * res);    
    for (unsigned int i = 0; i < res * res; i++) {
        x[i] = -0.5 + 1 / res * (i % res);
        y[i] = -0.5 + 1 / res * (i / res);
        z[i] = 1.0;

    }

    //transform matrix to physical coords
    float affine[9] = {.3, 0, 0, 0, .3, 0, 0, 0, .3};
    thrust::device_vector<float> affine_vec(affine, affine+9);
   
    // Phi_c(xres, yres, t)
    float* Phi_c;
    cudaMallocManaged(&Phi_c, sizeof(float) * res * res * Npts);
    thrust::device_vector<float> gx_vec(gx, gx + Npts);
    thrust::device_vector<float> gy_vec(gy, gy + Npts);
    thrust::device_vector<float> x_vec(x, x + res * res);
    thrust::device_vector<float> y_vec(y, y + res * res);
    thrust::device_vector<float> z_vec(z, z + res * res);   
    getBc_spiral(gx_vec, gy_vec, x_vec, y_vec, z_vec, tres, B0, affine_vec, Phi_c);     //Phi_c is float*

    float* Gphase, * Ophase;
    cudaMallocManaged(&Gphase, sizeof(float) * res * res);
    cudaMallocManaged(&Ophase, sizeof(float) * res * res);
    for (unsigned int i = 0; i < Npts; i++) {

        // acquisition
        

        // recon


    }
    