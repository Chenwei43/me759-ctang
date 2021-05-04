import numba as nb
import numpy as np
import cupy as cp
from math import ceil
from sigpy import backend, config, util
import os

acq_kernel = cp.ReductionKernel(
    'T gphase, T ophase, T imt',
    'T signal',
    'imt*gphase*ophase',
    'a+b',
    'signal=a',
    identity='0',
    name='acq'
)

recon_kernel = cp.ElementwiseKernel(
    'T gphaseconj, T ophaseconj, T signal, S wx, S wy',
    'T Dimage',
    'Dimage = signal * gphaseconj * ophaseconj * wx * wy',
    name='pre_recon'
)

weights_kernel = cp.ElementwiseKernel(
    'S kx, S fr, S fw',
    'S w',
    'w = 1 / (1 + expf((fabsf(kx) - fr) / fw))',
    name='calc_dcf'
)

calc_Gphase_kernel = cp.RawKernel(r'''
#include <cupy/complex.cuh>
extern "C" __global__
void calc_gphase(const float* kx, const float* ky, const float* x, const float* y, const complex<float> I, 
    complex<float>* gphase){
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int imagesize = sizeof(x)/sizeof(x[0]);
    for (unsigned int xi=0; xi<imagesize; xi++){
        //gphase[tid*imagesize + xi] = exp(I * 2.f * 3.1415927f * (kx[tid] * x[xi] + ky[tid] * y[xi]));
        gphase[tid*imagesize + xi] = kx[tid] * x[xi] + ky[tid] * y[xi];
    }   
}
''', name='calc_gphase')

calc_Ophase_kernel = cp.RawKernel(r'''
#include <cupy/complex.cuh>
extern "C" __global__
void calc_ophase(const float* t, const float* offmap, const float* fmax, const complex<float> I, 
    complex<float>* ophase){
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int imagesize = sizeof(offmap)/sizeof(offmap[0]);
    for (unsigned int xi=0; xi<imagesize; xi++){
        //ophase[tid*imagesize + xi] = exp(I * 2.f * 3.1415927f * (t[tid] * offmap[xi]));
        ophase[tid*imagesize + xi] = t[tid] * offmap[xi];
    }   
}
''', name='calc_ophase')
