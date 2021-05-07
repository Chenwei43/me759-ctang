import cupy as cp
import numpy as np
import numba as nb
import sigpy as sp
from math import ceil
import torch
from sigpy import backend, config, util
import os
import scipy.io
from grid_ops import gridding, interpolate

device = sp.Device(0)
xp = device.xp


def scale_kw(kw, kxA, kyA, kzA, acc=1):
    Npts = kxA.shape[0]

    stride_y = kw.shape[0]
    stride_z = kw.shape[1]
    for index in range(Npts):

        ii = index % stride_y
        tempi = (index - ii) / stride_y
        jj = tempi % stride_z
        kk = (tempi - jj) / stride_z

        kx = kxA(ii, jj, kk)
        ky = kyA(ii, jj, kk)
        kz = kzA(ii, jj, kk)

        kr = np.sqrt(kx * kx + ky * ky + kz * kz)
        scale = kr / 128.0 * (acc - 1) + 1

    kw[ii, jj, kk] *= scale**3
    return kw

samples_per_spiral = 12100
Narms = 1
Niter = 100
dcf_dwin = 2.1

X = xp.zeros((4 * 512, 4 * 512, 1))
#X = xp.ones((samples_per_spiral, Narms, 1))
Kweight = xp.ones((samples_per_spiral, Narms, 1))
Kweight2 = xp.zeros((samples_per_spiral, Narms, 1))

k = scipy.io.loadmat('k.mat')
temp = k['k']* 1e2   # in m-1
kx = xp.real(temp)
ky = xp.imag(temp)
kz = xp.ones(kx.shape, kx.dtype)

for i in range(Niter):
    X[:] = 0
    Kweight2[:] = 0

    X = gridding(X, Kweight, X.shape,width=dcf_dwin, kernel_x='poly2', kernel_y='poly2', kernel_z=None)
    Kweight2 = interpolate(X, Kweight2, width=dcf_dwin, kernel_x='poly2', kernel_y='poly2', kernel_z=None)

    kw_sum = 0.0;
    for j in range(Kweight.shape[0]):
        if Kweight2[i] < 1e-3:
            Kweight[i] = 0.0
        else:
            Kweight[i] /= Kweight2[i]
            kw_sum += xp.abs(Kweight[i])

kw = scale_kw(Kweight, kx, ky, kz)

