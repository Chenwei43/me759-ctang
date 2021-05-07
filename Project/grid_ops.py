import cupy as cp
import numpy as np
import numba as nb
import sigpy as sp
from math import ceil
import torch
from sigpy import backend, config, util
import os

KERNELS = ['poly', 'poly2']

def interpolate(input, coord, width=2, kernel_x='poly2', kernel_y='poly2', kernel_z=None):
    ndim = coord.shape[-1]
    nkernel = 3

    batch_shape = input.shape[:-ndim]
    batch_size = util.prod(batch_shape)

    pts_shape = coord.shape[:-1]
    # print(pts_shape)
    npts = util.prod(pts_shape)

    xp = backend.get_array_module(input)

    coord = sp.fourier._scale_coord(coord, input.shape, 1.0)

    input = input.reshape([batch_size] + list(input.shape[-ndim:]))
    coord = coord.reshape([npts, ndim])
    output = xp.zeros([batch_size, npts], dtype=input.dtype)

    if np.isscalar(width):
        width = xp.array([width] * nkernel, coord.dtype)
    else:
        width = xp.array(width, coord.dtype)

    # print(f'batchshape {batch_shape}')  # batch_size=1
    # print(f'input shape {input.shape}') # (batchsize, Nframe, 192, 192, 192)
    # print(f'coord shape {coord.shape}') # (135500, 4)
    # print(f'output shape {output.shape}')   # (1, 135500)
    # print(f'width {width}')
    _interpolate_cuda[kernel_x][kernel_y][kernel_z][ndim - 1](input, coord, width, output, size=npts)

    return output.reshape(batch_shape + pts_shape)


def gridding(input, coord, shape, width=2, kernel_x='poly2', kernel_y='poly2', kernel_z=None):
    xp = backend.get_array_module(input)

    ndim = coord.shape[-1]
    nkernel = 3  # for kernel_xyz and kernel_t

    batch_shape = shape[:-ndim]
    batch_size = util.prod(batch_shape)

    pts_shape = coord.shape[:-1]
    npts = util.prod(pts_shape)

    input = input.reshape([batch_size, npts])
    coord = coord.reshape([npts, ndim])
    output = xp.zeros([batch_size] + list(shape[-ndim:]), dtype=input.dtype)
    # print('in gridding()')
    # print(f'batchshape {batch_shape}')
    # print(f'input shape {input.shape}')
    # print(f'coord shape {coord.shape}')
    # print(f'output shape {output.shape}')

    if xp.isscalar(width):
        width = xp.array([width] * nkernel, coord.dtype)
    else:
        width = xp.array(width, coord.dtype)

    _gridding_cuda[kernel_x][kernel_y][kernel_z][ndim - 1](input, coord, width, output, size=npts)


    return output.reshape(shape)


mod_cuda = """
__device__ inline int mod(int x, int n) {
    return (x % n + n) % n;
}
"""

_poly_kernel = """
    S x2 = x*x;
    S x4 = x2*x2;
    S x3 = x*x2;
    S x5 = x*x4;

    return 1 + 0.04522831*x - 3.36020304*x2 + 1.12417012*x3 + 2.82448025*x4 - 1.63447764*x5;

"""

_poly2_kernel = """
    S x2 = x*x;
    S x4 = x2*x2;
    S x3 = x*x2;
    S x5 = x*x4;

    return 1 + 0.03056504*x - 3.01961845*x2 + 0.6679865*x3 + 2.77924058*x4 - 1.45923643*x5;
"""


# TODO kernel normalization??

def _get_kernel_cuda(kernel, name):
    kname = f'__device__ inline S {name}(S x) {{'
    if kernel == 'poly':
        kernel = _poly_kernel
    elif kernel == 'poly2':
        kernel = _poly2_kernel

    kernel = kname + '\n' + kernel

    return kernel


def _get_gridding_cuda(kernel_x, kernel_y, kernel_z):
    _gridding1_cuda = cp.ElementwiseKernel(
        'raw T input, raw S coord, raw S width',
        'raw T output',
        """
        const int ndim = 1;
        const int batch_size = output.shape()[0];
        const int nx = output.shape()[1];

        const int coord_idx[] = {i, 0};
        const S kx = coord[coord_idx];
        const int x0 = ceil(kx - width[ndim - 1] / 2.0);
        const int x1 = floor(kx + width[ndim - 1] / 2.0);

        for (int x = x0; x < x1 + 1; x++) {
            const S w = kernel_x(((S) x - kx) / (width[ndim - 1] / 2.0));
            for (int b = 0; b < batch_size; b++) {
                const int input_idx[] = {b, i};
                const T v = (T) w * input[input_idx];
                const int output_idx[] = {b, mod(x, nx)};
                atomicAdd(&output[output_idx], v);
            }
        }
        """,
        name='gridding1',
        preamble=kernel_x + mod_cuda,
        reduce_dims=False)

    _gridding2_cuda = cp.ElementwiseKernel(
        'raw T input, raw S coord, raw S width', 'raw T output', """
                const int ndim = 2;
                const int batch_size = output.shape()[0];
                const int ny = output.shape()[1];
                const int nx = output.shape()[2];

                const int coordx_idx[] = {i, 1};
                const S kx = coord[coordx_idx];
                const int coordy_idx[] = {i, 0};
                const S ky = coord[coordy_idx];

                const int x0 = ceil(kx - width[ndim - 1] / 2.0);
                const int y0 = ceil(ky - width[ndim - 2] / 2.0);

                const int x1 = floor(kx + width[ndim - 1] / 2.0);
                const int y1 = floor(ky + width[ndim - 2] / 2.0);

                for (int y = y0; y < y1 + 1; y++) {
                    const S wy = kernel_y(
                        ((S) y - ky) / (width[ndim - 2] / 2.0));
                    for (int x = x0; x < x1 + 1; x++) {
                        const S w = wy * kernel_x(((S) x - kx) / (width[ndim - 1] / 2.0));
                        for (int b = 0; b < batch_size; b++) {
                            const int input_idx[] = {b, i};
                            const T v = (T) w * input[input_idx];
                            const int output_idx[] = {b, mod(y, ny), mod(x, nx)};
                            atomicAdd(&output[output_idx], v);
                        }
                    }
                }
                """, name='gridding2', preamble=kernel_x + kernel_y + mod_cuda,
        reduce_dims=False)

    _gridding3_cuda = cp.ElementwiseKernel(
        'raw T input, raw S coord, raw S width', 'raw T output', """
                const int ndim = 3;
                const int batch_size = output.shape()[0];
                const int nz = output.shape()[1];
                const int ny = output.shape()[2];
                const int nx = output.shape()[3];

                const int coordz_idx[] = {i, 0};
                const S kz = coord[coordz_idx];
                const int coordy_idx[] = {i, 1};
                const S ky = coord[coordy_idx];
                const int coordx_idx[] = {i, 2};
                const S kx = coord[coordx_idx];

                const int x0 = ceil(kx - width[ndim - 1] / 2.0);
                const int y0 = ceil(ky - width[ndim - 2] / 2.0);
                const int z0 = ceil(kz - width[ndim - 3] / 2.0);

                const int x1 = floor(kx + width[ndim - 1] / 2.0);
                const int y1 = floor(ky + width[ndim - 2] / 2.0);
                const int z1 = floor(kz + width[ndim - 3] / 2.0);

                for (int z = z0; z < z1 + 1; z++) {
                    const S wz = kernel_z(((S) z - kz) / (width[ndim - 3] / 2.0));
                    for (int y = y0; y < y1 + 1; y++) {
                        const S wy = wz * kernel_y(((S) y - ky) / (width[ndim - 2] / 2.0));
                        for (int x = x0; x < x1 + 1; x++) {
                            const S w = wy * kernel_x(((S) x - kx) / (width[ndim - 1] / 2.0));
                            for (int b = 0; b < batch_size; b++) {
                                const int input_idx[] = {b, i};
                                const T v = (T) w * input[input_idx];
                                const int output_idx[] = {
                                    b, mod(z, nz), mod(y, ny), mod(x, nx)};
                                atomicAdd(&output[output_idx], v);
                            }
                        }
                    }
                }
                """, name='gridding3', preamble=kernel_x + kernel_y + kernel_z + mod_cuda,
        reduce_dims=False)
    return _gridding1_cuda, _gridding2_cuda, _gridding3_cuda


def _get_interpolate_cuda(kernel_x, kernel_y, kernel_z):
    _interpolate1_cuda = cp.ElementwiseKernel(
        'raw T input, raw S coord, raw S width, raw S param',
        'raw T output',
        """
        const int ndim = 1;
        const int batch_size = input.shape()[0];
        const int nx = input.shape()[1];
        const int coord_idx[] = {i, 0};
        const S kx = coord[coord_idx];
        const int x0 = ceil(kx - width[ndim - 1] / 2.0);
        const int x1 = floor(kx + width[0] / 2.0);
        for (int x = x0; x < x1 + 1; x++) {
            const S w = kernel_x(((S) x - kx) / (width[ndim - 1] / 2.0), param[0]);
            for (int b = 0; b < batch_size; b++) {
                const int input_idx[] = {b, mod(x, nx)};
                const T v = (T) w * input[input_idx];
                const int output_idx[] = {b, i};
                output[output_idx] += v;
            }
        }
        """,
        name='interpolate1',
        preamble=kernel_x + mod_cuda,
        reduce_dims=False)

    _interpolate2_cuda = cp.ElementwiseKernel(
        'raw T input, raw S coord, raw S width, raw S param',
        'raw T output',
        """
        const int ndim = 2;
        const int batch_size = input.shape()[0];
        const int ny = input.shape()[1];
        const int nx = input.shape()[2];
        const int coordx_idx[] = {i, 1};
        const S kx = coord[coordx_idx];
        const int coordy_idx[] = {i, 0};
        const S ky = coord[coordy_idx];
        const int x0 = ceil(kx - width[ndim - 1] / 2.0);
        const int y0 = ceil(ky - width[ndim - 2] / 2.0);
        const int x1 = floor(kx + width[ndim - 1] / 2.0);
        const int y1 = floor(ky + width[ndim - 2] / 2.0);
        for (int y = y0; y < y1 + 1; y++) {
            const S wy = kernel_y(((S) y - ky) / (width[ndim - 2] / 2.0), param[0]);
            for (int x = x0; x < x1 + 1; x++) {
                const S w = wy * kernel_x(((S) x - kx) / (width[ndim - 1] / 2.0), param[1]);
                for (int b = 0; b < batch_size; b++) {
                    const int input_idx[] = {b, mod(y, ny), mod(x, nx)};
                    const T v = (T) w * input[input_idx];
                    const int output_idx[] = {b, i};
                    output[output_idx] += v;
                }
            }
        }
        """,
        name='interpolate2',
        preamble=kernel_x + kernel_y + mod_cuda,
        reduce_dims=False)

    _interpolate3_cuda = cp.ElementwiseKernel(
        'raw T input, raw S coord, raw S width, raw S param', 'raw T output', """
        const int ndim = 3;
        const int batch_size = input.shape()[0];
        const int nz = input.shape()[1];
        const int ny = input.shape()[2];
        const int nx = input.shape()[3];
        const int coordz_idx[] = {i, 0};
        const S kz = coord[coordz_idx];
        const int coordy_idx[] = {i, 1};
        const S ky = coord[coordy_idx];
        const int coordx_idx[] = {i, 2};
        const S kx = coord[coordx_idx];
        const int x0 = ceil(kx - width[ndim - 1] / 2.0);
        const int y0 = ceil(ky - width[ndim - 2] / 2.0);
        const int z0 = ceil(kz - width[ndim - 3] / 2.0);
        const int x1 = floor(kx + width[ndim - 1] / 2.0);
        const int y1 = floor(ky + width[ndim - 2] / 2.0);
        const int z1 = floor(kz + width[ndim - 3] / 2.0);
        for (int z = z0; z < z1 + 1; z++) {
            const S wz = kernel_z(((S) z - kz) / (width[ndim - 3] / 2.0), param[0]);
            for (int y = y0; y < y1 + 1; y++) {
                const S wy = wz * kernel_y(((S) y - ky) / (width[ndim - 2] / 2.0), param[1]);
                for (int x = x0; x < x1 + 1; x++) {
                    const S w = wy * kernel_x(((S) x - kx) / (width[ndim - 1] / 2.0), param[2]);
                    for (int b = 0; b < batch_size; b++) {
                        const int input_idx[] = {b, mod(z, nz), mod(y, ny),
                            mod(x, nx)};
                        const T v = (T) w * input[input_idx];
                        const int output_idx[] = {b, i};
                        output[output_idx] += v;
                    }
                }
            }
        }
        """, name='interpolate3', preamble=kernel_x + kernel_y + kernel_z + mod_cuda,
        reduce_dims=False)

    return _interpolate1_cuda, _interpolate2_cuda, _interpolate3_cuda


_interpolate_cuda = {}
_gridding_cuda = {}
for kernel_x in KERNELS:
    _interpolate_cuda[kernel_x] = {}
    _gridding_cuda[kernel_x] = {}
    kernx = _get_kernel_cuda(kernel_x, 'kernel_x')
    for kernel_y in KERNELS:
        _interpolate_cuda[kernel_x][kernel_y] = {}
        _gridding_cuda[kernel_x][kernel_y] = {}
        kerny = _get_kernel_cuda(kernel_y, 'kernel_y')
        for kernel_z in KERNELS:
            kernz = _get_kernel_cuda(kernel_z, 'kernel_z')
            _interpolate_cuda[kernel_x][kernel_y][kernel_z] = _get_interpolate_cuda(kernx, kerny, kernz)
            _gridding_cuda[kernel_x][kernel_y][kernel_z] = _get_gridding_cuda(kernx, kerny, kernz)


