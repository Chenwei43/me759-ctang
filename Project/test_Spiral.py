"""
2D spiral concomitant field correction viability test
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import sigpy as sp
import sigpy.mri
from gradient_system import *
from scan_prescription import *
from calcBc import getBcSpiral
from dft_offres_ops import *

device = sp.Device(0)
xp = device.xp

grad_system = GradientSystem()
scan = ScanPrescription()

gammabar = 42.58e6 # Hz/T
gamma = 2.6752e8    #rad/(s*T)
Tres = 4e-6 #s
Npe = 39
Nfreq = 39
t_offset = 0.0
# Nslice = 1
# sliceThick = 4e-3   # in m
# sliceSpacing = 0
# grid_res = (39,39, 1)
# fov_z = Nslice * (sliceThick+sliceSpacing)

# Create gradients
# TODO write up vds
import scipy.io
g = scipy.io.loadmat('g.mat')
gradients = g['g'] * 1e-2   # in T/m
plt.plot(gradients[:,0])
plt.show()

t = np.arange(0,gradients.shape[0]*Tres, Tres)

k = scipy.io.loadmat('k.mat')
temp = k['k']* 1e2   # in m-1
kcoords = np.stack((np.real(temp), np.imag(temp)))
kcoords = kcoords[...,:Npe*Nfreq]
# kz= np.linspace(-fov_z/2, fov_z/2, Nslice)
# kcoords = np.tile(kcoords,(1,Nslice, 1))

# kz = np.tile(kz, (kcoords.shape[0],1,1))
# kcoords = np.concatenate((kcoords,kz), axis=1)
kcoords = sp.to_device(np.transpose(kcoords,(-1,1,0)), device)   #(npts, nslice,ndim)
ones = xp.ones(kcoords.shape[:-1]+(1,))
plt.scatter(kcoords[:100,0,0].get(),kcoords[:100,0,1].get())
plt.show()

# TODO: how to get image domain coords
# kcoords_cart = sp.gridding(ones, kcoords, grid_res)
# plt.imshow(sp.to_device(kcoords_cart[:,:,0], sp.cpu_device))
# plt.show()
reso = 1/(2*np.sqrt(kcoords[-1,0,0]**2 + kcoords[-1,0,1]**2))   # in m
[x, y] = xp.meshgrid(xp.linspace(-0.5,0.5,256),xp.linspace(-0.5,0.5,256))


affine = np.identity(3)
z = xp.zeros(x.shape, x.dtype)
Bc = getBcSpiral(gradients[:,0], gradients[:,1], x, y, z, affine, B0=3)
offmap = gamma * np.cumsum(Bc) * Tres   #(xres,yres,npts)

# phantom
sl_amps = [0.2, 1., 1.2, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]
sl_scales = [[.6624, .874, .780],  # gray big
             [.1100, .310, .220],  # right black
             [.1600, .410, .280],  # left black
             [.2100, .250, .410],  # gray center blob
             [.0460, .046, .050],
             [.0460, .046, .050],
             [.0460, .046, .050],  # left small dot
             [.0230, .023, .020],  # mid small dot
             [.0230, .023, .020]]
sl_offsets = [[0., -.0184, 0],
              [.22, 0., 0],
              [-.22, 0., 0],
              [0., .35, -.15],
              [0., .1, .25],
              [0., -.1, .25],
              [-.08, -.605, 0],
              [0., -.606, 0],
              [.06, -.605, 0]]
sl_angles = [[0, 0, 0],
             [-18, 0, 10],
             [18, 0, 10],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]
water = sp.sim.phantom([256,256], sl_amps, sl_scales, sl_offsets, sl_angles,dtype=np.complex64)
water = np.flipud(water)
water = sp.backend.to_device(water,device)
plt.imshow(np.abs(water.get()), cmap='gray')
plt.show()

# for sl in range(Nslice):
sl=0
s = xp.zeros(kcoords.shape[0], xp.complex64)
imPreSum = xp.zeros(water.shape+(kcoords.shape[0],), xp.complex64)
imEst = xp.zeros(water.shape, xp.complex64)

fr = 0.9 * Nfreq / 2.0
fw = 0.1 * Nfreq / 2.0
wx = weights_kernel(kcoords[:, sl, 0], fr, fw)
wy = weights_kernel(kcoords[:, sl, 0], fr, fw)

# Gphase = xp.zeros((256, 256, kcoords.shape[0]), dtype='complex64')
# calc_Gphase_kernel((1,),(1024,), (kx, ky, x, y, 1j, Gphase))
# Gphase = xp.exp(1j*2.0*math.pi*Gphase)
# Ophase = xp.zeros((256, 256,  kcoords.shape[0]), dtype='complex64')
# calc_Ophase_kernel((1,), (1024,), (t, offmap, fmax, 1j, Ophase))
# Ophase = xp.exp(1j * 2.0 * math.pi * Ophase)
for pos in range(gradients.shape[0]):
    # sampling
    Gphase = xp.exp(1j * 2.0 * math.pi * (kcoords[pos, sl, 0] * x + kcoords[pos, sl, 1] * y))
    Ophase = xp.exp(1j * offmap[:,:,pos] )
    s[pos] = acq_kernel(Gphase, Ophase, water)

    # DFT recon
    imPreSum[:,:,pos] = recon_kernel(xp.conj(Gphase), xp.conj(Ophase), s[pos], wx[pos], wy[pos])
imEst = xp.sum(imPreSum, axis=2)


