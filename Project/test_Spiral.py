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
Npe = 110
Nfreq = 110
t_offset = 0.0
# Nslice = 1
# sliceThick = 4e-3   # in m
# sliceSpacing = 0
# grid_res = (39,39, 1)
# fov_z = Nslice * (sliceThick+sliceSpacing)

# TODO: calc dcf for vds spiral
# import scipy.io
# g = scipy.io.loadmat('g.mat')
# gradients = g['g'] * 1e-2   # in T/m
# gradients = xp.array(gradients, dtype='float32')
# plt.plot(gradients[:,0].get())
# plt.show()
#
# t = np.arange(0,gradients.shape[0]*Tres, Tres)

# k = scipy.io.loadmat('k.mat')
# temp = k['k']* 1e2   # in m-1
# kcoords = np.stack((np.real(temp), np.imag(temp)))
# kcoords = kcoords[...,:Npe*Nfreq]
# kcoords = kcoords.astype('float32')

Tread = 30.0*1e-3
t = np.linspace(0, Tread,Nfreq*Nfreq)
t = t.astype('float32')
tt = np.sqrt(t/Tread)
kx = Nfreq/2*tt*np.cos( 2*math.pi*Nfreq/2*tt)
kx =kx.astype('float32')
ky = Nfreq/2*tt*np.sin( 2*math.pi*Nfreq/2*tt)
ky =ky.astype('float32')

#kx,ky = -ky,kx
ky = -ky
kx = xp.array(kx)
ky=xp.array(ky)
kcoords = xp.stack((kx,ky))
kcoords = kcoords[None, ...]
kcoords = sp.to_device(np.transpose(kcoords,(-1,0,1)), device)   #(npts, nslice,ndim)

k=kx+1j*ky
g = 1/gamma*(k-xp.concatenate((k[1:],xp.zeros(1,'complex64'))))/Tres;
gradients = np.stack((xp.real(g), xp.imag(g)), axis=-1)
plt.plot(np.abs(g[:200].get()))
plt.show()

#kcoords = sp.to_device(np.transpose(kcoords,(-1,1,0)), device)   #(npts, nslice,ndim)
ones = xp.ones(kcoords.shape[:-1]+(1,))
plt.scatter(kcoords[:200,0,0].get(),kcoords[:200,0,1].get())
plt.show()

# kcoords_cart = sp.gridding(ones, kcoords, grid_res)
# plt.imshow(sp.to_device(kcoords_cart[:,:,0], sp.cpu_device))
# plt.show()
reso = 1/(2*np.sqrt(kcoords[-1,0,0]**2 + kcoords[-1,0,1]**2))   # in m
[x, y] = xp.meshgrid(xp.linspace(-0.5,0.5,256, dtype='float32'),xp.linspace(-0.5,0.5,256, dtype='float32'))


affine = np.identity(3, dtype='float32').flatten() * .3   # scale to meters
z = xp.ones(x.shape, x.dtype) * 1
Bc = getBcSpiral(gradients[:,0], gradients[:,1], x, y, z, affine, B0=3)
offmap = gamma * np.cumsum(Bc, axis=-1) * Tres   #(xres,yres,npts)

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

imPreSum2 = xp.zeros(water.shape+(kcoords.shape[0],), xp.complex64)
imEst2 = xp.zeros(water.shape, xp.complex64)

fr = 0.9 * Nfreq / 2.0
fw = 0.1 * Nfreq / 2.0
wx = weights_kernel(kcoords[:, sl, 0], fr, fw)
wy = weights_kernel(kcoords[:, sl, 0], fr, fw)
# wx = xp.ones(Npe * Nfreq, xp.float32)
# wy = xp.ones(Npe * Nfreq, xp.float32)


# Gphase = xp.zeros((256, 256, kcoords.shape[0]), dtype='complex64')
# calc_Gphase_kernel((1,),(1024,), (kx, ky, x, y, 1j, Gphase))
# Gphase = xp.exp(1j*2.0*math.pi*Gphase)
# Ophase = xp.zeros((256, 256,  kcoords.shape[0]), dtype='complex64')
# calc_Ophase_kernel((1,), (1024,), (t, offmap, fmax, 1j, Ophase))
# Ophase = xp.exp(1j * 2.0 * math.pi * Ophase)
for pos in range(Npe * Nfreq):
    # sampling
    Gphase = xp.exp(1j * 2.0 * math.pi * (kcoords[pos, sl, 0] * x + kcoords[pos, sl, 1] * y))
    Ophase = xp.exp(1j * offmap[:,:,pos] )
    s[pos] = acq_kernel(Gphase, Ophase, water)

    # DFT recon
    ignoreOphase = xp.ones(Gphase.shape, Gphase.dtype)
    imPreSum[:,:,pos] = recon_kernel(xp.conj(Gphase), xp.conj(Ophase), s[pos], wx[pos], wy[pos])
    imPreSum2[:, :, pos] = recon_kernel(xp.conj(Gphase), xp.conj(ignoreOphase), s[pos], wx[pos], wy[pos])
imEst = xp.sum(imPreSum, axis=2)
imEst2 = xp.sum(imPreSum2, axis=2)
plt.figure()
plt.imshow(np.abs(imEst.get()-imEst2.get()), cmap='gray')
plt.colorbar()
plt.axis('off')
plt.title('Difference')
plt.show()

fig = plt.figure(figsize=(12,3))
ax1 = plt.subplot(141)
ax2 = plt.subplot(142)
ax3 = plt.subplot(143)
ax4 = plt.subplot(144)
ax1.set_title("With correction")
ax1.imshow(np.abs(imEst.get()), cmap='gray')
ax1.axis('off')
ax2.set_title("without correction")
ax2.imshow(np.abs(imEst2.get()), cmap='gray')
ax2.axis('off')
ax3.set_title("Difference")
ax3.imshow(np.abs(imEst.get()-imEst2.get()), cmap='gray')
ax3.axis('off')
ax4.set_title("Truth")
ax4.imshow(np.abs(water.get()), cmap='gray')
ax4.axis('off')
plt.show()
fig.savefig(r'D:\Courses\ME759\me759-ctang\Project\phantom_simpleSpiral.png')
