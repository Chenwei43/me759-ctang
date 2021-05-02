# -*- coding: utf-8 -*-
import numpy as np
import sigpy as sp
import wave
import h5py
import math


class GradientSystem(object):
    """ Gradient system class
    Args:
        slew: Slew rate in T/m/s
        gmax: Max gradient in mT/m
    Calls:
        check_slew:
    """

    def __init__(self, smax=200, gmax=50e-3, dt=4e-6, verbose=True):

        self.smax = smax
        self.gmax = gmax
        self.verbose = verbose
        self.dt = dt

    def slew_calc(self, gradient=None, dt=None):

        xp = sp.get_array_module(gradient)
        with sp.get_device(gradient):
            slew_wave = xp.diff(gradient) / dt

        return slew_wave

    def check_slew(self, gradient=None, axis=None, dt=4e-6):

        slew_wave = self.slew_calc()
        xp = sp.get_array_module(gradient)

        max_slew = xp.max(slew_wave)

        if self.verbose:
            print(f'Max slew waveform = {max_slew}, max_slew system = {self.smax}')

        if max_slew > self.smax:
            return False
        else:
            return True


def trapezoid_gradient(area, smax, gmax, dt=4e-6, device=sp.cpu_device, sinusoid_factor=1.0):
    """Function to design a trapezoid gradient based on area.
    Args:
        area (float): desired area of gradient (T/m*s)
        smax (float): max slew of gradient ( T/m/s)
        gmax (float): maximum gradient amplitude (T/m)
        dt (float): update rate of gradients (s)
        device (Device): device to perform design

    Returns:
        array: gradient waveform

    """
    print('Design Trapezoid with:')
    print(f'\tGmax = {gmax}')
    print(f'\tSmax = {smax}')
    print(f'\tArea = {area}')
    print(f'\tSinusoid factor  = {sinusoid_factor}')

    if area < 1e-9:
        gradient = np.zeros((1,),dtype=np.float32)
        return sp.to_device(gradient.astype(np.float32), device)

    risetime = dt * math.ceil(gmax / smax / dt)
    print(f'Risetime = {1e6 * risetime}')
    area_ramp = risetime * gmax

    if sinusoid_factor==1:
        if area < area_ramp:
            # Triangle gradient
            pw_ga = dt * math.ceil(np.sqrt(area / smax) / dt)
            pw_g = 0

            rampup = np.arange(math.ceil(pw_ga / dt)) * smax * dt
            gradient = np.concatenate([rampup, np.flip(rampup)])

        else:
            # Trapezoid
            pw_ga = risetime
            pw_g = dt * math.ceil((area - gmax * risetime) / gmax / dt)

            rampup = np.arange(1 + math.ceil(pw_ga / dt)) * smax * dt
            rampup *= gmax / np.max(rampup)
            flat = np.ones(math.ceil(pw_g / dt)) * gmax
            gradient = np.concatenate([rampup, flat, np.flip(rampup)])
    else:
        if area < area_ramp:
            # Triangle gradient
            pw_ga = dt * math.ceil(np.sqrt(area / smax) / dt)
            pw_g = 0

            rampup =  np.sin(2*math.pi*np.arange(math.ceil(pw_ga / dt)) *sinusoid_factor)
            gradient = np.concatenate([rampup, np.flip(rampup)])


        else:
            # Trapezoid
            pw_ga = risetime
            pw_g = dt * math.ceil((area - gmax * risetime) / gmax / dt)

            rampup = np.sin(2 * math.pi * np.arange(1+math.ceil(pw_ga / dt)) * sinusoid_factor)
            rampup *= gmax / np.max(rampup)
            flat = np.ones(math.ceil(pw_g / dt)) * gmax
            gradient = np.concatenate([rampup, flat, np.flip(rampup)])


    # Adjust area for rounding
    scale = area / np.sum(dt * gradient)
    print(f'Scale = {scale}')
    gradient *= scale

    gradient = sp.to_device(gradient.astype(np.float32), device)
    print(f'trapezoid_gradient dtype {gradient.dtype}')
    return gradient
