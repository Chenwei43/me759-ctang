# -*- coding: utf-8 -*-
import numpy as np
import sigpy as sp
import wave
import h5py
import math

class ScanPrescription(object):
    def __init__(self):
        # Imaging and Simulation Paramaters
        self.TR = 4e-3
        self.gam = 42.58e6 # Hz/T
        self.bwmax = 125e3 # Hz
        self.fov = 0.32 # m
        self.gmax_bw = self.bwmax / ( self.gam*self.fov ) # Hz / ( Hz/T) / m -> ( T / m )
        self.resolution = 1.25e-3 #m
        self.kmax = 1 / (2*self.resolution) # m^-1
        self.area_kmax = self.kmax / self.gam # (1/m) / ( Hz/T)
