#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  test_recoverstats.py
#
#  Copyright 2016 Bruno S <bruno.sanchez.63@gmail.com>
#

import os
import shlex
import subprocess

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits

import propercoadd as pc

# =============================================================================
#     PSF measure test by propercoadd
# =============================================================================
datapath = os.path.abspath('/home/bruno/Documentos/reduccionTolar/20151212/Landolt_C53')

frame = os.path.join(datapath, 'Landolt_C53-006.fit')

sim = pc.SingleImage(frame, imagefile=True)

#fitted_models = sim.fit_psf_sep()

# =============================================================================
#    PSF spatially variant
# =============================================================================
psf_basis = sim._kl_from_stars
a_fields = sim._kl_a_fields

test_dir = os.path.abspath('./test_images/real_image_test/')

plt.figure(figsize=(16,16))
plt.imshow(np.log10(fits.getdata(frame)), interpolation='none')
plt.colorbar(orientation='horizontal')
plt.savefig(os.path.join(test_dir, 'test_frame.png'))
plt.close()

subplots = int(np.sqrt(len(psf_basis)) + 1)
plt.figure(figsize=(16, 16))
for i in range(len(psf_basis)):
    plt.subplot(subplots, subplots, i+1);
    plt.imshow(psf_basis[i], interpolation='none')
    plt.colorbar(orientation='horizontal')
plt.savefig(os.path.join(test_dir, 'psf_basis.png'))
plt.close()

x, y = np.mgrid[:sim.imagedata.shape[0], :sim.imagedata.shape[1]]
plt.figure(figsize=(16, 16))
for i in range(len(a_fields)):
    plt.subplot(subplots, subplots, i+1);
    plt.imshow(a_fields[i](x, y))
    plt.colorbar(orientation='horizontal')
plt.savefig(os.path.join(test_dir, 'a_fields.png'))
plt.close()





