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

from imsim import simtools
import propercoadd as pc

# =============================================================================
#     PSF measure test by propercoadd
# =============================================================================

frames = []
for theta in [0, 45, 105, 150]:
    N = 512  # side
    X_FWHM = 6 + 12*theta/180
    Y_FWHM = 9
    t_exp = 1
    max_fw = max(X_FWHM, Y_FWHM)
    test_dir = os.path.abspath('./test_images/psf_basis_kl')

    x = np.linspace(6*max_fw, N-6*max_fw, 7)
    y = np.linspace(6*max_fw, N-6*max_fw, 7)
    xy = simtools.cartesian_product([x, y])


    SN =  1000. # SN para poder medir psf
    weights = list(np.linspace(10, 100, len(xy)))
    m = simtools.delta_point(N, center=False, xy=xy, weights=weights)
    im = simtools.image(m, N, t_exp, X_FWHM, Y_FWHM=Y_FWHM, theta=theta,
                        SN=SN, bkg_pdf='poisson')
    frames.append(im)

frame = np.zeros((1024, 1024))
for j in range(2):
    for i in range(2):
        frame[i*512:(i+1)*512, j*512:(j+1)*512] = frames[i+2*j]



sim = pc.SingleImage(frame)

fitted_models = sim.fit_psf_sep()

# =============================================================================
#    PSF spatially variant
# =============================================================================
from astropy.modeling import models
from astropy.modeling import fitting


x_sds = np.zeros(len(fitted_models))
y_sds = np.zeros(len(fitted_models))
x = np.zeros(len(fitted_models))
y = np.zeros(len(fitted_models))
th = np.zeros(len(fitted_models))
amplitudes = np.zeros(len(fitted_models))

i = 0
for g in fitted_models:
    x_sds[i] = g.x_stddev.value
    y_sds[i] = g.y_stddev.value
    amplitudes[i] = g.amplitude.value
    th[i] = g.theta*180/np.pi
    x[i] = round(g.x_mean.value - 0.5)
    y[i] = round(g.y_mean.value - 0.5)
    i += 1

fwhm_x = 2.335*np.mean(x_sds)
fwhm_y = 2.335*np.mean(y_sds)
fwhm = max(fwhm_x, fwhm_y)
mean_th = round(np.mean(th))

print 'X Fwhm = {}, Y Fwhm = {}, Mean Theta = {}'.format(fwhm_x, fwhm_y, mean_th)

psf_basis = sim._kl_from_stars
a_fields = sim._kl_a_fields

plt.imshow(frame)
plt.colorbar()
plt.savefig(os.path.join(test_dir, 'test_frame.png'))
plt.close()


plt.figure(figsize=(18, 6))
for i in range(len(psf_basis)):
    plt.subplot(1, len(psf_basis), i+1);
    plt.imshow(psf_basis[i], interpolation='none')
    plt.colorbar()
plt.savefig(os.path.join(test_dir, 'psf_basis.png'))
plt.close()

x, y = np.mgrid[:1024, :1024]
plt.figure(figsize=(18, 6))
for i in range(len(a_fields)):
    plt.subplot(1, len(psf_basis), i+1);
    plt.imshow(a_fields[i](x, y))
    plt.colorbar()
plt.savefig(os.path.join(test_dir, 'a_fields.png'))
plt.close()





