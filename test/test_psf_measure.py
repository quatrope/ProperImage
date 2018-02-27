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
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import stats
import sep

from astropy.convolution import convolve
from astropy.convolution import convolve_fft
from astropy.time import Time
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.stats import signal_to_noise_oir_ccd
from astropy.table import Table
from astropy.modeling import fitting
from astropy.modeling import models
from astropy.nddata.utils import extract_array

from photutils import psf
from photutils import daofind

from properimage import simtools
from properimage import propercoadd as pc

# =============================================================================
#     PSF measure test by propercoadd
# =============================================================================
N = 512  # side
X_FWHM = 6
Y_FWHM = 7
theta = 78
t_exp = 1
max_fw = max(X_FWHM, Y_FWHM)
test_dir = os.path.abspath('../test_images/measure_psf')

x = np.linspace(6*max_fw, N-6*max_fw, 7)
y = np.linspace(6*max_fw, N-6*max_fw, 7)
xy = simtools.cartesian_product([x, y])


SN =  1000. # SN para poder medir psf
weights = list(np.linspace(10, 100, len(xy)))
m = simtools.delta_point(N, center=False, xy=xy, weights=weights)
im = simtools.image(m, N, t_exp, X_FWHM, SN,
                    Y_FWHM=Y_FWHM, theta=theta, bkg_pdf='poisson')

sim = pc.SingleImage(im, imagefile=False, sim=True)

fitted_models = sim.fit_psf_sep()

x_sds = [g.x_stddev for g in fitted_models]
y_sds = [g.y_stddev for g in fitted_models]
th = [g.theta*180/np.pi for g in fitted_models]
amplitudes = [g.amplitude for g in fitted_models]

fwhm_x = 2.335*np.mean(x_sds)
fwhm_y = 2.335*np.mean(y_sds)
mean_th = round(np.mean(th))
fwhm = max(fwhm_x, fwhm_y)

print('X Fwhm = {}, Y Fwhm = {}, Mean Theta = {}'.format(fwhm_x, fwhm_y, mean_th))

# =============================================================================
#    PSF spatially variant
# =============================================================================
covMat = np.zeros(shape=(len(fitted_models), len(fitted_models)))

renders = [g.render() for g in fitted_models]

for i in range(len(fitted_models)):
    for j in range(len(fitted_models)):
        if i<=j:
            psfi_render = renders[i]
            psfj_render = renders[j]

            inner = np.vdot(psfi_render.flatten()/np.sum(psfi_render),
                            psfj_render.flatten()/np.sum(psfj_render))

            covMat[i, j] = inner
            covMat[j, i] = inner
import ipdb; ipdb.set_trace()

valh, vech = np.linalg.eigh(covMat)

power = valh/np.sum(abs(valh))
cum = 0
cut = 0
while cum < 0.90:
    cut -= 1
    cum += abs(power[cut])

#  Build psf basis
N_psf_basis = abs(cut)
lambdas = valh[cut:]
xs = vech[:,cut:]
psf_basis = []
for i in range(N_psf_basis):
    psf_basis.append(np.tensordot(xs[:,i], renders, axes=[0,0]))



# =============================================================================
#       Manual test
# =============================================================================

runtest = False#input('Run Manual test?')
if runtest:
    prf_model = models.Gaussian2D(x_stddev=1, y_stddev=1)
    fitter = fitting.LevMarLSQFitter()
    indices = np.indices(sim.bkg_sub_img.shape)
    model_fits = []
    best_big = srcs['tnpix']>=p_sizes[0]**2.
    best_small = srcs['tnpix']<=p_sizes[2]**2.
    best_flag = srcs['flag']<31
    best_srcs = srcs[ best_big & best_flag & best_small]
    fitshape = (4*FWHM, 4*FWHM)
    prf_model.x_mean = fitshape[0]/2.
    prf_model.y_mean = fitshape[1]/2.

    for row in best_srcs:
        position = (row['y'], row['x'])
        y = extract_array(indices[0], fitshape, position)
        x = extract_array(indices[1], fitshape, position)
        sub_array_data = extract_array(sim.bkg_sub_img,
                                        fitshape, position,
                                        fill_value=sim.bkg.globalrms)
        prf_model.x_mean = position[1]
        prf_model.y_mean = position[0]
        fit = fitter(prf_model, x, y, sub_array_data)
        print(row['x'],row['y'],row['flux'],row['tnpix'],row['a'],row['b'])
        print(fit)
        res = sub_array_data - fit(x,y)

        if np.sum(res*res) < sim.bkg.globalrms*fitshape[0]**2:
            model_fits.append(fit)
            plt.subplot(131)
            plt.imshow(fit(x, y), interpolation='none')
            plt.title('fit')
            plt.subplot(132)
            plt.title('sub_array')
            plt.imshow(sub_array_data, interpolation='none')
            plt.subplot(133)
            plt.title('residual')
            plt.imshow(sub_array_data - fit(x,y), interpolation='none')
            plt.show()
            continue_loop = input('continue loop?')
            if not continue_loop: break
