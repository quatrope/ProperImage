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

from imsim import simtools
import propercoadd as pc


N = 1024  # side
FWHM = 6
test_dir = os.path.abspath('./test_images/measure_psf')

x = np.linspace(6*FWHM, N-6*FWHM, 10)
y = np.linspace(6*FWHM, N-6*FWHM, 10)
xy = simtools.cartesian_product([x, y])


SN =  1000. # SN para poder medir psf
weights = list(np.linspace(10, 100, len(xy)))
m = simtools.delta_point(N, center=False, xy=xy, weights=weights)
im = simtools.image(m, N, t_exp=1, FWHM=FWHM, SN=SN, bkg_pdf='poisson')

sim = pc.SingleImage(im)
sim.subtract_back()

srcs = sep.extract(sim.bkg_sub_img, thresh=30*sim.bkg.globalrms)
posflux = srcs[['x','y', 'flux']]


fitted_models = sim.fit_psf_sep()


#Manual test

prf_model = models.Gaussian2D(x_stddev=1, y_stddev=1)
fitter = fitting.LevMarLSQFitter()
indices = np.indices(sim.bkg_sub_img.shape)
model_fits = []
best_srcs = srcs[srcs['flag'] < 31]
fitshape = (3*FWHM, 3*FWHM)
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
    print fit
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
        if not continue_loop:
            break
