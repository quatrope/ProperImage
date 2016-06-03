#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  test_recoverstats.py
#
#  Copyright 2016 Bruno S <bruno.sanchez.63@gmail.com>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
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


N = 512  # side
FWHM = 12
test_dir = os.path.abspath('./test_images/measure_psf')

x = np.linspace(5*FWHM, N-5*FWHM, 3)
y = np.linspace(5*FWHM, N-5*FWHM, 3)
xy = simtools.cartesian_product([x, y])


SN =  100. # SN para poder medir psf
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
best_srcs = srcs[srcs['flag'] == 0]
fitshape = (4*FWHM, 4*FWHM)
prf_model.x_mean = fitshape[0]/2.
prf_model.y_mean = fitshape[1]/2.

for row in best_srcs:
    position = (row['y'], row['x'])
    y = extract_array(indices[0], fitshape, position)
    x = extract_array(indices[1], fitshape, position)
    sub_array_data = extract_array(sim.bkg_sub_img,
                                    fitshape, position,
                                    fill_value=sim.bkg.globalback)
    prf_model.x_mean = position[1]
    prf_model.y_mean = position[0]
    fit = fitter(prf_model, x, y, sub_array_data)
    print fit
    model_fits.append(fit)
    plt.subplot(131)
    plt.imshow(fit(x, y))
    plt.title('fit')
    plt.subplot(132)
    plt.title('sub_array')
    plt.imshow(sub_array_data)
    plt.subplot(133)
    plt.title('residual')
    plt.imshow(sub_array_data - fit(x,y))
    plt.show()

## Things are running somewhat like expected

# Again and simpler
N = 128  # side
FWHM = 6
SN =  100. # SN para poder medir psf
m = simtools.delta_point(N, center=False, xy=[[50, 64]])
im = simtools.image(m, N, t_exp=1, FWHM=FWHM, SN=SN, bkg_pdf='gaussian')

fitshape = (64,64)#(6*FWHM, 6*FWHM)
prf_model = models.Gaussian2D(x_stddev=3, y_stddev=3,
                                x_mean=fitshape[0], y_mean=fitshape[1])
fitter = fitting.LevMarLSQFitter()

indices = np.indices(im)
position = (50, 64)
prf_model.y_mean, prf_model.x_mean = position
y = extract_array(indices[0], fitshape, position)
x = extract_array(indices[1], fitshape, position)
sub_array_data = extract_array(im, fitshape, position)

fit = fitter(prf_model, x, y, sub_array_data)

print fit

plt.subplot(221)
plt.imshow(im)
plt.subplot(222)
plt.imshow(fit(x, y))
plt.title('fit')
plt.subplot(223)
plt.imshow(sub_array_data)
plt.subplot(224)
plt.imshow(sub_array_data - fit(x,y))
plt.show()

print 'hola'
