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
N = 512  # side
X_FWHM = 6
Y_FWHM = 7
theta = 78
t_exp = 1
max_fw = max(X_FWHM, Y_FWHM)
test_dir = os.path.abspath('./test_images/measure_psf')

x = np.linspace(6*max_fw, N-6*max_fw, 7)
y = np.linspace(6*max_fw, N-6*max_fw, 7)
xy = simtools.cartesian_product([x, y])


SN =  1000. # SN para poder medir psf
weights = list(np.linspace(10, 100, len(xy)))
m = simtools.delta_point(N, center=False, xy=xy, weights=weights)
im = simtools.image(m, N, t_exp, X_FWHM, Y_FWHM=Y_FWHM, theta=theta,
                    SN=SN, bkg_pdf='poisson')



sim = pc.SingleImage(im)
sim.subtract_back()
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
amplitudes = np.zeros(len(fitted_models))

i = 0
for g in fitted_models:
    x_sds[i] = g.x_stddev.value
    y_sds[i] = g.y_stddev.value
    amplitudes[i] = g.amplitude.value
    x[i] = round(g.x_mean.value - 0.5)
    y[i] = round(g.y_mean.value - 0.5)
    i += 1

fwhm_x = 2.335*np.mean(x_sds)
fwhm_y = 2.335*np.mean(y_sds)
fwhm = max(fwhm_x, fwhm_y)

print fwhm_x, fwhm_y


xpol_model = models.Polynomial2D(degree=3)
ypol_model = models.Polynomial2D(degree=3)

fit_p = fitting.LevMarLSQFitter()










