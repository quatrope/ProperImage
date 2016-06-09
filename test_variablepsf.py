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
N = 1024  # side
FWHM = 8
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
fitted_models = sim.fit_psf_sep()

# =============================================================================
#    PSF spatially variant
# =============================================================================
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

#renders = [g.render() for g in fitted_models]
