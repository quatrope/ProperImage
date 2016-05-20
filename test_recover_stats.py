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

from astropy.convolution import convolve
from astropy.convolution import convolve_fft
from astropy.time import Time
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.stats import signal_to_noise_oir_ccd

from imsim import simtools
import propercoadd as pc


N = 1024  # side
FWHM = 10
test_dir = os.path.abspath('./test_images/recover_stats')

x = np.linspace(5*FWHM, N-5*FWHM, 10)
y = np.linspace(5*FWHM, N-5*FWHM, 10)
xy = simtools.cartesian_product([x, y])

now = '2016-05-17T00:00:00.1234567'
t = Time(now)

filenames = []

for i in range(30):
    SN =  2. #+ i
    weights = list(np.linspace(1, 10000, len(xy)))
    m = simtools.delta_point(N, center=False, xy=xy, weights=weights)
    im = simtools.image(m, N, t_exp=1, FWHM=FWHM, SN=SN, bkg_pdf='poisson')

    filenames.append(
        simtools.capsule_corp(im, t, t_exp=1, i=int(i).zfill(2), zero=3.1415,
            path=test_dir))


