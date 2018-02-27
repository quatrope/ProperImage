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

from astropy.io import fits

from properimage import single_image as si
from properimage import plot

# =============================================================================
#     PSF measure test by propercoadd
# =============================================================================
datapath = os.path.abspath('/home/bruno/Data/Tolar2017/20170624/NGC_6544')

frame = os.path.join(datapath, 'NGC_6544-010.fit')

d = fits.getdata(frame)

plot.plot_S(d, path='./test/test_images/real_image_test/frame.png')

# =============================================================================
#    PSF spatially variant
# =============================================================================


with si.SingleImage(frame) as sim:
    a_f, psf_b = sim.get_variable_psf(inf_loss=0.1)
    x, y = sim.get_afield_domain()
    normal_image = sim.normal_image
    interp = sim.interped
    print((sim.n_sources))
    S = sim.s_component

plot.plot_psfbasis(psf_b,
                    path='./test/test_images/real_image_test/psf_basis.png')

plot.plot_afields(a_f, x, y,
                   path='./test/test_images/real_image_test/a_fields.png')

plot.plot_S(S, path='./test/test_images/real_image_test/S.png')
plot.plot_S(interp, path='./test/test_images/real_image_test/interped.png')
plot.plot_S(normal_image, path='./test/test_images/real_image_test/norm.png')

