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

from properimage import propercoadd as pc
from properimage import utils


reload(utils)
reload(pc)
# =============================================================================
#     PSF measure test by propercoadd
# =============================================================================
datapath = os.path.abspath('/home/bruno/Documentos/Data/ESO085-030')

frame = os.path.join(datapath, 'eso085-030-030.fit')

d = fits.getdata(frame)

utils.plot_S(d, path='./test/test_images/real_image_test/frame.png')

#fitted_models = sim.fit_psf_sep()

# =============================================================================
#    PSF spatially variant
# =============================================================================


with pc.SingleImage(frame, imagefile=True) as sim:
    a_f, psf_b = sim.get_variable_psf(pow_th=0.01)
    S = sim.s_component

utils.plot_psfbasis(psf_b,
                    path='./test/test_images/real_image_test/psf_basis.png')

utils.plot_afields(a_f, S.shape,
                   path='./test/test_images/real_image_test/a_fields.png')

utils.plot_S(S, path='./test/test_images/real_image_test/S.png')

