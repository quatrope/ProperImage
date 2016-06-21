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
datapath = os.path.abspath('/home/bos0109/DATA/Data/Tolar2015/CAMPAÑA_LIGO_OBSERVACIONES_MACON/20151212/preprocessed/Landolt_C53')

S = np.zeros((1365, 1365))

for root, dirs, files in os.walk(datapath):
    for afile in files:
        frame = os.path.join(root, afile)
        sim = pc.SingleImage(frame, imagefile=True)
        s_comp = sim.s_component()

        S += s_comp

test_dir = os.path.abspath('./test_images/real_coadd_test/')

if not os.path.exists(test_dir):
    os.mkdir(test_dir)

plt.figure(figsize=(16,16))
plt.imshow(np.log10(S), interpolation='none')
plt.colorbar(orientation='horizontal')
plt.savefig(os.path.join(test_dir, 'S.png'))
plt.close()



