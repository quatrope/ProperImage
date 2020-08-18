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


# =============================================================================
#     PSF measure test by propercoadd
# =============================================================================
#datapath = os.path.abspath('/home/bos0109/DATA/Data/Tolar2015/CAMPAÑA_LIGO_OBS
#ERVACIONES_MACON/20151212/preprocessed/Landolt_C53')

datapath = os.path.abspath(
           '/home/bruno/Documentos/Data/ESO085-030')

for root, dirs, files in os.walk(datapath):
    fs = [os.path.join(root, afile) for afile in files]
    print('files to process: {}'.format(fs))

    with pc.ImageEnsemble(fs, pow_th=0.01) as ensemble:
        R, S = ensemble.calculate_R(n_procs=4, return_S=True)


test_dir = os.path.abspath('./test/test_images/real_coadd_test/')

if not os.path.exists(test_dir):
    os.mkdir(test_dir)


utils.plot_S(S, path=os.path.join(test_dir,'S.png'))

utils.plot_R(R, path=os.path.join(test_dir,'R.png'))

utils.encapsule_S(S, path=os.path.join(test_dir,'S.fits'))

utils.encapsule_R(R, path=os.path.join(test_dir,'R.fits'))
