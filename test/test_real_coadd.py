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

sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits

from propercoadd import propercoadd as pc


# =============================================================================
#     PSF measure test by propercoadd
# =============================================================================
#datapath = os.path.abspath('/home/bos0109/DATA/Data/Tolar2015/CAMPAÑA_LIGO_OBS
#ERVACIONES_MACON/20151212/preprocessed/Landolt_C53')

datapath = os.path.abspath('/home/bruno/Documentos/reduccionTolar/20151212/subset/')

S = np.zeros((1365, 1365))

for root, dirs, files in os.walk(datapath):
    fs = [os.path.join(root, afile) for afile in files]
    ensemble = pc.ImageEnsemble(fs)
    S = ensemble.calculate_S(n_procs=4)

test_dir = os.path.abspath('./test_images/real_coadd_test/')

if not os.path.exists(test_dir):
    os.mkdir(test_dir)

plt.figure(figsize=(16,16))
plt.imshow(np.log10(S), interpolation='none')
plt.colorbar(orientation='horizontal')
plt.savefig(os.path.join(test_dir, 'S.png'))
plt.close()

with file(os.path.join(test_dir,'S.npy'), 'w') as f:
    np.save(f, S)


def fftwn(array, nthreads=4):
    array = array.astype('complex').copy()
    outarray = array.copy()
    fft_forward = fftw3.Plan(array, outarray, direction='forward',
            flags=['estimate'], nthreads=nthreads)
    fft_forward.execute()
    return outarray

def ifftwn(array, nthreads=4):
    array = array.astype('complex').copy()
    outarray = array.copy()
    fft_backward = fftw3.Plan(array, outarray, direction='backward',
            flags=['estimate'], nthreads=nthreads)
    fft_backward.execute()
    return outarray / np.size(array)
