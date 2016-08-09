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
from astropy.time import Time
from astropy.io import fits

from imsim import simtools
from propercoadd import propercoadd as pc

# =============================================================================
#     PSF measure test by propercoadd
# =============================================================================
test_dir = os.path.abspath('/home/bruno/Devel/zackay_code/properimage/test/test_images/varpsf_coadd')
filenames = []
for k in range(10):
    frames = []
    for theta in [0, 45, 105, 150]:
        N = 256  # side
        X_FWHM = 6 + 12*theta/180
        Y_FWHM = 9
        t_exp = 1
        max_fw = max(X_FWHM, Y_FWHM)

        x = np.linspace(6*max_fw, N-6*max_fw, 4)
        y = np.linspace(6*max_fw, N-6*max_fw, 4)
        xy = simtools.cartesian_product([x, y])

        SN =  100. # SN para poder medir psf
        weights = list(np.linspace(10, 100, len(xy)))
        m = simtools.delta_point(N, center=False, xy=xy, weights=weights)
        im = simtools.image(m, N, t_exp, X_FWHM, Y_FWHM=Y_FWHM, theta=theta,
                            SN=SN, bkg_pdf='poisson')
        frames.append(im)

    frame = np.zeros((2*N, 2*N))
    for j in range(2):
        for i in range(2):
            frame[i*N:(i+1)*N, j*N:(j+1)*N] = frames[i+2*j]
    now = '2016-05-17T00:00:00.1234567'
    t = Time(now)
    filenames.append(
        simtools.capsule_corp(frame, t, t_exp=1, i=k,
                              zero=3.1415, path=test_dir))
# =============================================================================
#   Coadd
# =============================================================================

S = np.zeros((N, N))
R = np.zeros((N, N))

ensemble = pc.ImageEnsemble(filenames)
#S = ensemble.calculate_S(n_procs=4)
R, S = ensemble.calculate_R(n_procs=4, return_S=True)

test_dir = os.path.join(test_dir, 'coadd')

#~ if isinstance(S, np.ma.masked_array):
    #~ S = S.filled(1.)

#~ if isinstance(R, np.ma.masked_array):
    #~ R = R.real.filled(1.)

if not os.path.exists(test_dir):
    os.mkdir(test_dir)

#~ with file(os.path.join(test_dir,'S.npy'), 'w') as f:
    #~ np.save(f, S)

#~ with file(os.path.join(test_dir,'R.npy'), 'w') as f:
    #~ np.save(f, R)

plt.figure(figsize=(16,16))
plt.imshow(np.log10(S), interpolation='none')
plt.colorbar(orientation='horizontal')
plt.savefig(os.path.join(test_dir, 'S.png'))
plt.close()

plt.figure(figsize=(16,16))
plt.imshow(np.log10(R), interpolation='none')
plt.colorbar(orientation='horizontal')
plt.savefig(os.path.join(test_dir, 'R.png'))
plt.close()

shdu = fits.PrimaryHDU(S)
shdulist = fits.HDUList([shdu])
shdulist.writeto(os.path.join(test_dir,'S.fits'), clobber=True)

rhdu = fits.PrimaryHDU(R)
rhdulist = fits.HDUList([rhdu])
rhdulist.writeto(os.path.join(test_dir,'R.fits'), clobber=True)



