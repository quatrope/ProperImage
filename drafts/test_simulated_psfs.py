#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  test_simulated_psfs.py
#
#  Copyright 2016 Bruno S <bruno@oac.unc.edu.ar>
#
# This file is part of ProperImage (https://github.com/toros-astro/ProperImage)
# License: BSD-3-Clause
# Full Text: https://github.com/toros-astro/ProperImage/blob/master/LICENSE.txt
#
import os
import shlex
import subprocess
import numpy as np

from astropy.io import fits
from astropy.time import Time
from properimage import numpydb as npdb
from properimage import simtools as sm
from properimage import propercoadd as pc


N = 1024  # side

test_dir = os.path.abspath('./test/test_images/several_stars_gramschmidt')
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

now = '2016-05-17T00:00:00.1234567'
t = Time(now)

SN = 1
weights = np.random.random(100)*20000 + 10

for xfwhm in [4, 5, 6]:
    for yfwhm in [2, 3, 7]:
        for theta in [10, 50, 90, 130]:
            filenames = []
            x = np.random.randint(low=30, high=900, size=100)
            y = np.random.randint(low=30, high=900, size=100)

            xy = [(x[i], y[i]) for i in range(100)]
            m = sm.delta_point(N, center=False, xy=xy, weights=weights)

            img_dir = os.path.join(test_dir, str(xfwhm)+'_'+str(yfwhm)+'_'+
                                   str(theta))
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)

            for i in range(12):
                im = sm.image(m, N, t_exp=1, X_FWHM=xfwhm, Y_FWHM=yfwhm,
                              theta=theta, SN=SN, bkg_pdf='poisson')
                filenames.append(sm.capsule_corp(im, t, t_exp=1, i=i,
                                zero=3.1415, path=img_dir))

            cmd = ' '
            for files in filenames:
                cmd += ' ' + files
            coadd = os.path.join(img_dir, 'coadd.fits')
            swarp = shlex.split('swarp '+cmd+' -IMAGEOUT_NAME '+coadd)
            subprocess.call(swarp)

            with pc.ImageEnsemble(filenames) as ensemble:
                # S = ensemble.calculate_S(n_procs=4)
                R, S = ensemble.calculate_R(n_procs=4, return_S=True)

                if isinstance(S, np.ma.masked_array):
                    S = S.filled(1.)

                if isinstance(R, np.ma.masked_array):
                    R = R.real.filled(1.)

                shdu = fits.PrimaryHDU(S)
                shdulist = fits.HDUList([shdu])
                shdulist.writeto(os.path.join(img_dir,'S.fits'), clobber=True)

                rhdu = fits.PrimaryHDU(R.real)
                rhdulist = fits.HDUList([rhdu])
                rhdulist.writeto(os.path.join(img_dir,'R.fits'), clobber=True)
