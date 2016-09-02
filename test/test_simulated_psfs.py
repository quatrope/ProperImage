#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  test_simulated_psfs.py
#
#  Copyright 2016 Bruno S <bruno@oac.unc.edu.ar>
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
import numpy as np

from astropy.io import fits
from astropy.time import Time
from properimage import numpydb as npdb
from properimage import simtools as sm
from properimage import propercoadd as pc

filenames = []
N = 1024  # side

test_dir = os.path.abspath('./test_images/several_stars')
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

now = '2016-05-17T00:00:00.1234567'
t = Time(now)

x = np.linspace(30, 900, 10)
y = np.linspace(30, 900, 10)
xy = sm.cartesian_product([x, y])
SN = 30
weights = list(np.linspace(1000, 200000, len(xy)))

for xfwhm in [3, 6]:
    for yfwhm in [2, 3, 7]:
        for theta in [10, 50, 90, 130]:
            m = sm.delta_point(N, center=False, xy=xy, weights=weights)
            im = sm.image(m, N, t_exp=1, X_FWHM=xfwhm, Y_FWHM=yfwhm,
                                theta=theta, SN=SN, bkg_pdf='poisson')
            img_dir = os.path.join(test_dir, str(xfwhm)+'_'+str(yfwhm)+'_'+
                                   str(theta))
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)

            for i in range(6):
                filenames.append(sm.capsule_corp(im, t, t_exp=1, i=i,
                                zero=3.1415, path=img_dir))


for root, dirs, files in os.walk(test_dir):
    fs = [os.path.join(root, afile) for afile in files]
    if len(fs) is 0: continue
    print 'files to process: {}'.format(fs)
    ensemble = pc.ImageEnsemble(fs)
    #S = ensemble.calculate_S(n_procs=4)
    R, S = ensemble.calculate_R(n_procs=4, return_S=True)

    if isinstance(S, np.ma.masked_array):
        S = S.filled(1.)

    if isinstance(R, np.ma.masked_array):
        R = R.real.filled(1.)

    shdu = fits.PrimaryHDU(S)
    shdulist = fits.HDUList([shdu])
    shdulist.writeto(os.path.join(root,'S.fits'), clobber=True)

    rhdu = fits.PrimaryHDU(R)
    rhdulist = fits.HDUList([rhdu])
    rhdulist.writeto(os.path.join(root,'R.fits'), clobber=True)




