#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  test_propersubtract.py
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
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
import numpy as np

from properimage import propercoadd as pc
from properimage import simtools as sm
from properimage import utils
from properimage import propersubtract as ps

reload(utils)

N = 1024  # side

test_dir = os.path.abspath('./test/test_images/test_zpoint')
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

now = '2016-05-17T00:00:00.1234567'
t = Time(now)

SN = 5.
theta = 0
xfwhm = 4
yfwhm = 3
weights = np.random.random(100)*20000 + 10

zero = 10  #for zero in [5, 10, 25]:
filenames = []
x = np.random.randint(low=30, high=900, size=100)
y = np.random.randint(low=30, high=900, size=100)

xy = [(x[i], y[i]) for i in range(100)]
m = sm.delta_point(N, center=False, xy=xy, weights=weights)

img_dir = os.path.join(test_dir, 'zp={}'.format(zero))
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

for i in range(50):
    im = sm.image(m, N, t_exp=i+1, X_FWHM=xfwhm, Y_FWHM=yfwhm,
                  theta=theta, SN=SN, bkg_pdf='poisson')
    filenames.append(sm.capsule_corp(im, t, t_exp=i+1, i=i,
                    zero=zero+i, path=img_dir))

ensemble = pc.ImageEnsemble(filenames)

zps, meanmags = utils.transparency(ensemble)
S_hat_stack, S_stack, S_hat, S, R_hat = ensemble.calculate_R(n_procs=6, debug=True)

ensemble._clean()

    #~ with pc.ImageEnsemble(filenames) as ensemble:
        #~ zp = utils.transparency(ensemble)
        # S = ensemble.calculate_S(n_procs=4)
        #~ R, S = ensemble.calculate_R(n_procs=4, return_S=True)

        #~ if isinstance(S, np.ma.masked_array):
            #~ S = S.filled(1.)

        #~ if isinstance(R, np.ma.masked_array):
            #~ R = R.real.filled(1.)

        #~ shdu = fits.PrimaryHDU(S)
        #~ shdulist = fits.HDUList([shdu])
        #~ shdulist.writeto(os.path.join(img_dir,'S.fits'), clobber=True)

        #~ rhdu = fits.PrimaryHDU(R.real)
        #~ rhdulist = fits.HDUList([rhdu])
        #~ rhdulist.writeto(os.path.join(img_dir,'R.fits'), clobber=True)
