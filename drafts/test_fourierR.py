#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  test_propersubtract.py
#
#  Copyright 2016 Bruno S <bruno@oac.unc.edu.ar>
#
# This file is part of ProperImage (https://github.com/toros-astro/ProperImage)
# License: BSD-3-Clause
# Full Text: https://github.com/toros-astro/ProperImage/blob/master/LICENSE.txt
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

test_dir = os.path.abspath('./test/test_images/test_fourierR')
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

now = '2016-05-17T00:00:00.1234567'
t = Time(now)

SN = 5.
theta = 0
xfwhm = 4
yfwhm = 3
weights = np.random.random(25)*20000 + 10

zero = 10  #for zero in [5, 10, 25]:
filenames = []
x = np.random.randint(low=30, high=900, size=25)
y = np.random.randint(low=30, high=900, size=25)

xy = [(x[i], y[i]) for i in range(25)]
m = sm.delta_point(N, center=False, xy=xy, weights=weights)

img_dir = os.path.join(test_dir, 'zp={}'.format(zero))
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

for i in range(40):
    im = sm.image(m, N, t_exp=i+1, X_FWHM=xfwhm, Y_FWHM=yfwhm,
                  theta=theta, SN=SN, bkg_pdf='poisson')
    filenames.append(sm.capsule_corp(im, t, t_exp=i+1, i=i,
                    zero=zero+i, path=img_dir))

ensemble = pc.ImageEnsemble(filenames)

S_hat_stack, S_stack, S_hat, S, R_hat = ensemble.calculate_R(n_procs=4,
                                                             debug=True)

S_inv = pc._ifftwn(S_hat)

d = S - S_inv

plt.hist(d.real.flatten(), log=True)
plt.title('Histograma diferencia pixeles fourier inverso')
plt.show()

plt.hist(d.imag.flatten(), log=True)
plt.title('Histograma diferencia pixeles fourier inverso')
plt.show()


comp1 = S_hat_stack[:, :, 0]
comp2 = S_hat_stack[:, :, 1]
comp3 = S_hat_stack[:, :, 2]
comp4 = S_hat_stack[:, :, 3]


s_hat_summed = np.ma.sum(S_hat_stack, axis=2)

d_hat = S_hat - s_hat_summed
plt.hist(d_hat.real.flatten(), log=True)
plt.show()
plt.hist(d_hat.imag.flatten(), log=True)
plt.show()

R = pc._ifftwn(R_hat)

fits.PrimaryHDU(R.real).writeto('R_.fits', clobber=True)

hat_std = np.ma.std(S_hat_stack, axis=2)
r_hat = np.ma.divide(S_hat, hat_std)

r = pc._ifftwn(r_hat)

fits.PrimaryHDU(r.real).writeto('r_.fits', clobber=True)

dr = R - r
plt.hist(dr.real.flatten(), log=True)
plt.show()
plt.hist(dr.imag.flatten(), log=True)
plt.show()


ensemble._clean()
