# -*- coding: utf-8 -*-
"""
Program that works simulating images with different SN and performs
the calculation of S statistic image from Ofek&Zackay2015a

This program doesn't calculate F_j, nor estimates psf or
background from images.
Created on Fri May 13 17:06:14 2016

@author: bruno
"""
import os
import shlex
import subprocess
import sys

#~ sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import stats
from astropy.convolution import convolve, convolve_fft
from astropy.time import Time
from astropy.io import fits

from properimage import simtools
from properimage import propercoadd as pc


N = 1024  # side
FWHM = 10
test_dir = os.path.abspath('./test/test_images/one_star')

x = np.linspace(5*FWHM, N-5*FWHM, 10)
y = np.linspace(5*FWHM, N-5*FWHM, 10)
xy = simtools.cartesian_product([x, y])

now = '2016-05-17T00:00:00.1234567'
t = Time(now)

filenames = []
for i in range(100):
    SN =  2. # + i
    weights = list(np.linspace(0.00001, 1, len(xy)))
    m = simtools.delta_point(N, center=False, xy=xy, weights=weights)
    im = simtools.image(m, N, t_exp=1, X_FWHM=FWHM, SN=SN, bkg_pdf='gaussian')

    filenames.append(
        simtools.capsule_corp(im, t, t_exp=1, i=i, zero=3.1415, path=test_dir))

S = np.zeros(shape=(N,N))
psf = simtools.Psf(N, FWHM)
for afile in filenames:
    px = pc.SingleImage(afile)
    # Now I must combine this images, normalizing by the var(noise)
    var = px.meta['std']
    conv = convolve_fft(px.imagedata, psf)
    S = S + conv/var**2.

fits.writeto(data=S, filename='./test_images/S.fits', overwrite=True)

swarp = shlex.split('swarp @test_images/one_star/imagelist.txt -c default.swarp')
subprocess.call(swarp)

plt.figure()
plt.subplot(121)
plt.imshow(S, interpolation='none')
plt.title('Ofek')
plt.subplot(122)
sw = fits.getdata('./coadd.fits')
plt.imshow(sw, interpolation='none')
plt.title('SWarp')
plt.savefig('./test_images/one_star/coadds.png')

plt.figure(figsize=(14,6))
plt.subplot(121)
plt.hist(S.flatten(), log=True)
plt.title('Ofek')
plt.subplot(122)
sw = fits.getdata('./coadd.fits')
plt.hist(sw.flatten(), log=True)
plt.title('SWarp')
plt.savefig('./test_images/one_star/coadds_hist.png')

