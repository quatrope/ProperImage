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
from scipy.stats import stats
import sep

from astropy.convolution import convolve
from astropy.convolution import convolve_fft
from astropy.time import Time
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.stats import signal_to_noise_oir_ccd
from astropy.table import Table

from photutils import psf
from photutils import daofind

from imsim import simtools
import propercoadd as pc


def residuals_psf_sub_table(outtab, title=None):
    plt.subplot(3,1,1)
    if title is not None:
        plt.title(title)
    plt.scatter(np.arange(len(outtab))+1,
        (outtab['flux_fit']-outtab['flux_0'])*100./outtab['flux_0'])
    plt.axhline(0, ls='--', c='k')
    plt.ylabel('fluxperc')
    plt.xlim(0.5, len(outtab)+.5)

    plt.subplot(3,1,2)
    plt.scatter(np.arange(len(outtab))+1, outtab['x_fit']-outtab['x_0'])
    plt.axhline(0, ls='--', c='k')
    plt.ylabel('dx')
    plt.xlim(0.5, len(outtab)+.5)

    plt.subplot(3,1,3)
    plt.scatter(np.arange(len(outtab))+1, outtab['y_fit']-outtab['y_0'])
    plt.axhline(0, ls='--', c='k')
    plt.ylabel('dy')
    plt.xlim(0.5, len(outtab)+.5)

    plt.tight_layout()
    plt.show()


def compare_psf_sub(subim, pim, im, kw1s={}, kw2s={}, kw3s={}, kw4s={}):
    subps = (2, 2)
    cborient = 'vertical'

    plt.subplot(2,2,1)
    plt.imshow(pim, **kw1s)
    plt.colorbar(orientation=cborient)
    plt.title('Base image')

    plt.subplot(2,2,2)
    plt.imshow(subim, **kw2s)
    plt.colorbar(orientation=cborient)
    plt.title('PSF subtracted image')
    #print("Subtracted image bkg-sub mean:", np.mean(subim-bkg), 'and SD:', np.std(subim-bkg))

    plt.subplot(2,2,3)
    plt.imshow(im, **kw3s)
    plt.colorbar(orientation=cborient)
    plt.title('Real noise-free images')

    plt.subplot(2,2,4)
    plt.imshow(pim-subim, **kw4s)
    plt.colorbar(orientation=cborient)
    plt.title('PSF images')
    plt.show()


N = 1024  # side
FWHM = 10
test_dir = os.path.abspath('./test_images/recover_stats')

x = np.linspace(5*FWHM, N-5*FWHM, 10)
y = np.linspace(5*FWHM, N-5*FWHM, 10)
xy = simtools.cartesian_product([x, y])

t = Time.now()


SN =  100. # SN para poder medir psf
weights = list(np.linspace(1, 10000, len(xy)))
m = simtools.delta_point(N, center=False, xy=xy, weights=weights)
im = simtools.image(m, N, t_exp=1, FWHM=FWHM, SN=SN, bkg_pdf='poisson')

sim = pc.SingleImage(im)
sim.subtract_back()

srcs = sep.extract(sim.bkg_sub_img, thresh=12*sim.bkg.globalrms)
posflux = srcs[['x','y', 'flux']]

psf_guess = psf.IntegratedGaussianPRF(flux=1, sigma=8)

psf_guess.flux.fixed = False
psf_guess.x_0.fixed = False
psf_guess.y_0.fixed = False
psf_guess.x_0.sigma = True

fitshape = (64,64)
intab = Table(names=['x_0', 'y_0', 'flux_0'], data=posflux)
#subimi = psf.subtract_psf(sim.bkg_sub_img, psf_guess, posflux)

outtabi = psf.psf_photometry(sim.bkg_sub_img, intab, psf_guess, fitshape,
    store_fit_info=True)
outtabi['flux_input'] = intab['flux_0']

# with daofind there are lots of garbage
found = daofind(sim.bkg_sub_img, threshold=5*sim.bkg.globalrms, fwhm=10,
    exclude_border=True)
intab2 = Table(names=['x_0', 'y_0', 'flux_0'], data=[found['xcentroid'],
    found['ycentroid'], found['flux']])
outtabi2 = psf.psf_photometry(sim.bkg_sub_img, intab2, psf_guess, fitshape,
    store_fit_info=True)
outtabi2['flux_input'] = intab2['flux_0']



