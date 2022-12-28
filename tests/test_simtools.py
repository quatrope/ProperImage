#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  test_simtools.py
#
#  Copyright 2020 QuatroPe
#
# This file is part of ProperImage (https://github.com/quatrope/ProperImage)
# License: BSD-3-Clause
# Full Text: https://github.com/quatrope/ProperImage/blob/master/LICENSE.txt
#

"""
test_simtools module from ProperImage
for analysis of astronomical images

Written by Bruno SANCHEZ

PhD of Astromoy - UNC
bruno@oac.unc.edu.ar

Instituto de Astronomia Teorica y Experimental (IATE) UNC
Cordoba - Argentina

Of 301
"""

import numpy as np
from numpy.random import default_rng

from properimage import simtools as sm

from scipy.ndimage import rotate

from .core import ProperImageTestCase

random = default_rng(seed=42)


class TestSimulationSuite(ProperImageTestCase):
    def test_Psf_module(self):
        module = np.sum(sm.Psf(100, 15))
        self.assertAlmostEqual(module, 1.0)

    def test_Psf_asymmetrical(self):
        psf1 = sm.Psf(100, 30, 25)
        psf2 = sm.Psf(100, 25, 30)
        delta = (psf1 - rotate(psf2, 90)) ** 2
        self.assertLess(delta.sum(), 0.01)

    def test_Psf_rotated(self):
        psf1 = sm.Psf(100, 30, theta=45)
        psf2 = sm.Psf(100, 30)
        np.testing.assert_almost_equal(psf1, rotate(psf2, 45))

    def test_astropyPsf_module(self):
        module = np.sum(sm.astropy_Psf(100, 15))
        self.assertAlmostEqual(module, 1.0)

    def test_airy_patron(self):
        size = random.integers(8, 32)
        width = random.integers(1, size)
        pattern1 = sm.airy_patron(size, width)
        np.testing.assert_equal(size, pattern1.shape[0])
        np.testing.assert_equal(size, pattern1.shape[1])

    def test_gal_sersic(self):
        size = 256
        n = random.random() * 4.0
        gal = sm.gal_sersic(size, n)
        np.testing.assert_equal(size, gal.shape[0])
        np.testing.assert_equal(size, gal.shape[1])

    def test_convol_gal_psf_fft(self):
        pat_size = random.integers(4, 8) * 2
        width = random.integers(1, pat_size / 2)
        pattern1 = sm.airy_patron(pat_size, width)
        gal_size = 128
        n = random.random() * 4.0
        gal = sm.gal_sersic(gal_size, n)
        conv = sm.convol_gal_psf_fft(gal, pattern1)
        np.testing.assert_equal(gal_size, conv.shape[1])

    def test_sim_varpsf(self):
        nstars = 20
        size = 128
        thetas = [0, 45, 105, 150]
        im1 = sm.sim_varpsf(nstars, SN=30.0, thetas=thetas, N=size, seed=42)
        im2 = sm.sim_varpsf(nstars, SN=30.0, thetas=thetas, N=size, seed=42)
        im3 = sm.sim_varpsf(nstars, SN=30.0, thetas=thetas, N=size, seed=None)

        assert np.all(im1 > 0.0)
        assert np.all(im1 != im3)
        assert im1.shape[0] == 2 * size
        np.testing.assert_array_almost_equal(im1, im2)
