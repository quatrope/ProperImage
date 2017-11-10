#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  test_simtools.py
#
#  Copyright 2016 Bruno S <bruno@oac.unc.edu.ar>
#

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

import unittest
import numpy as np
from scipy.ndimage.interpolation import rotate
import simtools as sm


class TestSimulationSuite(unittest.TestCase):

    def test_Psf_module(self):
        module = np.sum(sm.Psf(100, 15))
        self.assertAlmostEqual(module, 1.)

    def test_Psf_asymmetrical(self):
        psf1 = sm.Psf(100, 30, 25)
        psf2 = sm.Psf(100, 25, 30)
        delta = (psf1 - rotate(psf2, 90))**2
        self.assertLess(delta.sum(), 0.01)

    def test_Psf_rotated(self):
        psf1 = sm.Psf(100, 30, theta=45)
        psf2 = sm.Psf(100, 30)
        np.testing.assert_almost_equal(psf1, rotate(psf2, 45))

    def test_astropyPsf_module(self):
        module = np.sum(sm.astropy_Psf(100, 15))
        self.assertAlmostEqual(module, 1.)


if __name__ == '__main__':
    unittest.main()
