#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  test_propercoadd.py
#
#  Copyright 2016 Bruno S <bruno@oac.unc.edu.ar>
#

import sys

sys.path.insert(0, os.path.abspath('..'))

import unittest
import numpy as np
import ccdproc
from astropy.io import fits

import propercoadd as pc

class TestImageStats(unittest.TestCase):

    def setUp(self):
        #setting different input objects
        self.ndarrayformat = 'numpy_array'
        self.ndarray_img_obj = np.empty((100, 100))

        self.ccddataformat = 'CCDData'
        self.ccddata_img_obj = ccdproc.CCDData(data=self.ndarray_img_obj,
                                               unit='adu')


    def test_array_init(self):
        pass
