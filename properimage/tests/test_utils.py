#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  test_utils.py
#
#  Copyright 2018 Bruno S <bruno@oac.unc.edu.ar>
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

"""
test_utils module from ProperImage
for analysis of astronomical images

Written by Bruno SANCHEZ

PhD of Astromoy - UNC
bruno@oac.unc.edu.ar

Instituto de Astronomia Teorica y Experimental (IATE) UNC
Cordoba - Argentina

Of 301
"""

import os
import numpy as np
import tempfile
from astropy.time import Time

from .core import ProperImageTestCase

from . import simtools as sm
from .. import single_image as si
from .. import utils


class UtilsBase(object):

    def setUp(self):
        print('setting up')
        self.tempdir = tempfile.mkdtemp()

        now = '2018-05-17T00:00:00.1234567'
        t = Time(now)

        N = 1024
        SN = 15.
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

        img_dir = os.path.join(self.tempdir, 'zp={}'.format(zero))

        for i in range(50):
            im = sm.image(m, N, t_exp=2*i+1, X_FWHM=xfwhm, Y_FWHM=yfwhm,
                          theta=theta, SN=SN, bkg_pdf='gaussian')
            filenames.append(sm.capsule_corp(im, t, t_exp=i+1, i=i,
                            zero=zero+i, path=img_dir))

        self.filenames = filenames




class TestChunkIt(ProperImageTestCase):

    def setUp(self):
        self.data = np.random.random(20)

    def testChunks(self):
        for i in range(len(self.data)):
            chunks = si.chunk_it(self.data, i+1)
            self.assertIsInstance(chunks, list)

            flat_list = [item for sublist in chunks for item in sublist]
            self.assertCountEqual(flat_list, list(self.data))
