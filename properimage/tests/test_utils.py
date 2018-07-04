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

import numpy as np

from .core import ProperImageTestCase

from .. import single_image as si
# from properimage import utils


class UtilsBase(object):
    pass


class TestChunkIt(ProperImageTestCase):

    def setUp(self):
        self.data = np.random.random(20)

    def testChunks(self):
        for i in range(len(self.data)):
            chunks = si.chunk_it(self.data, i+1)
            self.assertIsInstance(chunks, list)

            flat_list = [item for sublist in chunks for item in sublist]
            self.assertCountEqual(flat_list, list(self.data))
