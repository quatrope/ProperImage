#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  test_single_image.py
#
#  Copyright 2017 Bruno S <bruno@oac.unc.edu.ar>
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

import unittest
import numpy as np
from astropy.io import fits

from properimage import single_image2 as s

### mock data
# a numpy array
mock_image_data = np.random.random((256, 256))
# a numpy array mask
mock_image_mask = np.random.randint(0, 2, size=(256, 256))
# a fits file
fits.writeto('/tmp/mockfits.fits', mock_image_data,
                               overwrite=True)
# a fits file mask
fits.writeto('/tmp/mockmask.fits', mock_image_mask,
                               overwrite=True)

# a hdu
mockimageHdu = fits.PrimaryHDU(mock_image_data)
# a hdulist
mockmaskHdu = fits.ImageHDU(mock_image_mask.astype('uint8'))
mock_masked_hdu = fits.HDUList([mockimageHdu, mockmaskHdu])

# a fits file with hdulist
mock_masked_hdu.writeto('/tmp/mockmasked.fits', overwrite=True)


class TestNpArray(unittest.TestCase):

    def setUp(self):
        self.si = s.SingleImage(mock_image_data)

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, 'ndarray')

class TestNpArrayMask(unittest.TestCase):

    def setUp(self):
        self.si = s.SingleImage(mock_image_data, mock_image_mask)

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, 'ndarray')

class TestFitsFile(unittest.TestCase):

    def setUp(self):
        self.si = s.SingleImage('/tmp/mockfits.fits')

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, '/tmp/mockfits.fits')


class TestFitsMask(unittest.TestCase):

    def setUp(self):
        self.si = s.SingleImage('/tmp/mockfits.fits', '/tmp/mockmask.fits')

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, '/tmp/mockfits.fits')


class TestHDU(unittest.TestCase):

    def setUp(self):
        self.si = s.SingleImage(mockimageHdu)

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, 'PrimaryHDU')

class TestHDUList(unittest.TestCase):

    def setUp(self):
        self.si = s.SingleImage(mock_masked_hdu)

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, 'HDUList')


class TestFitsExtension(unittest.TestCase):

    def setUp(self):
        self.si = s.SingleImage('/tmp/mockmasked.fits')

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, '/tmp/mockmasked.fits')



if __name__=='__main__':
    unittest.main()
