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
mock_image_data[123, 123] = np.nan
# a numpy array mask
mock_image_mask = np.random.randint(0, 2, size=(256, 256))
mock_image_mask[123, 123] = 1
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

h_fitsfile = {'SIMPLE':True, 'BITPIX':-64, 'NAXIS':2,
              'NAXIS1':256, 'NAXIS2':256, 'EXTEND':True}


class TestNpArray(unittest.TestCase):

    def setUp(self):
        self.si = s.SingleImage(mock_image_data)

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, 'ndarray')

    def testPixeldata(self):
        np.testing.assert_array_equal(mock_image_data, self.si.pixeldata.data)

    def testMask(self):
        nanmask = np.zeros((256, 256))
        nanmask[123, 123] = 1
        np.testing.assert_array_equal(nanmask, self.si.pixeldata.mask)

    def testHeader(self):
        self.assertDictEqual(self.si.header, {})

class TestNpArrayMask(unittest.TestCase):

    def setUp(self):
        self.si = s.SingleImage(mock_image_data, mock_image_mask)

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, 'ndarray')

    def testPixeldata(self):
        np.testing.assert_array_equal(mock_image_data, self.si.pixeldata.data)

    def testMask(self):
        np.testing.assert_array_equal(mock_image_mask, self.si.pixeldata.mask)

    def testHeader(self):
        self.assertDictEqual(self.si.header, {})

class TestFitsFile(unittest.TestCase):

    def setUp(self):
        self.si = s.SingleImage('/tmp/mockfits.fits')

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, '/tmp/mockfits.fits')

    def testPixeldata(self):
            np.testing.assert_array_equal(mock_image_data, self.si.pixeldata.data)

    def testMask(self):
        nanmask = np.zeros((256, 256))
        nanmask[123, 123] = 1
        np.testing.assert_array_equal(nanmask, self.si.pixeldata.mask)

    def testHeader(self):
        self.assertDictEqual(dict(self.si.header), h_fitsfile)

class TestFitsMask(unittest.TestCase):

    def setUp(self):
        self.si = s.SingleImage('/tmp/mockfits.fits', '/tmp/mockmask.fits')

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, '/tmp/mockfits.fits')

    def testPixeldata(self):
            np.testing.assert_array_equal(mock_image_data, self.si.pixeldata.data)

    def testMask(self):
        np.testing.assert_array_equal(mock_image_mask, self.si.pixeldata.mask)

class TestHDU(unittest.TestCase):

    def setUp(self):
        self.si = s.SingleImage(mockimageHdu)

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, 'PrimaryHDU')

    def testPixeldata(self):
            np.testing.assert_array_equal(mock_image_data, self.si.pixeldata.data)

    def testMask(self):
        nanmask = np.zeros((256, 256))
        nanmask[123, 123] = 1
        np.testing.assert_array_equal(nanmask, self.si.pixeldata.mask)

    def testHeader(self):
        self.assertDictEqual(dict(self.si.header), h_fitsfile)

class TestHDUList(unittest.TestCase):

    def setUp(self):
        self.si = s.SingleImage(mock_masked_hdu)

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, 'HDUList')

    def testPixeldata(self):
        np.testing.assert_array_equal(mock_image_data, self.si.pixeldata.data)

    def testMask(self):
        np.testing.assert_array_equal(mock_image_mask, self.si.pixeldata.mask)


class TestFitsExtension(unittest.TestCase):

    def setUp(self):
        self.si = s.SingleImage('/tmp/mockmasked.fits')

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, '/tmp/mockmasked.fits')

    def testPixeldata(self):
        np.testing.assert_array_equal(mock_image_data, self.si.pixeldata.data)

    def testMask(self):
        np.testing.assert_array_equal(mock_image_mask, self.si.pixeldata.mask)




if __name__=='__main__':
    unittest.main()
