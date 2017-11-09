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

import sys
import os
import shutil
import tempfile
import unittest

import mock

import numpy as np

from astropy.io import fits

from properimage import single_image2 as s
from properimage import simtools


class TestSingleImage(object):

    def setUp(self):
        print('setting up')
        self.tempdir = tempfile.mkdtemp()

        ### mock data
        psf = simtools.Psf(13, 2.5)

        # a numpy array
        self.mock_image_data = np.random.random((256, 256))*10. + 50
        self.mock_image_data[123, 123] = np.nan

        for i in range(30):
            x, y = np.random.randint(2, 240, size=2)
            #~ print x, y
            self.mock_image_data[x:x+13, y:y+13] += psf*float(i+1)*20.

        # a numpy array mask
        self.mock_image_mask = np.zeros((256, 256))
        self.mock_image_mask[123, 123] = 1

        for i in range(6):
            x, y = np.random.randint(20, 240, size=2)
            l, h = np.random.randint(2, 6, size=2)
            self.mock_image_mask[x:x+l, y:y+h] = 1

        # a fits file
        self.mockfits_path = os.path.join(self.tempdir, 'mockfits.fits')
        fits.writeto(self.mockfits_path, self.mock_image_data,
                                       overwrite=True)
        # a fits file mask
        self.mockmask_path = os.path.join(self.tempdir, 'mockmask.fits')
        fits.writeto(self.mockmask_path, self.mock_image_mask,
                                       overwrite=True)

        # a hdu
        self.mockimageHdu = fits.PrimaryHDU(self.mock_image_data)
        # a hdulist
        self.mockmaskHdu = fits.ImageHDU(self.mock_image_mask.astype('uint8'))
        self.mock_masked_hdu = fits.HDUList([self.mockimageHdu, self.mockmaskHdu])

        # a fits file with hdulist
        self.masked_hdu_path = os.path.join(self.tempdir, 'mockmasked.fits')
        self.mock_masked_hdu.writeto(self.masked_hdu_path, overwrite=True)

        self.h_fitsfile = {'SIMPLE':True, 'BITPIX':-64, 'NAXIS':2,
                      'NAXIS1':256, 'NAXIS2':256, 'EXTEND':True}

    def tearDown(self):
        if os.path.isdir(self.tempdir):
            shutil.rmtree(self.tempdir)
        try:
            self.si._clean()
        except: pass

    def testPixeldata(self):
        np.testing.assert_array_equal(self.mock_image_data,
                                      self.si.pixeldata.data)

    def testBackground(self):
        self.assertIsInstance(self.si.background, np.ndarray)

    def testNSources(self):
        self.assertNotEqual(self.si.n_sources, 0)

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, 'ndarray')

    def testBkgSubImg(self):
        self.assertIsInstance(self.si.bkg_sub_img, np.ndarray)

    def testMask(self):
        np.testing.assert_array_equal(self.mock_image_mask, self.si.mask)


class TestNpArray(TestSingleImage, unittest.TestCase):

    def setUp(self):
        super(TestNpArray, self).setUp()
        self.si = s.SingleImage(self.mock_image_data)

    def testMask(self):
        nanmask = np.zeros((256, 256))
        nanmask[123, 123] = 1
        np.testing.assert_array_equal(nanmask, self.si.mask)

    def testHeader(self):
        self.assertDictEqual(self.si.header, {})


class TestNpArrayMask(TestSingleImage, unittest.TestCase):

    def setUp(self):
        super(TestNpArrayMask, self).setUp()
        self.si = s.SingleImage(self.mock_image_data, self.mock_image_mask)

    def testHeader(self):
        self.assertDictEqual(self.si.header, {})


class TestFitsFile(TestSingleImage, unittest.TestCase):

    def setUp(self):
        super(TestFitsFile, self).setUp()
        self.si = s.SingleImage(self.mockfits_path)

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, self.mockfits_path)

    def testMask(self):
        nanmask = np.zeros((256, 256))
        nanmask[123, 123] = 1
        np.testing.assert_array_equal(nanmask, self.si.mask)

    def testHeader(self):
        self.assertDictEqual(dict(self.si.header), self.h_fitsfile)


class TestFitsMask(TestSingleImage, unittest.TestCase):

    def setUp(self):
        super(TestFitsMask, self).setUp()
        self.si = s.SingleImage(self.mockfits_path, self.mockmask_path)

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, self.mockfits_path)

    def testHeader(self):
        self.assertDictEqual(dict(self.si.header), self.h_fitsfile)


class TestHDU(TestSingleImage, unittest.TestCase):

    def setUp(self):
        super(TestHDU, self).setUp()
        self.si = s.SingleImage(self.mockimageHdu)

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, 'PrimaryHDU')

    def testMask(self):
        nanmask = np.zeros((256, 256))
        nanmask[123, 123] = 1
        np.testing.assert_array_equal(nanmask, self.si.pixeldata.mask)

    def testHeader(self):
        self.assertDictEqual(dict(self.si.header), self.h_fitsfile)


class TestHDUList(TestSingleImage, unittest.TestCase):

    def setUp(self):
        super(TestHDUList, self).setUp()
        self.si = s.SingleImage(self.mock_masked_hdu)

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, 'HDUList')

    def testHeader(self):
        self.assertDictEqual(dict(self.si.header), self.h_fitsfile)



class TestFitsExtension(TestSingleImage, unittest.TestCase):

    def setUp(self):
        super(TestFitsExtension, self).setUp()
        self.si = s.SingleImage(self.masked_hdu_path)

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, self.masked_hdu_path)

    def testHeader(self):
        self.assertDictEqual(dict(self.si.header), self.h_fitsfile)


if __name__=='__main__':
    unittest.main()
