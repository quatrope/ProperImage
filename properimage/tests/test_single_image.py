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

mock_image_data = np.random.random((256, 256))
mock_image_mask = np.random.randint(0, 2, size=(256, 256))
fits.writeto('/tmp/mockfits.fits', mock_image_data,
                               overwrite=True)
fits.writeto('/tmp/mockmask.fits', mock_image_mask,
                               overwrite=True)

mockimageHdu = fits.PrimaryHDU(mock_image_data)
mockmaskHdu = fits.ImageHDU(mock_image_mask.astype('uint8'))
mock_masked_hdu = fits.HDUList([mockimageHdu, mockmaskHdu])

mock_masked_hdu.writeto('/tmp/mockmasked.fits', overwrite=True)


class TestSingleImage(unittest.TestCase):

    def test_attachedto_fits(self):
        self.assertEqual(s.SingleImage('/tmp/mockfits.fits').attached_to,
                         '/tmp/mockfits.fits')

    def test_attachedto_array(self):
        self.assertEqual(s.SingleImage(mock_image_data).attached_to,
                         'ndarray')

    def test_attachedto_hdu(self):
        self.assertEqual(s.SingleImage(mock_masked_hdu).attached_to, 'HDUList')

    #~ def test_



if __name__=='__main__':
    unittest.main()
