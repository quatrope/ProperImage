#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  test_single_image.py
#
#  Copyright 2020 QuatroPe
#
# This file is part of ProperImage (https://github.com/quatrope/ProperImage)
# License: BSD-3-Clause
# Full Text: https://github.com/quatrope/ProperImage/blob/master/LICENSE.txt
#

"""
test_single_image module from ProperImage
for analysis of astronomical images

Written by Bruno SANCHEZ

PhD of Astromoy - UNC
bruno@oac.unc.edu.ar

Instituto de Astronomia Teorica y Experimental (IATE) UNC
Cordoba - Argentina

Of 301
"""

import os
import shutil
import tempfile
import warnings

from astropy.io import fits

import numpy as np
from numpy.random import default_rng

from properimage import SingleImage, simtools

import pytest

from .core import ProperImageTestCase

warnings.filterwarnings("ignore")

random = default_rng(seed=42)


class SingleImageBase(object):
    def setUp(self):
        print("setting up")
        self.tempdir = tempfile.mkdtemp()

        # mock data
        psf = simtools.Psf(11, 2.5, 3.0)

        # a numpy array
        self.mock_image_data = random.random((256, 256)) * 50.0 + 350
        self.mock_image_data[123, 123] = np.nan

        for i in range(30):
            x = random.integers(7, 220)
            y = random.integers(7, 120)
            # print x, y
            self.mock_image_data[x : x + 11, y : y + 11] += (
                psf * float(i + 1) * 2000.0
            )

        psf = simtools.Psf(11, 3.0, 1.9)
        for i in range(30):
            x = random.integers(7, 220)
            y = random.integers(122, 220)
            # print x, y
            self.mock_image_data[x : x + 11, y : y + 11] += (
                psf * float(i + 1) * 2000.0
            )

        # a numpy array mask
        self.mock_image_mask = np.zeros((256, 256))
        self.mock_image_mask[123, 123] = 1

        for i in range(6):
            x, y = random.integers(20, 240, size=2)
            l, h = random.integers(2, 6, size=2)
            self.mock_image_mask[x : x + l, y : y + h] = random.integers(
                0, 32, size=(l, h)
            )

        # a fits file
        self.mockfits_path = os.path.join(self.tempdir, "mockfits.fits")
        fits.writeto(self.mockfits_path, self.mock_image_data, overwrite=True)
        # a fits file mask
        self.mockmask_path = os.path.join(self.tempdir, "mockmask.fits")
        fits.writeto(self.mockmask_path, self.mock_image_mask, overwrite=True)

        # a hdu
        self.mockimageHdu = fits.PrimaryHDU(self.mock_image_data)
        # a hdulist
        self.mockmaskHdu = fits.ImageHDU(self.mock_image_mask.astype("uint8"))
        self.mock_masked_hdu = fits.HDUList(
            [self.mockimageHdu, self.mockmaskHdu]
        )

        # a fits file with hdulist
        self.masked_hdu_path = os.path.join(self.tempdir, "mockmasked.fits")
        self.mock_masked_hdu.writeto(self.masked_hdu_path, overwrite=True)

        self.h_fitsfile = {
            "SIMPLE": True,
            "BITPIX": -64,
            "NAXIS": 2,
            "NAXIS1": 256,
            "NAXIS2": 256,
            "EXTEND": True,
        }

    def tearDown(self):
        if os.path.isdir(self.tempdir):
            shutil.rmtree(self.tempdir)
        try:
            self.si._clean()
        except OSError:
            pass

    def testdata(self):
        np.testing.assert_allclose(
            self.mock_image_data, self.si.data.data, rtol=0.15
        )

    def testRepr(self):
        self.assertIsInstance(self.si.__repr__(), str)
        self.assertTrue("SingleImage" in self.si.__repr__())

    def testBackground(self):
        self.assertIsInstance(self.si.background, np.ndarray)

    def testNSources(self):
        self.assertNotEqual(self.si.n_sources, 0)

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, "ndarray")

    def testBkgSubImg(self):
        self.assertIsInstance(self.si.bkg_sub_img, np.ndarray)

    def testMask(self):
        np.testing.assert_allclose(
            self.mock_image_mask <= self.si.maskthresh,
            self.si.mask,
            rtol=0.2,
            atol=3,
        )

    def testStampPos(self):
        self.assertIsInstance(self.si.stamps_pos, np.ndarray)
        if self.si.n_sources != 0:
            self.assertIsInstance(self.si.stamps_pos[0], np.ndarray)
            self.assertEqual(self.si.n_sources, len(self.si.stamps_pos))

    def testBestSources(self):
        self.assertIsInstance(self.si.best_sources, np.ndarray)
        self.assertGreaterEqual(len(self.si.best_sources), 1)

    def testUpdateBestSources(self):
        self.si.update_sources()
        self.assertIsInstance(self.si.best_sources, np.ndarray)
        self.assertGreaterEqual(len(self.si.best_sources), 1)

    def testCovMat(self):
        self.assertIsInstance(self.si.cov_matrix, np.ndarray)
        np.testing.assert_array_equal(self.si.cov_matrix, self.si.cov_matrix.T)

    def testEigenV(self):
        self.assertIsInstance(self.si.eigenv, tuple)
        self.assertIsInstance(self.si.eigenv[0], np.ndarray)
        self.assertIsInstance(self.si.eigenv[1], np.ndarray)

    def testInfLoss(self):
        self.assertEqual(self.si.inf_loss, 0.2)
        self.si.inf_loss = 0.01
        self.assertEqual(self.si.inf_loss, 0.01)

    def testPsfBasis(self):
        self.assertIsInstance(self.si.kl_basis, list)
        self.assertGreaterEqual(len(self.si.kl_basis), 1)

    def testAFields(self):
        self.assertIsInstance(self.si.kl_afields, list)
        self.assertGreaterEqual(len(self.si.kl_basis), 1)
        if len(self.si.kl_basis) == 1:
            self.assertIsNone(self.si.kl_afields[0], None)

    def testAFieldsLowInfLoss(self):
        self.si.inf_loss = 0.0002
        self.assertIsInstance(self.si.kl_afields, list)
        self.assertGreaterEqual(len(self.si.kl_basis), 1)
        if len(self.si.kl_basis) == 1:
            self.assertIsNone(self.si.kl_afields[0], None)

    def testGetAFieldDomain(self):
        self.assertIsInstance(self.si.get_afield_domain(), tuple)
        self.assertIsInstance(self.si.get_afield_domain()[0], np.ndarray)

    def testGetVariablePsf(self):
        self.assertIsInstance(self.si.get_variable_psf(), list)
        self.assertIsInstance(self.si.get_variable_psf(inf_loss=0.002), list)
        self.assertIsInstance(self.si.get_variable_psf(shape=(23, 23)), list)

    def testNormalImage(self):
        self.assertIsInstance(self.si.normal_image, np.ndarray)
        self.assertFalse(np.isnan(np.sum(self.si.normal_image)))

    def testSComponent(self):
        self.assertIsInstance(self.si.s_component, np.ndarray)
        self.assertFalse(np.isnan(np.sum(self.si.s_component)))

    def testSHatComp(self):
        self.assertIsInstance(self.si.s_hat_comp, np.ndarray)
        self.assertFalse(np.isnan(np.sum(self.si.s_hat_comp)))

    def testPsfHatSqNorm(self):
        self.assertIsInstance(self.si.psf_hat_sqnorm(), np.ndarray)
        self.assertFalse(np.isnan(np.sum(self.si.psf_hat_sqnorm())))

    def testZP(self):
        self.assertEqual(self.si.zp, 1)

    def testSetZP(self):
        self.si.zp = 12.0
        self.assertEqual(self.si.zp, 12)

    def testVar(self):
        self.assertIsInstance(self.si.var, float)

    def testInterped(self):
        self.assertIsInstance(self.si.interped, np.ndarray)
        self.assertFalse(np.isnan(np.sum(self.si.interped)))

    def testInterpedHat(self):
        self.assertIsInstance(self.si.interped_hat, np.ndarray)
        self.assertFalse(np.isnan(np.sum(self.si.interped_hat)))

    def testPSqNorm(self):
        self.assertIsInstance(self.si.p_sqnorm(), np.ndarray)

    def testPsfBasisNorm(self):
        for an_inf_loss in [0.1, 0.2, 0.05, 0.15]:
            afields, psfs = self.si.get_variable_psf(an_inf_loss)
            xs, ys = self.si.data.shape
            for i in range(10):
                xc = random.integers(xs - 60) + 30
                yc = random.integers(ys - 60) + 30
                psfxy = self.si.get_psf_xy(xc, yc)
                np.testing.assert_approx_equal(
                    np.sum(psfxy), 1.0, significant=2
                )

    @pytest.mark.xfail()
    def testPSFInfLossChange(self):
        original_si = self.si.inf_loss
        self.si.get_variable_psf(inf_loss=0)
        self.assertEqual(original_si, self.si.inf_loss)


class TestNpArray(SingleImageBase, ProperImageTestCase):
    def setUp(self):
        super(TestNpArray, self).setUp()
        print(self.mock_image_data.shape)
        self.si = SingleImage(self.mock_image_data)

    def testMask(self):
        nanmask = np.zeros((256, 256))
        nanmask[123, 123] = 1
        np.testing.assert_array_equal(nanmask, self.si.mask)

    def testHeader(self):
        self.assertDictEqual(self.si.header, {})


class TestNpArrayMask(SingleImageBase, ProperImageTestCase):
    def setUp(self):
        super(TestNpArrayMask, self).setUp()
        self.si = SingleImage(self.mock_image_data, self.mock_image_mask)

    def testHeader(self):
        self.assertDictEqual(self.si.header, {})


class TestFitsFile(SingleImageBase, ProperImageTestCase):
    def setUp(self):
        super(TestFitsFile, self).setUp()
        self.si = SingleImage(self.mockfits_path)

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, self.mockfits_path)

    def testMask(self):
        nanmask = np.zeros((256, 256))
        nanmask[123, 123] = 1
        np.testing.assert_array_equal(nanmask, self.si.mask)

    def testHeader(self):
        self.assertDictEqual(dict(self.si.header), self.h_fitsfile)


class TestFitsMask(SingleImageBase, ProperImageTestCase):
    def setUp(self):
        super(TestFitsMask, self).setUp()
        self.si = SingleImage(self.mockfits_path, mask=self.mockmask_path)

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, self.mockfits_path)

    def testHeader(self):
        self.assertDictEqual(dict(self.si.header), self.h_fitsfile)


class TestHDU(SingleImageBase, ProperImageTestCase):
    def setUp(self):
        super(TestHDU, self).setUp()
        self.si = SingleImage(self.mockimageHdu)

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, "PrimaryHDU")

    def testMask(self):
        nanmask = np.zeros((256, 256))
        nanmask[123, 123] = 1
        np.testing.assert_array_equal(nanmask, self.si.data.mask)

    def testHeader(self):
        self.assertDictEqual(dict(self.si.header), self.h_fitsfile)


class TestHDUList(SingleImageBase, ProperImageTestCase):
    def setUp(self):
        super(TestHDUList, self).setUp()
        self.si = SingleImage(self.mock_masked_hdu)

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, "HDUList")

    def testHeader(self):
        self.assertDictEqual(dict(self.si.header), self.h_fitsfile)


class TestFitsExtension(SingleImageBase, ProperImageTestCase):
    def setUp(self):
        super(TestFitsExtension, self).setUp()
        self.si = SingleImage(self.masked_hdu_path)

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, self.masked_hdu_path)

    def testHeader(self):
        self.assertDictEqual(dict(self.si.header), self.h_fitsfile)


class TestNoSources(ProperImageTestCase):
    def setUp(self):
        self.no_sources_image = random.random((256, 256))

    def testNoSources(self):
        with self.assertRaises(ValueError):
            SingleImage(self.no_sources_image)


class TestNpArrayMinSrcs(SingleImageBase, ProperImageTestCase):
    def setUp(self):
        super(TestNpArrayMinSrcs, self).setUp()
        print(self.mock_image_data.shape)
        self.si = SingleImage(self.mock_image_data, min_sources=6)

    def testMask(self):
        nanmask = np.zeros((256, 256))
        nanmask[123, 123] = 1
        np.testing.assert_array_equal(nanmask, self.si.mask)

    def testHeader(self):
        self.assertDictEqual(self.si.header, {})


class TestNpArrayCtxt(SingleImageBase, ProperImageTestCase):
    def setUp(self):
        super(TestNpArrayCtxt, self).setUp()
        print(self.mock_image_data.shape)
        self.si = SingleImage(self.mock_image_data)

    def testCtxt(self):
        with SingleImage(self.mock_image_data) as sim:
            np.testing.assert_array_equal(self.si.data.data, sim.data.data)


# Test with picky star stamp strategy
class TestNpArrayPicky(SingleImageBase, ProperImageTestCase):
    def setUp(self):
        super(TestNpArrayPicky, self).setUp()
        print(self.mock_image_data.shape)
        self.si = SingleImage(self.mock_image_data, strict_star_pick=True)

    def testMask(self):
        nanmask = np.zeros((256, 256))
        nanmask[123, 123] = 1
        np.testing.assert_array_equal(nanmask, self.si.mask)

    def testHeader(self):
        self.assertDictEqual(self.si.header, {})


class TestNpArrayMaskPicky(SingleImageBase, ProperImageTestCase):
    def setUp(self):
        super(TestNpArrayMaskPicky, self).setUp()
        self.si = SingleImage(
            self.mock_image_data, self.mock_image_mask, strict_star_pick=True
        )

    def testHeader(self):
        self.assertDictEqual(self.si.header, {})


class TestFitsFilePicky(SingleImageBase, ProperImageTestCase):
    def setUp(self):
        super(TestFitsFilePicky, self).setUp()
        self.si = SingleImage(self.mockfits_path, strict_star_pick=True)

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, self.mockfits_path)

    def testMask(self):
        nanmask = np.zeros((256, 256))
        nanmask[123, 123] = 1
        np.testing.assert_array_equal(nanmask, self.si.mask)

    def testHeader(self):
        self.assertDictEqual(dict(self.si.header), self.h_fitsfile)


class TestFitsMaskPicky(SingleImageBase, ProperImageTestCase):
    def setUp(self):
        super(TestFitsMaskPicky, self).setUp()
        self.si = SingleImage(
            self.mockfits_path, mask=self.mockmask_path, strict_star_pick=True
        )

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, self.mockfits_path)

    def testHeader(self):
        self.assertDictEqual(dict(self.si.header), self.h_fitsfile)


class TestHDUPicky(SingleImageBase, ProperImageTestCase):
    def setUp(self):
        super(TestHDUPicky, self).setUp()
        self.si = SingleImage(self.mockimageHdu, strict_star_pick=True)

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, "PrimaryHDU")

    def testMask(self):
        nanmask = np.zeros((256, 256))
        nanmask[123, 123] = 1
        np.testing.assert_array_equal(nanmask, self.si.data.mask)

    def testHeader(self):
        self.assertDictEqual(dict(self.si.header), self.h_fitsfile)


class TestHDUListPicky(SingleImageBase, ProperImageTestCase):
    def setUp(self):
        super(TestHDUListPicky, self).setUp()
        self.si = SingleImage(self.mock_masked_hdu, strict_star_pick=True)

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, "HDUList")

    def testHeader(self):
        self.assertDictEqual(dict(self.si.header), self.h_fitsfile)


class TestFitsExtensionPicky(SingleImageBase, ProperImageTestCase):
    def setUp(self):
        super(TestFitsExtensionPicky, self).setUp()
        self.si = SingleImage(self.masked_hdu_path)

    def testAttachedTo(self):
        self.assertEqual(self.si.attached_to, self.masked_hdu_path)

    def testHeader(self):
        self.assertDictEqual(dict(self.si.header), self.h_fitsfile)


class TestNoSourcesPicky(ProperImageTestCase):
    def setUp(self):
        self.no_sources_image = random.random((256, 256))

    def testNoSources(self):
        with self.assertRaises(ValueError):
            SingleImage(self.no_sources_image)
