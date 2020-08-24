#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  test_propercoadd.py
#
#  Copyright 2018 Bruno S <bruno@oac.unc.edu.ar>
#
# This file is part of ProperImage (https://github.com/toros-astro/ProperImage)
# License: BSD-3-Clause
# Full Text: https://github.com/toros-astro/ProperImage/blob/master/LICENSE.txt
#
"""
test_propercoadd module from ProperImage,
for coadding astronomical images.

Written by Bruno SANCHEZ

PhD of Astromoy - UNC
bruno@oac.unc.edu.ar

Instituto de Astronomia Teorica y Experimental (IATE) UNC
Cordoba - Argentina

Of 301

"""

import os
import tempfile
import shutil

import numpy as np
from astropy.io import fits

from properimage import propercoadd as pc
from properimage import single_image as si
from properimage import simtools

from .core import ProperImageTestCase


class PropercoaddBase(object):
    def setUp(self):
        print("setting up")
        self.tempdir = tempfile.mkdtemp()
        self.paths = []
        # mock data
        psf = simtools.Psf(11, 2.5, 3.0)

        self.mock_image_data = np.random.random((256, 256)) * 10.0
        for i in range(50):
            x = np.random.randint(7, 220)
            y = np.random.randint(7, 120)
            # print x, y
            self.mock_image_data[x : x + 11, y : y + 11] += (
                psf * float(i + 1) * 2000.0
            )

        psf = simtools.Psf(11, 3.0, 1.9)
        for i in range(50):
            x = np.random.randint(7, 220)
            y = np.random.randint(122, 220)
            # print x, y
            self.mock_image_data[x : x + 11, y : y + 11] += (
                psf * float(i + 1) * 2000.0
            )

        # generate 4 images to coadd
        for j in range(4):
            # a numpy array
            image_data = (
                self.mock_image_data
                + np.random.random((256, 256)) * 50.0
                + 350
            )

            image_data[123, 123] = np.nan

            # a fits file
            mockfits_path = os.path.join(
                self.tempdir, "mockfits_{}.fits".format(j)
            )
            fits.writeto(mockfits_path, image_data, overwrite=True)
            self.paths.append(mockfits_path)

    def testChunkIt(self):
        imgs = [si.SingleImage(img) for img in self.paths]

        for i in range(len(imgs)):
            chunks = si.chunk_it(imgs, i + 1)

            self.assertIsInstance(chunks, list)

            flat_list = [item for sublist in chunks for item in sublist]
            self.assertCountEqual(flat_list, imgs)

    def tearDown(self):
        if os.path.isdir(self.tempdir):
            shutil.rmtree(self.tempdir)
        try:
            for animg in self.imgs:
                animg._clean()
        except OSError:
            pass


class TestCoaddSingleCore(PropercoaddBase, ProperImageTestCase):
    def setUp(self):
        super(TestCoaddSingleCore, self).setUp()
        self.imgs = [si.SingleImage(img) for img in self.paths]

    def testCoaddSingleCore(self):
        R, P_r, mask = pc.stack_R(self.imgs, align=False, n_procs=1)
        self.assertIsInstance(R, np.ndarray)
        self.assertIsInstance(P_r, np.ndarray)
        self.assertIsInstance(mask, np.ndarray)


class TestCoadd2Core(PropercoaddBase, ProperImageTestCase):
    def setUp(self):
        super(TestCoadd2Core, self).setUp()
        self.imgs = [si.SingleImage(img) for img in self.paths]

    def testCoadd2Core(self):
        R, P_r, mask = pc.stack_R(self.imgs, align=False, n_procs=2)
        self.assertIsInstance(R, np.ndarray)
        self.assertIsInstance(P_r, np.ndarray)
        self.assertIsInstance(mask, np.ndarray)


class TestCoaddMultCore1(PropercoaddBase, ProperImageTestCase):
    def setUp(self):
        super(TestCoaddMultCore1, self).setUp()
        self.imgs = [si.SingleImage(img) for img in self.paths]

    def testCoaddMultipleCores(self):
        R2, P_r2, mask2 = pc.stack_R(self.imgs, align=False, n_procs=2)
        R, P_r, mask = pc.stack_R(self.imgs, align=False, n_procs=1)

        np.testing.assert_allclose(R.real, R2.real, rtol=0.2, atol=0.5)
        np.testing.assert_allclose(P_r.real, P_r2.real, rtol=0.2, atol=0.5)
        np.testing.assert_allclose(mask, mask2, rtol=0.2, atol=1)


class TestCoaddMultCore2(PropercoaddBase, ProperImageTestCase):
    def setUp(self):
        super(TestCoaddMultCore2, self).setUp()
        self.imgs = [si.SingleImage(img) for img in self.paths]

    def testCoaddMultipleCores(self):
        R2, P_r2, mask2 = pc.stack_R(self.imgs, align=False, n_procs=4)
        R, P_r, mask = pc.stack_R(self.imgs, align=False, n_procs=2)

        np.testing.assert_allclose(R.real, R2.real, rtol=0.2, atol=0.5)
        np.testing.assert_allclose(P_r.real, P_r2.real, rtol=0.2, atol=0.5)
        np.testing.assert_allclose(mask, mask2, rtol=0.2, atol=1)
