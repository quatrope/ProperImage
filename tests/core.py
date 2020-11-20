#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  core.py
#
#  Copyright 2020 QuatroPe
#
# This file is part of ProperImage (https://github.com/quatrope/ProperImage)
# License: BSD-3-Clause
# Full Text: https://github.com/quatrope/ProperImage/blob/master/LICENSE.txt
#
"""Propercoadd Tests case base"""


# =============================================================================
# IMPORTS
# =============================================================================

import unittest

import numpy as np
import numpy.testing as npt


# =============================================================================
# BASE CLASS
# =============================================================================


class ProperImageTestCase(unittest.TestCase):
    def assertAllClose(self, a, b, **kwargs):
        return npt.assert_allclose(a, b, **kwargs)

    def assertArrayEqual(self, a, b, **kwargs):
        return npt.assert_array_equal(a, b, **kwargs)

    def assertAll(self, arr, **kwargs):
        assert np.all(arr), "'{}' is not all True".format(arr)
