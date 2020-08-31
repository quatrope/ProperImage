#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  test_plot.py
#
#  Copyright 2017 Bruno S <bruno@oac.unc.edu.ar>
#
# This file is part of ProperImage (https://github.com/toros-astro/ProperImage)
# License: BSD-3-Clause
# Full Text: https://github.com/toros-astro/ProperImage/blob/master/LICENSE.txt
#

"""
Pytest configuration

Written by Bruno SANCHEZ, JB Cabral

PhD of Astromoy - UNC
bruno@oac.unc.edu.ar

Instituto de Astronomia Teorica y Experimental (IATE) UNC
Cordoba - Argentina Of 301
"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

from properimage import single_image as s


# =============================================================================
# CONSTANTS
# =============================================================================

# FIX the random state
random = np.random.RandomState(42)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def random_simage():
    pixel = random.random((128, 128)) * 5.0
    # Add some stars to it
    star = [[35, 38, 35], [38, 90, 39], [35, 39, 34]]
    for i in range(25):
        x, y = random.randint(120, size=2)
        pixel[x : x + 3, y : y + 3] = star

    mask = random.randint(2, size=(128, 128))
    for i in range(10):
        mask = mask & random.randint(2, size=(128, 128))

    img = s.SingleImage(pixel, mask)

    return img
