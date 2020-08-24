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
test_plot module from ProperImage
for analysis of astronomical images

Written by Bruno SANCHEZ

PhD of Astromoy - UNC
bruno@oac.unc.edu.ar

Instituto de Astronomia Teorica y Experimental (IATE) UNC
Cordoba - Argentina

Of 301
"""

from properimage import plot

from .core import ProperImageTestCase


class TestPrimes(ProperImageTestCase):
    def test9(self):
        self.assertEqual(plot.primes(9), 3)

    def test45045(self):
        self.assertEqual(plot.primes(45045), 143)

    def test3(self):
        self.assertEqual(plot.primes(3), 3)

    def test1(self):
        self.assertEqual(plot.primes(1), 1)
