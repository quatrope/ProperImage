#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  test_plot.py
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

from .. import plot

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
