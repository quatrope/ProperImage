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

# el siguiente código va a ser escrito como funciones. Esto se trata mas que
# nada por que los plots son mucho mas fáciles de probar con esta forma.
# lo ideal es qur todos los tests sigan esta forma si usamos pytest.

import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal

import pytest

from properimage import plot


# =============================================================================
# TEST API
# =============================================================================


def test_plot_default_ax(random_simage):
    img = random_simage
    ax = img.plot()
    assert ax is plt.gca()


@check_figures_equal()
def test_plot_default(random_simage, fig_test, fig_ref):
    img = random_simage

    # fig test
    test_ax = fig_test.subplots()
    img.plot(ax=test_ax)

    # expected
    exp_ax = fig_ref.subplots()
    img.plot.imshow(ax=exp_ax)


def test_plot_invalid_plot(random_simage):
    img = random_simage
    with pytest.raises(ValueError):
        img.plot("_foo")

    with pytest.raises(ValueError):
        img.plot("foo")


    with pytest.raises(ValueError):
        img.plot("si")


# =============================================================================
# imshow
# =============================================================================


@check_figures_equal()
def test_plot_imshow_method(random_simage, fig_test, fig_ref):
    img = random_simage

    # fig test
    test_ax = fig_test.subplots()
    img.plot.imshow(ax=test_ax)

    # expected
    exp_ax = fig_ref.subplots()
    exp_ax.imshow(img.data, origin="lower")
    exp_ax.set_title(f"SingleImage {img.data.shape}")


@check_figures_equal()
def test_plot_imshow_str(random_simage, fig_test, fig_ref):
    img = random_simage

    # fig test
    test_ax = fig_test.subplots()
    img.plot("imshow", ax=test_ax)

    # expected
    exp_ax = fig_ref.subplots()
    exp_ax.imshow(img.data, origin="lower")
    exp_ax.set_title(f"SingleImage {img.data.shape}")
