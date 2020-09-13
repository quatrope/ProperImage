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

# =============================================================================
# IMPORTS
# =============================================================================

import itertools as it

from astropy.stats import sigma_clipped_stats

import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal

import pytest

from properimage import plot

# =============================================================================
# TEST PRIMES
# =============================================================================


@pytest.mark.parametrize(
    "test_input, expected", [(9, 3), (45045, 143), (3, 3), (1, 1)]
)
def test_primes(test_input, expected):
    assert plot.primes(test_input) == expected


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


# =============================================================================
# TEST auto_psf_coef
# =============================================================================


# @check_figures_equal()
# def test_plot_autopsf_coef(random_4psf_simage, fig_test, fig_ref):
#     img = random_4psf_simage

#     # fig test
#     # test_ax = fig_test.subplots()
#     # img.plot.autopsf_coef(ax=test_ax)

#     # expected
#     a_fields, psf_basis = img.get_variable_psf()
#     print(len(a_fields))
#     import ipdb; ipdb.set_trace()
#     x, y = img.get_afield_domain()

#     # here we plot
#     N = len(a_fields)  # axis needed
#     p = plot.primes(N)

#     rows = (N // p) + (N % p)
#     subplots = (p, rows)

#     size = 4
#     width = size * subplots[0]
#     height = size * subplots[1]

#     fig_ref.set_size_inches(w=width, h=height)
#     exp_axs = fig_ref.subplots(*subplots)

#     cmap_kw = {"shrink": 0.75, "aspect": 30}

#     title_tpl = r"$a_{j}$,$\sum a_{j}={sum:4.3e}$"
#     for idx, a_field, ax in zip(range(N), a_fields, it.chain(*exp_axs)):

#         a = a_field(x, y)
#         mean, med, std = sigma_clipped_stats(a)

#         img = ax.imshow(a, vmax=med + 2 * std, vmin=med - 2 * std)
#         fig.colorbar(img, ax=ax, **cmap_kw)

#         title = title_tpl.format(j=idx + 1, sum=np.sqrt(np.sum(a ** 2)))
#         ax.set_title(title)


