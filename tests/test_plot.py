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

from unittest import mock

from astropy.stats import sigma_clipped_stats

import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal

import numpy as np

from properimage import plot

import pytest

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


def test_plot_imshow_default_ax(random_simage):
    img = random_simage
    ax = img.plot.imshow()
    assert ax is plt.gca()


@check_figures_equal()
def test_plot_imshow(random_simage, fig_test, fig_ref):
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
    img.plot.imshow(ax=exp_ax)


# =============================================================================
# TEST auto_psf
# =============================================================================


def test_plot_autopsf_default_axes(random_4psf_simage):
    simg = random_4psf_simage

    a_fields, psf_basis = simg.get_variable_psf(inf_loss=0.15)
    axs = simg.plot.autopsf(inf_loss=0.15)

    assert np.size(axs) >= len(psf_basis)


def test_plot_autopsf_too_few_axis(random_4psf_simage):
    simg = random_4psf_simage
    with pytest.raises(ValueError):
        simg.plot.autopsf(inf_loss=0.15, axs=[plt.gca()])


@check_figures_equal()
def test_plot_autopsf(random_4psf_simage, fig_test, fig_ref):
    simg = random_4psf_simage

    # expected
    a_fields, psf_basis = simg.get_variable_psf(inf_loss=0.15)

    xsh, ysh = psf_basis[0].shape

    N = len(psf_basis)
    p = plot.primes(N)

    if N == 2:
        subplots = (1, 2)
    if N == 3:
        subplots = (1, 3)
    elif p == N:
        subplots = (round(np.sqrt(N)), round(np.sqrt(N) + 1))
    else:
        rows = N // p
        rows += N % p
        subplots = (rows, p)

    height = plot.DEFAULT_HEIGHT * subplots[0]
    width = plot.DEFAULT_WIDTH * subplots[1]

    fig_ref.set_size_inches(w=width, h=height)
    exp_axs = fig_ref.subplots(*subplots)

    kwargs = {"interpolation": "none"}
    cmap_kw = {"shrink": 0.85}
    iso_kw = {"colors": "black", "alpha": 0.5}

    title_tpl = r"$\sum p_{j:d} = {sum:4.3e}$"
    for idx, psf_basis, ax in zip(range(N), psf_basis, np.ravel(exp_axs)):

        img = ax.imshow(psf_basis, **kwargs)
        title = title_tpl.format(j=idx + 1, sum=np.sum(psf_basis))
        ax.set_title(title)

        fig_ref.colorbar(img, ax=ax, **cmap_kw)

        ax.contour(np.arange(xsh), np.arange(ysh), psf_basis, **iso_kw)

    # fig test
    fig_test.set_size_inches(w=width, h=height)
    test_axs = fig_test.subplots(*subplots)
    simg.plot.autopsf(axs=test_axs, inf_loss=0.15, iso=True)


@check_figures_equal()
def test_plot_autopsf_str(random_4psf_simage, fig_test, fig_ref):
    simg = random_4psf_simage

    with mock.patch("matplotlib.pyplot.gcf", return_value=fig_test):
        simg.plot("autopsf", inf_loss=0.15, iso=True)

    with mock.patch("matplotlib.pyplot.gcf", return_value=fig_ref):
        simg.plot.autopsf(inf_loss=0.15, iso=True)


# =============================================================================
# TEST auto_psf_coef
# =============================================================================


def test_plot_autopsf_coef_no_coef(random_4psf_simage):
    simg = random_4psf_simage

    with pytest.raises(plot.NoDataToPlot):
        simg.plot.autopsf_coef(inf_loss=1.0)


def test_plot_autopsf_coef_default_axes(random_4psf_simage):
    simg = random_4psf_simage

    a_fields, psf_basis = simg.get_variable_psf(inf_loss=0.15)
    axs = simg.plot.autopsf_coef(inf_loss=0.15)

    assert np.size(axs) >= len(a_fields)


def test_plot_autopsf_coef_too_few_axis(random_4psf_simage):
    simg = random_4psf_simage
    with pytest.raises(ValueError):
        simg.plot.autopsf_coef(inf_loss=0.15, axs=[plt.gca()])


@check_figures_equal()
def test_plot_autopsf_coef(random_4psf_simage, fig_test, fig_ref):
    simg = random_4psf_simage

    # expected
    a_fields, psf_basis = simg.get_variable_psf(inf_loss=0.15)
    x, y = simg.get_afield_domain()

    # here we plot
    N = len(a_fields)  # axis needed
    p = plot.primes(N)

    if N == 2:
        subplots = (1, 2)
    if N == 3:
        subplots = (1, 3)
    elif p == N:
        subplots = (round(np.sqrt(N)), round(np.sqrt(N) + 1))
    else:
        rows = int((N // p) + (N % p))
        subplots = (rows, p)

    width = plot.DEFAULT_WIDTH * subplots[0]
    height = plot.DEFAULT_HEIGHT * subplots[1]

    fig_ref.set_size_inches(w=width, h=height)
    exp_axs = fig_ref.subplots(*subplots)

    cmap_kw = {"shrink": 0.75, "aspect": 30}

    title_tpl = r"$a_{j}$,$\sum a_{j}={sum:4.3e}$"
    for idx, a_field, ax in zip(range(N), a_fields, np.ravel(exp_axs)):

        a = a_field(x, y)
        mean, med, std = sigma_clipped_stats(a)

        img = ax.imshow(a, vmax=med + 2 * std, vmin=med - 2 * std)
        fig_ref.colorbar(img, ax=ax, **cmap_kw)

        title = title_tpl.format(j=idx + 1, sum=np.sqrt(np.sum(a**2)))
        ax.set_title(title)

    # fig test
    fig_test.set_size_inches(w=width, h=height)
    test_axs = fig_test.subplots(*subplots)
    simg.plot.autopsf_coef(axs=test_axs, inf_loss=0.15)


@check_figures_equal()
def test_plot_autopsf_coef_str(random_4psf_simage, fig_test, fig_ref):
    simg = random_4psf_simage

    with mock.patch("matplotlib.pyplot.gcf", return_value=fig_test):
        simg.plot("autopsf_coef", inf_loss=0.15)

    with mock.patch("matplotlib.pyplot.gcf", return_value=fig_ref):
        simg.plot.autopsf_coef(inf_loss=0.15)
