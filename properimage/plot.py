#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  plot.py
#
#  Copyright 2020 QuatroPe
#
# This file is part of ProperImage (https://github.com/quatrope/ProperImage)
# License: BSD-3-Clause
# Full Text: https://github.com/quatrope/ProperImage/blob/master/LICENSE.txt
#

"""plot module from ProperImagefor coadding astronomical images.

This module contains plotting utilities and plugins.
"""

import logging

from astropy.stats import sigma_clipped_stats

import attr

import matplotlib.pyplot as plt

import numpy as np


# =============================================================================
# CONSTANTS
# =============================================================================

FONT = {
    "family": "sans-serif",
    "sans-serif": ["Computer Modern Sans serif"],
    "weight": "regular",
    "size": 12,
}

TEXT = {"usetex": True}

DEFAULT_HEIGHT = 4
DEFAULT_WIDTH = 4


logger = logging.getLogger()

# =============================================================================
# EXCEPTIONS
# =============================================================================


class NoDataToPlot(ValueError):
    """Exception for empty data."""

    pass


# =============================================================================
# FUNCTIONS
# =============================================================================


def primes(n):
    """Get maximum prime number factor."""
    divisors = [d for d in range(2, n // 2 + 1) if n % d == 0]
    prims = [
        d for d in divisors if all(d % od != 0 for od in divisors if od != d)
    ]
    if len(prims) >= 4:
        return prims[-1] * prims[-2]
    elif len(prims) == 0:
        return n

    return max(prims)


def plot_S(S, path=None, nbook=False):
    """Plot an S-type image subtraction.

    Parameters
    ----------
    S : array_like
        Image to plot
    path : str, optional
        Path to store the plot
    nbook : bool, optional
        Whether we are plotting in a notebook

    Returns
    -------
    ax : matplotlib axes
    """
    if isinstance(S, np.ma.masked_array):
        S = S.filled()
    mean, med, std = sigma_clipped_stats(S)
    plt.imshow(
        S,
        vmax=med + 4 * std,
        vmin=med - 4 * std,
        interpolation="none",
        cmap="viridis",
    )
    plt.tight_layout()
    plt.colorbar()
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
    if not nbook:
        plt.close()
    return plt.gca()


def plot_R(R, path=None, nbook=False):
    """Plot an D-type image subtraction.

    Parameters
    ----------
    R : array_like
        Image to plot
    path : str, optional
        Path to store the plot
    nbook : bool, optional
        Whether we are plotting in a notebook

    Returns
    -------
    ax : matplotlib axes
    """
    if isinstance(R[0, 0], complex):
        R = R.real
    if isinstance(R, np.ma.masked_array):
        R = R.filled()
    plt.imshow(np.log10(R), interpolation="none", cmap="viridis")
    plt.tight_layout()
    plt.colorbar()
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
    if not nbook:
        plt.close()
    return


# =============================================================================
# PLOT API
# =============================================================================


@attr.s(frozen=True, repr=False, eq=False, order=False)
class Plot:
    """Plotting plug-in object for SingleImage class."""

    DEFAULT_PLOT = "imshow"

    si = attr.ib()

    def __call__(self, plot=None, **kwargs):
        """Execute plot."""
        if plot is not None and (
            plot.startswith("_") or not hasattr(self, plot)
        ):
            raise ValueError(f"Ivalid plot method '{plot}'")
        method = getattr(self, plot or Plot.DEFAULT_PLOT)
        if not callable(method):
            raise ValueError(f"Ivalid plot method '{plot}'")
        return method(**kwargs)

    def imshow(self, ax=None, **kwargs):
        """Plot 2d image."""
        if ax is None:
            fig = plt.gcf()
            ax = plt.gca()
            fig.set_size_inches(h=DEFAULT_HEIGHT, w=DEFAULT_WIDTH)

        kwargs.setdefault("origin", "lower")

        ax.imshow(self.si.data, **kwargs)

        ax.set_title(f"SingleImage {self.si.data.shape}")
        return ax

    def autopsf(
        self,
        axs=None,
        iso=False,
        inf_loss=None,
        shape=None,
        cmap_kw=None,
        iso_kw=None,
        **kwargs,
    ):
        """Plot autopsf basis components."""
        _, psf_basis = self.si.get_variable_psf(inf_loss=inf_loss, shape=shape)

        # here we plot
        N = len(psf_basis)

        if axs is None:
            p = primes(N)

            fig = plt.gcf()

            if N == 1:
                subplots = (1, 1)
            if N == 2:
                subplots = (1, 2)
            if N == 3:
                subplots = (1, 3)
            elif p == N:
                subplots = (round(np.sqrt(N)), round(np.sqrt(N) + 1))
            else:
                rows = int((N // p) + (N % p))
                subplots = (rows, p)

            height = DEFAULT_HEIGHT * subplots[0]
            width = DEFAULT_WIDTH * subplots[1]

            fig.set_size_inches(w=width, h=height)
            axs = fig.subplots(*subplots)

        if N > np.size(axs):
            raise ValueError(
                f"You must provide at least {N} axs. Found {len(axs)}"
            )

        xsh, ysh = psf_basis[0].shape

        kwargs.setdefault("interpolation", "none")

        iso_kw = iso_kw or {}
        iso_kw = {"colors": "black", "alpha": 0.5}

        cmap_kw = cmap_kw or {}
        cmap_kw.setdefault("shrink", 0.85)

        title_tpl = r"$\sum p_{j:d} = {sum:4.3e}$"
        for idx, psf_basis, ax in zip(range(N), psf_basis, np.ravel(axs)):
            fig = ax.get_figure()

            img = ax.imshow(psf_basis, **kwargs)
            title = title_tpl.format(j=idx + 1, sum=np.sum(psf_basis))
            ax.set_title(title)

            fig.colorbar(img, ax=ax, **cmap_kw)

            if iso:
                ax.contour(np.arange(xsh), np.arange(ysh), psf_basis, **iso_kw)

        return axs

    def autopsf_coef(
        self, axs=None, inf_loss=None, shape=None, cmap_kw=None, **kwargs
    ):
        """Plot autopsf basis component coefficients."""
        a_fields, _ = self.si.get_variable_psf(inf_loss=inf_loss, shape=shape)
        if a_fields == [None]:
            raise NoDataToPlot("No coeficients for this PSF")

        x, y = self.si.get_afield_domain()

        # here we plot
        N = len(a_fields)  # axis needed

        if axs is None:
            p = primes(N)

            fig = plt.gcf()

            if N == 2:
                subplots = (1, 2)
            if N == 3:
                subplots = (1, 3)
            elif p == N:
                subplots = (round(np.sqrt(N)), round(np.sqrt(N) + 1))
            else:
                rows = int((N // p) + (N % p))
                subplots = (rows, p)

            height = DEFAULT_HEIGHT * subplots[0]
            width = DEFAULT_WIDTH * subplots[1]

            fig.set_size_inches(w=width, h=height)
            axs = fig.subplots(*subplots)

        if N > np.size(axs):
            raise ValueError(
                f"You must provide at least {N} axs. Found {len(axs)}"
            )

        cmap_kw = cmap_kw or {}
        cmap_kw.setdefault("shrink", 0.75)
        cmap_kw.setdefault("aspect", 30)

        title_tpl = r"$a_{j}$,$\sum a_{j}={sum:4.3e}$"
        for idx, a_field, ax in zip(range(N), a_fields, np.ravel(axs)):
            fig = ax.get_figure()

            a = a_field(x, y)

            _, med, std = sigma_clipped_stats(a)

            imshow_kw = kwargs.copy()
            imshow_kw.setdefault("vmax", med + 2 * std)
            imshow_kw.setdefault("vmin", med - 2 * std)

            img = ax.imshow(a, **imshow_kw)
            fig.colorbar(img, ax=ax, **cmap_kw)

            title = title_tpl.format(j=idx + 1, sum=np.sqrt(np.sum(a**2)))
            ax.set_title(title)

        return axs
