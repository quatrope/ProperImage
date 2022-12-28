#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  simtools.py
#
#  Copyright 2020 QuatroPe
#
# This file is part of ProperImage (https://github.com/quatrope/ProperImage)
# License: BSD-3-Clause
# Full Text: https://github.com/quatrope/ProperImage/blob/master/LICENSE.txt
#

"""
simtools module of ProperImage package for astronomical image analysis.

This module contains utilities for mocking images, and simulating data.
"""

import os
from functools import reduce


# from astropy.convolution import convolve_fft
from astropy.io import fits
from astropy.modeling import models
from astropy.time import Time

import numpy as np
from numpy.random import default_rng

import scipy as sp
from scipy import signal as sg
from scipy import stats
from scipy.ndimage import rotate

random_def = default_rng(seed=110112)


def Psf(N, X_FWHM, Y_FWHM=0, theta=0):
    """Mock a point spread function PSF, of size NxN.

    It uses a gaussian in both axis.
    """
    if Y_FWHM == 0:
        Y_FWHM = X_FWHM

    a = np.zeros((int(N), int(N)))
    mu = (N - 1) / 2.0
    sigma_x = X_FWHM / 2.335
    sigma_y = Y_FWHM / 2.335
    sigma = max(sigma_x, sigma_y)
    tail_len = min(int(5 * sigma), N / 2)
    mu_int = int(mu)
    for i in range(int(mu_int - tail_len), int(mu_int + tail_len), 1):
        for j in range(int(mu_int - tail_len), int(mu_int + tail_len), 1):
            a[i, j] = stats.norm.pdf(
                i, loc=mu, scale=sigma_x
            ) * stats.norm.pdf(j, loc=mu, scale=sigma_y)
    if theta != 0:
        a = rotate(a, theta)
    return a / np.sum(a)


def astropy_Psf(N, FWHM):
    """Mock a gaussian point spread function PSF, of size NxN.

    Psf es una funcion que proporciona una matriz 2D con una gaussiana
    simétrica en ambos ejes. con N se especifica el tamaño en pixeles que
    necesitamos y con FWHM el ancho sigma de la gaussiana en pixeles.
    """
    psf = np.zeros((N, N))
    mu = (N - 1) / 2.0
    sigma = FWHM / 2.335
    model = models.Gaussian2D(
        amplitude=1.0, x_mean=mu, y_mean=mu, x_stddev=sigma, y_stddev=sigma
    )
    tail_len = int(7 * sigma)
    mu_int = int(mu)
    i = range(mu_int - tail_len, mu_int + tail_len, 1)
    for ii, jj in cartesian_product([i, i]):
        psf[ii, jj] = model(ii, jj)
    return psf / np.sum(psf)


def _airy_func(rr, width, amplitude=1.0):
    """Evaluate Airy Disc function.

    For a simple radially symmetric airy function, returns the value at a given
    (normalized) radius
    """
    r = float(rr) / width
    return amplitude * (2.0 * sp.special.j1(r) / r) ** 2


def airy_patron(N, width):
    """Generate Airy disc pattern in a 2D array.

    Function that generates an Airy disc pattern, in a 2D array,
    which represents the telescope pupil optical transfer function,
    with a functional form of

      sin(theta) = 1.22 (Lambda / D)

    with theta is the distance from the pattern center to its first
    minimum, Lambda is the light wavelength, and D is the pupil
    aperture diameter.

    Parameters
    ----------
    N: integer, the array size (in pixels)
    width: the theta angle already estimated in pixel units.
    """
    mu = (N - 1) / 2.0
    a = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            r_pix = np.sqrt((i - mu) ** 2 + (j - mu) ** 2)
            a[i, j] = _airy_func(r_pix, width)
    return a


def convol_gal_psf_fft(gal, a):
    """Convolve two matrices.

    Esta funcion convoluciona dos matrices, a pesar de su nombre es una FFT
    para matrices 2D, y se usa con el patron de airy tambien

    convol_gal_psf_fft(gal, a)

    retorna la convolucion de gal x a, usando la misma forma matricial que gal.

    FASTER
    """
    b = sg.fftconvolve(gal, a, mode="same")
    return b


def perfilsersic(r_e, I_e, n, r):
    """Evaluate a Sersic Profile.

    funcion que evalua a un dado radio r el valor de
    brillo correspondiente a un perfil de sersic
         r_e  :  Radio de escala
         I_e  :  Intensidad de escala
         n    :  Indice de Sersic
         r    :  Radio medido desde el centro en pixeles
    """
    b = 1.999 * n - 0.327
    I_r = I_e * np.exp(-b * (((r / r_e) ** (1 / float(n))) - 1))
    I_r = I_r / (I_e * np.exp(-b * (((0.0 / r_e) ** (1 / float(n))) - 1)))
    return I_r


def gal_sersic(N, n):
    """Generate a 2D NxN matrix with a Sersic galaxy profile."""
    gal = np.zeros((N, N))
    mu = (N - 1) / 2.0  # calculo posicion del pixel central
    # radio de escala, tomado como un
    # sexto del ancho de la imagen para que la galaxia no ocupe toda la imagen
    R_e = (N - 1) / 6.0
    for i in range(N - 1):
        for j in range(N - 1):
            r_pix = np.sqrt((i - mu) ** 2 + (j - mu) ** 2)
            if r_pix <= (4.0 * R_e):
                gal[i, j] = perfilsersic(R_e, 10, n, r_pix)
            else:
                gal[i, j] = 0
    return gal


def cartesian_product(arrays):
    """Create a cartesian product array from a list of arrays.

    It is used to create x-y coordinates array from x and y arrays.

    Stolen from stackoverflow
    http://stackoverflow.com/a/11146645
    """
    broadcastable = np.ix_(*arrays)
    broadcasted = np.broadcast_arrays(*broadcastable)
    rows, cols = reduce(np.multiply, broadcasted[0].shape), len(broadcasted)
    out = np.empty(rows * cols, dtype=broadcasted[0].dtype)
    start, end = 0, rows
    for a in broadcasted:
        out[start:end] = a.reshape(-1)
        start, end = end, end + rows
    return out.reshape(cols, rows).T


def delta_point(N, center=True, xy=None, weights=None):
    """Create delta sources in a square NxN matrix.

    If center is True (default) it will locate a unique delta
    in the center of the image.
    Else it will need a xy list of positions where to put the deltas.
    It can handle weights for different source flux.

    Returns a NxN numpy array.

    Example
    -------
    N = 100
    x = np.random.integers(10, 80, size=10)
    y = np.random.integers(10, 80, size=10)

    m = delta_point(N, center=False, xy=zip(x, y))
    """
    m = np.zeros(shape=(N, N))

    if center:
        m[int(N / 2.0), int(N / 2.0)] = 1.0
    else:
        if weights is None:
            weights = np.ones(shape=len(xy))
        j = -1
        for x, y in xy:
            w = weights[j]
            m[int(x), int(y)] = 1.0 * w
            j -= 1
    return m


def image(
    MF,
    N,
    t_exp,
    X_FWHM,
    SN,
    Y_FWHM=0,
    theta=0,
    bkg_pdf="poisson",
    std=None,
    seed=None,
    bias=100.0,
):
    """Generate an image with noise and a seeing profile for stars.

    Parameters
    ----------
    MF : array_like
        master image for image creating
    FWHM : float
        Seeing FWHM in pixels
    t_exp : float
        Exposure time of image
    N : int
        Final image size
    SN : float
        Signal to Noise ratio related to sky noise
    bkg_pdf : str, "poisson" or "gaussian"
        probability distribution of sky noise
    std : float
        For gaussian kind of background the standard deviation.
    """
    random = default_rng(seed=seed) if seed is not None else random_def

    FWHM = max(X_FWHM, Y_FWHM)
    PSF = Psf(5 * FWHM, X_FWHM, Y_FWHM, theta)
    IM = convol_gal_psf_fft(MF, PSF)

    b = np.max(IM)  # image[int(N2/2), int(N2/2)]

    if bkg_pdf == "poisson":
        mean = b / SN
        C = random.poisson(mean, (N, N)).astype(np.float32)

    elif bkg_pdf == "gaussian":
        mean = 0
        std = b / SN
        C = random.normal(mean, std, (N, N))
    return C + IM + bias


def store_fits(gal, t, t_exp, i, zero, path="."):
    """Store an array as a fits image.

    gal : array_like
        Base image to store
    t : astropy.time.Time
        Time of observation
    t_exp : float
        Exposure time of image
    i : int
        Image number
    zero : float
        Photometric zeropoint
    """
    file = fits.PrimaryHDU(gal)
    hdulist = fits.HDUList([file])
    hdr = hdulist[0].header
    if isinstance(t, Time):
        tt = t
    else:
        tt = Time(t, format="jd", scale="utc")
    day = tt.iso[0:10]
    hour = tt.iso[11:24]
    jd = tt.jd
    hdr.set("TIME-OBS", hour)
    hdr.set("DATE-OBS", day)
    hdr.set("EXPTIME", t_exp)
    hdr.set("JD", jd)
    hdr.set("ZERO_P", zero)
    path_fits = os.path.join(path, ("image00" + str(i) + ".fits"))
    hdulist.writeto(path_fits, overwrite=True)
    return path_fits


def sim_varpsf(nstars, SN=30.0, thetas=[0, 45, 105, 150], N=512, seed=None):
    """Simulate an image with variable PSF.

    Args:
        nstars : int
            Number of stars
        SN (float, optional): SN ratio. Defaults to 30.0.
        thetas (list, optional): angles of PSF. Defaults to [0, 45, 105, 150].
        N (int, optional): size of image. Defaults to 512.
        seed (int, optional): random seed. Defaults to None.

    Returns:
        frame : array_like
            Simulated image.
    """
    random = default_rng(seed=seed) if seed is not None else random_def

    frames = []
    for theta in thetas:
        X_FWHM = 5 + 5.0 * theta / 180.0
        Y_FWHM = 5
        bias = 100.0
        t_exp = 1
        max_fw = max(X_FWHM, Y_FWHM)

        max_pix = N - 3 * max_fw
        min_pix = 3 * max_fw

        x = random.integers(low=min_pix, high=max_pix, size=nstars // 4)
        y = random.integers(low=min_pix, high=max_pix, size=nstars // 4)

        weights = list(np.linspace(100.0, 10000.0, len(x)))
        m = delta_point(N, center=False, xy=zip(x, y), weights=weights)
        im = image(
            m,
            N,
            t_exp,
            X_FWHM,
            Y_FWHM=Y_FWHM,
            theta=theta,
            SN=SN,
            bkg_pdf="gaussian",
            seed=seed,
            bias=bias,
        )
        frames.append(im + bias)

    frame = np.zeros((2 * N, 2 * N))
    for j in range(2):
        for i in range(2):
            frame[i * N : (i + 1) * N, j * N : (j + 1) * N] = frames[i + 2 * j]

    return frame
