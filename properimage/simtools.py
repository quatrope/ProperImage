#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2017 Bruno S <bruno@oac.unc.edu.ar>
#
# This file is part of ProperImage (https://github.com/toros-astro/ProperImage)
# License: BSD-3-Clause
# Full Text: https://github.com/toros-astro/ProperImage/blob/master/LICENSE.txt
#

"""
simtools module from ProperImage, with
utilities for mocking images, and simulating data

Written by Bruno SANCHEZ

PhD of Astromoy - UNC
bruno@oac.unc.edu.ar

Instituto de Astronomia Teorica y Experimental (IATE) UNC
Cordoba - Argentina

Of 301
"""

import os
from functools import reduce
import numpy as np

import scipy as sp
from scipy import signal as sg
from scipy.ndimage.interpolation import rotate
from scipy import stats

# from astropy.convolution import convolve_fft
from astropy.modeling import models
from astropy.io import fits
from astropy.time import Time


def Psf(N, X_FWHM, Y_FWHM=0, theta=0):
    """Psf mocks a point spread function, of size NxN, with a symmetric
    gaussian in both axis.
    theta is in degrees

    FASTER
    %timeit simtools.Psf(128, 10)
    1 loops, best of 3: 234 ms per loop

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
    """Psf es una funcion que proporciona una matriz 2D con una gaussiana
    simétrica en ambos ejes. con N se especifica el tamaño en pixeles que
    necesitamos y con FWHM el ancho sigma de la gaussiana en pixeles

    %timeit simtools.astropy_Psf(128, 10)
    1 loops, best of 3: 338 ms per loop
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
    """
    For a simple radially symmetric airy function, returns the value at a given
    (normalized) radius
    """
    r = float(rr) / width
    return amplitude * (2.0 * sp.special.j1(r) / r) ** 2


def airy_patron(N, width):
    """Esta funcion genera un patron de airy, en una matriz 2D
     el cual es la impronta
     del espejo del telescopio y sigue una relacion de

       sin(theta) = 1.22 (Lambda / D)

     donde theta es la distancia desde el centro del patron a el primer
     minimo del mismo, lambda es la longitud de onda de la radiacion que
     colecta el telescopio, y D es el diametro del objetivo del telescopio
     N es el tamaño de la matriz en pixeles
     width es el theta ya calculado. Es importante saber que este theta
     depende del CCD también, ya que esta construida la funcion en pixeles
    """
    mu = (N - 1) / 2.0
    a = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            r_pix = np.sqrt((i - mu) ** 2 + (j - mu) ** 2)
            a[i, j] = _airy_func(r_pix, width)
    return a


def convol_gal_psf_fft(gal, a):
    """
    Esta funcion convoluciona dos matrices, a pesar de su nombre es una FFT
    para matrices 2D, y se usa con el patron de airy tambien

    convol_gal_psf_fft(gal, a)

    retorna la convolucion de gal x a, usando la misma forma matricial que gal.

    FASTER
    """
    b = sg.fftconvolve(gal, a, mode="same")
    return b


def perfilsersic(r_e, I_e, n, r):
    """
    funcion que evalua a un dado radio r el valor de
    brillo correspondiente a un perfil de sersic
         r_e  :  Radio de escala
         I_e  :  Intensidad de escala
         n    :  Indice de Sersic
         r    :  Radio medido desde el centro en pixeles
    """
    b = 1.999 * n - 0.327
    I_r = I_e * np.exp(-b * (((r / r_e) ** (1 / np.float(n))) - 1))
    I_r = I_r / (I_e * np.exp(-b * (((0.0 / r_e) ** (1 / np.float(n))) - 1)))
    return I_r


def gal_sersic(N, n):
    """
    esta funcion genera una matriz 2D que posee en sus componentes
    un perfil de sersic centrado en la matriz cuadrada, con un tamaño N
    y que posee un indice de sersic n
    """
    gal = np.zeros((N, N))
    mu = (N - 1) / 2.0  # calculo posicion del pixel central
    # radio de escala, tomado como un
    # sexto del ancho de la imagen para que la galaxia no ocupe toda la imagen
    R_e = (N - 1) / 6.0
    for i in range(N - 1):
        for j in range(N - 1):
            r_pix = np.sqrt((i - mu) ** 2 + (j - mu) ** 2)
            if r_pix <= (4.5 * R_e):
                gal[i, j] = perfilsersic(R_e, 10, n, r_pix)
            else:
                gal[i, j] = 0
    return gal


def cartesian_product(arrays):
    """
    Creates a cartesian product array from a list of arrays.

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
    """
    Function to create delta sources in a square NxN matrix.
    If center is True (default) it will locate a unique delta
    in the center of the image.
    Else it will need a xy list of positions where to put the deltas.
    It can handle weights for different source flux.

    Returns a NxN numpy array.

    Example:
    N = 100
    x = [np.random.randint(10, 80) for j in range(10)]
    y = [np.random.randint(10, 80) for j in range(10)]
    xy = [(x[i], y[i]) for i in range(10)]

    m = delta_point(N, center=False, xy=xy)
    """
    m = np.zeros(shape=(N, N))
    if center is False:
        if weights is None:
            weights = list(np.repeat(1.0, len(xy)))
        j = -1
        for x, y in xy:
            w = weights[j]
            m[int(x), int(y)] = 1.0 * w
            j -= 1
    else:
        m[int(N / 2.0), int(N / 2.0)] = 1.0
    return m


def image(
    MF, N2, t_exp, X_FWHM, SN, Y_FWHM=0, theta=0, bkg_pdf="poisson", std=None
):
    """
    funcion que genera una imagen con ruido y seeing a partir
    de un master frame, y la pixeliza hasta tamaño N2

    Parametros
    ----------
    IMC : imagen Master
    FWHM : tamaño en pixeles del FWHM del seeing
    t_exp : tiempo exposicion de la imagen
    N2 : tamaño de la imagen final, pixelizada
    SN : es la relacion señal ruido con el fondo del cielo
    bkg_pdf : distribucion de probabilidad del ruido background
    std : en caso que bkg_pdf sea gaussian, valor de std
    """
    N = np.shape(MF)[0]
    FWHM = max(X_FWHM, Y_FWHM)
    PSF = Psf(5 * FWHM, X_FWHM, Y_FWHM, theta)
    IM = convol_gal_psf_fft(MF, PSF)

    if N != N2:
        image = IM  # pixelize(N/N2, IM)  overriden
    else:
        image = IM

    b = np.max(image)  # image[int(N2/2), int(N2/2)]

    if bkg_pdf == "poisson":
        mean = b / SN
        print("mean = {}, b = {}, SN = {}".format(mean, b, SN))
        C = np.random.poisson(mean, (N2, N2)).astype(np.float32)

    elif bkg_pdf == "gaussian":
        mean = 0
        std = b / SN
        print("mean = {}, std = {}, b = {}, SN = {}".format(mean, std, b, SN))
        C = np.random.normal(mean, std, (N2, N2))
    bias = 100.0
    F = C + image + bias
    return F


def capsule_corp(gal, t, t_exp, i, zero, path=".", round_int=False):
    """
    funcion que encapsula las imagenes generadas en fits
    gal        :   Imagen (matriz) a encapsular
    t          :   Tiempo en dias julianos de esta imagen
    t_exp      :   Tiempo de exposicion de la imagen
    i          :   Numero de imagen
    zero       :   Punto cero de la fotometria
    """
    if round_int:
        gal = gal.astype(int)

    file1 = fits.PrimaryHDU(gal)
    hdulist = fits.HDUList([file1])
    hdr = hdulist[0].header
    if t.__class__.__name__ == "Time":
        dia = t.iso[0:10]
        hora = t.iso[11:24]
        jd = t.jd
    else:
        time = Time(t, format="jd", scale="utc")
        dia = time.iso[0:10]
        hora = time.iso[11:24]
        jd = time.jd
    hdr.set("TIME-OBS", hora)
    hdr.set("DATE-OBS", dia)
    hdr.set("EXPTIME", t_exp)
    hdr.set("JD", jd)
    hdr.set("ZERO_P", zero)
    path_fits = os.path.join(path, ("image00" + str(i) + ".fits"))
    hdulist.writeto(path_fits, clobber=True)
    return path_fits


def sim_varpsf(nstars, SN=3.0, thetas=[0, 45, 105, 150], N=512):
    frames = []
    for theta in thetas:
        X_FWHM = 5 + 5.0 * theta / 180.0
        Y_FWHM = 5
        bias = 100.0
        t_exp = 1
        max_fw = max(X_FWHM, Y_FWHM)

        x = np.random.randint(
            low=6 * max_fw, high=N - 6 * max_fw, size=nstars / 4
        )
        y = np.random.randint(
            low=6 * max_fw, high=N - 6 * max_fw, size=nstars / 4
        )
        xy = np.array([(x[i], y[i]) for i in range(nstars / 4)])

        weights = list(np.linspace(100.0, 10000.0, len(xy)))
        m = delta_point(N, center=False, xy=xy, weights=weights)
        im = image(
            m,
            N,
            t_exp,
            X_FWHM,
            Y_FWHM=Y_FWHM,
            theta=theta,
            SN=SN,
            bkg_pdf="gaussian",
        )
        frames.append(im + bias)

    frame = np.zeros((2 * N, 2 * N))
    for j in range(2):
        for i in range(2):
            frame[i * N : (i + 1) * N, j * N : (j + 1) * N] = frames[i + 2 * j]

    return frame
