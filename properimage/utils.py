#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  utils.py
#
#  Copyright 2016 Bruno S <bruno@oac.unc.edu.ar>
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

"""utils module from ProperImage,
for coadding astronomical images.

Written by Bruno SANCHEZ

PhD of Astromoy - UNC
bruno@oac.unc.edu.ar

Instituto de Astronomia Teorica y Experimental (IATE) UNC
Cordoba - Argentina

Of 301
"""

import os

import numpy as np

from scipy import sparse
import scipy.ndimage as ndimage

from numpy.lib.recfunctions import append_fields

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve, convolve_fft

from astroML import crossmatch as cx

import astroalign as aa

aa.PIXEL_TOL = 0.3
aa.NUM_NEAREST_NEIGHBORS = 5
aa.MIN_MATCHES_FRACTION = 0.6


def encapsule_S(S, path=None):
    if isinstance(S, np.ma.core.MaskedArray):
        mask = S.mask.astype("int")
        data = S.data
        hdu_data = fits.PrimaryHDU(data)
        hdu_data.scale(type="float32")
        hdu_mask = fits.ImageHDU(mask, uint="uint8")
        hdu_mask.header["IMG_TYPE"] = "BAD_PIXEL_MASK"
        hdu = fits.HDUList([hdu_data, hdu_mask])
    else:
        hdu = fits.PrimaryHDU(S)
    if path is not None:
        hdu.writeto(path, overwrite=True)
    else:
        return hdu


def encapsule_R(R, path=None, header=None):
    if isinstance(R[0, 0], np.complex):
        R = R.real
    if isinstance(R, np.ma.core.MaskedArray):
        mask = R.mask.astype("int")
        data = R.data
        hdu_data = fits.PrimaryHDU(data, uint="int16")
        hdu_data.scale(type="int16")
        hdu_mask = fits.ImageHDU(mask, uint="uint8")
        hdu_mask.header["IMG_TYPE"] = "BAD_PIXEL_MASK"
        hdu = fits.HDUList([hdu_data, hdu_mask])
    else:
        hdu = fits.PrimaryHDU(R)
    if header is not None:
        pass
    if path is not None:
        hdu.writeto(path, overwrite=True)
    else:
        return hdu


def matching(
    master, cat, masteridskey=None, angular=False, radius=1.5, masked=False
):
    """Function to match stars between frames.
    """
    if masteridskey is None:
        masterids = np.arange(len(master))
        master["masterindex"] = masterids
        idkey = "masterindex"
    else:
        idkey = masteridskey

    if angular:
        masterRaDec = np.empty((len(master), 2), dtype=np.float64)
        try:
            masterRaDec[:, 0] = master["RA"]
            masterRaDec[:, 1] = master["Dec"]
        except KeyError:
            masterRaDec[:, 0] = master["ra"]
            masterRaDec[:, 1] = master["dec"]
        imRaDec = np.empty((len(cat), 2), dtype=np.float64)
        try:
            imRaDec[:, 0] = cat["RA"]
            imRaDec[:, 1] = cat["Dec"]
        except KeyError:
            imRaDec[:, 0] = cat["ra"]
            imRaDec[:, 1] = cat["dec"]
        radius2 = radius / 3600.0
        dist, ind = cx.crossmatch_angular(
            masterRaDec, imRaDec, max_distance=radius2 / 2.0
        )
        dist_, ind_ = cx.crossmatch_angular(
            imRaDec, masterRaDec, max_distance=radius2 / 2.0
        )
    else:
        masterXY = np.empty((len(master), 2), dtype=np.float64)
        masterXY[:, 0] = master["x"]
        masterXY[:, 1] = master["y"]
        imXY = np.empty((len(cat), 2), dtype=np.float64)
        imXY[:, 0] = cat["x"]
        imXY[:, 1] = cat["y"]
        dist, ind = cx.crossmatch(masterXY, imXY, max_distance=radius)
        dist_, ind_ = cx.crossmatch(imXY, masterXY, max_distance=radius)

    IDs = np.zeros_like(ind_) - 13133
    for i in range(len(ind_)):
        if dist_[i] != np.inf:
            # dist_o = dist_[i]
            ind_o = ind_[i]
            if dist[ind_o] != np.inf:
                # dist_s = dist[ind_o]
                ind_s = ind[ind_o]
                if ind_s == i:
                    IDs[i] = master[idkey][ind_o]

    # print((len(IDs), len(ind_), len(ind)))
    # print(("Matching result::  IDs > 0. => {}".format(sum(IDs > 0))))
    if masked:
        mask = IDs > 0
        return (IDs, mask)
    return IDs


def transparency(images, master=None):
    """Transparency calculator, using Ofek method."""

    if master is None:
        p = len(images)
        master = images[0]
        imglist = images[1:]
    else:
        # master is a separated file
        p = len(images) + 1
        imglist = images

    mastercat = master.best_sources
    try:
        mastercat = append_fields(
            mastercat,
            "sourceid",
            np.arange(len(mastercat)),
            usemask=False,
            dtypes=int,
        )
    except ValueError:
        pass

    detect = np.repeat(True, len(mastercat))
    #  Matching the sources
    for img in imglist:
        newcat = img.best_sources
        ids, mask = matching(
            mastercat,
            newcat,
            masteridskey="sourceid",
            angular=False,
            radius=2.0,
            masked=True,
        )
        try:
            newcat = append_fields(newcat, "sourceid", ids, usemask=False)
        except ValueError:
            newcat["sourceid"] = ids

        for i in range(len(mastercat)):
            if mastercat[i]["sourceid"] not in ids:
                detect[i] = False
        newcat.sort(order="sourceid")
        img.update_sources(newcat)
    try:
        mastercat = append_fields(
            mastercat, "detected", detect, usemask=False, dtypes=bool
        )
    except ValueError:
        mastercat["detected"] = detect

    # Now populating the vector of magnitudes
    q = sum(mastercat["detected"])

    if q != 0:
        m = np.zeros(p * q)
        # here 20 is a common value for a zp, and is only for weighting
        m[:q] = (
            -2.5 * np.log10(mastercat[mastercat["detected"]]["flux"]) + 20.0
        )

        j = 0
        for row in mastercat[mastercat["detected"]]:
            for img in imglist:
                cat = img.best_sources
                imgrow = cat[cat["sourceid"] == row["sourceid"]]
                m[q + j] = -2.5 * np.log10(imgrow["flux"]) + 20.0
                j += 1
        # print mastercat['detected']
        master.update_sources(mastercat)

        # print("p={}, q={}".format(p, q))
        ident = sparse.identity(q)
        col = np.repeat(1.0, q)
        sparses = []
        for j in range(p):
            ones_col = np.zeros((q, p))
            ones_col[:, j] = col
            sparses.append([sparse.csc_matrix(ones_col), ident])

        H = sparse.bmat(sparses)

        P = sparse.linalg.lsqr(H, m)
        zps = P[0][:p]

        meanmags = P[0][p:]

        return zps, meanmags
    else:
        return np.ones(p), np.nan


def convolve_psf_basis(image, psf_basis, a_fields, x, y):
    imconvolved = np.zeros_like(image)
    for j in range(len(psf_basis)):
        a = a_fields[j](x, y) * image
        # a = a_fields[j] * image
        psf = psf_basis[j]

        imconvolved += convolve(a, psf, boundary="extend")

    return imconvolved


def fftconvolve_psf_basis(image, psf_basis, a_fields, x, y):
    imconvolved = np.zeros_like(image)
    for j in range(len(psf_basis)):
        a = a_fields[j](x, y) * image
        psf = psf_basis[j]

        imconvolved += convolve_fft(
            a, psf, interpolate_nan=True, allow_huge=True
        )

    return imconvolved


def lucy_rich(
    image, psf_basis, a_fields, adomain, iterations=50, clip=True, fft=False
):
    # direct_time = np.prod(image.shape + psf.shape)
    # fft_time =  np.sum([n*np.log(n) for n in image.shape + psf.shape])

    # see whether the fourier transform convolution method or the direct
    # convolution method is faster (discussed in scikit-image PR #1792)
    # time_ratio = 40.032 * fft_time / direct_time

    if fft:
        convolve_method = fftconvolve_psf_basis
    else:
        convolve_method = convolve_psf_basis

    image = image.astype(np.float)
    image = np.ma.masked_invalid(image).filled(np.nan)
    # x, y = np.mgrid[:image.shape[0], :image.shape[1]]
    x, y = adomain

    im_deconv = 0.5 * np.ones(image.shape)
    psf_mirror = [psf[::-1, ::-1] for psf in psf_basis]

    for _ in range(iterations):
        rela_blur = image / convolve_method(
            im_deconv, psf_basis, a_fields, x, y
        )
        im_deconv *= convolve_method(rela_blur, psf_mirror, a_fields, x, y)

    if clip:
        im_deconv = np.ma.masked_invalid(im_deconv).filled(-1.0)

    return im_deconv


def align_for_diff(refpath, newpath, newmask=None):
    """Function to align two images using their paths,
    and returning newpaths for differencing.
    We will allways rotate and align the new image to the reference,
    so it is easier to compare differences along time series.
    """
    ref = np.ma.masked_invalid(fits.getdata(refpath))
    new = fits.getdata(newpath)
    hdr = fits.getheader(newpath)
    if newmask is not None:
        new = np.ma.masked_array(new, mask=fits.getdata(newmask))
    else:
        new = np.ma.masked_invalid(new)

    dest_file = "aligned_" + os.path.basename(newpath)
    dest_file = os.path.join(os.path.dirname(newpath), dest_file)

    try:
        new2 = aa.register(new.filled(np.median(new)), ref)
    except ValueError:
        ref = ref.astype(float)
        new = new.astype(float)
        new2 = aa.register(new, ref)

    hdr.set("comment", "aligned img " + newpath + " to " + refpath)
    if isinstance(new2, np.ma.masked_array):
        hdu = fits.HDUList(
            [
                fits.PrimaryHDU(new2.data, header=hdr),
                fits.ImageHDU(new2.mask.astype("uint8")),
            ]
        )
        hdu.writeto(dest_file, overwrite=True)
    else:
        fits.writeto(dest_file, new2, hdr, overwrite=True)

    return dest_file


def align_for_diff_crop(refpath, newpath, bordersize=50):
    """Function to align two images using their paths,
    and returning newpaths for differencing.
    We will allways rotate and align the new image to the reference,
    so it is easier to compare differences along time series.

    This special function differs from aligh_for_diff since it
    crops the images, so they do not have borders with problems.
    """
    ref = fits.getdata(refpath)
    hdr_ref = fits.getheader(refpath)

    dest_file_ref = "cropped_" + os.path.basename(refpath)
    dest_file_ref = os.path.join(os.path.dirname(refpath), dest_file_ref)

    hdr_ref.set("comment", "cropped img " + refpath + " to " + newpath)
    ref2 = ref[bordersize:-bordersize, bordersize:-bordersize]
    fits.writeto(dest_file_ref, ref2, hdr_ref, overwrite=True)

    new = fits.getdata(newpath)
    hdr_new = fits.getheader(newpath)

    dest_file_new = "aligned_" + os.path.basename(newpath)
    dest_file_new = os.path.join(os.path.dirname(newpath), dest_file_new)

    try:
        new2 = aa.align_image(ref, new)
    except ValueError:
        ref = ref.astype(float)
        new = new.astype(float)
        new2 = aa.align_image(ref, new)

    hdr_new.set("comment", "aligned img " + newpath + " to " + refpath)
    new2 = new2[bordersize:-bordersize, bordersize:-bordersize]
    fits.writeto(dest_file_new, new2, hdr_new, overwrite=True)

    return [dest_file_new, dest_file_ref]


def align_for_coadd(imglist):
    """Function to align a group of images for coadding, it uses
    the
    """
    ref = imglist[0]
    for animg in imglist[1:]:
        aa.estimate_transform("affine", animg.data, ref.data)

    pass


def find_S_local_maxima(S_image, threshold=2.5, neighborhood_size=5):
    # std = np.std(S_image)
    # mean = np.mean(S_image)
    # threshold = threshold * std
    mean, median, std = sigma_clipped_stats(S_image, maxiters=3)
    # data_max = filters.maximum_filter(S_image, neighborhood_size)
    # maxima = (S_image == data_max)
    # data_min = filters.minimum_filter(S_image, neighborhood_size)
    # diff = ((data_max - data_min) > threshold)
    # maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label((S_image - mean) / std > threshold)
    xy = np.array(
        ndimage.center_of_mass(S_image, labeled, range(1, num_objects + 1))
    )
    # x = xy[:, 0]
    # y = xy[:, 1]
    cat = []
    for x, y in xy:
        cat.append((y, x, (S_image[int(x), int(y)] - mean) / std))

    return cat
