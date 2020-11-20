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
from scipy.spatial import cKDTree
import scipy.ndimage as ndimage
from numpy.lib.recfunctions import append_fields
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import astroalign as aa

aa.PIXEL_TOL = 0.3
aa.NUM_NEAREST_NEIGHBORS = 5
aa.MIN_MATCHES_FRACTION = 0.6


def store_img(img, path=None):
    if isinstance(img[0, 0], np.complex):
        img = img.real

    if isinstance(img, np.ma.core.MaskedArray):
        mask = img.mask.astype("int")
        data = img.data
        hdu_data = fits.PrimaryHDU(data)
        hdu_data.scale(type="float32")
        hdu_mask = fits.ImageHDU(mask, uint="uint8")
        hdu_mask.header["IMG_TYPE"] = "BAD_PIXEL_MASK"
        hdu = fits.HDUList([hdu_data, hdu_mask])
    else:
        hdu = fits.PrimaryHDU(img)
    if path is not None:
        hdu.writeto(path, overwrite=True)
    else:
        return hdu


def crossmatch(X1, X2, max_distance=np.inf):
    """Cross-match the values between X1 and X2
    By default, this uses a KD Tree for speed.
    Parameters
    ----------
    X1 : array_like
        first dataset, shape(N1, D)
    X2 : array_like
        second dataset, shape(N2, D)
    max_distance : float (optional)
        maximum radius of search.  If no point is within the given radius,
        then inf will be returned.
    Returns
    -------
    dist, ind: ndarrays
        The distance and index of the closest point in X2 to each point in X1
        Both arrays are length N1.
        Locations with no match are indicated by
        dist[i] = inf, ind[i] = N2
    """
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)

    N1, D = X1.shape
    N2, D2 = X2.shape

    if D != D2:
        raise ValueError("Arrays must have the same second dimension")

    kdt = cKDTree(X2)

    dist, ind = kdt.query(X1, k=1, distance_upper_bound=max_distance)

    return dist, ind


def _matching(master, cat, masteridskey=None, radius=1.5, masked=False):
    """Function to match stars between frames."""

    if masteridskey is None:
        masterids = np.arange(len(master))
        master["masterindex"] = masterids
        idkey = "masterindex"
    else:
        idkey = masteridskey

    masterXY = np.empty((len(master), 2), dtype=np.float64)
    masterXY[:, 0] = master["x"]
    masterXY[:, 1] = master["y"]
    imXY = np.empty((len(cat), 2), dtype=np.float64)
    imXY[:, 0] = cat["x"]
    imXY[:, 1] = cat["y"]
    dist, ind = crossmatch(masterXY, imXY, max_distance=radius)
    dist_, ind_ = crossmatch(imXY, masterXY, max_distance=radius)

    IDs = np.zeros_like(ind_) - 13133
    for i in range(len(ind_)):
        if dist_[i] != np.inf:
            ind_o = ind_[i]
            if dist[ind_o] != np.inf:
                ind_s = ind[ind_o]
                if ind_s == i:
                    IDs[i] = master[idkey][ind_o]

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
        ids, mask = _matching(
            mastercat,
            newcat,
            masteridskey="sourceid",
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
        master.update_sources(mastercat)

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

        return np.asarray(zps), np.asarray(meanmags)
    else:
        return np.ones(p), np.nan


def _align_for_diff(refpath, newpath, newmask=None):
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
        new2 = aa.register(ref, new.filled(np.median(new)))
    except ValueError:
        ref = ref.astype(float)
        new = new.astype(float)
        new2 = aa.register(ref, new)

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


def _align_for_coadd(imglist):
    """
    Function to align a group of images for coadding, it uses
    the astroalign `align_image` tool.
    """
    ref = imglist[0]
    new_list = [ref]
    for animg in imglist[1:]:
        registrd, registrd_mask = aa.register(
            animg.data, ref.data, propagate_mask=True
        )
        # [: ref.data.shape[0], : ref.data.shape[1]],  Deprecated
        new_list.append(
            type(animg)(registrd, mask=registrd_mask, borders=False)
        )
    return new_list


def find_S_local_maxima(S_image, threshold=2.5, neighborhood_size=5):
    mean, median, std = sigma_clipped_stats(S_image, maxiters=3)
    labeled, num_objects = ndimage.label((S_image - mean) / std > threshold)
    xy = np.array(
        ndimage.center_of_mass(S_image, labeled, range(1, num_objects + 1))
    )
    cat = []
    for x, y in xy:
        cat.append((y, x, (S_image[int(x), int(y)] - mean) / std))

    return cat


def chunk_it(seq, num):
    """Creates chunks of a sequence suitable for data parallelism using
    multiprocessing.

    Parameters
    ----------
    seq: list, array or sequence like object. (indexable)
        data to separate in chunks

    num: int
        number of chunks required

    Returns
    -------
    Sorted list.
    List of chunks containing the data splited in num parts.

    """
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last) : int(last + avg)])
        last += avg
    try:
        return sorted(out, reverse=True)
    except TypeError:
        return out
    except ValueError:
        return out
