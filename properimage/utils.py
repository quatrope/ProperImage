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
#
import os
import numpy as np
from scipy import sparse
from numpy.lib.recfunctions import append_fields
from astropy.io import fits
from astropy.convolution import convolve, convolve_fft
from astroML import crossmatch as cx
import matplotlib.pyplot as plt

import astroalign as aa
from . import simtools

font = {'family'        : 'sans-serif',
        'sans-serif'    : ['Computer Modern Sans serif'],
        'weight'        : 'regular',
        'size'          : 12}

text = {'usetex'        : True}

plt.rc('font', **font)
plt.rc('text', **text)


def plot_psfbasis(psf_basis, path=None, nbook=False, size=4, **kwargs):
    psf_basis.reverse()
    N = len(psf_basis)
    p = primes(N)
    if N == 2:
        subplots = (2, 1)
    elif p == N:
        subplots = (np.rint(np.sqrt(N)),  np.rint(np.sqrt(N)))
    else:
        subplots = (N/float(p), p)

    plt.figure(figsize=(size*subplots[0], size*subplots[1]))
    for i in range(len(psf_basis)):
        plt.subplot(subplots[1], subplots[0], i+1)
        plt.imshow(psf_basis[i], interpolation='none', cmap='viridis')
        plt.title(r'$p_i, i = {}$'.format(i+1)) #, interpolation='linear')
        plt.tight_layout()
        #plt.colorbar(shrink=0.85)
    if path is not None:
        plt.savefig(path)
    if not nbook:
        plt.close()

    return

def plot_afields(a_fields, shape, path=None, nbook=False, size=4, **kwargs):
    if a_fields is None:
        print 'No a_fields were calculated. Only one Psf Basis'
        return
    a_fields.reverse()
    N = len(a_fields)
    p = primes(N)
    if N == 2:
        subplots = (2, 1)
    elif p == N:
        subplots = (np.rint(np.sqrt(N)),  np.rint(np.sqrt(N)))
    else:
        subplots = (N/float(p), p)

    plt.figure(figsize=(size*subplots[0], size*subplots[1]), **kwargs)
    x, y = np.mgrid[:shape[0], :shape[1]]
    for i in range(len(a_fields)):
        plt.subplot(subplots[1], subplots[0], i+1)
        plt.imshow(a_fields[i](x, y), cmap='viridis')
        plt.title(r'$a_i, i = {}$'.format(i+1))
        plt.tight_layout()
        #plt.colorbar(shrink=0.85, aspect=30)
    if path is not None:
        plt.savefig(path)
    if not nbook:
        plt.close()
    return

def encapsule_S(S, path=None):
    if isinstance(S, np.ma.core.MaskedArray):
        mask = S.mask.astype('uint8')
        data = S.data
        hdu_data = fits.PrimaryHDU(data)
        hdu_mask = fits.ImageHDU(mask, uint='int8')
        hdu_mask.header['IMG_TYPE'] = 'BAD_PIXEL_MASK'
        hdu = fits.HDUList([hdu_data, hdu_mask])
    else:
        hdu = fits.PrimaryHDU(S)
    if path is not None:
        hdu.writeto(path, clobber=True)
    else:
        return hdu

def encapsule_R(R, path=None):
    if isinstance(R[0, 0] , np.complex):
        R = R.real
    hdu = fits.PrimaryHDU(R)
    if path is not None:
        hdu.writeto(path, clobber=True)
    else:
        return hdu

def plot_S(S, path=None, nbook=False):
    if isinstance(S, np.ma.masked_array):
        S = S.filled()
    plt.imshow(np.log10(S), interpolation='none', cmap='viridis')
    plt.tight_layout()
    plt.colorbar()
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
    if not nbook:
        plt.close()
    return

def plot_R(R, path=None, nbook=False):
    if isinstance(R[0, 0] , np.complex):
        R = R.real
    if isinstance(R, np.ma.masked_array):
        R = R.filled()
    plt.imshow(np.log10(R), interpolation='none', cmap='viridis')
    plt.tight_layout()
    plt.colorbar()
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
    if not nbook:
        plt.close()
    return

def sim_varpsf(nstars, test_dir, SN=3., thetas=[0, 45, 105, 150], N=512):
    frames = []
    for theta in thetas:
        X_FWHM = 5 + 5.*theta/180.
        Y_FWHM = 5
        bias = 100.
        t_exp = 1
        max_fw = max(X_FWHM, Y_FWHM)

        x = np.random.randint(low=6*max_fw, high=N-6*max_fw, size=nstars/4)
        y = np.random.randint(low=6*max_fw, high=N-6*max_fw, size=nstars/4)
        xy = [(x[i], y[i]) for i in range(nstars/4)]

        weights = list(np.linspace(1000., 100000., len(xy)))
        m = simtools.delta_point(N, center=False, xy=xy, weights=weights)
        im = simtools.image(m, N, t_exp, X_FWHM, Y_FWHM=Y_FWHM, theta=theta,
                            SN=SN, bkg_pdf='poisson')
        frames.append(im+bias)

    frame = np.zeros((2*N, 2*N))
    for j in range(2):
        for i in range(2):
            frame[i*N:(i+1)*N, j*N:(j+1)*N] = frames[i+2*j]

    return frame

def sim_ref_new(x, y, SN=2.):
    pass

def primes(n):
    divisors = [ d for d in range(2,n//2+1) if n % d == 0 ]
    prims = [ d for d in divisors if \
             all( d % od != 0 for od in divisors if od != d ) ]
    if len(prims) is 0:
        return n
    else:
        return max(prims)

def matching(master, cat, masteridskey=None,
             angular=False, radius=1.5, masked=False):
    """Function to match stars between frames.
    """
    if masteridskey is None:
        masterids = np.arange(len(master))
        master['masterindex'] = masterids
        idkey = 'masterindex'
    else:
        idkey = masteridskey

    if angular:
        masterRaDec = np.empty((len(master), 2), dtype=np.float64)
        try:
            masterRaDec[:, 0] = master['RA']
            masterRaDec[:, 1] = master['Dec']
        except:
            masterRaDec[:, 0] = master['ra']
            masterRaDec[:, 1] = master['dec']
        imRaDec = np.empty((len(cat), 2), dtype=np.float64)
        try:
            imRaDec[:, 0] = cat['RA']
            imRaDec[:, 1] = cat['Dec']
        except:
            imRaDec[:, 0] = cat['ra']
            imRaDec[:, 1] = cat['dec']
        radius2 = radius/3600.
        dist, ind = cx.crossmatch_angular(masterRaDec, imRaDec,
                                          max_distance=radius2/2.)
        dist_, ind_ = cx.crossmatch_angular(imRaDec, masterRaDec,
                                            max_distance=radius2/2.)
    else:
        masterXY = np.empty((len(master), 2), dtype=np.float64)
        masterXY[:, 0] = master['x']
        masterXY[:, 1] = master['y']
        imXY = np.empty((len(cat), 2), dtype=np.float64)
        imXY[:, 0] = cat['x']
        imXY[:, 1] = cat['y']
        dist, ind = cx.crossmatch(masterXY, imXY, max_distance=radius)
        dist_, ind_ = cx.crossmatch(imXY, masterXY, max_distance=radius)

    match = ~np.isinf(dist)
    match_ = ~np.isinf(dist_)

    IDs = np.zeros_like(ind_) - 13133
    for i in xrange(len(ind_)):
        if dist_[i] != np.inf:
            dist_o = dist_[i]
            ind_o = ind_[i]
            if dist[ind_o] != np.inf:
                dist_s = dist[ind_o]
                ind_s = ind[ind_o]
                if ind_s == i:
                    IDs[i] = master[idkey][ind_o]

    print len(IDs), len(ind_), len(ind)
    if masked:
        mask = IDs > 0
        return(IDs, mask)
    return(IDs)

def transparency(images, master=None, ensemble=True):
    """Transparency calculator, using Ofek method."""

    if ensemble:
        # master is the first file of ensemble
        p = len(images)
        master = images.atoms[0]
        imglist = images.atoms[1:]
    else:
        for img in images:
            if not isinstance(img, pc.SingleImage):
                img = pc.SingleImage(img)

        if master is None:
            p = len(images)
            master = images[0]
            imglist = images[1:]
        else:
            # master is a separated file
            p = len(images) + 1
            imglist = images
            if not isinstance(master, pc.SingleImage):
                master = pc.SingleImage(master)

    mastercat = master._best_srcs['sources']
    mastercat = append_fields(mastercat, 'sourceid',
                              np.arange(len(mastercat)),
                              usemask=False,
                              dtypes=int)

    detect = np.repeat(True, len(mastercat))
    #  Matching the sources
    for img in imglist:
        newcat = img._best_srcs['sources']

        ids, mask = matching(mastercat, newcat, masteridskey='sourceid',
                             angular=False, radius=1., masked=True)

        newcat = append_fields(newcat, 'sourceid', ids,
                               usemask=False)

        for i in xrange(len(mastercat)):
            if mastercat[i]['sourceid'] not in ids:
                detect[i] = False
        newcat.sort(order='sourceid')
        img._best_srcs['sources'] = newcat
    mastercat = append_fields(mastercat, 'detected',
                              detect,
                              usemask=False,
                              dtypes=bool)

    # Now populating the vector of magnitudes
    q = sum(mastercat['detected'])

    m = np.zeros(p*q)
    # here 20 is a common value for a zp, and is only for weighting
    m[:q] = -2.5*np.log10(mastercat[mastercat['detected']]['flux']) + 20.

    j = 0
    for row in mastercat[mastercat['detected']]:
        for img in imglist:
            cat = img._best_srcs['sources']
            imgrow = cat[cat['sourceid'] == row['sourceid']]
            m[q+j] = -2.5*np.log10(imgrow['flux']) + 20.
            j += 1
    #print mastercat['detected']
    master._best_srcs['sources'] = mastercat

    print p, q
    ident = sparse.identity(q)
    col = np.repeat(1., q)
    sparses = []
    for j in xrange(p):
        ones_col = np.zeros((q, p))
        ones_col[:, j] = col
        sparses.append([sparse.csc_matrix(ones_col), ident])

    H = sparse.bmat(sparses)

    P = sparse.linalg.lsqr(H, m)
    zps = P[0][:p]

    meanmags = P[0][p:]

    return zps, meanmags

def convolve_psf_basis(image, psf_basis, a_fields, x, y):
    imconvolved = np.zeros_like(image)
    for j in range(len(psf_basis)):
        a = a_fields[j](x, y) * image
        psf = psf_basis[j]

        imconvolved += convolve(a, psf, boundary='extend')

    return imconvolved

def fftconvolve_psf_basis(image, psf_basis, a_fields, x, y):
    imconvolved = np.zeros_like(image)
    for j in range(len(psf_basis)):
        a = a_fields[j](x, y) * image
        psf = psf_basis[j]

        imconvolved += convolve_fft(a, psf, interpolate_nan=True,
                                    allow_huge=True)

    return imconvolved

def lucy_rich(image, psf_basis, a_fields, iterations=50, clip=True, fft=False):
    #~ direct_time = np.prod(image.shape + psf.shape)
    #~ fft_time =  np.sum([n*np.log(n) for n in image.shape + psf.shape])

    #~ # see whether the fourier transform convolution method or the direct
    #~ # convolution method is faster (discussed in scikit-image PR #1792)
    #~ time_ratio = 40.032 * fft_time / direct_time

    if fft:
        convolve_method = fftconvolve_psf_basis
    else:
        convolve_method = convolve_psf_basis

    image = image.astype(np.float)
    image = np.ma.masked_invalid(image).filled(np.nan)
    x, y = np.mgrid[:image.shape[0], :image.shape[1]]

    im_deconv = 0.5 * np.ones(image.shape)
    psf_mirror = [psf[::-1, ::-1] for psf in psf_basis]

    for _ in range(iterations):
        rela_blur = image/convolve_method(im_deconv, psf_basis, a_fields, x, y)
        im_deconv *= convolve_method(rela_blur, psf_mirror, a_fields, x, y)

    if clip:
        im_deconv = np.ma.masked_invalid(im_deconv).filled(-1.)

    return im_deconv

def align_for_diff(refpath, newpath):
    """Function to align two images using their paths,
    and returning newpaths for differencing.
    We will allways rotate and align the new image to the reference,
    so it is easier to compare differences along time series.
    """
    ref = fits.getdata(refpath)
    new = fits.getdata(newpath)
    hdr = fits.getheader(newpath)

    dest_file = 'aligned_'+os.path.basename(newpath)
    dest_file = os.path.join(os.path.dirname(newpath), dest_file)

    new2 = aa.align_image(ref, new)

    hdr.set('comment', 'aligned img '+newpath+' to '+refpath)
    fits.writeto(dest_file, new2, hdr, clobber=True)

    return dest_file
