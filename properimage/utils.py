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
import numpy as np
from numpy.lib.recfunctions import append_fields
from astropy.io import fits
from astroML import crossmatch as cx
import matplotlib.pyplot as plt

from . import simtools

font = {'family'        : 'sans-serif',
        'sans-serif'    : ['Computer Modern Sans serif'],
        'weight'        : 'regular',
        'size'          : 14}

text = {'usetex'        : True,
        'fontsize'      : 14}

plt.rc('font', **font)
plt.rc('text', **text)



def plot_psfbasis(psf_basis, path=None, nbook=False, **kwargs):
    psf_basis.reverse()
    N = len(psf_basis)
    p = primes(N)
    if p == N:
        subplots = (np.rint(np.sqrt(N)),  np.rint(np.sqrt(N)))
    else:
        subplots = (N/float(p), p)

    plt.figure(figsize=(4*subplots[0], 4*subplots[1]))
    for i in range(len(psf_basis)):
        plt.subplot(subplots[1], subplots[0], i+1)
        plt.imshow(psf_basis[i])
        plt.title(r'$a_i, i = {}$'.format(i+1)) #, interpolation='linear')
        plt.tight_layout()
        plt.colorbar()
    if path is not None:
        plt.savefig(path)
    if not nbook:
        plt.close()

    return

def plot_afields(a_fields, shape, path=None, nbook=False, **kwargs):
    if a_fields is None:
        print 'No a_fields were calculated. Only one Psf Basis'
        return
    a_fields.reverse()
    N = len(a_fields)
    p = primes(N)
    if p == N:
        subplots = (np.rint(np.sqrt(N)),  np.rint(np.sqrt(N)))
    else:
        subplots = (N/float(p), p)

    plt.figure(figsize=(4*subplots[0], 4*subplots[1]), **kwargs)
    x, y = np.mgrid[:shape[0], :shape[1]]
    for i in range(len(a_fields)):
        plt.subplot(subplots[1], subplots[0], i+1)
        plt.imshow(a_fields[i](x, y))
        plt.title(r'$a_i, i = {}$'.format(i+1))
        plt.tight_layout()
        plt.colorbar()
    if path is not None:
        plt.savefig(path)
    if not nbook:
        plt.close()
    return


def plot_S():
    return


def plot_R():
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
    for i in range(len(ind_)):
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
        if master is None:
            p = len(images)
            master = images.atoms[0]
            imglist = images.atoms[1:]
        else:
            p = len(images) + 1

        mastercat = master._best_srcs['sources']
        mastercat = append_fields(mastercat, 'sourceid',
                                  np.arange(len(mastercat)),
                                  usemask=False,
                                  dtypes=int)

        detect = np.repeat(True, len(mastercat))

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
        mastercat = append_fields(mastercat, 'detected',
                                  detect,
                                  usemask=False)
        q = sum(mastercat['detected'])

        print mastercat['detected']
