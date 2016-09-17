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
from astropy.io import fits
import matplotlib.pyplot as plt

font = {'family'        : 'sans-serif',
        'sans-serif'    : ['Computer Modern Sans serif'],
        'weight'        : 'regular',
        'size'          : 12}

text = {'usetex'        : True,
        'fontsize'      : 12}

plt.rc('font', **font)
plt.rc('text', **text)



def plot_psfbasis(psf_basis, path=None):
    psf_basis.reverse()
    N = len(psf_basis)
    p = primes(N)
    if p == N:
        subplots = (np.rint(np.sqrt(N)),  np.rint(np.sqrt(N)))
    else:
        subplots = (p, N/float(p))

    plt.figure(figsize=(3*subplots[0], 3*subplots[1]))
    for i in range(len(psf_basis)):
        plt.subplot(subplots[1], subplots[0], i+1)
        plt.imshow(psf_basis[i])
        plt.title(r'$a_i, i = {}$'.format(i+1)) #, interpolation='linear')
        plt.tight_layout()
        plt.colorbar()
    if path is not None:
        plt.savefig(path)
    plt.close()

    return

def plot_afields(a_fields, shape, path=None):
    if a_fields is None:
        print 'No a_fields were calculated. Only one Psf Basis'
        return
    a_fields.reverse()
    N = len(a_fields)
    p = primes(N)
    if p == N:
        subplots = (np.rint(np.sqrt(N)),  np.rint(np.sqrt(N)))
    else:
        subplots = (p, N/float(p))

    plt.figure(figsize=(3*subplots[0], 3*subplots[1]))
    x, y = np.mgrid[:shape[0], :shape[1]]
    for i in range(len(a_fields)):
        plt.subplot(subplots[1], subplots[0], i+1)
        plt.imshow(a_fields[i](x, y))
        plt.title(r'$a_i, i = {}$'.format(i+1))
        plt.tight_layout()
        plt.colorbar()
    if path is not None:
        plt.savefig(path)
    plt.close()
    return


def plot_S():
    return


def plot_R():
    return


def primes(n):
    divisors = [ d for d in range(2,n//2+1) if n % d == 0 ]
    prims = [ d for d in divisors if \
             all( d % od != 0 for od in divisors if od != d ) ]
    if len(prims) is 0:
        return n
    else:
        return max(prims)
