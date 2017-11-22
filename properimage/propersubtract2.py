#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  propersubtract.py
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

"""propersubtract module from ProperImage,
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

from scipy import optimize
from scipy.ndimage import center_of_mass
from scipy.ndimage.fourier import fourier_shift
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip
import sep
import time
from . import single_image as s
from . import utils

try:
    import pyfftw
    _fftwn = pyfftw.interfaces.numpy_fft.fftn
    _ifftwn = pyfftw.interfaces.numpy_fft.ifftn
except:
    _fftwn = np.fft.fft2
    _ifftwn = np.fft.ifft2



def diff(ref, new, align=True, inf_loss=0.2, beta=False, shift=False, iterative=False):
    """Function that takes a list of SingleImage instances
    and performs a stacking using properimage R estimator

    """

    #~ if align:
        #~ img_list = utils.align_for_coadd([ref, new])
        #~ for an_img in img_list:
            #~ an_img.update_sources()
    #~ else:
        #~ img_list = [ref, new]
    t0 = time.time()
    zps, meanmags = utils.transparency([ref, new])
    ref.zp = zps[0]
    new.zp = zps[1]
    n_zp = new.zp
    r_zp = ref.zp
    r_var = ref.var
    n_var = new.var

    a_ref, psf_ref = ref.get_variable_psf(inf_loss)
    a_new, psf_new = new.get_variable_psf(inf_loss)

    dx_ref, dy_ref = center_of_mass(psf_ref[0])
    dx_new, dy_new = center_of_mass(psf_new[0])

    psf_ref_hat = _fftwn(psf_ref[0], s=ref.pixeldata.shape)
    psf_new_hat = _fftwn(psf_new[0], s=new.pixeldata.shape)

    D_hat_r = fourier_shift(psf_new_hat * ref.interped_hat, (dx_ref, dy_ref))
    D_hat_n = fourier_shift(psf_ref_hat * new.interped_hat, (dx_new, dy_new))

    if beta:
        new_back = sep.Background(new.interped).back()
        ref_back = sep.Background(ref.interped).back()
        gamma = new_back - ref_back
        b = 1
        #start with beta=1

        if shift:
            def cost_beta(vec, gamma=gamma):
                b, dx, dy = vec[:]

                gammap = gamma/np.sqrt(new.var**2 + b**2 * ref.var**2)

                norm  = b*b*(ref.var**2 * psf_new_hat*psf_new_hat.conj())
                norm += new.var**2 * psf_ref_hat * psf_ref_hat.conj()

                cost = _ifftwn(D_hat_n/np.sqrt(norm)) - \
                       _ifftwn(fourier_shift((D_hat_r/np.sqrt(norm))*b, (dx,dy))) -\
                       _ifftwn(fourier_shift(_fftwn(gammap), (dx, dy)))
                cost = np.absolute(cost*cost.conj())

                return sigma_clipped_stats(cost[50:-50, 50:50], sigma=5.)[1]

            tbeta0 = time.time()
            vec0 = [n_zp/r_zp, 0., 0.]
            bounds = ([0.5, -2.9, -2.9], [2., 2.9, 2.9])
            solv_beta = optimize.least_squares(cost_beta,
                                               vec0, ftol=1e-10,
                                               jac='3-point',
                                               bounds=bounds)
            tbeta1 = time.time()

            if solv_beta.success:
                print('Found that beta = {}'.format(solv_beta.x))
                print('Took only {} awesome seconds'.format(tbeta1-tbeta0))
                print('The solution was with cost {}'.format(solv_beta.cost))
                beta, dx, dy = solv_beta.x
            else:
                print('Least squares could not find our beta  :(')
                print('Beta is overriden to be the zp ratio again')
                beta =  n_zp/r_zp
                dx = 0.
                dy = 0.

        elif iterative:
            def beta_next(b, gamma=gamma):
                gammap = gamma/np.sqrt(new.var**2 + b**2 * ref.var**2)

                norm  = b*b*(ref.var**2 * psf_new_hat*psf_new_hat.conj())
                norm += new.var**2 * psf_ref_hat * psf_ref_hat.conj()

                b_n = (_ifftwn(D_hat_n/np.sqrt(norm)) - gammap)/_ifftwn(D_hat_r/np.sqrt(norm))

                b_next = sigma_clipped_stats(b_n)[1]
                return b_next
            bi = 1
            print('Start iteration')
            ti = time.time()
            bf = beta_next(bi)
            n_iter = 1
            while np.abs(bf-bi) > 0.002 or n_iter>25:
                bi = bf
                bf = beta_next(bf)
                n_iter += 1
            b = bf
            tf = time.time()
            print('b = {}. Finished on {} iterations, and {} time'.format(b, n_iter, tf-ti))
            dx = dy = 0.

        else:
            def cost_beta(vec, gamma=gamma):
                b = vec[0]
                gammap = gamma/np.sqrt(new.var**2 + b**2 * ref.var**2)

                norm  = b*b*(ref.var**2 * psf_new_hat*psf_new_hat.conj())
                norm += new.var**2 * psf_ref_hat * psf_ref_hat.conj()

                cost = _ifftwn(D_hat_n/np.sqrt(norm)) - \
                       _ifftwn((D_hat_r/np.sqrt(norm))*beta) - gammap
                cost = np.absolute(cost*cost.conj())

                return sigma_clipped_stats(cost[50:-50, 50:50], sigma=5.)[1]

            dx = 0
            dy = 0
            tbeta0 = time.time()
            vec0 = [new.zp/ref.zp]
            bounds = ([0.01], [20.])
            solv_beta = optimize.least_squares(cost_beta,
                                               vec0, ftol=1e-9,
                                               jac='3-point',
                                               bounds=bounds)
            tbeta1 = time.time()
            if solv_beta.success:
                print('Found that beta = {}'.format(solv_beta.x))
                print('Took only {} awesome seconds'.format(tbeta1-tbeta0))
                print('The solution was with cost {}'.format(solv_beta.cost))
                beta = solv_beta.x
            else:
                print('Least squares could not find our beta  :(')
                print('Beta is overriden to be the zp ratio again')
                beta =  n_zp/r_zp


    else:
        b = new.zp/ref.zp
        dx = 0.
        dy = 0.

    norm  = b*b*(ref.var**2 * psf_new_hat*psf_new_hat.conj())
    norm += new.var**2 * psf_ref_hat * psf_ref_hat.conj()

    if dx==0. and dy==0.:
        D_hat = (D_hat_n - b * D_hat_r)/np.sqrt(norm)
    else:
        D_hat = (D_hat_n - fourier_shift(b*D_hat_r, (dx,dy)))/np.sqrt(norm)
    D = _ifftwn(D_hat)

    d_zp = new.zp/np.sqrt(ref.var**2 * b*b + new.var**2 )
    P_hat =(psf_ref_hat * psf_new_hat * b)/(np.sqrt(norm)*d_zp)

    P = _ifftwn(P_hat).real
    dx_p, dy_p = center_of_mass(P)

    S_hat = fourier_shift(d_zp * D_hat * P_hat.conjugate(), (dx_p, dy_p))

    kr=_ifftwn(b*new.zp*psf_ref_hat.conj()*psf_new_hat*psf_new_hat.conj()/norm)
    kn=_ifftwn(b*new.zp*psf_new_hat.conj()*psf_ref_hat*psf_ref_hat.conj()/norm)

    V_en = _ifftwn(_fftwn(new.pixeldata.filled(0)+1.) * _fftwn(kn**2, s=new.pixeldata.shape))
    V_er = _ifftwn(_fftwn(ref.pixeldata.filled(0)+1.) * _fftwn(kr**2, s=ref.pixeldata.shape))

    S_corr = _ifftwn(S_hat)/np.sqrt(V_en + V_er)
    print('S_corr sigma_clipped_stats ')
    print('mean = {}, median = {}, std = {}\n'.format(*sigma_clipped_stats(S_corr.real.flatten())))
    print('Subtraction performed in {} seconds'.format(time.time()-t0))

    #import ipdb; ipdb.set_trace()
    return D, P, S_corr.real


def get_transients(self, threshold=2.5, neighborhood_size=5.):
    S = self.subtract()[2]
    threshold = np.std(S) * threshold
    cat = u.find_S_local_maxima(S, threshold=threshold,
                                neighborhood_size=neighborhood_size)

    return cat
















