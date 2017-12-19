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
import astroalign as aa
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

aa.PIXEL_TOL=0.5
eps = np.finfo(np.float64).eps

def diff(ref, new, align=True, inf_loss=0.25, beta=True, shift=True, iterative=False):
    """Function that takes a list of SingleImage instances
    and performs a stacking using properimage R estimator

    """

    if not isinstance(ref, s.SingleImage):
        ref = s.SingleImage(ref)

    if not isinstance(new, s.SingleImage):
        new = s.SingleImage(new)

    if align:
        registered = aa.register(new.pixeldata, ref.pixeldata)
        new._clean()
        new = s.SingleImage(registered.data, mask=registered.mask)
        #~ new.pixeldata = registered
        #~ new.pixeldata.mask = registered.mask

    #~ make sure that the alignement has delivered arrays of size
    if new.pixeldata.data.shape != ref.pixeldata.data.shape:
        import ipdb; ipdb.set_trace()

    t0 = time.time()
    mix_mask = np.ma.mask_or(new.pixeldata.mask, ref.pixeldata.mask)

    zps, meanmags = utils.transparency([ref, new])
    print zps
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
    #~ print(dx_new, dy_new)
    if dx_new < 0. or dy_new<0.:
        import ipdb; ipdb.set_trace()

    psf_ref_hat = _fftwn(psf_ref[0], s=ref.pixeldata.shape, norm='ortho')
    psf_new_hat = _fftwn(psf_new[0], s=new.pixeldata.shape, norm='ortho')

    psf_ref_hat[np.where(psf_ref_hat.real==0)] = eps
    psf_new_hat[np.where(psf_new_hat.real==0)] = eps

    psf_ref_hat_conj = psf_ref_hat.conj()
    psf_new_hat_conj = psf_new_hat.conj()

    D_hat_r = fourier_shift(psf_new_hat * ref.interped_hat, (-dx_new, -dy_new))
    D_hat_n = fourier_shift(psf_ref_hat * new.interped_hat, (-dx_ref, -dy_ref))
    #~ D_hat_r = psf_new_hat * ref.interped_hat
    #~ D_hat_n = psf_ref_hat * new.interped_hat

    if beta:
        new_back = sep.Background(new.interped).back()
        ref_back = sep.Background(ref.interped).back()
        gamma = new_back - ref_back
        b = n_zp/r_zp
        #start with beta=1

        if shift:
            def cost_beta(vec, gamma=gamma):
                b, dx, dy = vec[:]

                #~ gammap = gamma/np.sqrt(new.var**2 + b**2 * ref.var**2)

                norm  = b**2 * ref.var**2 * psf_new_hat*psf_new_hat_conj
                norm += new.var**2 * psf_ref_hat * psf_ref_hat_conj

                cost = _ifftwn(D_hat_n/np.sqrt(norm), norm='ortho') - \
                       _ifftwn(fourier_shift((D_hat_r/np.sqrt(norm))*b, (dx,dy)), norm='ortho') #-\
                       #~ _ifftwn(fourier_shift(_fftwn(gammap), (dx, dy)))
                cost = np.absolute(cost*cost.conj())[50:-50, 50:-50].flatten()

                #~ return sigma_clipped_stats(cost[50:-50, 50:-50], sigma=5.)[1]
                return np.abs(sigma_clipped_stats(cost, sigma=5.)[2] -1)


            tbeta0 = time.time()
            vec0 = [n_zp/r_zp, 0., 0.]
            bounds = ([0.1, -2.9, -2.9], [15., 2.9, 2.9])
            solv_beta = optimize.least_squares(cost_beta,
                                               vec0, ftol=1e-10,
                                               jac='3-point',
                                               bounds=bounds)
            tbeta1 = time.time()

            if solv_beta.success:
                print('Found that beta = {}'.format(solv_beta.x))
                print('Took only {} awesome seconds'.format(tbeta1-tbeta0))
                print('The solution was with cost {}'.format(solv_beta.cost))
                b, dx, dy = solv_beta.x
            else:
                print('Least squares could not find our beta  :(')
                print('Beta is overriden to be the zp ratio again')
                b =  n_zp/r_zp
                dx = 0.
                dy = 0.

        elif iterative:
            def beta_next(b, gamma=gamma):
                gammap = gamma/np.sqrt(new.var**2 + b**2 * ref.var**2)

                norm  = b**2 *ref.var**2 * psf_new_hat * psf_new_hat_conj
                norm += new.var**2 * psf_ref_hat * psf_ref_hat_conj

                b_n = (_ifftwn(D_hat_n/np.sqrt(norm), norm='ortho') - gammap)/_ifftwn(D_hat_r/np.sqrt(norm), norm='ortho')

                #b_n = _ifftwn(D_hat_n/np.sqrt(norm))/_ifftwn(D_hat_r/np.sqrt(norm))

                ab = np.absolute(b_n)
                bb = ab[(np.percentile(ab, q=97)>ab)*(ab>np.percentile(ab, q=55))]

                #~ import matplotlib.pyplot as plt
                #~ plt.hist(bb.real.flatten(), log=True, bins=150)
                #~ plt.vlines(sigma_clipped_stats(bb, sigma=12)[1], 0, 1000)
                #~ plt.show()
                print('Sigma clip on beta values')
                print(sigma_clipped_stats(bb.real, iters=3, sigma=5.))
                b_next = sigma_clipped_stats(bb.real, iters=3, sigma=5.)[1]
                #~ b_next = sigma_clipped_stats(ab)[0]
                if b_next==0.:
                    return b
                #~ b_next = np.mean(b_n)
                return b_next

            bi = new.zp/ref.zp # 1
            print('Start iteration')
            ti = time.time()
            bf = beta_next(bi)
            n_iter = 1
            while np.abs(bf-bi) > 0.002 and n_iter<45:
                bi = bf
                bf = beta_next(bi)
                n_iter += 1
            b = bf
            tf = time.time()
            print('b = {}. Finished on {} iterations, and {} time\n'.format(b, n_iter, tf-ti))
            dx = dy = 0.

        else:
            def cost_beta(vec, gamma=gamma):
                b = vec[0]
                #~ gammap = gamma/np.sqrt(new.var**2 + b**2 * ref.var**2)

                norm  = b*b*(ref.var**2 * psf_new_hat*psf_new_hat_conj)
                norm += new.var**2 * psf_ref_hat * psf_ref_hat_conj

                cost = _ifftwn(D_hat_n/np.sqrt(norm), norm='ortho') - \
                       _ifftwn((D_hat_r/np.sqrt(norm))*b, norm='ortho') #- gammap
                cost = np.absolute(cost*cost.conj())[50:-50, 50:-50].flatten()

                return sigma_clipped_stats(cost, sigma=5.)[2]
                #~ return np.std(cost[50:-50, 50:-50].flatten())

            dx = 0
            dy = 0
            tbeta0 = time.time()
            vec0 = [new.zp/ref.zp]
            bounds = ([0.01], [15.])
            solv_beta = optimize.least_squares(cost_beta,
                                               vec0, ftol=1e-9,
                                               jac='3-point',
                                               bounds=bounds)
            tbeta1 = time.time()
            if solv_beta.success:
                print('Found that beta = {}'.format(solv_beta.x))
                print('Took only {} awesome seconds'.format(tbeta1-tbeta0))
                print('The solution was with cost {}'.format(solv_beta.cost+1))
                b = solv_beta.x
            else:
                print('Least squares could not find our beta  :(')
                print('Beta is overriden to be the zp ratio again')
                b =  n_zp/r_zp


    else:
        b = new.zp/ref.zp
        dx = 0.
        dy = 0.

    norm  = b**2 * ref.var**2 * psf_new_hat * psf_new_hat_conj
    norm += new.var**2 * psf_ref_hat * psf_ref_hat_conj

    if dx==0. and dy==0.:
        D_hat = (D_hat_n - b * D_hat_r)/np.sqrt(norm)
    else:
        D_hat = (D_hat_n - fourier_shift(b*D_hat_r, (dx,dy)))/np.sqrt(norm)
    D = _ifftwn(D_hat, norm='ortho')

    d_zp = new.zp/np.sqrt(ref.var**2 * b**2 + new.var**2 )
    P_hat =(psf_ref_hat * psf_new_hat * b)/(np.sqrt(norm)*d_zp)

    P = _ifftwn(P_hat, norm='ortho').real
    dx_p, dy_p = center_of_mass(P)

    S_hat = fourier_shift(d_zp * D_hat * P_hat.conj(), (dx_p, dy_p))

    kr=_ifftwn(b*new.zp*psf_ref_hat_conj*psf_new_hat*psf_new_hat_conj/norm, norm='ortho')
    kn=_ifftwn(b*new.zp*psf_new_hat_conj*psf_ref_hat*psf_ref_hat_conj/norm, norm='ortho')

    V_en = _ifftwn(_fftwn(new.pixeldata.filled(0)+1., norm='ortho') * _fftwn(kn**2, s=new.pixeldata.shape), norm='ortho')
    V_er = _ifftwn(_fftwn(ref.pixeldata.filled(0)+1., norm='ortho') * _fftwn(kr**2, s=ref.pixeldata.shape), norm='ortho')

    S_corr = _ifftwn(S_hat, norm='ortho')/np.sqrt(V_en + V_er)
    print('S_corr sigma_clipped_stats ')
    print('mean = {}, median = {}, std = {}\n'.format(*sigma_clipped_stats(S_corr.real.flatten(), sigma=200)))
    print('Subtraction performed in {} seconds\n\n'.format(time.time()-t0))

    #import ipdb; ipdb.set_trace()
    return D, P, S_corr.real


def get_transients(self, threshold=2.5, neighborhood_size=5.):
    S = self.subtract()[2]
    threshold = np.std(S) * threshold
    cat = u.find_S_local_maxima(S, threshold=threshold,
                                neighborhood_size=neighborhood_size)

    return cat
















