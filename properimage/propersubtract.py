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

import os
import numpy as np
from scipy import optimize
from astropy.stats import sigma_clipped_stats, sigma_clip
import time
from . import propercoadd as pc
from . import utils as u

try:
    import pyfftw
    _fftwn = pyfftw.interfaces.numpy_fft.fftn
    _ifftwn = pyfftw.interfaces.numpy_fft.ifftn
except:
    _fftwn = np.fft.fft2
    _ifftwn = np.fft.ifft2



class ImageSubtractor(object):
    def __init__(self, refpath, newpath, align=True, crop=False,
                 solve_beta=False, calc_zps=True, border=50, shape=None):

        if align:
            if crop:
                new, refpath = u.align_for_diff_crop(refpath, newpath, border)
            else:
                new = u.align_for_diff(refpath, newpath)

            new = u.align_for_diff(refpath, newpath)
            self.ens = pc.ImageEnsemble([refpath, new])
        else:
            self.ens = pc.ImageEnsemble([refpath, newpath])

        self.sb = solve_beta
        self.zp = calc_zps
        self.psfshape = shape
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._clean()

    def _clean(self):
        self.ens._clean()

    def subtract(self):
        t0 = time.time()
        ref = self.ens.atoms[0]
        new = self.ens.atoms[1]

        shape = ref.imagedata.shape
        if self.psfshape is not None:
            _, psf_ref = ref.get_variable_psf(shape=self.psfshape)
            _, psf_new = new.get_variable_psf(shape=self.psfshape)
        else:
            _, psf_ref = ref.get_variable_psf(shape=self.psfshape)
            _, psf_new = new.get_variable_psf(shape=self.psfshape)


        psf_ref = psf_ref[0]/np.sum(psf_ref[0])
        psf_new = psf_new[0]/np.sum(psf_new[0])

        psf_ref_hat = _fftwn(psf_ref, s=shape)
        psf_new_hat = _fftwn(psf_new, s=shape)

        ref_shift = np.zeros_like(psf_ref)
        ref_shift[np.where(psf_ref==np.max(psf_ref))] = 1.
        ref_shift = _fftwn(ref_shift, s=shape)

        new_shift = np.zeros_like(psf_new)
        new_shift[np.where(psf_new==np.max(psf_new))] = 1.
        new_shift = _fftwn(new_shift, s=shape)

        if self.zp:
            zps = self.ens.transparencies
            r_zp = zps[0]
            n_zp = zps[1]

            print 'Ref_zp = {}, New_zp = {}'.format(r_zp, n_zp)
        else:
            r_zp = 1.
            n_zp = 1.

        r_var = ref.bkg.globalrms
        n_var = new.bkg.globalrms

        D_hat_r = psf_new_hat * _fftwn(ref.bkg_sub_img) * ref_shift.conjugate()
        D_hat_n = psf_ref_hat * _fftwn(new.bkg_sub_img) * new_shift.conjugate()

        if (self.sb or (r_zp==1.0 and n_zp==1.0)):
            from scipy.ndimage.fourier import fourier_shift

            #~ gamma = new.bkg.back()-ref.bkg.back()
            def cost_beta(vec):
                beta, dx, dy = vec[:]

                norm  = beta*beta*(r_var*r_var*np.absolute(psf_new_hat)**2)
                norm += n_var*n_var * np.absolute(psf_ref_hat)**2

                #~ gamma_p = gamma/np.sqrt(n_var**2 + r_var**2 * beta**2)

                cost = _ifftwn(D_hat_n/np.sqrt(norm)) - \
                       _ifftwn(fourier_shift((D_hat_r/np.sqrt(norm))*beta, (dx,dy)))

                #return np.sqrt(np.average(np.square(cost[50:-50, 50:-50])))
                clipped = sigma_clip(cost.real[50:-50, 50:-50], 8)
                return clipped.filled(0).reshape(-1)
                #return cost.real[50:-50, 50:-50].reshape(-1)


            tbeta0 = time.time()
            vec0 = [n_zp/r_zp, 0., 0.]
            bounds = ([0.1, -0.5, -0.5], [3., 0.5, 0.5])
            solv_beta = optimize.least_squares(cost_beta,
                                               vec0,
                                               bounds=bounds)
            tbeta1 = time.time()

            if solv_beta.success:
                print 'Found that beta = {}'.format(solv_beta.x)
                print 'Took only {} awesome seconds'.format(tbeta1-tbeta0)
                print 'The solution was with cost {}'.format(solv_beta.cost)
                beta, dx, dy = solv_beta.x
            else:
                print 'Least squares could not find our beta  :('
                print 'Beta is overriden to be the zp ratio again'
                beta =  n_zp/r_zp
                dx = 0.
                dy = 0.
        else:
            beta = n_zp/r_zp
            dx = 0.
            dy = 0.
        norm  = beta*beta*(r_var*r_var*np.absolute(psf_new_hat)**2)
        norm += n_var*n_var * np.absolute(psf_ref_hat)**2

        if dx==0. and dy==0.:
            D_hat = (D_hat_n - beta * D_hat_r)/np.sqrt(norm)
        else:
            D_hat = (D_hat_n - fourier_shift(beta*D_hat_r, (dx,dy)))/np.sqrt(norm)
        D = _ifftwn(D_hat)

        d_zp = n_zp/np.sqrt(r_var*r_var*beta*beta + n_var*n_var)
        P_hat =(psf_ref_hat * psf_new_hat * beta)/(np.sqrt(norm)*d_zp)

        P = _ifftwn(P_hat).real
        shift = np.zeros_like(P)
        shift[np.where(P==np.max(P))] = 1.
        shift = _fftwn(shift, s=shape)

        S_hat = d_zp * D_hat * P_hat.conjugate() * shift

        kr=_ifftwn(beta*n_zp*psf_ref_hat.conjugate()*np.absolute(psf_new_hat)**2/norm)
        kn=_ifftwn(beta*n_zp*psf_new_hat.conjugate()*np.absolute(psf_ref_hat)**2/norm)

        V_en = _ifftwn(_fftwn(new.imagedata+1.)*_fftwn(kn*kn, s=shape))
        V_er = _ifftwn(_fftwn(ref.imagedata+1.)*_fftwn(kr*kr, s=shape))

        S_corr = _ifftwn(S_hat)/np.sqrt(V_en + V_er)
        print 'S_corr sigma_clipped_stats '
        print 'mean = {}, median = {}, std = {}\n'.format(*sigma_clipped_stats(S_corr.real))
        print 'Subtraction performed in {} seconds'.format(time.time()-t0)

        #import ipdb; ipdb.set_trace()
        return D, P, S_corr.real

    def get_transients(self, threshold=2.5, neighborhood_size=5.):
        S = self.subtract()[2]
        threshold = np.std(S) * threshold
        cat = u.find_S_local_maxima(S, threshold=threshold,
                                    neighborhood_size=neighborhood_size)

        return cat






