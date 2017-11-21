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



def diff(ref, new, align=True, inf_loss=0.2, beta=True, shift=True):
    """Function that takes a list of SingleImage instances
    and performs a stacking using properimage R estimator

    """

    if align:
        img_list = utils.align_for_coadd(si_list)
        for an_img in img_list:
            an_img.update_sources()
    else:
        img_list = si_list

    zps, meanmags = utils.transparency([ref, new])
    ref.zp = zps[0]
    new.zp = zps[1]

    shift_ref = np.where(psf_ref==np.max(psf_ref))
    shift_new = np.where(psf_new==np.max(psf_new))

    psf_ref = ref.get_variable_psf(inf_loss)
    psf_new = new.get_variable_psf(inf_loss)

    psf_ref_hat = _fftwn(psf_ref, s=ref.pixeldata.shape)
    psf_new_hat = _fftwn(psf_new, s=new.pixeldata.shape)

    D_hat_r = fourier_shift(psf_new_hat * _fftwn(ref.interp), -1*shift_ref)
    D_hat_n = fourier_shift(psf_ref_hat * _fftwn(new.interp), -1*shift_new)

    if beta:
        new_back = sep.Background(new.interped).back()
        ref_back = sep.Background(ref.interped).back()
        gamma = new_back - ref_back

        #start with beta=1

        if shift:
            def cost_beta(vec):
                b, dx, dy = vec[:]
                b = 1
                gamma = gamma/np.sqrt(new.var**2 + b*b* * ref.var**2)

                norm  = b*b*(ref.var**2 * psf_new_hat*psf_new_hat.conj())
                norm += new.var**2 * psf_ref_hat * psf_ref_hat.conj()

                cost = _ifftwn(D_hat_n/np.sqrt(norm)) - \
                       _ifftwn(fourier_shift((D_hat_r/np.sqrt(norm))*beta, (dx,dy))) -\
                       _ifftwn(fourier_shift(_fftwn(gamma), (dx, dy))
                cost = np.absolute(cost*cost.conj())

                return sigma_clipped_stats(cost[50:-50, 50:50], sigma=5.)[1]
        else:
            def cost_beta(beta)







