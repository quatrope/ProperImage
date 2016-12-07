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
    def __init__(self, refpath, newpath):

        #~ newpath = u.align_for_diff(refpath, newpath)

        self.ens = pc.ImageEnsemble([refpath, newpath])


    def subtract(self):
        ref = self.ens.atoms[0]
        new = self.ens.atoms[1]

        shape = ref.imagedata.shape

        _, psf_ref = ref.get_variable_psf()
        _, psf_new = new.get_variable_psf()

        psf_ref_hat = _fftwn(psf_ref[0], s=shape)
        psf_new_hat = _fftwn(psf_new[0], s=shape)

        r_zp = ref.zp
        n_zp = new.zp

        r_var = ref.bkg.globalrms
        n_var = new.bkg.globalrms

        D_hat_r = n_zp * psf_new_hat.conjugate() * _fftwn(ref.imagedata)
        D_hat_n = r_zp * psf_ref_hat.conjugate() * _fftwn(new.imagedata)

        norm  = r_var*r_var * r_zp*r_zp * psf_ref_hat*psf_ref_hat.conjugate()
        norm += n_var*n_var * n_zp*n_zp * psf_new_hat*psf_new_hat.conjugate()

        D_hat = (D_hat_n - D_hat_r)/np.sqrt(norm)

        #~ psf_rate = 1.  # this is a wrong value, but we can't calculate it now
        #~ lam = 1. + ((n_zp/n_var)/(r_zp/r_var))**2

        #~ D_hat = s_n/lam - (1. - 1./lam)*s_r
        D = _ifftwn(D_hat)

        #~ print lam
        return D, D_hat






