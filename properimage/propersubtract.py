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

        new = u.align_for_diff(refpath, newpath)

        self.ens = pc.ImageEnsemble([refpath, new])


    def subtract(self):
        ref = self.ens.atoms[0]
        new = self.ens.atoms[1]

        s_r = _fftwn(ref.s_component)
        s_n = _fftwn(new.s_component)

        r_zp = ref.zp
        n_zp = new.zp

        r_var = ref.bkg.globalrms
        n_var = new.bkg.globalrms

        psf_rate = 1.  # this is a wrong value, but we can't calculate it now
        lam = 1. + ((n_zp/n_var)/(r_zp/r_var))**2

        D_hat = s_n/lam - (1. - 1./lam)*s_r
        D = _ifftwn(D_hat)

        print lam
        return D, D_hat






