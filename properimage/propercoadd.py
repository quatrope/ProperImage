#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  propercoadd.py
#
#  Copyright 2017 Bruno S <bruno@oac.unc.edu.ar>
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

from multiprocessing import Queue
from collections import MutableSequence

import numpy as np

from astropy.io import fits

from . import utils
from .combinator import StackCombinator
from .single_image import SingleImage, chunk_it


try:
    import cPickle as pickle
except:
    import pickle

try:
    import pyfftw
    _fftwn = pyfftw.interfaces.numpy_fft.fftn
    _ifftwn = pyfftw.interfaces.numpy_fft.ifftn
except:
    _fftwn = np.fft.fft2
    _ifftwn = np.fft.ifft2



def stack_R(si_list, align=True, inf_loss=0.2, n_procs=2):
    """Function that takes a list of SingleImage instances
    and performs a stacking using properimage R estimator

    """
    if align:
        img_list = utils.align_for_coadd(si_list)
        for an_img in img_list:
            an_img.update_sources()
    else:
        img_list = si_list

    shapex = np.min([an_img.pixeldata.shape[0] for an_img in img_list])
    shapey = np.min([an_img.pixeldata.shape[1] for an_img in img_list])
    global_shape = (shapex, shapey)

    zps, meanmags = utils.transparency(img_list)
    for j, an_img in enumerate(img_list):
        an_img.zp = zps[j]
        an_img._setup_kl_a_fields(inf_loss)


    if n_procs>1:
        queues = []
        procs = []
        for chunk in chunk_it(img_list, n_procs):
            queue = Queue()
            proc = StackCombinator(chunk, queue, shape=global_shape,
                              stack=True, fourier=False)
            print 'starting new process'
            proc.start()

            queues.append(queue)
            procs.append(proc)

        print 'all chunks started, and procs appended'

        S_hat = np.zeros(global_shape).astype(np.complex128)
        P_hat = np.zeros(global_shape).astype(np.complex128)
        for q in queues:
            serialized = q.get()
            print 'loading pickles'
            s_hat_comp, psf_hat_sum = pickle.loads(serialized)
            np.add(s_hat_comp, S_hat, out=S_hat, casting='same_kind')
            np.add(psf_hat_sum, P_hat, out=P_hat, casting='same_kind')
        P_r_hat = np.sqrt(P_hat)
        P_r = _ifftwn(P_r_hat)
        P_r = P_r/np.sum(P_r)
        R = _ifftwn(S_hat/np.sqrt(P_hat))

        print 'S calculated, now starting to join processes'

        for proc in procs:
            print 'waiting for procs to finish'
            proc.join()

        print 'processes finished, now returning R'
    else:
        S_hat = np.zeros(global_shape).astype(np.complex128)
        P_hat = np.zeros(global_shape).astype(np.complex128)
        for an_img in img_list:
            np.add(an_img.s_hat_comp(), S_hat, out=S_hat, casting='same_kind')
            np.add(((an_img.zp/an_img.var)**2)*an_img.psf_hat_sqnorm(), P_hat,
                   out=P_hat, casting='same_kind')
        P_r_hat = np.sqrt(P_hat)
        P_r = _ifftwn(P_r_hat)
        P_r = P_r/np.sum(P_r)
        R = _ifftwn(S_hat/P_r_hat)

    return R, P_r
