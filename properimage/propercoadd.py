#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  propercoadd.py
#
#  Copyright 2017 Bruno S <bruno@oac.unc.edu.ar>
#
# This file is part of ProperImage (https://github.com/toros-astro/ProperImage)
# License: BSD-3-Clause
# Full Text: https://github.com/toros-astro/ProperImage/blob/master/LICENSE.txt
#
"""propercoadd module from ProperImage,
for coadding astronomical images.

Written by Bruno SANCHEZ

PhD of Astromoy - UNC
bruno@oac.unc.edu.ar

Instituto de Astronomia Teorica y Experimental (IATE) UNC
Cordoba - Argentina

Of 301
"""

from multiprocessing import Queue

# from collections import MutableSequence

import numpy as np

from scipy.ndimage.fourier import fourier_shift

from . import utils
from .combinator import StackCombinator
from .single_image import SingleImage as si
from .single_image import chunk_it

try:
    import cPickle as pickle  # noqa
except ImportError:
    import pickle

try:
    import pyfftw

    _fftwn = pyfftw.interfaces.numpy_fft.fftn  # noqa
    _ifftwn = pyfftw.interfaces.numpy_fft.ifftn  # noqa
except ImportError:
    _fftwn = np.fft.fft2
    _ifftwn = np.fft.ifft2


def stack_R(si_list, align=True, inf_loss=0.2, n_procs=2):
    """Function that takes a list of SingleImage instances
    and performs a stacking using properimage R estimator

    """

    for i_img, animg in enumerate(si_list):
        if not isinstance(animg, si):
            si_list[i_img] = si(animg)

    if align:
        img_list = utils.align_for_coadd(si_list)
        for an_img in img_list:
            an_img.update_sources()
    else:
        img_list = si_list

    shapex = np.min([an_img.data.shape[0] for an_img in img_list])
    shapey = np.min([an_img.data.shape[1] for an_img in img_list])
    global_shape = (shapex, shapey)

    zps, meanmags = utils.transparency(img_list)
    for j, an_img in enumerate(img_list):
        an_img.zp = zps[j]
        an_img._setup_kl_a_fields(inf_loss)

    psf_shapes = [an_img.stamp_shape[0] for an_img in img_list]
    psf_shape = np.max(psf_shapes)
    psf_shape = (psf_shape, psf_shape)

    if n_procs > 1:
        queues = []
        procs = []
        for chunk in chunk_it(img_list, n_procs):
            queue = Queue()
            proc = StackCombinator(
                chunk, queue, shape=global_shape, stack=True, fourier=False
            )
            print("starting new process")
            proc.start()

            queues.append(queue)
            procs.append(proc)

        print("all chunks started, and procs appended")

        S_hat = np.zeros(global_shape, dtype=np.complex128)
        P_hat = np.zeros(global_shape, dtype=np.complex128)
        mix_mask = np.zeros(global_shape, dtype=np.bool)
        for q in queues:
            serialized = q.get()
            print("loading pickles")
            s_hat_comp, psf_hat_sum, mask = pickle.loads(serialized)
            np.add(s_hat_comp, S_hat, out=S_hat)  # , casting='same_kind')
            np.add(psf_hat_sum, P_hat, out=P_hat)  # , casting='same_kind')
            mix_mask = np.ma.mask_or(mix_mask, mask)

        P_r_hat = np.sqrt(P_hat)
        P_r = _ifftwn(fourier_shift(P_r_hat, psf_shape))
        P_r = P_r / np.sum(P_r)
        R = _ifftwn(S_hat / np.sqrt(P_hat))

        print("S calculated, now starting to join processes")

        for proc in procs:
            print("waiting for procs to finish")
            proc.join()

        print("processes finished, now returning R")
    else:
        S_hat = np.zeros(global_shape, dtype=np.complex128)
        P_hat = np.zeros(global_shape, dtype=np.complex128)
        mix_mask = img_list[0].data.mask

        for an_img in img_list:
            np.add(an_img.s_hat_comp, S_hat, out=S_hat)
            np.add(
                ((an_img.zp / an_img.var) ** 2) * an_img.psf_hat_sqnorm(),
                P_hat,
                out=P_hat,
            )
            mix_mask = np.ma.mask_or(mix_mask, an_img.data.mask)
        P_r_hat = np.sqrt(P_hat)
        P_r = _ifftwn(fourier_shift(P_r_hat, psf_shape))
        P_r = P_r / np.sum(P_r)
        R = _ifftwn(S_hat / P_r_hat)

    return R, P_r, mix_mask
