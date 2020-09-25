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
from multiprocessing import Process

import numpy as np

from scipy.ndimage.fourier import fourier_shift

from . import utils
from .single_image import SingleImage as si
import logging

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


class StackCombinator(Process):
    """Combination engine.
    An engine for image combination in parallel, using multiprocessing.Process
    class.
    Uses an ensemble of images and a queue to calculate the propercoadd of
    the list of images.

    Parameters
    ----------
    img_list: list or tuple
        list of SingleImage instances used in the combination process

    queue: multiprocessing.Queue instance
        an instance of multiprocessing.Queue class where to pickle the
        intermediate results.

    shape: shape of the images being coadded.

    stack: boolean, default True
        Whether to stack the results for coadd or just obtain individual
        image calculations.
        If True it will pickle in queue a coadded image of the chunk's images.
        If False it will pickle in queue a list of individual matched filtered
        images.

    fourier: boolean, default False.
        Whether to calculate individual fourier transform of each s_component
        image.
        If stack is True this parameter will be ignored.
        If stack is False, and fourier is True, the pickled object will be a
        tuple of two values, with the first one containing the list of
        s_components, and the second one containing the list of fourier
        transformed s_components.

    Returns
    -------
    Combinator process
        An instance of Combinator.
        This can be launched like a multiprocessing.Process

    Example
    -------
    queue1 = multiprocessing.Queue()
    queue2 = multiprocessing.Queue()
    p1 = Combinator(list1, queue1)
    p2 = Combinator(list2, queue2)

    p1.start()
    p2.start()

    #results are in queues
    result1 = queue1.get()
    result2 = queue2.get()

    p1.join()
    p2.join()

    """

    def __init__(
        self,
        img_list,
        queue,
        shape,
        stack=True,
        fourier=False,
        *args,
        **kwargs,
    ):
        super(StackCombinator, self).__init__(*args, **kwargs)
        self.list_to_combine = img_list
        self.queue = queue
        self.global_shape = shape
        logging.getLogger("StackCombinator").info(self.global_shape)
        # self.zps = ensemble.transparencies

    def run(self):
        S_hat = np.zeros(self.global_shape).astype(np.complex128)
        psf_hat_sum = np.zeros(self.global_shape).astype(np.complex128)
        mix_mask = self.list_to_combine[0].data.mask

        for an_img in self.list_to_combine:
            np.add(an_img.s_hat_comp, S_hat, out=S_hat, casting="same_kind")
            np.add(
                ((an_img.zp / an_img.var) ** 2) * an_img.psf_hat_sqnorm(),
                psf_hat_sum,
                out=psf_hat_sum,
            )  # , casting='same_kind')
            # psf_hat_sum = ((an_img.zp/an_img.var)**2)*an_img.psf_hat_sqnorm()
            mix_mask = np.ma.mask_or(mix_mask, an_img.data.mask)

        serialized = pickle.dumps([S_hat, psf_hat_sum, mix_mask])
        self.queue.put(serialized)
        return


def stack_R(si_list, align=True, inf_loss=0.2, n_procs=2):
    """Stack a list of SingleImage instances using properimage R estimator.

    Parameters
    ----------
    si_list: list(SingleImage)
        A list of SingleImage objects.

    align: boolean
        Perform alignment before stacking.

    inf_loss: boolean
        Tolerance of the fraction of information lost when calculating PSF
        fits. Default is 0.25 (25% loss).

    Returns
    -------
    R: numpy array
        Stack of images
    P_r: numpy array
        PSF of stack
    mix_mask: numpy array
        The mask of R
    """
    logger = logging.getLogger()
    for i_img, animg in enumerate(si_list):
        if not isinstance(animg, si):
            si_list[i_img] = si(animg)

    if align:
        img_list = utils._align_for_coadd(si_list)
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
        for chunk in utils.chunk_it(img_list, n_procs):
            queue = Queue()
            proc = StackCombinator(
                chunk, queue, shape=global_shape, stack=True, fourier=False
            )
            logger.info("starting new process")
            proc.start()

            queues.append(queue)
            procs.append(proc)

        logger.info("all chunks started, and procs appended")

        S_hat = np.zeros(global_shape, dtype=np.complex128)
        P_hat = np.zeros(global_shape, dtype=np.complex128)
        mix_mask = np.zeros(global_shape, dtype=np.bool)
        for q in queues:
            serialized = q.get()
            logger.info("loading pickles")
            s_hat_comp, psf_hat_sum, mask = pickle.loads(serialized)
            np.add(s_hat_comp, S_hat, out=S_hat)  # , casting='same_kind')
            np.add(psf_hat_sum, P_hat, out=P_hat)  # , casting='same_kind')
            mix_mask = np.ma.mask_or(mix_mask, mask)

        P_r_hat = np.sqrt(P_hat)
        P_r = _ifftwn(fourier_shift(P_r_hat, psf_shape))
        P_r = P_r / np.sum(P_r)
        R = _ifftwn(S_hat / np.sqrt(P_hat))

        logger.info("S calculated, now starting to join processes")

        for proc in procs:
            logger.info("waiting for procs to finish")
            proc.join()

        logger.info("processes finished, now returning R")
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
