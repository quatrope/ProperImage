#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  combinator.py
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

"""combinator module from ProperImage,
for coadding astronomical images.

Written by Bruno SANCHEZ

PhD of Astromoy - UNC
bruno@oac.unc.edu.ar

Instituto de Astronomia Teorica y Experimental (IATE) UNC
Cordoba - Argentina

Of 301
"""
import numpy as np

from multiprocessing import Process

try:
    import cPickle as pickle  # noqa
except ImportError:
    import pickle

try:
    import pyfftw  # noqa

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
        print(self.global_shape)
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
