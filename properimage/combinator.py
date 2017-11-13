#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  image_stats.py
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


class Combinator(Process):
    """Combination engine.
    An engine for image combination in parallel, using multiprocessing.Process
    class.
    Uses an ensemble of images and a queue to calculate the propercoadd of
    the list of images.

    Parameters
    ----------
    ensemble: list or tuple
        list of SingleImage instances used in the combination process

    queue: multiprocessing.Queue instance
        an instance of multiprocessing.Queue class where to pickle the
        intermediate results.

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
    def __init__(self, ensemble, queue, shape, stack=True, fourier=False,
                 *args, **kwargs):
        super(Combinator, self).__init__(*args, **kwargs)
        self.list_to_combine = ensemble
        self.queue = queue
        self.stack = stack
        self.fourier = fourier
        self.global_shape = shape
        print self.global_shape
        # self.zps = ensemble.transparencies

    def run(self):
        if self.stack:
            S = np.zeros(self.global_shape)
            for img in self.list_to_combine:
                s_comp = np.ma.MaskedArray(img.s_component, img.mask)
                print 'S component obtained, summing arrays'
                S = np.ma.add(s_comp[:self.global_shape[0],
                                     :self.global_shape[1]], S)

            print 'chunk processed, now pickling'
            serialized = pickle.dumps(S)
            self.queue.put(serialized)
            return
        else:
            S_stack = []
            for img in self.list_to_combine:
                if np.any(np.isnan(img.s_component)):
                    import ipdb; ipdb.set_trace()
                s_comp = np.ma.MaskedArray(img.s_component, img.mask)
                print 'S component obtained'
                S_stack.append(s_comp)

            if self.fourier:
                S_hat_stack = []
                for s_c in S_stack:
                    sh = _fftwn(s_c)
                    S_hat_stack.append(np.ma.masked_invalid(sh))
                print 'Fourier transformed'
                print 'chunk processed, now pickling'
                serialized = pickle.dumps((S_stack, S_hat_stack))
            else:
                print 'chunk processed, now pickling'
                serialized = pickle.dumps(S_stack)
            self.queue.put(serialized)
            return
