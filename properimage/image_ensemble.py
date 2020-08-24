#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  image_ensemble.py
#
#  Copyright 2017 Bruno S <bruno@oac.unc.edu.ar>
#
# This file is part of ProperImage (https://github.com/toros-astro/ProperImage)
# License: BSD-3-Clause
# Full Text: https://github.com/toros-astro/ProperImage/blob/master/LICENSE.txt
#

"""image_ensemble module from ProperImage,
for coadding astronomical images.

Written by Bruno SANCHEZ

PhD of Astromoy - UNC
bruno@oac.unc.edu.ar

Instituto de Astronomia Teorica y Experimental (IATE) UNC
Cordoba - Argentina

Of 301
"""

from multiprocessing import Queue
from collections import MutableSequence

import numpy as np

from . import utils
from . import combinator as cm
from . import single_image as si


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


class ImageEnsemble(MutableSequence):
    """Processor for several images that uses SingleImage as an atomic
    processing unit. It deploys the utilities provided in the mentioned
    class and combines the results, making possible to coadd and subtract
    astronomical images with optimal techniques.

    Parameters
    ----------
    imgpaths: List or tuple of path of images. At this moment it should be a
    fits file for each image.

    Returns
    -------
    An instance of ImageEnsemble

    """

    def __init__(
        self, imglist, masklist=None, inf_loss=0.1, align=True, *arg, **kwargs
    ):
        super(ImageEnsemble, self).__init__(*arg, **kwargs)

        if masklist is not None:
            self.imglist = zip(imglist, masklist)
        else:
            self.masklist = np.repeat(masklist, len(imglist))
            self.imglist = zip(imglist, self.masklist)
        self.inf_loss = inf_loss

    def __setitem__(self, i, v):
        self.imglist[i] = v

    def __getitem__(self, i):
        return self.imglist[i]

    def __delitem__(self, i):
        del self.imglist[i]

    def __len__(self):
        return len(self.imgl)

    def insert(self, i, v):
        self.imgl.insert(i, v)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._clean()

    @property
    def atoms(self):
        """Property method.
        Transforms the list of images into a list of 'atoms'
        that are instances of the SingleImage class.
        This atoms are capable of compute statistics of Psf on every image,
        and are the main unit of image processing.

        Parameters
        ----------
        None parameters are passed, it is a property.

        Returns
        -------
        A list of instances of SingleImage class, one per each image in the
        list of images passed to ImageEnsemble.

        """
        if not hasattr(self, "_atoms"):
            self._atoms = [
                si.SingleImage(im[0], mask=im[1]) for im in self.imglist
            ]
        elif len(self._atoms) is not len(self.imglist):
            self._atoms = [
                si.SingleImage(im[0], mask=im[1]) for im in self.imglist
            ]
        return self._atoms

    @property
    def global_shape(self):
        if not hasattr(self, "_global_shape"):
            shapex = np.min([at.data.shape[0] for at in self.atoms])
            shapey = np.min([at.data.shape[1] for at in self.atoms])
            self._global_shape = (shapex, shapey)
            print(self._global_shape)
        return self._global_shape

    @property
    def transparencies(self):
        zps, meanmags = utils.transparency(self.atoms)
        self._zps = zps
        for j, anatom in enumerate(self.atoms):
            anatom.zp = zps[j]
        return self._zps

    def calculate_S(self, n_procs=2):
        """Method for properly coadding images given by Zackay & Ofek 2015
        (http://arxiv.org/abs/1512.06872, and http://arxiv.org/abs/1512.06879)
        It uses multiprocessing for parallelization of the processing of each
        image.

        Parameters
        ----------
        n_procs: int
            number of processes for computational parallelization. Should not
            be greater than the number of cores of the machine.

        Returns
        -------
        S: np.array 2D of floats
            S image, calculated by the SingleImage method s_component.

        """
        queues = []
        procs = []
        for chunk in si.chunk_it(self.atoms, n_procs):
            queue = Queue()
            proc = cm.Combinator(
                chunk,
                queue,
                shape=self.global_shape,
                stack=True,
                fourier=False,
            )
            print("starting new process")
            proc.start()

            queues.append(queue)
            procs.append(proc)

        print("all chunks started, and procs appended")

        S = np.zeros(self.global_shape)
        for q in queues:
            serialized = q.get()
            print("loading pickles")
            s_comp = pickle.loads(serialized)
            print(s_comp.shape)
            S = np.ma.add(
                s_comp[: self.global_shape[0], : self.global_shape[1]], S
            )

        print("S calculated, now starting to join processes")

        for proc in procs:
            print("waiting for procs to finish")
            proc.join()

        print("processes finished, now returning S")
        return S

    def calculate_R(self, n_procs=2, return_S=False, debug=False):
        """Method for properly coadding images given by Zackay & Ofek 2015
        (http://arxiv.org/abs/1512.06872, and http://arxiv.org/abs/1512.06879)
        It uses multiprocessing for parallelization of the processing of each
        image.

        Parameters
        ----------
        n_procs: int
            number of processes for computational parallelization. Should not
            be greater than the number of cores of the machine.

        Returns
        -------
        R: np.array 2D of floats
            R image, calculated by the ImageEnsemble method.

        """
        queues = []
        procs = []
        for chunk in si.chunk_it(self.atoms, n_procs):
            queue = Queue()
            proc = cm.Combinator(chunk, queue, fourier=True, stack=False)
            print("starting new process")
            proc.start()

            queues.append(queue)
            procs.append(proc)

        print("all chunks started, and procs appended")

        S_stk = []
        S_hat_stk = []

        for q in queues:
            serialized = q.get()
            print("loading pickles")
            s_list, s_hat_list = pickle.loads(serialized)

            S_stk.extend(s_list)
            S_hat_stk.extend(s_hat_list)

        S_stack = np.stack(S_stk, axis=-1)
        # S_stack = np.tensordot(S_stack, self.transparencies, axes=(-1, 0))

        S_hat_stack = np.stack(S_hat_stk, axis=-1)

        # real_s_hat = S_hat_stack.real
        # imag_s_hat = S_hat_stack.imag

        # real_std = np.ma.std(real_s_hat, axis=2)
        # imag_std = np.ma.std(imag_s_hat, axis=2)

        # hat_std = real_std + 1j* imag_std

        S = np.ma.sum(S_stack, axis=2)

        # S_hat = _fftwn(S)
        S_hat = np.ma.sum(S_hat_stack, axis=2)

        hat_std = np.ma.std(S_hat_stack, axis=2)
        R_hat = np.ma.divide(S_hat, hat_std)

        R = _ifftwn(R_hat)

        for proc in procs:
            print("waiting for procs to finish")
            proc.join()

        if debug:
            return [S_hat_stack, S_stack, S_hat, S, R_hat]
        if return_S:
            print("processes finished, now returning R, S")
            return R, S
        else:
            print("processes finished, now returning R")
            return R

    def _clean(self):
        """Method to end the sequence processing stage. This is the end
        of the ensemble's life. It empties the memory and cleans the numpydbs
        created for each atom.

        """
        for anatom in self.atoms:
            anatom._clean()
