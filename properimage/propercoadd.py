#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Propercoadd module from ProperImage,
for coadding astronomical images.

Written by Bruno SANCHEZ

PhD of Astromoy - UNC
bruno@oac.unc.edu.ar

Instituto de Astronomia Teorica y Experimental (IATE) UNC
Cordoba - Argentina

Of 301
"""

from multiprocessing import Process
from multiprocessing import Queue
from collections import MutableSequence
import numpy as np
from scipy.stats import stats
from scipy import signal as sg
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.modeling import fitting
from astropy.modeling import models
from astropy.convolution import convolve_fft
from astropy.nddata.utils import extract_array
from photutils import psf
from astroML import crossmatch as cx
import sep
from . import numpydb as npdb

try:
    import cPickle as pickle
except:
    import pickle

try:
    import pyfftw
    _fftn = pyfftw.interfaces.numpy_fft.fftn
    _ifftn = pyfftw.interfaces.numpy_fft.ifftn
except:
    _fftn = np.fft.fft2
    _ifftn = np.fft.ifft2


class ImageEnsemble(MutableSequence):
    """Processor for several images that uses SingleImage as an atomic processing
    unit. It deploys the utilities provided in the mentioned class and combines
    the results, making possible to coadd and subtract astronomical images with
    optimal techniques.

    Parameters
    ----------
    imgpaths: List or tuple of path of images. At this moment it should be a
    fits file for each image.

    Returns
    -------
    An instance of ImageEnsemble

    """
    def __init__(self, imgpaths, *arg, **kwargs):
        super(ImageEnsemble, self).__init__(*arg, **kwargs)
        self.imgl = imgpaths
        self.global_shape = fits.getdata(imgpaths[0]).shape
        print self.global_shape

    def __setitem__(self, i, v):
        self.imgl[i] = v

    def __getitem__(self, i):
        return self.imgl[i]

    def __delitem__(self, i):
        del self.imgl[i]

    def __len__(self):
        return len(self.imgl)

    def insert(self, i, v):
        self.imgl.insert(i, v)

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
        if not hasattr(self, '_atoms'):
            self._atoms = [SingleImage(im, imagefile=True) for im in self.imgl]
        elif len(self._atoms) is not len(self.imgl):
            self._atoms = [SingleImage(im, imagefile=True) for im in self.imgl]
        return self._atoms

    def transparencies(self):
        pass

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
        for chunk in chunk_it(self.atoms, n_procs):
            queue = Queue()
            proc = Combinator(chunk, queue, stack=True, fourier=False)
            print 'starting new process'
            proc.start()

            queues.append(queue)
            procs.append(proc)

        print 'all chunks started, and procs appended'

        S = np.zeros(self.global_shape)
        for q in queues:
            serialized = q.get()
            print 'loading pickles'
            s_comp = pickle.loads(serialized)

            S = np.ma.add(s_comp, S)

        print 'S calculated, now starting to join processes'

        #~ for proc in procs:
            #~ print 'waiting for procs to finish'
            #~ proc.join()

        print 'processes finished, now returning S'
        return S

    def calculate_R(self, n_procs=2, return_S=False):
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
        for chunk in chunk_it(self.atoms, n_procs):
            queue = Queue()
            proc = Combinator(chunk, queue, fourier=True, stack=False)
            print 'starting new process'
            proc.start()

            queues.append(queue)
            procs.append(proc)

        print 'all chunks started, and procs appended'

        S_stk = []
        S_hat_stk = []

        for q in queues:
            serialized = q.get()
            print 'loading pickles'
            s_list, s_hat_list = pickle.loads(serialized)

            S_stk.extend(s_list)
            S_hat_stk.extend(s_hat_list)

        S_stack = np.stack(S_stk, axis=-1)
        S_hat_stack = np.stack(S_hat_stk, axis=-1)

        #~ for proc in procs:
            #~ print 'waiting for procs to finish'
            #~ proc.join()

        S = np.ma.sum(S_stack, axis=2)
        S_hat = fftwn(S)
        hat_std = np.ma.std(S_hat_stack, axis=2)
        R_hat = np.ma.divide(S_hat, hat_std)

        R = ifftwn(R_hat)

        if return_S:
            print 'processes finished, now returning R, S'
            return R, S
        else:
            print 'processes finished, now returning R'
            return R


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
    def __init__(self, ensemble, queue, stack=True, fourier=False,
                 *args, **kwargs):
        super(Combinator, self).__init__(*args, **kwargs)
        self.list_to_combine = ensemble
        self.queue = queue
        self.stack = stack
        self.fourier = fourier

    def run(self):
        if self.stack:
            shape = self.list_to_combine[0].imagedata.shape
            S = np.zeros(shape)
            for img in self.list_to_combine:
                s_comp = np.ma.masked_array(img.s_component,
                                            np.isnan(img.s_component),
                                            fill_value=0)
                print 'S component obtained, summing arrays'
                S = np.ma.add(s_comp, S)

            print 'chunk processed, now pickling'
            serialized = pickle.dumps(S)
            self.queue.put(serialized)

        else:
            S_stack = []
            for img in self.list_to_combine:
                if np.any(np.isnan(img.s_component)):
                    import ipdb; ipdb.set_trace()
                s_comp = np.ma.masked_array(img.s_component,
                                            np.isnan(img.s_component),
                                            fill_value=0)
                print 'S component obtained'
                S_stack.append(s_comp)

            if self.fourier:
                S_hat_stack = []
                for s_c in S_stack:
                    sh = fftwn(s_c)
                    S_hat_stack.append(np.ma.masked_array(sh,
                                       np.isnan(sh), fill_value=0))
                print 'Fourier transformed'
                print 'chunk processed, now pickling'
                serialized = pickle.dumps((S_stack, S_hat_stack))
            else:
                print 'chunk processed, now pickling'
                serialized = pickle.dumps(S_stack)

            self.queue.put(serialized)


class SingleImage(object):
    """Atomic processor class for a single image.
    Contains several tools for PSF measures, and different coadding
    building structures.

    It includes the pixel matrix data, as long as some descriptions.
    For statistical values of the pixel matrix the class' methods need to be
    called.


    Parameters
    ----------
    img : `~numpy.ndarray` or :class:`~ccdproc.CCDData`,
                `~astropy.io.fits.HDUList`  or a `str` naming the filename.
        The image object to work with

    imagefile : `bool`
        optional information regarding if the img is a fits file
        Default to None, and a guessing attempt will be made.

    """
    def __init__(self, img=None, imagefile=True, sim=False, meta={}):
        if not imagefile:
            self._attached_to = img.__class__.__name__
        else:
            self._attached_to = img

        if imagefile:
            self.imagedata = fits.getdata(img)
            if not self.imagedata.dtype == 'uint16':
                self.imagedata = self.imagedata.byteswap().newbyteorder()

            else:
                self.imagedata = self.imagedata.astype('float')
        else:
            self.imagedata = img

        if sim:
            self.meta = meta
        else:
            imgstats = ImageStats(self.imagedata, 'numpy_array')
            imgstats.calc_stats()
            # imgstats.summary()
            self.meta = imgstats.full_description

        if np.any(np.isnan(self.imagedata)):
            self.imagedata = np.ma.masked_array(self.imagedata,
                                    np.isnan(self.imagedata)).filled(35000.)

        if np.any(self.imagedata < 0.):
            self.imagedata = np.ma.masked_array(self.imagedata,
                                                self.imagedata < 0.).filled(13)

    def __repr__(self):
        return 'SingleImage instance for {}'.format(self._attached_to)

    def sigma_clip_bkg(self):
        """Determine background using sigma clipping stats.
        Sets the bkg, bkg_mean, and bkg_std attributes of the
        SingleImage instance.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        if not hasattr(self, 'bkg'):
            self.bkg = sigma_clip(self.imagedata, iters=10)
            self.bkg_mean = self.bkg.mean()
            self.bkg_sd = self.bkg.std()

    @property
    def bkg_sub_img(self):
        """Image background subtracted property of SingleImage.
        The background is estimated using sep. This also sets an attribute,
        called self._masked that contains a mask for nans.

        Returns
        -------
        numpy.array 2D
            a background subtracted image is returned

        """
        if not hasattr(self, 'bkg_sub_img'):
            self.bkg = sep.Background(self.imagedata)
            self._bkg_sub_img = self.imagedata - self.bkg
            self._masked = np.ma.masked_array(self._bkg_sub_img,
                                              np.isnan(self._bkg_sub_img))
        return self._bkg_sub_img

    def _fit_models_psf(self, best_srcs, indices, fitshape, prf_model, fitter):
        """Hidden method, that fits prf models to selected stars.

        Parameters
        ----------
        best_srcs: sep extraction data structure

        indices: numpy indices extraction

        fitshape: tuple or sequence, givin axis dimensions and sizes of
            data patches arrays

        prf_model: astropy psf model.

        fitter: astropy.modeling fitter. LevMaqFitter is good one.

        Returns
        -------
        model_fits: a list of fitted astropy models

        """
        model_fits = []
        for row in best_srcs:
            position = (row['y'], row['x'])
            y = extract_array(indices[0], fitshape, position)
            x = extract_array(indices[1], fitshape, position)
            sub_array_data = extract_array(self.bkg_sub_img,
                                           fitshape,
                                           position,
                                           fill_value=self.bkg.globalback)
            try:
                prf_model.x_0 = position[1]
                prf_model.y_0 = position[0]
            except:
                prf_model.x_mean = position[1]
                prf_model.y_mean = position[0]

            fit = fitter(prf_model, x, y, sub_array_data)
            resid = sub_array_data - fit(x, y)
            if np.sum(np.square(resid)) < 5*self.bkg.globalrms*fitshape[0]**2:
                model_fits.append(fit)
        print 'succesful fits = {}'.format(len(model_fits))
        return model_fits

    def fit_psf_sep(self, model='astropy-Gaussian2D'):
        """Fit and calculate the Psf of an image using sep source detection.

        Parameters
        ----------
        model: str
            'photutils-IntegratedGaussianPRF' or
            'astropy-Gaussian2D

        Returns
        -------
        a list of astropy models, fitted to the stars of the image.

        """
        # calculate x, y, flux of stars
        best_srcs = self._best_srcs['sources']
        fitshape = self._best_srcs['fitshape']
        print 'Fitshape = {}'.format(fitshape)

        fitter = fitting.LevMarLSQFitter()
        indices = np.indices(self.bkg_sub_img.shape)
        size = max(fitshape)

        if model == 'photutils-IntegratedGaussianPRF':
            prf_model = psf.IntegratedGaussianPRF(x_0=size/2., y_0=size/2.,
                                                  sigma=size/3.)
            prf_model.fixed['flux'] = False
            prf_model.fixed['sigma'] = False
            prf_model.fixed['x_0'] = False
            prf_model.fixed['y_0'] = False

        elif model == 'astropy-Gaussian2D':
            prf_model = models.Gaussian2D(x_stddev=1, y_stddev=1)

        model_fits = self._fit_models_psf(best_srcs, indices, fitshape,
                                          prf_model, fitter)

        return model_fits

    @property
    def _best_srcs(self):
        """Property, a table of best sources detected in the image.

        """
        if not hasattr(self, '_best_sources'):
            try:
                srcs = sep.extract(self.bkg_sub_img,
                                   thresh=5*self.bkg.globalrms)
            except Exception:
                sep.set_extract_pixstack(700000)
                srcs = sep.extract(self.bkg_sub_img,
                                   thresh=5*self.bkg.globalrms)
            except ValueError:
                srcs = sep.extract(self.bkg_sub_img.byteswap().newbyteorder(),
                                   thresh=5*self.bkg.globalrms)

            if len(srcs) < 10:
                try:
                    srcs = sep.extract(self.bkg_sub_img,
                                       thresh=2.5*self.bkg.globalrms)
                except Exception:
                    sep.set_extract_pixstack(900000)
                    srcs = sep.extract(self.bkg_sub_img,
                                       thresh=2.5*self.bkg.globalrms)
            if len(srcs) < 10:
                print 'No sources detected'

            print 'raw sources = {}'.format(len(srcs))

            p_sizes = np.percentile(srcs['npix'], q=[15, 55, 65])

            best_big = srcs['npix'] >= p_sizes[0]
            best_small = srcs['npix'] <= p_sizes[2]
            best_flag = srcs['flag'] <= 2
            fluxes_quartiles = np.percentile(srcs['flux'], q=[15, 85])
            low_flux = srcs['flux'] > fluxes_quartiles[0]
            hig_flux = srcs['flux'] < fluxes_quartiles[1]

            # best_srcs = srcs[best_big & best_flag & best_small & hig_flux & low_flux]
            best_srcs = srcs[best_flag & best_small & low_flux]

            p_sizes = np.sqrt(np.percentile(best_srcs['npix'], q=[15, 55, 65]))
            if not p_sizes[1] < 12:
                fitshape = (int(p_sizes[1]), int(p_sizes[1]))
            else:
                fitshape = (12, 12)

            # if len(best_srcs) > 130:
            #     jj = np.random.choice(len(best_srcs), 130, replace=False)
            #     best_srcs = best_srcs[jj]

            print 'Sources good to calculate = {}'.format(len(best_srcs))
            self._best_sources = {'sources': best_srcs, 'fitshape': fitshape}

            self.db = npdb.NumPyDB_cPickle('test_db', mode='load')
            pos = []
            jj = 0
            for row in best_srcs:
                position = [row['y'], row['x']]
                sub_array_data = extract_array(self.bkg_sub_img,
                                               fitshape, position,
                                               fill_value=self.bkg.globalrms)
                sub_array_data = sub_array_data/np.sum(sub_array_data)
                # Patch.append(sub_array_data)
                self.db.dump(sub_array_data, jj)
                pos.append(position)
                jj += 1
            # self._best_sources['patches'] = np.array(Patch)
            self._best_sources['positions'] = np.array(pos)
            self._best_sources['n_sources'] = jj
            # self._best_sources['detected'] = srcs
        return self._best_sources

    def _covMat_from_stars(self):
        """Determines the covariance matrix of the psf measured directly from
        the detected stars in the image.

        """
        # calculate x, y, flux of stars
        # best_srcs = self._best_srcs['sources']
        fitshape = self._best_srcs['fitshape']
        print 'Fitshape = {}'.format(fitshape)

        # best_srcs = best_srcs[best_srcs['flag']<=1]
        # renders = self._best_srcs['patches']
        nsources = self._best_srcs['n_sources']
        covMat = np.zeros(shape=(nsources, nsources))

        for i in range(nsources):
            for j in range(nsources):
                if i <= j:
                    psfi_render = self.db.load(i)
                    psfj_render = self.db.load(j)

                    inner = np.vdot(psfi_render.flatten()/np.sum(psfi_render),
                                    psfj_render.flatten()/np.sum(psfj_render))

                    covMat[i, j] = inner
                    covMat[j, i] = inner
        return covMat

    def _covMat_psf(self):
        """Determines the covariance matrix for psf gaussian models fitted to
        detected stars in the image

        """
        fitted_models = self.fit_psf_sep()
        covMat = np.zeros(shape=(len(fitted_models), len(fitted_models)))

        renders = [g.render() for g in fitted_models]
        shapes = [np.array(r.shape) for r in renders]
        maxshape = np.array((np.max(shapes), np.max(shapes)))

        for i in range(len(fitted_models)):
            for j in range(len(fitted_models)):
                if i <= j:
                    psfi_render = renders[i]
                    psfj_render = renders[j]
                    shapei = maxshape - np.array(psfi_render.shape)
                    shapej = maxshape - np.array(psfj_render.shape)
                    psfi_render = np.pad(psfi_render,
                                         [[int(shapei[0]/2.),
                                           int(round(shapei[0]/2.))],
                                          [int(shapei[1]/2.),
                                           int(round(shapei[1]/2.))]],
                                         'edge')

                    psfj_render = np.pad(psfj_render,
                                         [[int(shapej[0]/2.),
                                           int(round(shapej[0]/2.))],
                                          [int(shapej[1]/2.),
                                           int(round(shapej[1]/2.))]],
                                         'edge')

                    inner = np.vdot(psfi_render.flatten()/np.sum(psfi_render),
                                    psfj_render.flatten()/np.sum(psfj_render))

                    covMat[i, j] = inner
                    covMat[j, i] = inner

                    renders[i] = psfi_render
                    renders[j] = psfj_render

        return [covMat, renders]

    @property
    def _kl_PSF(self, pow_th=0.99):
        """Determines the KL psf_basis from PSF gaussian models fitted to
        stars detected in the field.

        """
        if not hasattr(self, 'psf_KL_basis_model'):
            covMat, renders = self._covMat_psf()
            valh, vech = np.linalg.eigh(covMat)

            power = abs(valh)/np.sum(abs(valh))
            cum = 0
            cut = 0
            if not pow_th == 1:
                while cum <= pow_th:
                    cut -= 1
                    cum += power[cut]
            else:
                cut = -len(valh)

            #  Build psf basis
            N_psf_basis = abs(cut)
            # lambdas = valh[cut:]  # unused variable
            xs = vech[:, cut:]
            # print cut, lambdas
            psf_basis = []
            for i in range(N_psf_basis):
                psf_basis.append(np.tensordot(xs[:, i], renders, axes=[0, 0]))

            self._psf_KL_basis_model = psf_basis

        return self._psf_KL_basis_model

    def _kl_from_stars(self, pow_th=0.99):
        """Determines the KL psf_basis from stars detected in the field.

        """
        if not hasattr(self, '_psf_KL_basis_stars'):
            covMat = self._covMat_from_stars()
            # renders = self._best_srcs['patches']
            valh, vech = np.linalg.eigh(covMat)

            power = abs(valh)/np.sum(abs(valh))
            cum = 0
            cut = 0
            if not pow_th == 1:
                while cum <= pow_th:
                    cut -= 1
                    cum += power[cut]
            else:
                cut = -len(valh)

            #  Build psf basis
            N_psf_basis = abs(cut)
            # lambdas = valh[cut:]  # unused variable
            xs = vech[:, cut:]
            # print lambdas
            psf_basis = []
            #~ for i in range(N_psf_basis):
                #~ psf_basis.append(np.tensordot(xs[:, i], renders, axes=[0, 0]))

            for i in range(N_psf_basis):
                base = np.zeros(self._best_srcs['fitshape'])
                for j in range(self._best_srcs['n_sources']):
                    base += xs[j, i] * self.db.load(j)
                psf_basis.append(base)

            del(base)
            self._psf_KL_basis_stars = psf_basis
            self._valh = valh

        return self._psf_KL_basis_stars

    def _kl_a_fields(self, pow_th=0.99, from_stars=True):
        """Calculate the coefficients of the expansion in basis of KLoeve.

        """
        if not hasattr(self, '_a_fields'):
            if from_stars:
                psf_basis = self._kl_from_stars(pow_th=pow_th)
            else:
                psf_basis = self._kl_PSF

            N_fields = len(psf_basis)

            best_srcs = self._best_srcs['sources']
            # fitshape = self._best_srcs['fitshape']  # unused variable

            flag_key = [col_name for col_name in best_srcs.dtype.fields.keys()
                        if 'flag' in col_name.lower()][0]

            mask = best_srcs[flag_key] <= 1
            # patches = self._best_srcs['patches'][mask]
            positions = self._best_srcs['positions'][mask]
            best_srcs = best_srcs[mask]

            # Each element in patches brings information about the real PSF
            # evaluated -or measured-, giving an interpolation point for a

            a_fields = []
            for i in range(N_fields):
                p_i = psf_basis[i].flatten()
                p_i_sq = np.sum(np.dot(p_i, p_i))

                measures = []
                x = positions[:, 0]
                y = positions[:, 1]
                for j in range(self._best_srcs['n_sources']):
                    if mask[j]:
                        Pval = self.db.load(j).flatten()
                        measures.append(np.dot(Pval, p_i)/p_i_sq)

                z = np.array(measures)
                # x_domain = [0, self.imagedata.shape[0]]
                # y_domain = [0, self.imagedata.shape[1]]
                a_field_model = models.Polynomial2D(degree=4)
                #     , x_domain=x_domain, y_domain=y_domain)
                fitter = fitting.LinearLSQFitter()
                a_fields.append(fitter(a_field_model, x, y, z))

            self._a_fields = a_fields
        return self._a_fields

    def get_variable_psf(self, from_stars=True, #delete_patches=False,
                         pow_th=0.90):
        a_fields = self._kl_a_fields(from_stars=from_stars,
                                     pow_th=pow_th)
        if from_stars:
            psf_basis = self._kl_from_stars(pow_th=pow_th)
        else:
            psf_basis = self._kl_PSF
        #if delete_patches:
        #    del(self._best_srcs['patches'])
        #    print 'Patches deleted!'
        return [a_fields, psf_basis]

    @property
    def normal_image(self):
        """Calculates the normalization image from kl

        """
        if not hasattr(self, '_normal_image'):
            a_fields, psf_basis = self.get_variable_psf()
            x, y = np.mgrid[:self.imagedata.shape[0],
                            :self.imagedata.shape[1]]
            conv = np.zeros_like(self.bkg_sub_img)

            for i in range(len(a_fields)):
                a = a_fields[i]
                a = a(x, y)
                psf_i = psf_basis[i]
                conv += convolve_fft(a, psf_i, psf_pad=True, # mode='same',
                                     fftn=fftwn, ifftn=ifftwn)
                # conv += sg.fftconvolve(a, psf_i, mode='same')

            self._normal_image = conv
        return self._normal_image

    @property
    def s_component(self):
        """Calculates the matched filter S (from propercoadd) component
        from the image.

        """
        if not hasattr(self, '_s_component'):
            var = self.meta['std']
            nrm = self.normal_image
            a_fields, psf_basis = self.get_variable_psf() #delete_patches=True)
            mfilter = np.zeros_like(self.bkg_sub_img)
            x, y = np.mgrid[:mfilter.shape[0], :mfilter.shape[1]]

            for i in range(len(a_fields)):
                a = a_fields[i]
                psf = psf_basis[i]
                cross = np.multiply(a(x, y), self._masked)
                # cross = convolve_fft(self.bkg_sub_img, psf)
                # import ipdb; ipdb.set_trace()
                conv = sg.correlate2d(cross, psf, mode='same')
                mfilter += conv

            mfilter = mfilter/nrm
            self._s_component = mfilter/var**2
        return self._s_component


class ImageStats(object):
    """
    A class with methods for calculating statistics for an astronomical image.

    It includes the pixel matrix data, as long as some descriptions.
    For statistical values of the pixel matrix the class' methods need to be
    called.
    Parameters
    ----------
    image_obj : `~numpy.ndarray` or :class:`~ccdproc.CCDData`,
                `~astropy.io.fits.HDUList`  or a `str` naming the filename.
        The image object to work with

    dataformat : `str`
        optional dataformat of the image_object.
        Default to None, and a guessing attempt will be made.
    """
    def __init__(self, image_obj, dataformat=None):
        self._attached_to = repr(image_obj)
        self.full_description = {}

        if dataformat is None:
            try:
                dataformat = image_obj.__class__.__name__
            except:
                raise InputError('Dataformat not set nor guessable')

        if dataformat not in ('CCDData', 'fits_file', 'numpy_array', 'hdu',
                            'ndarray', 'str', 'HDUList'):
            raise InputError('Dataformat not recognized, try one of these \
            \n CCDData, fits_file, numpy_array, hdu')

        if dataformat == 'CCDData':
            self.pixmatrix = image_obj.data
            assert isinstance(self.pixmatrix, np.array)
        elif dataformat == 'fits_file' or dataformat == 'str':
            self.pixmatrix = fits.open(image_obj)[0].data
        elif dataformat == 'numpy_array' or dataformat == 'ndarray':
            self.pixmatrix = image_obj
            self.pixmatrix = np.ma.masked_array(self.pixmatrix,
                                                np.isnan(self.pixmatrix))
        elif dataformat == 'HDUList':
            self.pixmatrix = image_obj[0].data
        else:
            self.pixmatrix = image_obj[0].data

    def __repr__(self):
        return 'ImageStats instance for {}'.format(self._attached_to)

    def pix_sd(self):
        sd = self.pixmatrix.std()
        self.full_description['std'] = sd
        return sd

    def pix_median(self):
        m = np.median(self.pixmatrix)
        self.full_description['median'] = m
        return m

    def count_hist(self):
        h = stats.histogram(self.pixmatrix.flatten(), numbins=30)
        self.full_description['histogram'] = h
        return h

    def pix_mean(self):
        m = np.mean(self.pixmatrix)
        self.full_description['mean'] = m
        return m

    def to1d(self):
        self._oneDdata = self.pixmatrix.flatten()
        return

    def calc_stats(self):
        self.sd = self.pix_sd()
        self.median = self.pix_median()
        # self.hist = self.count_hist()
        self.mean = self.pix_mean()
        # self.to1d()
        return

    def summary(self):
        self.to1d()
        self.summ = stats.describe(self._oneDdata)
        # print self.summ
        return


def chunk_it(seq, num):
    """Creates chunks of a sequence suitable for data parallelism using
    multiprocessing.

    Parameters
    ----------
    seq: list, array or sequence like object. (indexable)
        data to separate in chunks

    num: int
        number of chunks required

    Returns
    -------
    Sorted list.
    List of chunks containing the data splited in num parts.

    """
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return sorted(out, reverse=True)


def fftwn(*dat):
    """Wrapper around the fftw library, returning a transform of fourier.
    Default number of threads is 4

    """
    return _fftn(*dat, threads=4)


def ifftwn(*dat):
    """Wrapper around the fftw library, returning an inverse transform of fourier.
    Default number of threads is 4

    """
    return _ifftn(*dat, threads=4)


def matching(master, cat, angular=False, radius=1.5):
    """Function to match stars between frames.
    """
    if angular:
        masterRaDec = np.empty((len(master), 2), dtype=np.float64)
        try:
            masterRaDec[:, 0] = master['RA']
            masterRaDec[:, 1] = master['Dec']
        except:
            masterRaDec[:, 0] = master['ra']
            masterRaDec[:, 1] = master['dec']
        imRaDec = np.empty((len(cat), 2), dtype=np.float64)
        try:
            imRaDec[:, 0] = cat['RA']
            imRaDec[:, 1] = cat['Dec']
        except:
            imRaDec[:, 0] = cat['ra']
            imRaDec[:, 1] = cat['dec']
        radius2 = radius/3600.
        dist, ind = cx.crossmatch_angular(masterRaDec, imRaDec,
                                          max_distance=radius2/2.)
        dist_, ind_ = cx.crossmatch_angular(imRaDec, masterRaDec,
                                            max_distance=radius2/2.)
    else:
        masterXY = np.empty((len(master), 2), dtype=np.float64)
        masterXY[:, 0] = master['x']
        masterXY[:, 1] = master['y']
        imXY = np.empty((len(cat), 2), dtype=np.float64)
        imXY[:, 0] = cat['x']
        imXY[:, 1] = cat['y']
        dist, ind = cx.crossmatch(masterXY, imXY, max_distance=radius)
        dist_, ind_ = cx.crossmatch(imXY, masterXY, max_distance=radius)
        #imRaDec = imXY

    match = ~np.isinf(dist)
    match_ = ~np.isinf(dist_)

    IDs = np.zeros_like(ind_) - 13133
    for i in range(len(ind_)):
        if dist_[i] != np.inf:
            dist_o = dist_[i]
            ind_o = ind_[i]
            if dist[ind_o] != np.inf:
                dist_s = dist[ind_o]
                ind_s = ind[ind_o]
                if ind_s == i:
                    try:
                        IDs[i] = master['SOURCE_ID'][ind_o]
                    except:
                        try:
                            IDs[i] = master['masterindex'][ind_o]
                        except: raise
                    #ff = master['SOURCE_ID'] == new_catID
                    #handshake = i + 1 == master['SOURCE_ID'][ff]
                    # if handshake is multiple is because multiple
                    # newsources point to a single
                    # mastersource and this needs to be iterated
                    #for h in handshake:
                    #    if h: IDs.append(new_catID)
        #if len(IDs) == i:
        #    IDs.append(-13133)
    #print len(IDs), len(ind_), len(ind)
    return(IDs)
