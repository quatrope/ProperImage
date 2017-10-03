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
#
"""single_image module from ProperImage,
for coadding astronomical images.

Written by Bruno SANCHEZ

PhD of Astromoy - UNC
bruno@oac.unc.edu.ar

Instituto de Astronomia Teorica y Experimental (IATE) UNC
Cordoba - Argentina

Of 301
"""

import os
from multiprocessing import Process, Queue
from collections import MutableSequence

from six.moves import range

import numpy as np

from scipy.stats import stats
from scipy import signal as sg

from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.modeling import fitting
from astropy.modeling import models
from astropy.convolution import convolve  # _fft, convolve
from astropy.nddata.utils import extract_array

from photutils import psf

import sep
from . import numpydb as npdb
from . import utils

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

chunk_it = utils.chunk_it


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
    def __init__(self, img=None, imagefile=True, sim=False,
                 meta={}, pow_th=0.9):
        self.pow_th = pow_th
        if not imagefile:
            self._attached_to = img.__class__.__name__
        else:
            self._attached_to = img

        if imagefile:
            self.header = fits.getheader(img)
            self.imagedata = fits.getdata(img)
            if not self.imagedata.dtype == 'uint16':
                self.imagedata = self.imagedata.byteswap().newbyteorder()
            elif self.imagedata.dtype == 'int32':
                self.imagedata = self.imagedata.byteswap().newbyteorder()
            else:
                self.imagedata = self.imagedata.astype('float')

            #~ try:
                #~ self.imagedata /= self.header['EXPTIME']
            #~ except:
                #~ pass
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

        self.dbname = os.path.abspath('._'+str(id(self))+'SingleImage')
        self.zp = 1.0  # in case is not setled.

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._clean()

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
    def masked(self):
        """This sets an attribute,
        called self._masked that contains a mask for nans.

        Returns
        -------
        numpy.array 2D
            a background subtracted image is returned

        """
        if not hasattr(self, '_masked'):
            mask1 = np.ma.masked_invalid(self.imagedata)
            mask2 = np.ma.masked_outside(self.imagedata, -50., 25000.)
            mask3 = sigma_clip(self.imagedata, sigma_upper=50)
            self._masked = np.ma.masked_array(self.imagedata, mask= mask1.mask & mask2.mask & mask3.mask)
            #~ print 'background subtracted image obtained'
        return self._masked

    @property
    def bkg_sub_img(self):
        """Image background subtracted property of SingleImage.
        The background is estimated using sep.

        Returns
        -------
        numpy.array 2D
            a background subtracted image is returned

        """
        if not hasattr(self, '_bkg_sub_img'):
            self.bkg = sep.Background(self.masked.data,
                                      mask=self.masked.mask)
            self._bkg_sub_img = self.imagedata - self.bkg

            #~ print 'background subtracted image obtained'
        return self._bkg_sub_img

    @property
    def _best_srcs(self):
        """Property, a table of best sources detected in the image.

        """
        if not hasattr(self, '_best_sources'):
            try:
                srcs = sep.extract(self.bkg_sub_img,
                                   thresh=6*self.bkg.globalrms,
                                   mask=self.masked.mask)
            except Exception:
                sep.set_extract_pixstack(700000)
                srcs = sep.extract(self.bkg_sub_img,
                                   thresh=8*self.bkg.globalrms,
                                   mask=self.masked.mask)
            except ValueError:
                srcs = sep.extract(self.bkg_sub_img.byteswap().newbyteorder(),
                                   thresh=8*self.bkg.globalrms,
                                   mask=self.masked.mask)

            if len(srcs) < 20:
                try:
                    srcs = sep.extract(self.bkg_sub_img,
                                       thresh=5*self.bkg.globalrms,
                                       mask=self.masked.mask)
                except Exception:
                    sep.set_extract_pixstack(900000)
                    srcs = sep.extract(self.bkg_sub_img,
                                       thresh=5*self.bkg.globalrms,
                                       mask=self.masked.mask)
            if len(srcs) < 10:
                print 'No sources detected'

            #~ print 'raw sources = {}'.format(len(srcs))

            p_sizes = np.percentile(srcs['npix'], q=[25, 55, 75])

            best_big = srcs['npix'] >= p_sizes[0]
            best_small = srcs['npix'] <= p_sizes[2]
            best_flag = srcs['flag'] <= 1
            fluxes_quartiles = np.percentile(srcs['flux'], q=[15, 85])
            low_flux = srcs['flux'] > fluxes_quartiles[0]
            # hig_flux = srcs['flux'] < fluxes_quartiles[1]

            # best_srcs = srcs[best_big & best_flag & best_small & hig_flux & low_flux]
            best_srcs = srcs[best_flag & best_small & low_flux & best_big]

            if self._shape is not None:
                fitshape = self._shape
            else:
                p_sizes = 3.*np.sqrt(np.percentile(best_srcs['npix'],
                                                q=[35, 65, 95]))

                if p_sizes[1] >= 21:
                    dx = int(p_sizes[1])
                    if dx % 2 != 1: dx += 1
                    fitshape = (dx, dx)
                else:
                    fitshape = (21, 21)

            if len(best_srcs) > 1800:
                jj = np.random.choice(len(best_srcs), 1800, replace=False)
                best_srcs = best_srcs[jj]

            print 'Sources good to calculate = {}'.format(len(best_srcs))
            self._best_sources = {'sources': best_srcs, 'fitshape': fitshape}

            self.db = npdb.NumPyDB_cPickle(self.dbname, mode='store')

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
            # self.db = npdb.NumPyDB_cPickle(self._dbname, mode='store')

            #~ print 'returning best sources\n'
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
                    psfi_render = self.db.load(i)[0]
                    psfj_render = self.db.load(j)[0]

                    inner = np.vdot(psfi_render.flatten()/np.sum(psfi_render),
                                    psfj_render.flatten()/np.sum(psfj_render))

                    covMat[i, j] = inner
                    covMat[j, i] = inner
        # print 'returning Covariance Matrix'
        return covMat

    @property
    def _kl_PSF(self):
        """Determines the KL psf_basis from PSF gaussian models fitted to
        stars detected in the field.

        """
        pow_th = self.pow_th
        if not hasattr(self, '_psf_KL_basis_model'):
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

            # print 'obtaining KL basis'
        return self._psf_KL_basis_model

    def _kl_from_stars(self, pow_th=None):
        """Determines the KL psf_basis from stars detected in the field.

        """
        if pow_th is None:
            pow_th = self.pow_th
        if not hasattr(self, '_psf_KL_basis_stars'):
            covMat = self._covMat_from_stars()
            # renders = self._best_srcs['patches']
            valh, vech = np.linalg.eigh(covMat)

            power = abs(valh)/np.sum(abs(valh))
                        #  THIS IS A REFACTORING OF THRESHOLDS IN PSF BASIS
            pw = power >= pow_th
            if sum(pw) == 0:
                cut = -1
            else:
                cut = -sum(pw)
            #  Build psf basis
            N_psf_basis = abs(cut)
            xs = vech[:, cut:]
            psf_basis = []
            # THIS IS AN IMPLEMENTATION OF numpydb METHOD FOR ARRAY STORING

            for i in range(N_psf_basis):
                base = np.zeros(self._best_srcs['fitshape'])
                for j in range(self._best_srcs['n_sources']):
                    try: pj = self.db.load(j)[0]
                    except: import ipdb; ipdb.set_trace()
                    base += xs[j, i] * pj
                    norm = np.sqrt(np.sum(base**2.))
                psf_basis.append(base/norm)
            del(base)
            self._psf_KL_basis_stars = psf_basis
            self._valh = valh
            #~ print 'obtainig KL basis, using k = {}'.format(N_psf_basis)
        return self._psf_KL_basis_stars

    def _kl_a_fields(self, pow_th=None, from_stars=True):
        """Calculate the coefficients of the expansion in basis of KLoeve.

        """
        if pow_th is None:
            pow_th = self.pow_th
        if not hasattr(self, '_a_fields'):
            if from_stars:
                psf_basis = self._kl_from_stars(pow_th=pow_th)
            else:
                psf_basis = self._kl_PSF

            N_fields = len(psf_basis)

            if N_fields == 1:
                self._a_fields = None
                return self._a_fields

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
            measures = np.zeros((N_fields, self._best_srcs['n_sources']))
            for i in range(N_fields):
                p_i = psf_basis[i].flatten()
                # p_i_sq = np.sqrt(np.sum(np.dot(p_i, p_i)))

                x = positions[:, 0]
                y = positions[:, 1]
                for j in range(self._best_srcs['n_sources']):
                    if mask[j]:
                        Pval = self.db.load(j)[0].flatten()
                        # redefinir Pval
                        for ii in range(i):
                            Pval -= measures[ii, j] * psf_basis[ii].flatten()

                        Pval_sq = np.sqrt(np.sum(np.dot(Pval, Pval)))
                        measures[i, j] = np.dot(Pval, p_i)/Pval_sq
                    else:
                        measures[i, j] = None

                z = measures[i, :]
                z = z[z > -10000.]
                a_field_model = models.Polynomial2D(degree=4)
                fitter = fitting.LinearLSQFitter()
                a_fields.append(fitter(a_field_model, x, y, z))

            self._a_fields = a_fields
            # print 'obtaining a fields'
        return self._a_fields

    def get_variable_psf(self, from_stars=True, pow_th=None, shape=None): #delete_patches=False,
        self._shape = shape
        if pow_th is None:
            pow_th = self.pow_th
        a_fields = self._kl_a_fields(from_stars=from_stars,
                                     pow_th=pow_th)
        if from_stars:
            psf_basis = self._kl_from_stars(pow_th=pow_th)
        else:
            psf_basis = self._kl_PSF

        #~ print 'returning variable psf'
        return [a_fields, psf_basis]

    @property
    def normal_image(self):
        """Calculates the normalization image from kl

        """
        if not hasattr(self, '_normal_image'):
            a_fields, psf_basis = self.get_variable_psf()

            if a_fields is None:
                a = np.ones(self.imagedata.shape)
                self._normal_image = convolve(a, psf_basis[0])

            else:
                x, y = np.mgrid[:self.imagedata.shape[0],
                                :self.imagedata.shape[1]]
                conv = np.zeros_like(self.bkg_sub_img)

                for i in range(len(a_fields)):
                    a = a_fields[i]
                    a = a(x, y)
                    psf_i = psf_basis[i]
                    conv += convolve(a, psf_i)#, psf_pad=True)#, # mode='same',
                                        # fftn=fftwn, ifftn=ifftwn)
                    # conv += sg.fftconvolve(a, psf_i, mode='same')

                self._normal_image = conv
            #~ print 'getting normal image'
        return self._normal_image

    @property
    def s_component(self):
        """Calculates the matched filter S (from propercoadd) component
        from the image.

        """
        if not hasattr(self, '_s_component'):
            mfilter = np.zeros_like(self.bkg_sub_img)
            x, y = np.mgrid[:mfilter.shape[0], :mfilter.shape[1]]

            a_fields, psf_basis = self.get_variable_psf() #delete_patches=True)

            # var = self.meta['std']
            var = self.bkg.globalrms
            nrm = self.normal_image

            if a_fields is None:
                #~ print 'starting matched filter'
                mfilter = sg.correlate2d(self._masked,
                                         psf_basis[0],
                                         mode='same')
            else:
                for i in range(len(a_fields)):
                    a = a_fields[i]
                    psf = psf_basis[i]
                    #~ print 'calculating Im . a_field({})'.format(i)
                    cross = np.multiply(a(x, y), self._masked)
                    # cross = convolve_fft(self.bkg_sub_img, psf)
                    #~ print 'starting matched filter'
                    conv = sg.correlate2d(cross, psf, mode='same')
                    #~ print 'stacking matched filter'
                    mfilter += conv

            #~ print 'matched filter succesful'
            mfilter = mfilter/nrm
            self._s_component = self.zp * mfilter/var**2
            #~ print 'getting s component'
        return self._s_component

    def _clean(self):
        print 'cleaning... '
        try:
            os.remove(self.dbname+'.dat')
            os.remove(self.dbname+'.map')
        except:
            print 'Nothing to clean. (Or something has failed)'


