#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  single_image2.py
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

from six.moves import range

import numpy as np
from numpy import ma

from scipy import signal as sg

from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.modeling import fitting
from astropy.modeling import models
from astropy.convolution import convolve  # _fft, convolve
from astropy.nddata.utils import extract_array

import sep

from . import numpydb as npdb
#from . import utils
from .image_stats import ImageStats

try:
    import pyfftw
    _fftwn = pyfftw.interfaces.numpy_fft.fft2
    _ifftwn = pyfftw.interfaces.numpy_fft.ifft2
except:
    _fftwn = np.fft.rfft2
    _ifftwn = np.fft.rifft2


class Bunch(dict):

    def __dir__(self):
        return self.keys()

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(attr)


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

    mask: `~numpy.ndarray` or a `str` naming the filename.
        The mask image
    """

    def __init__(self, img=None, mask=None, maskthresh=None, stamp_shape=None):
        self.__img = img
        self.attached_to = img
        self.zp = 1.
        self.pixeldata = img
        self.header = img
        self.mask = mask
        self._bkg = maskthresh
        self.stamp_shape = stamp_shape
        self.inf_loss = 0.1
        self.dbname = os.path.abspath('._'+str(id(self))+'SingleImage')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._clean()

    def __repr__(self):
        return 'SingleImage instance for {}'.format(self.attached_to)

    def _clean(self):
        print('cleaning... ')
        try:
            os.remove(self.dbname+'.dat')
            os.remove(self.dbname+'.map')
        except:
            print('Nothing to clean. (Or something has failed)')

    @property
    def attached_to(self):
        return self.__attached_to

    @attached_to.setter
    def attached_to(self, img):
        if isinstance(img, str):
            self.__attached_to = img
        else:
            self.__attached_to = img.__class__.__name__

    @property
    def pixeldata(self):
        return self.__pixeldata

    @pixeldata.setter
    def pixeldata(self, img):
        if isinstance(img, str):
            self.__pixeldata = ma.asarray(fits.getdata(img)).astype('<f8')
        elif isinstance(img, np.ndarray):
            self.__pixeldata = ma.asarray(img).astype('<f8')
        elif isinstance(img, fits.HDUList):
            if img[0].is_image:
                self.__pixeldata = ma.asarray(img[0].data).astype('<f8')

    @property
    def header(self):
        return self.__header
    @header.setter
    def header(self, img):
        if isinstance(img, str):
            self.__header = fits.getheader(img)
        elif isinstance(img, np.ndarray):
            self.__header = {}
        elif isinstance(img, fits.HDUList):
            self.__header = img[0].header
        elif isinstance(img, fits.PrimaryHDU):
            self.__header = img.header

    @property
    def mask(self):
        return self.__pixeldata.mask

    @mask.setter
    def mask(self, mask):
        if isinstance(mask, str):
            self.__pixeldata.mask = fits.getdata(mask)
        elif isinstance(mask, np.ndarray):
            self.__pixeldata.mask = mask
        elif mask is None:
            # check the fits file
            if self.attached_to=='HDUList':
                if self.header['EXTEND']:
                    self.__pixeldata.mask = self.__img[1].data
            elif self.attached_to=='PrimaryHDU':
                self.__pixeldata = ma.masked_invalid(self.__img.data)
            elif isinstance(self.__img, str):
                ff = fits.open(self.attached_to)
                if ff[0].header['EXTEND']:
                    try:
                        self.__pixeldata.mask = ff[1].data
                    except IndexError:
                        self.__pixeldata = ma.masked_invalid(self.__pixeldata)
            else:
                self.__pixeldata = ma.masked_invalid(self.__pixeldata)

    @property
    def background(self):
        """Image background subtracted property of SingleImage.
        The background is estimated using sep.

        Returns
        -------
        numpy.array 2D
            a background estimation image is returned
        """
        return self.__bkg.back()

    @property
    def _bkg(self):
        return self.__bkg

    @_bkg.setter
    def _bkg(self, maskthresh=None):
        if maskthresh is not None:
            back = sep.Background(self.pixeldata.data,
                                  mask=self.mask,
                                  maskthresh=maskthresh)
            self.__bkg = back
        else:
            back = sep.Background(self.pixeldata.data,
                                  mask=self.mask)
            self.__bkg = back

    @property
    def bkg_sub_img(self):
        return self.pixeldata - self.__bkg

    @property
    def stamp_shape(self):
        return self.__stamp_shape

    @stamp_shape.setter
    def stamp_shape(self, shape):
        if shape is None:
            percent = np.percentile(self.best_sources['npix'], q=65)
            p_sizes = 3.*np.sqrt(percent)

            if p_sizes > 21:
                dx = int(p_sizes)
                if dx % 2 != 1: dx += 1
                shape = (dx, dx)
            else:
                shape = (21, 21)
        self.__stamp_shape = shape

    @property
    def best_sources(self):
        """Property, a dictionary of best sources detected in the image.
        Keys are:
            fitshape: tuple, the size of the stamps on each source detected
            sources: a table of sources, with the imformation from sep
            positions: an array, with the position of each source stamp
            n_sources: the total number of sources extracted
        """
        if not hasattr(self, '_best_sources'):
            # print('looking for srcs')
            try:
                srcs = sep.extract(self.bkg_sub_img.data,
                                   thresh=10*self.__bkg.globalrms,
                                   mask=self.mask)
            except Exception:
                sep.set_extract_pixstack(700000)
                srcs = sep.extract(self.bkg_sub_img.data,
                                   thresh=8*self.__bkg.globalrms,
                                   mask=self.mask)
            if len(srcs) < 20:
                try:
                    srcs = sep.extract(self.bkg_sub_img.data,
                                       thresh=5*self.__bkg.globalrms,
                                       mask=self.mask)
                except Exception:
                    sep.set_extract_pixstack(900000)
                    srcs = sep.extract(self.bkg_sub_img.data,
                                       thresh=5*self.__bkg.globalrms,
                                       mask=self.mask)
            if len(srcs)==0:
                raise ValueError('Few sources detected on image')

            p_sizes = np.percentile(srcs['npix'], q=[25, 55, 75])

            best_big = srcs['npix'] >= p_sizes[0]
            best_small = srcs['npix'] <= p_sizes[2]
            best_flag = srcs['flag'] <= 1

            fluxes_quartiles = np.percentile(srcs['flux'], q=[15, 85])
            low_flux = srcs['flux'] >= fluxes_quartiles[0]
            hig_flux = srcs['flux'] <= fluxes_quartiles[1]

            best_srcs = srcs[best_big & best_flag & best_small & hig_flux & low_flux]

            if len(best_srcs) > 1800:
                jj = np.random.choice(len(best_srcs), 1800, replace=False)
                best_srcs = best_srcs[jj]

            print('Sources found = {}'.format(len(best_srcs)))
            self._best_sources = best_srcs

        return self._best_sources

    @property
    def stamps_pos(self):
        _cond = (hasattr(self, '_shape') and
                 self._shape!=self.stamp_shape and
                 self._shape is not None)
        if not hasattr(self, '_stamps_pos') or _cond:
            if _cond:
                self.stamp_shape = self._shape
            self.db = npdb.NumPyDB_cPickle(self.dbname, mode='store')

            pos = []
            jj = 0
            for row in self.best_sources:
                position = (row['y'], row['x'])
                sub_array_data = extract_array(self.bkg_sub_img.data,
                                               self.stamp_shape, position,
                                               fill_value=self._bkg.globalrms)
                sub_array_data = sub_array_data/np.sum(sub_array_data)

                self.db.dump(sub_array_data, jj)
                pos.append(position)
                jj += 1

            self._stamps_pos = np.array(pos)
            self._n_sources = jj
        return self._stamps_pos

    @property
    def n_sources(self):
        try:
            return self._n_sources
        except AttributeError:
            s = self.stamps_pos
            return self._n_sources

    @property
    def cov_matrix(self):
        """Determines the covariance matrix of the psf measured directly from
        the stamps of the detected stars in the image.

        """
        if not hasattr(self, '_covMat'):
            covMat = np.zeros(shape=(self.n_sources, self.n_sources))

            for i in range(self.n_sources):
                for j in range(self.n_sources):
                    if i <= j:
                        psfi_render = self.db.load(i)[0]
                        psfj_render = self.db.load(j)[0]

                        inner = np.vdot(psfi_render.flatten()/np.sum(psfi_render),
                                        psfj_render.flatten()/np.sum(psfj_render))

                        covMat[i, j] = inner
                        covMat[j, i] = inner
            self._covMat = covMat
        return self._covMat


    @property
    def eigenv(self):
        if not hasattr(self, '_eigenv'):
            try:
                self._eigenv = np.linalg.eigh(self.cov_matrix)
            except:
                raise
        return self._eigenv

    @property
    def kl_basis(self):
        if not hasattr(self, '_kl_basis'):
            self._setup_kl_basis()
        return self._kl_basis

    def _setup_kl_basis(self, inf_loss=None):
        """Determines the KL psf_basis from
        stars detected in the field."""

        if not hasattr(self, '_kl_basis') or self.inf_loss!=inf_loss:
            if inf_loss is not None:
                self.inf_loss = inf_loss

            valh, vech = self.eigenv
            power = abs(valh)/np.sum(abs(valh))
            pw = 1
            cut = 1
            for elem in power[::-1]:
                pw -= elem
                if pw > self.inf_loss:
                    cut += 1

            #  Build psf basis
            n_basis = abs(cut)
            xs = vech[:, -cut:]
            psf_basis = []
            for i in range(n_basis):
                base = np.zeros(self.stamp_shape)
                for j in range(self.n_sources):
                    try:
                        pj = self.db.load(j)[0]
                    except:
                        import ipdb; ipdb.set_trace()
                    base += xs[j, i] * pj
                    norm = np.sqrt(np.sum(base**2.))
                psf_basis.append(base/norm)
            del(base)
            self._kl_basis = psf_basis

    @property
    def kl_afields(self):
        if not hasattr(self, '_a_fields'):
            self._setup_kl_a_fields()
        return self._a_fields

    def get_afield_domain(self):
        x, y = np.mgrid[:self.pixeldata.data.shape[0],
                        :self.pixeldata.data.shape[1]]
        return x, y


    def _setup_kl_a_fields(self, inf_loss=None):
        """Calculate the coefficients of the expansion in basis of KLoeve.

        """
        if not hasattr(self, '_a_fields') or self.inf_loss!=inf_loss:
            if inf_loss is not None:
                self.inf_loss = inf_loss

            psf_basis = self.kl_basis

            n_fields = len(psf_basis)

            if n_fields == 1:
                self._a_fields = [None]
                return self._a_fields

            best_srcs = self.best_sources
            # fitshape = self._best_srcs['fitshape']  # unused variable

            flag_key = [col_name for col_name in best_srcs.dtype.fields.keys()
                        if 'flag' in col_name.lower()][0]

            mask = best_srcs[flag_key] <= 1
            # patches = self._best_srcs['patches'][mask]
            positions = self.stamps_pos[mask]
            best_srcs = best_srcs[mask]

            # Each element in patches brings information about the real PSF
            # evaluated -or measured-, giving an interpolation point for a

            a_fields = []
            measures = np.zeros((n_fields, self.n_sources))
            for i in range(n_fields):
                p_i = psf_basis[i].flatten()
                # p_i_sq = np.sqrt(np.sum(np.dot(p_i, p_i)))

                x = positions[:, 0]
                y = positions[:, 1]
                for j in range(self.n_sources):
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

    def get_variable_psf(self, inf_loss=None, shape=None):
        """Method to obtain the space variant PSF determination,
        according to Lauer 2002 method with Karhunen Loeve transform.

        Parameters
        ----------
        pow_th: float, between 0 and 1. It sets the minimum amount of
        information that a PSF-basis of the Karhunen Loeve transformation
        should have in order to be taken into account. A high value will return
        only the most significant components. Default is 0.9

        shape: tuple, the size of the stamps for source extraction.
        This value affects the _best_srcs property, and should be settled in
        the SingleImage instancing step. At this stage it will override the
        value settled in the instancing step only if _best_srcs hasn't been
        called yet, which is the case if you are performing context managed
        image subtraction. (See propersubtract module)

        Returns
        -------
        [a_fields, psf_basis]: a list of two lists.
        Basically it returns a sequence of psf-basis elements with its
        associated coefficient.

        The psf_basis elements are numpy arrays of the given fitshape (or shape)
        size.

        The a_fields are astropy.fitting fitted model functions, which need
        arguments to return numpy array fields (for example a_fields[0](x, y)).
        These can be generated using
        x, y = np.mgrid[:self.imagedata.shape[0],
                        :self.imagedata.shape[1]]

        """
        if shape is not None:
            self._shape = shape
        if inf_loss is not None:
            self.inf_loss = inf_loss

        self._setup_kl_basis(inf_loss)
        self._setup_kl_a_fields(inf_loss)

        a_fields = self.kl_afields
        psf_basis = self.kl_basis

        return [a_fields, psf_basis]

    @property
    def normal_image(self):
        """Calculates the PSF normalization image given in Lauer 2002,
        for this image using the psf-basis elements and coefficients.

        """
        if not hasattr(self, '_normal_image'):
            a_fields, psf_basis = self.get_variable_psf()

            if a_fields[0] is None:
                a = np.ones_like(self.pixeldata.data)
                self._normal_image = convolve(a, psf_basis[0])

            else:
                x, y = self.get_afield_domain()
                conv = np.zeros_like(self.pixeldata.data)

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
        from the image. Uses the measured psf, and is space variant capable.

        """
        if not hasattr(self, '_s_component'):
            mfilter = np.zeros_like(self.pixeldata.data)
            x, y = self.get_afield_domain()

            a_fields, psf_basis = self.get_variable_psf()

            var = self._bkg.globalrms
            nrm = self.normal_image

            if a_fields[0] is None:
                mfilter = sg.correlate2d(self.bkg_sub_img,
                                         psf_basis[0],
                                         mode='same')
            else:
                for i in range(len(a_fields)):
                    a = a_fields[i]
                    psf = psf_basis[i]

                    cross = np.multiply(a(x, y), self.bkg_sub_img)
                    conv = sg.correlate2d(cross, psf, mode='same')
                    mfilter += conv

            mfilter = mfilter/nrm
            self._s_component = self.zp * mfilter/var**2
        return self._s_component



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




