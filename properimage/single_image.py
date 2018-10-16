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

"""single_image module from ProperImage,
for analysis of astronomical images.

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

# from scipy import signal as sg
from scipy.ndimage import convolve as convolve_scp
from scipy.ndimage.fourier import fourier_shift
from scipy.ndimage import center_of_mass

from astropy.io import fits
# from astropy.stats import sigma_clip
from astropy.stats import sigma_clipped_stats

from astropy.modeling import fitting
from astropy.modeling import models
from astropy.convolution import convolve_fft
# from astropy.convolution import convolve
from astropy.convolution import interpolate_replace_nans
from astropy.convolution import Box2DKernel
# from astropy.convolution import Gaussian2DKernel
from astropy.nddata.utils import extract_array

from astroscrappy import detect_cosmics
import sep

from . import numpydb as npdb
from . import utils

try:
    import pyfftw
    _fftwn = pyfftw.interfaces.numpy_fft.fft2  # noqa
    _ifftwn = pyfftw.interfaces.numpy_fft.ifft2  # noqa
except ImportError:
    _fftwn = np.fft.fft2
    _ifftwn = np.fft.ifft2


def conv(*arg, **kwargs):
    return(convolve_fft(fftn=_fftwn, ifftn=_ifftwn, *arg, **kwargs))


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

    def __init__(self, img=None, mask=None, maskthresh=None, stamp_shape=None,
                 borders=True, crop=((0, 0), (0, 0)), min_sources=None,
                 strict_star_pick=False, smooth_psf=True):
        self.borders = borders  # try to find zero border padding?
        self.crop = crop  # crop edge?
        self._strict_star_pick = strict_star_pick  # pick stars VERY carefully?
        if min_sources is not None:
            self.min_sources = min_sources
        self.__img = img
        self.attached_to = img
        self.zp = 1.
        self.maskthresh = maskthresh
        self.pixeldata = img
        self.header = img
        self.mask = mask
        self._bkg = maskthresh
        self.stamp_shape = stamp_shape
        self.inf_loss = 0.2
        self._smooth_autopsf = smooth_psf
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
        except OSError:
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
            self.__pixeldata = ma.asarray(fits.getdata(img)).astype('<f4')
        elif isinstance(img, np.ndarray):
            self.__pixeldata = ma.MaskedArray(img, mask=False).astype('<f4')
        elif isinstance(img, fits.HDUList):
            if img[0].is_image:
                self.__pixeldata = ma.asarray(img[0].data).astype('<f4')
        elif isinstance(img, fits.PrimaryHDU):
            if img.is_image:
                self.__pixeldata = ma.asarray(img.data).astype('<f4')
        if self.borders:
            sx, sy = self.__pixeldata.shape
            line = self.__pixeldata.data[sx // 2, :]
            pxsum = 0
            for x, px in enumerate(line):
                pxsum += px
                if pxsum > 0.:
                    ldx = x
                    break
            for dx in range(ldx):
                if not np.sum(self.__pixeldata.data[:dx, :]) == 0:
                    ldx = dx
                    break
            pxsum = 0
            for x, px in enumerate(np.flip(line, axis=0)):
                pxsum += px
                if pxsum > 0.:
                    rdx = sx - x
                    break
            for dx in range(x):
                if not np.sum(self.__pixeldata.data[-dx-1:, :]) == 0:
                    rdx = sx - dx
                    break
            col = self.__pixeldata.data[:, sy // 2]
            pxsum = 0
            for y, px in enumerate(col):
                pxsum += px
                if pxsum > 0.:
                    ldy = y
                    break
            for dy in range(ldy):
                if not np.sum(self.__pixeldata.data[:, :dy]) == 0:
                    ldy = dy
                    break
            pxsum = 0
            for y, px in enumerate(np.flip(line, axis=0)):
                pxsum += px
                if pxsum > 0.:
                    rdy = sy - y
                    break
            for dy in range(y):
                if not np.sum(self.__pixeldata.data[:, -dy-1:]) == 0:
                    rdy = sy - dy
                    break
            self.__pixeldata = self.__pixeldata[ldx:rdx, ldy:rdy]
        if not np.sum(self.crop) == 0.:
            dx, dy = self.crop
            self.__pixeldata = self.__pixeldata[dx[0]:-dx[1], dy[0]:-dy[1]]

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
            fitsmask = fits.getdata(mask)
            if np.median(fitsmask) == 0:
                self.__pixeldata.mask = fitsmask >= self.maskthresh
            else:
                self.__pixeldata.mask = fitsmask <= self.maskthresh
        # if the mask is a separated array
        elif isinstance(mask, np.ndarray):
            self.__pixeldata.mask = mask
        # if the mask is not given
        elif mask is None:
            # check the fits file and try to find it as an extension
            if self.attached_to == 'PrimaryHDU':
                self.__pixeldata = ma.masked_invalid(self.__img.data)

            elif self.attached_to == 'ndarray':
                self.__pixeldata = ma.masked_invalid(self.__img)
            elif self.attached_to == 'HDUList':
                if self.header['EXTEND']:
                    fitsmask = self.__img[1].data
                    if np.median(fitsmask) == 0:
                        self.__pixeldata.mask = fitsmask >= self.maskthresh
                    else:
                        self.__pixeldata.mask = fitsmask <= self.maskthresh
                else:
                    self.__pixeldata = ma.masked_invalid(self.__img[0].data)
            # if a path is given where we find a fits file search on extensions
            else:
                try:
                    ff = fits.open(self.attached_to)
                    if 'EXTEND' in ff[0].header.keys():
                        if ff[0].header['EXTEND']:
                            try:
                                fitsmask = ff[1].data
                                if np.median(fitsmask) == 0:
                                    self.__pixeldata.mask = fitsmask >= \
                                                            self.maskthresh
                                else:
                                    self.__pixeldata.mask = fitsmask <= \
                                                            self.maskthresh
                            except IndexError:
                                self.__pixeldata = ma.masked_invalid(
                                    self.__pixeldata.data)
                except IOError:
                    self.__pixeldata = ma.masked_invalid(self.__pixeldata)
        else:
            self.__pixeldata = ma.masked_greater(self.__pixeldata, 48000.)

        mask_lower = ma.masked_less(self.__pixeldata, -50.)
        mask_greater = ma.masked_greater(self.__pixeldata, 48000.)

        self.__pixeldata.mask = ma.mask_or(self.__pixeldata.mask,
                                           mask_lower.mask)
        self.__pixeldata.mask = ma.mask_or(self.__pixeldata.mask,
                                           mask_greater.mask)

    @property
    def background(self):
        """Image background subtracted property of SingleImage.
        The background is estimated using sep.

        Returns
        -------
        numpy.array 2D
            a background estimation image is returned
        """
        print("Background level = {}, rms = {}".format(self.__bkg.globalback,
                                                       self.__bkg.globalrms))
        return self.__bkg.back()

    @property
    def _bkg(self):
        return self.__bkg

    @_bkg.setter
    def _bkg(self, maskthresh):
        if self.mask.any():
            if maskthresh is not None:
                back = sep.Background(self.pixeldata.data, mask=self.mask)  # ,
                # maskthresh=maskthresh)
                self.__bkg = back
            else:
                back = sep.Background(self.pixeldata.data,
                                      mask=self.mask)
                self.__bkg = back
        else:
            # print(self.pixeldata.data.shape)
            back = sep.Background(self.pixeldata.data)
            self.__bkg = back

    @property
    def bkg_sub_img(self):
        return self.pixeldata - self.__bkg

    @property
    def stamp_shape(self):
        return self.__stamp_shape

    @stamp_shape.setter
    def stamp_shape(self, shape):
        if not hasattr(self, '__stamp_shape'):
            if shape is None:
                percent = np.percentile(self.best_sources['npix'], q=65)
                p_sizes = 3.*np.sqrt(percent)

                if p_sizes > 5:
                    dx = int(p_sizes)
                    if dx % 2 != 1:
                        dx += 1
                    shape = (dx, dx)
                else:
                    shape = (5, 5)
                print(('stamps will be {} x {}'.format(*shape)))
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
                                   thresh=8*self.__bkg.globalrms,
                                   mask=self.mask, minarea=9)
            except Exception:
                try:
                    sep.set_extract_pixstack(3600000)
                    srcs = sep.extract(self.bkg_sub_img.data,
                                       thresh=35*self.__bkg.globalrms,
                                       mask=self.mask, minarea=9)
                except Exception:
                    raise
            if len(srcs) < self.min_sources:
                print("""found {} sources, looking for at least {}.
                       Trying again""".format(len(srcs), self.min_sources))
                old_srcs = srcs
                try:
                    srcs = sep.extract(self.bkg_sub_img.data,
                                       thresh=3*self.__bkg.globalrms,
                                       mask=self.mask, minarea=5)
                except Exception:
                    sep.set_extract_pixstack(900000)
                    srcs = sep.extract(self.bkg_sub_img.data,
                                       thresh=3*self.__bkg.globalrms,
                                       mask=self.mask, minarea=9)
                if len(old_srcs) > len(srcs):
                    srcs = old_srcs

            if len(srcs) == 0:
                raise ValueError('Few sources detected on image')
            elif len(srcs) == 1:
                m, med, st = sigma_clipped_stats(self.bkg_sub_img.data.flatten())  # noqa
                if st <= 0.1:
                    raise ValueError('Image is constant, possible saturated')
                if m >= 65535.:
                    raise ValueError('Image is saturated')
                else:
                    raise ValueError('only one sources. Possible saturation')

            p_sizes = np.percentile(srcs['npix'], q=[20, 50, 80])

            best_big = srcs['npix'] >= p_sizes[0]
            best_small = srcs['npix'] <= p_sizes[2]
            best_flag = srcs['flag'] == 0

            fluxes_quartiles = np.percentile(srcs['flux'], q=[15, 85])
            low_flux = srcs['flux'] >= fluxes_quartiles[0]
            hig_flux = srcs['flux'] <= fluxes_quartiles[1]

            best_srcs = srcs[best_big & best_flag & best_small &
                             hig_flux & low_flux]

            if len(best_srcs) == 0:
                print('Best sources are too few- Using everything we have!')
                best_srcs = srcs
                # raise ValueError('Few sources detected on image')

            if len(best_srcs) > 1800:
                jj = np.random.choice(len(best_srcs), 1800, replace=False)
                best_srcs = best_srcs[jj]

            print(('Sources found = {}'.format(len(best_srcs))))
            self._best_sources = best_srcs

        return self._best_sources

    def update_sources(self, new_sources=None):
        del(self._best_sources)
        if new_sources is None:
            foo = self.best_sources  # noqa
            foo = self.stamp_shape  # noqa
            foo = self.n_sources  # noqa
        else:
            self._best_sources = new_sources
        return

    @property
    def stamps_pos(self):
        _cond = (hasattr(self, '_shape') and
                 self._shape != self.stamp_shape and
                 self._shape is not None)

        if not hasattr(self, '_stamps_pos') or _cond:
            if _cond:
                self.stamp_shape = self._shape
            self.db = npdb.NumPyDB_cPickle(self.dbname, mode='store')
            pos = []
            jj = 0
            to_del = []
            sx, sy = self.stamp_shape

            def check_margin(stamp, rms=self._bkg.globalrms):
                check_shape = False

                l_marg = stamp[0, :]
                r_marg = stamp[-1, :]
                t_marg = stamp[:, -1]
                b_marg = stamp[:, 0]

                margs = np.hstack([b_marg, t_marg, r_marg, l_marg])
                margs_m = np.mean(margs)
                margs_s = np.std(margs)

                if margs_m+margs_s > 3.5*rms or margs_m-margs_s < 3.5*rms:
                    check_shape = True

                return check_shape

            n_cand_srcs = len(self.best_sources)
            if n_cand_srcs <= 20:
                self.strict_star_pick = False
            for row in self.best_sources:
                position = (row['y'], row['x'])
                sub_array_data = extract_array(self.interped,
                                               self.stamp_shape, position,
                                               mode='partial',
                                               fill_value=self._bkg.globalrms)

                if np.any(np.isnan(sub_array_data)):
                    to_del.append(jj)
                    jj += 1
                    continue

                # there should be some checkings on the used stamps
                # check the margins so they are large enough
                check_shape = True
                new_shape = self.stamp_shape
                while check_shape:
                    check_shape = check_margin(sub_array_data)
                    new_shape = (new_shape[0]+2, new_shape[1]+2)
                    sub_array_data = extract_array(self.interped,
                                                   new_shape, position,
                                                   mode='partial',
                                                   fill_value=self._bkg.globalrms)  # noqa
                    # print check_shape, new_shape
                    if new_shape[0]-self.stamp_shape[0] >= 6:
                        check_shape = False

                new_shape = sub_array_data.shape
                # Normalize to unit sum
                sub_array_data += np.abs(np.min(sub_array_data))
                sub_array_data = sub_array_data/np.sum(sub_array_data)

                pad_dim = ((self.stamp_shape[0]-new_shape[0]+6)/2,
                           (self.stamp_shape[1]-new_shape[1]+6)/2)

                if not pad_dim == (0, 0):
                    sub_array_data = np.pad(sub_array_data, [pad_dim, pad_dim],
                                            'linear_ramp', end_values=0)

                if self._strict_star_pick:
                    #  Checking if the peak is off center
                    xcm, ycm = np.where(sub_array_data ==
                                        np.max(sub_array_data))
                    xcm = np.array([xcm[0], ycm[0]])

                    delta = xcm - np.asarray(new_shape)/2.
                    if np.sqrt(np.sum(delta**2)) > new_shape[0]/5.:
                        to_del.append(jj)
                        jj += 1
                        continue

                    #  Checking if it has outliers
                    sd = np.std(sub_array_data)
                    if sd > 0.15:
                        to_del.append(jj)
                        jj += 1
                        continue

                    if np.any(sub_array_data.flatten() > 0.5):
                        to_del.append(jj)
                        jj += 1
                        continue

                    outl_cat = utils.find_S_local_maxima(sub_array_data,
                                                         threshold=3.5,
                                                         neighborhood_size=5)
                    if len(outl_cat) > 1:
                        to_del.append(jj)
                        jj += 1
                        continue

                    # thrs = [outl[-1] for outl in outl_cat]
                    if len(outl_cat) is not 0:
                        ymax, xmax, thmax = outl_cat[0]
                        # np.where(thrs==np.max(thrs))[0][0]]
                        xcm = np.array([xmax, ymax])
                        delta = xcm - np.asarray(new_shape)/2.
                        if np.sqrt(np.sum(delta**2)) > new_shape[0]/5.:
                            to_del.append(jj)
                            jj += 1
                            continue

                # if everything was fine we store
                pos.append(position)
                self.db.dump(sub_array_data, len(pos)-1)
                jj += 1

            if n_cand_srcs - len(to_del) >= 15:
                self._best_sources = np.delete(self._best_sources,
                                               to_del, axis=0)

            self.stamp_shape = (self.stamp_shape[0] + 6,
                                self.stamp_shape[1] + 6)
            self._stamps_pos = np.array(pos)
            self._n_sources = len(pos)
        return self._stamps_pos

    @property
    def n_sources(self):
        try:
            return self._n_sources
        except AttributeError:
            s = self.stamps_pos  # noqa
            return self._n_sources

    @property
    def cov_matrix(self):
        """Determines the covariance matrix of the psf measured directly from
        the stamps of the detected stars in the image.

        """
        if not hasattr(self, '_covMat'):
            covMat = np.zeros(shape=(self.n_sources, self.n_sources))

            if self.n_sources < 120:
                m = np.zeros((self.stamp_shape[0] * self.stamp_shape[1],
                              self.n_sources))
                for i in range(self.n_sources):
                    psfi_render = self.db.load(i)[0]
                    m[:, i] = psfi_render.flatten()
                covMat = np.cov(m, rowvar=False)
                self._m = m
            else:
                for i in range(self.n_sources):
                    psfi_render = self.db.load(i)[0].flatten()
                    for j in range(self.n_sources):
                        if i <= j:
                            psfj_render = self.db.load(j)[0]

                            inner = np.vdot(psfi_render,  # .flatten(),
                                            psfj_render.flatten())
                            if inner is np.nan:
                                import ipdb
                                ipdb.set_trace()

                            covMat[i, j] = inner
                            covMat[j, i] = inner
            self._covMat = covMat
        return self._covMat

    @property
    def eigenv(self):
        if not hasattr(self, '_eigenv'):
            try:
                self._eigenv = np.linalg.eigh(self.cov_matrix)
            except np.linalg.LinAlgError:
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
        inf_loss_update = (inf_loss is not None) and \
                          (self.inf_loss != inf_loss)

        if not hasattr(self, '_kl_basis') or inf_loss_update:
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
                else:
                    break

            #  Build psf basis
            n_basis = abs(cut)
            xs = vech[:, -cut:]
            psf_basis = []
            if hasattr(self, '_m'):
                for i in range(n_basis):
                    eig = xs[:, i]
                    base = np.matmul(self._m, eig).reshape(self.stamp_shape)
                    base = base/np.sum(base)
                    base = base - np.abs(np.min(base))
                    base = base/np.sum(base)
                    psf_basis.append(base)
            else:
                for i in range(n_basis):
                    base = np.zeros(self.stamp_shape)
                    for j in range(self.n_sources):
                        pj = self.db.load(j)[0]
                        base += xs[j, i] * pj
                    base = base/np.sum(base)
                    base = base - np.abs(np.min(base))
                    base = base/np.sum(base)
                    psf_basis.append(base)
                    del(base)
            psf_basis.reverse()
            if self._smooth_autopsf:
                from skimage import filters
                new_psfs = []
                for a_psf in psf_basis:
                    new_psfs.append(filters.gaussian(a_psf, sigma=1.5,
                                                     preserve_range=True))
                    new_psfs[-1] = new_psfs[-1]/np.sum(new_psfs[-1])
                psf_basis = new_psfs
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

    def _setup_kl_a_fields(self, inf_loss=None, updating=False):
        """Calculate the coefficients of the expansion in basis of KLoeve.

        """
        inf_loss_update = (inf_loss is not None) and \
                          (self.inf_loss != inf_loss)

        if not hasattr(self, '_a_fields') or inf_loss_update or updating:
            if inf_loss is not None:
                self._setup_kl_basis(inf_loss)
                self.inf_loss = inf_loss
            # get the psf basis
            psf_basis = self.kl_basis
            n_fields = len(psf_basis)

            # if only one psf then the afields is  [None]
            if n_fields == 1:
                self._a_fields = [None]
                return self._a_fields

            # get the sources for observations
            best_srcs = self.best_sources  # noqa
            positions = self.stamps_pos
            x = positions[:, 0]
            y = positions[:, 1]

            # Each element in patches brings information about the real PSF
            # evaluated -or measured-, giving an interpolation point for a
            a_fields = []
            measures = np.zeros((n_fields, self.n_sources))
            for k in range(self.n_sources):
                Pval = self.db.load(k)[0].flatten()
                Pval = Pval/np.sum(Pval)
                for i in range(n_fields):
                    p_i = psf_basis[i].flatten()  # starting from bottom
                    p_i_sq = np.sqrt(np.sum(np.dot(p_i, p_i)))

                    Pval_sq = np.sqrt(np.sum(np.dot(Pval, Pval)))
                    m = np.dot(Pval, p_i)
                    m = m/(Pval_sq*p_i_sq)
                    measures[i, k] = m
                # else:
                    # measures[i, k] = None

            for i in range(n_fields):
                z = measures[i, :]
                a_field_model = models.Polynomial2D(degree=3)
                fitter = fitting.LinearLSQFitter()
                fit = fitter(a_field_model, x, y, z)

                a_fields.append(fit)
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

        The psf_basis elements are numpy arrays of the given fitshape
        (or shape) size.

        The a_fields are astropy.fitting fitted model functions, which need
        arguments to return numpy array fields (for example a_fields[0](x, y)).
        These can be generated using
        x, y = np.mgrid[:self.imagedata.shape[0],
                        :self.imagedata.shape[1]]

        """
        if shape is not None:
            self._shape = shape

        updating = (inf_loss is not None) and (self.inf_loss != inf_loss)

        self._setup_kl_basis(inf_loss)
        self._setup_kl_a_fields(inf_loss, updating=updating)

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
                self._normal_image = convolve_fft(a, psf_basis[0])

            else:
                x, y = self.get_afield_domain()
                conv = np.zeros_like(self.pixeldata.data)
                for a, psf_i in zip(a_fields, psf_basis):
                    conv += convolve_scp(a(x, y), psf_i)
                self._normal_image = conv
        return self._normal_image

    @property
    def min_sources(self):
        if not hasattr(self, '_min_sources'):
            return 20
        else:
            return self._min_sources

    @min_sources.setter
    def min_sources(self, n):
        self._min_sources = n

    @property
    def maskthresh(self):
        if not hasattr(self, '_maskthresh'):
            return 16
        else:
            return self._maskthresh

    @maskthresh.setter
    def maskthresh(self, thresh):
        if thresh is not None:
            self._maskthresh = thresh
        else:
            self._maskthresh = 16

    @property
    def zp(self):
        if not hasattr(self, '_zp'):
            return 1.0
        else:
            return self._zp

    @zp.setter
    def zp(self, val):
        self._zp = val

    @property
    def var(self):
        return self._bkg.globalrms

    @property
    def s_hat_comp(self):
        if not hasattr(self, '_s_hat_comp') or \
                self._s_hat_inf_loss != self.inf_loss:

            a_fields, psf_basis = self.get_variable_psf()

            var = self._bkg.globalrms
            nrm = self.normal_image
            dx, dy = center_of_mass(psf_basis[0])

            if a_fields[0] is None:
                s_hat = self.interped_hat * \
                          _fftwn(psf_basis[0],
                                 s=self.pixeldata.shape,
                                 norm='ortho').conj()

                s_hat = fourier_shift(s_hat, (+dx, +dy))
            else:
                s_hat = np.zeros_like(self.pixeldata.data, dtype=np.complex128)
                x, y = self.get_afield_domain()
                im_shape = self.pixeldata.shape
                for a, psf in zip(a_fields, psf_basis):
                    conv = _fftwn(self.interped * a(x, y)/nrm, norm='ortho') *\
                           _fftwn(psf, s=im_shape, norm='ortho').conj()
                    conv = fourier_shift(conv, (+dx, +dy))
                    np.add(conv, s_hat, out=s_hat)

            self._s_hat_comp = (self.zp/(var**2)) * s_hat
            self._s_hat_inf_loss = self.inf_loss

        return self._s_hat_comp

    @property
    def s_component(self):
        """Calculates the matched filter S (from propercoadd) component
        from the image. Uses the measured psf, and is psf space
        variant capable.

        """
        if not hasattr(self, '_s_component'):
            self._s_component = _ifftwn(self.s_hat_comp, norm='ortho').real
            self._s_inf_loss = self.inf_loss

        if self._s_inf_loss != self.inf_loss:
            self._s_component = _ifftwn(self.s_hat_comp, norm='ortho').real
        return self._s_component

    @property
    def interped(self):
        if not hasattr(self, '_interped'):
            kernel = Box2DKernel(5)  # Gaussian2DKernel(stddev=2.5) #

            crmask, _ = detect_cosmics(indat=np.ascontiguousarray(
                                             self.bkg_sub_img.filled(-9999)),
                                       inmask=self.bkg_sub_img.mask,
                                       sigclip=6.,
                                       cleantype='medmask')
            self.bkg_sub_img.mask = np.ma.mask_or(self.bkg_sub_img.mask,
                                                  crmask)
            self.bkg_sub_img.mask = np.ma.mask_or(self.bkg_sub_img.mask,
                                                  np.isnan(self.bkg_sub_img))

            print(('Masked pixels: ', np.sum(self.bkg_sub_img.mask)))
            img = self.bkg_sub_img.filled(np.nan)
            img_interp = interpolate_replace_nans(img, kernel, convolve=conv)

            while np.any(np.isnan(img_interp)):
                img_interp = interpolate_replace_nans(img_interp, kernel,
                                                      convolve=conv)
            # clipped = sigma_clip(self.bkg_sub_img,
                # iters=5, sigma_upper=40).filled(np.nan)
            # img_interp = interpolate_replace_nans(img_interp, kernel)
            self._interped = img_interp

        return self._interped

    @property
    def interped_hat(self):
        if not hasattr(self, '_interped_hat'):
            self._interped_hat = _fftwn(self.interped, norm='ortho')
        return self._interped_hat

    def psf_hat_sqnorm(self):
        psf_basis = self.kl_basis
        a_fields = self.kl_afields
        if a_fields is None:
            p_hat = _fftwn(psf_basis[0], s=self.pixeldata.shape, norm='ortho')
            p_hat_sqnorm = p_hat * p_hat.conj()
        else:
            p_hat_sqnorm = np.zeros(self.pixeldata.shape, dtype=np.complex128)
            for a_psf in psf_basis:
                psf_hat = _fftwn(a_psf, s=self.pixeldata.shape, norm='ortho')
                np.add(psf_hat*psf_hat.conj(), p_hat_sqnorm, out=p_hat_sqnorm)

        return p_hat_sqnorm

    def p_sqnorm(self):
        phat = self.psf_hat_sqnorm()
        p = _ifftwn(phat, norm='ortho')
        print(np.sum(p))
        return _ifftwn(fourier_shift(phat, (self.stamp_shape[0]/2,
                                            self.stamp_shape[1]/2)),
                       norm='ortho')


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
    try:
        return sorted(out, reverse=True)
    except TypeError:
        return out
    except ValueError:
        return out
