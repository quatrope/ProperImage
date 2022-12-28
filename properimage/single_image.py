#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  single_image.py
#
#  Copyright 2020 QuatroPe
#
# This file is part of ProperImage (https://github.com/quatrope/ProperImage)
# License: BSD-3-Clause
# Full Text: https://github.com/quatrope/ProperImage/blob/master/LICENSE.txt
#

"""single_image module of ProperImage package for astronomical image analysis.

This module contains main class SingleImage, and derived classes.
"""

import logging
import os
import pathlib
import tempfile

from astropy.convolution import Box2DKernel
from astropy.convolution import convolve_fft
from astropy.convolution import interpolate_replace_nans
from astropy.io import fits
from astropy.modeling import fitting, models
from astropy.nddata.utils import extract_array
from astropy.stats import sigma_clipped_stats

from astroscrappy import detect_cosmics

import numpy as np
from numpy import ma

from scipy.ndimage import center_of_mass
from scipy.ndimage import convolve as convolve_scp
from scipy.ndimage import fourier_shift

import sep

from six.moves import range

import tinynpydb as npdb

from . import plot, utils

try:
    import pyfftw

    _fftwn = pyfftw.interfaces.numpy_fft.fft2  # noqa
    _ifftwn = pyfftw.interfaces.numpy_fft.ifft2  # noqa
except ImportError:
    _fftwn = np.fft.fft2
    _ifftwn = np.fft.ifft2


# =============================================================================
# CONSTANTS
# =============================================================================

TEMP_DIR = tempfile.mkdtemp(suffix="_properimage")

TEMP_PATH = pathlib.Path(TEMP_DIR)

logger = logging.getLogger(__name__)

# =============================================================================
# API
# =============================================================================


def conv(*arg, **kwargs):
    """Bake scipy convolution function with custom FFTW implementation."""
    return convolve_fft(fftn=_fftwn, ifftn=_ifftwn, *arg, **kwargs)


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

    def __init__(
        self,
        img=None,
        mask=None,
        maskthresh=None,
        stamp_shape=None,
        borders=False,
        crop=((0, 0), (0, 0)),
        min_sources=None,
        strict_star_pick=False,
        smooth_psf=False,
        gain=None,
    ):
        """Create instance of SingleImage."""
        self.borders = borders  # try to find zero border padding?
        self.crop = crop  # crop edge?
        self._strict_star_pick = strict_star_pick  # pick stars VERY carefully?
        if min_sources is not None:
            self.min_sources = min_sources
        self.__img = img
        self.attached_to = img
        self.zp = 1.0
        self.header = img
        self.gain = gain
        self.maskthresh = maskthresh
        self.data = img
        self.mask = mask
        self._bkg = maskthresh
        self.stamp_shape = stamp_shape
        self.inf_loss = 0.2
        self._smooth_autopsf = smooth_psf
        self.dbname = str(TEMP_PATH / f"{id(self)}_SingleImage")

        self._plot = plot.Plot(self)

    def __enter__(self):
        """Open context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context manager."""
        self._clean()

    def __repr__(self):
        """Output representation of instance."""
        return f"SingleImage {self.data.shape[0]}, {self.data.shape[1]}"

    def _clean(self):
        logger.info("cleaning... ")
        try:
            os.remove(self.dbname + ".dat")
            os.remove(self.dbname + ".map")
        except OSError:
            logger.warning("Nothing to clean. (Or something has failed)")

    @property
    def attached_to(self):
        """Property.

        Information on the original data from which we create this instance.
        """
        return self.__attached_to

    @attached_to.setter
    def attached_to(self, img):
        """Property setter."""
        if isinstance(img, str):
            self.__attached_to = img
        else:
            self.__attached_to = img.__class__.__name__

    @property
    def data(self):
        """Get data property value."""
        return self.__data

    @data.setter
    def data(self, img):
        """Set data property."""
        if isinstance(img, str):
            self.__data = ma.asarray(fits.getdata(img)).astype("<f4")
        elif isinstance(img, np.ndarray):
            self.__data = ma.MaskedArray(img, mask=False).astype("<f4")
        elif isinstance(img, fits.HDUList):
            if img[0].is_image:
                self.__data = ma.asarray(img[0].data).astype("<f4")
        elif isinstance(img, fits.PrimaryHDU):
            if img.is_image:
                self.__data = ma.asarray(img.data).astype("<f4")
        if self.borders:
            sx, sy = self.__data.shape
            line = self.__data.data[sx // 2, :]
            pxsum = 0
            for x, px in enumerate(line):
                pxsum += px
                if pxsum > 0.0:
                    ldx = x
                    break
            for dx in range(ldx):
                if not np.sum(self.__data.data[:dx, :]) == 0:
                    ldx = dx
                    break
            pxsum = 0
            for x, px in enumerate(np.flip(line, axis=0)):
                pxsum += px
                if pxsum > 0.0:
                    rdx = sx - x
                    break
            for dx in range(x):
                if not np.sum(self.__data.data[-dx - 1 :, :]) == 0:
                    rdx = sx - dx
                    break
            col = self.__data.data[:, sy // 2]
            pxsum = 0
            for y, px in enumerate(col):
                pxsum += px
                if pxsum > 0.0:
                    ldy = y
                    break
            for dy in range(ldy):
                if not np.sum(self.__data.data[:, :dy]) == 0:
                    ldy = dy
                    break
            pxsum = 0
            for y, px in enumerate(np.flip(line, axis=0)):
                pxsum += px
                if pxsum > 0.0:
                    rdy = sy - y
                    break
            for dy in range(y):
                if not np.sum(self.__data.data[:, -dy - 1 :]) == 0:
                    rdy = sy - dy
                    break
            self.__data = self.__data[ldx:rdx, ldy:rdy]
        if not np.sum(self.crop) == 0.0:
            dx, dy = self.crop
            self.__data = self.__data[dx[0] : -dx[1], dy[0] : -dy[1]]
        self.__data = self.__data * self.__gain
        self.__data.soften_mask()

    @property
    def gain(self):
        """Get gain property value."""
        return self.__gain

    @gain.setter
    def gain(self, imggain):
        if imggain is None:
            try:
                self.__gain = self.header["GAIN"]
            except KeyError:
                self.__gain = 1.0
        else:
            self.__gain = imggain

    @property
    def header(self):
        """Get header property value."""
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
        """Get mask property value."""
        return self.__data.mask

    @mask.setter
    def mask(self, mask):
        if isinstance(mask, str):
            fitsmask = fits.getdata(mask)
            if np.median(fitsmask) == 0:
                self.__data.mask = fitsmask >= self.maskthresh
            else:
                self.__data.mask = fitsmask <= self.maskthresh
        # if the mask is a separated array
        elif isinstance(mask, np.ndarray):
            self.__data.mask = mask
        # if the mask is not given
        elif mask is None:
            # check the fits file and try to find it as an extension
            if self.attached_to == "PrimaryHDU":
                self.__data = ma.masked_invalid(self.__img.data)

            elif self.attached_to == "ndarray":
                self.__data = ma.masked_invalid(self.__img)
            elif self.attached_to == "HDUList":
                if self.header["EXTEND"]:
                    fitsmask = self.__img[1].data
                    if np.median(fitsmask) == 0:
                        self.__data.mask = fitsmask >= self.maskthresh
                    else:
                        self.__data.mask = fitsmask <= self.maskthresh
                else:
                    self.__data = ma.masked_invalid(self.__img[0].data)
            # if a path is given where we find a fits file search on extensions
            else:
                try:
                    ff = fits.open(self.attached_to)
                    if "EXTEND" in ff[0].header.keys():
                        if ff[0].header["EXTEND"]:
                            try:
                                fitsmask = ff[1].data
                                if np.median(fitsmask) == 0:
                                    self.__data.mask = (
                                        fitsmask >= self.maskthresh
                                    )
                                else:
                                    self.__data.mask = (
                                        fitsmask <= self.maskthresh
                                    )
                            except IndexError:
                                self.__data = ma.masked_invalid(
                                    self.__data.data
                                )
                except IOError:
                    self.__data = ma.masked_invalid(self.__data)
        else:
            masked = ma.masked_greater(self.__data, 65000.0)
            if not np.sum(~masked.mask) < 1000.0:
                self.__data = masked

        mask_lower = ma.masked_less(self.__data, -50.0)
        mask_greater = ma.masked_greater(self.__data, 65000.0)

        self.__data.mask = ma.mask_or(self.__data.mask, mask_lower.mask)
        self.__data.mask = ma.mask_or(self.__data.mask, mask_greater.mask)

        # this procedure makes the mask grow seven times, using 2 or more
        # neighbor pixels masked. This is useful for avoiding ripples from fft
        for i_enlarge in range(7):
            enlarged_mask = convolve_scp(
                self.__data.mask.astype(int), np.ones((3, 3))
            )
            enlarged_mask = enlarged_mask.astype(int) > 2
            self.__data.mask = ma.mask_or(self.__data.mask, enlarged_mask)

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
    def _bkg(self, maskthresh):
        if self.mask.any():
            if maskthresh is not None:
                back = sep.Background(self.data.data, mask=self.mask)
                self.__bkg = back
            else:
                back = sep.Background(self.data.data, mask=self.mask)
                self.__bkg = back
        else:
            back = sep.Background(self.data.data)
            self.__bkg = back

    @property
    def bkg_sub_img(self):
        """Get background subtracted image."""
        return self.data - self.__bkg

    @property
    def stamp_shape(self):
        """Get star stamp shapes."""
        return self.__stamp_shape

    @stamp_shape.setter
    def stamp_shape(self, shape):
        if not hasattr(self, "__stamp_shape"):
            if shape is None:
                percent = np.percentile(self.best_sources["npix"], q=65)
                p_sizes = 3.0 * np.sqrt(percent)

                if p_sizes > 5:
                    dx = int(p_sizes)
                    if dx % 2 != 1:
                        dx += 1
                    shape = (dx, dx)
                else:
                    shape = (5, 5)
        self.__stamp_shape = shape

    @property
    def best_sources(self):
        """Get best_sources property value.

        This is a dictionary of best sources detected in the image.
        Keys are:
            fitshape: tuple, the size of the stamps on each source detected
            sources: a table of sources, with the imformation from sep
            positions: an array, with the position of each source stamp
            n_sources: the total number of sources extracted
        """
        if not hasattr(self, "_best_sources"):
            try:
                srcs = sep.extract(
                    self.bkg_sub_img.data,
                    thresh=8 * self.__bkg.globalrms,
                    mask=self.mask,
                    minarea=9,
                )
            except Exception:
                try:
                    sep.set_extract_pixstack(3600000)
                    srcs = sep.extract(
                        self.bkg_sub_img.data,
                        thresh=35 * self.__bkg.globalrms,
                        mask=self.mask,
                        minarea=9,
                    )
                except Exception:
                    raise
            if len(srcs) < self.min_sources:
                old_srcs = srcs
                try:
                    srcs = sep.extract(
                        self.bkg_sub_img.data,
                        thresh=3 * self.__bkg.globalrms,
                        mask=self.mask,
                        minarea=5,
                    )
                except Exception:
                    sep.set_extract_pixstack(900000)
                    srcs = sep.extract(
                        self.bkg_sub_img.data,
                        thresh=3 * self.__bkg.globalrms,
                        mask=self.mask,
                        minarea=9,
                    )
                if len(old_srcs) > len(srcs):
                    srcs = old_srcs

            if len(srcs) == 0:
                raise ValueError("Few sources detected on image")
            elif len(srcs) == 1:
                m, med, st = sigma_clipped_stats(
                    self.bkg_sub_img.data.flatten()
                )  # noqa
                if st <= 0.1:
                    raise ValueError("Image is constant, possible saturated")
                if m >= 65535.0:
                    raise ValueError("Image is saturated")
                else:
                    raise ValueError("Only one source. Possible saturation")

            p_sizes = np.percentile(srcs["npix"], q=[20, 50, 80])

            best_big = srcs["npix"] >= p_sizes[0]
            best_small = srcs["npix"] <= p_sizes[2]
            best_flag = srcs["flag"] == 0

            fluxes_quartiles = np.percentile(srcs["flux"], q=[15, 85])
            low_flux = srcs["flux"] >= fluxes_quartiles[0]
            hig_flux = srcs["flux"] <= fluxes_quartiles[1]

            best_srcs = srcs[
                best_big & best_flag & best_small & hig_flux & low_flux
            ]

            if len(best_srcs) < 5:
                self.warning(
                    "Best sources are too few- Using everything we have!"
                )
                best_srcs = srcs

            if len(best_srcs) > 1800:
                jj = np.random.choice(len(best_srcs), 1800, replace=False)
                best_srcs = best_srcs[jj]

            self._best_sources = best_srcs

        return self._best_sources

    def update_sources(self, new_sources=None):
        """Update best source property."""
        del self._best_sources
        if new_sources is None:
            foo = self.best_sources  # noqa
            foo = self.stamp_shape  # noqa
            foo = self.n_sources  # noqa
        else:
            self._best_sources = new_sources
        return

    @property
    def stamps_pos(self):
        """Get star stamp positions propery value."""
        _cond = (
            hasattr(self, "_shape")
            and self._shape != self.stamp_shape
            and self._shape is not None
        )

        if not hasattr(self, "_stamps_pos") or _cond:
            if _cond:
                self.stamp_shape = self._shape
            self.db = npdb.NumPyDB(self.dbname, mode="store")
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

                if (
                    margs_m + margs_s > 3.5 * rms
                    or margs_m - margs_s < 3.5 * rms
                ):
                    check_shape = True

                return check_shape

            n_cand_srcs = len(self.best_sources)
            if n_cand_srcs <= 20:
                self.strict_star_pick = False

            for row in self.best_sources:
                position = (row["y"], row["x"])
                sub_array_data = extract_array(
                    self.interped,
                    self.stamp_shape,
                    position,
                    mode="partial",
                    fill_value=self._bkg.globalrms,
                )

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
                    new_shape = (new_shape[0] + 2, new_shape[1] + 2)
                    sub_array_data = extract_array(
                        self.interped,
                        new_shape,
                        position,
                        mode="partial",
                        fill_value=self._bkg.globalrms,
                    )  # noqa
                    if new_shape[0] - self.stamp_shape[0] >= 6:
                        check_shape = False

                new_shape = sub_array_data.shape
                # Normalize to unit sum
                sub_array_data += np.abs(np.min(sub_array_data))
                star_bg = np.where(
                    sub_array_data
                    < np.percentile(np.abs(sub_array_data), q=70)
                )
                sub_array_data -= np.median(sub_array_data[star_bg])

                pad_dim = (
                    (self.stamp_shape[0] - new_shape[0] + 6) / 2,
                    (self.stamp_shape[1] - new_shape[1] + 6) / 2,
                )

                if not pad_dim == (0, 0):
                    sub_array_data = np.pad(
                        sub_array_data,
                        [pad_dim, pad_dim],
                        "linear_ramp",
                        end_values=0,
                    )
                sub_array_data = sub_array_data / np.sum(sub_array_data)

                if self._strict_star_pick:
                    #  Checking if the peak is off center
                    xcm, ycm = np.where(
                        sub_array_data == np.max(sub_array_data)
                    )
                    xcm = np.array([xcm[0], ycm[0]])

                    delta = xcm - np.asarray(new_shape) / 2.0
                    if np.sqrt(np.sum(delta**2)) > new_shape[0] / 5.0:
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

                    outl_cat = utils.find_S_local_maxima(
                        sub_array_data, threshold=3.5, neighborhood_size=5
                    )
                    if len(outl_cat) > 1:
                        to_del.append(jj)
                        jj += 1
                        continue

                    if len(outl_cat) != 0:
                        ymax, xmax, thmax = outl_cat[0]
                        xcm = np.array([xmax, ymax])
                        delta = xcm - np.asarray(new_shape) / 2.0
                        if np.sqrt(np.sum(delta**2)) > new_shape[0] / 5.0:
                            to_del.append(jj)
                            jj += 1
                            continue

                # if everything was fine we store
                pos.append(position)
                self.db.dump(sub_array_data, len(pos) - 1)
                jj += 1

            if n_cand_srcs - len(to_del) >= 15:
                self._best_sources = np.delete(
                    self._best_sources, to_del, axis=0
                )
            else:
                logger.warning(
                    """Attempted to use strict star pick, but we had less
                    than 15 srcs to work on. Overriding this."""
                )
            # Adding extra border to ramp to 0
            self.stamp_shape = (
                self.stamp_shape[0] + 6,
                self.stamp_shape[1] + 6,
            )
            logger.warning(
                "updating stamp shape to ({},{})".format(
                    self.stamp_shape[0], self.stamp_shape[1]
                )
            )
            self._stamps_pos = np.array(pos)
            self._n_sources = len(pos)
            logger.info(f"We have {self._n_sources} total sources detected.")
        return self._stamps_pos

    @property
    def n_sources(self):
        """Get number of source property value."""
        try:
            return self._n_sources
        except AttributeError:
            s = self.stamps_pos  # noqa
            return self._n_sources

    @property
    def cov_matrix(self):
        """Get covariance matrix property value.

        Determines the covariance matrix of the psf measured directly from
        the stamps of the detected stars in the image.

        """
        if not hasattr(self, "_covMat"):
            covMat = np.zeros(shape=(self.n_sources, self.n_sources))

            if self.n_sources <= 250:
                m = np.zeros(
                    (self.stamp_shape[0] * self.stamp_shape[1], self.n_sources)
                )
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

                            inner = np.vdot(
                                psfi_render,  # .flatten(),
                                psfj_render.flatten(),
                            )
                            if inner is np.nan:
                                raise ValueError("PSFs inner prod is invalid")

                            covMat[i, j] = inner
                            covMat[j, i] = inner
            self._covMat = covMat
        return self._covMat

    @property
    def eigenv(self):
        """Get the eigen values of covariance matrix."""
        if not hasattr(self, "_eigenv"):
            try:
                self._eigenv = np.linalg.eigh(self.cov_matrix)
            except np.linalg.LinAlgError:
                # print('cov matx, nsrcs', self.cov_matrix, self.n_sources)
                raise ValueError(f"LinAlgError. Mat = {self.cov_matrix}")

        return self._eigenv

    @property
    def kl_basis(self):
        """Get the Karhunen-Loeve basis elements."""
        if not hasattr(self, "_kl_basis"):
            self._setup_kl_basis()
        return self._kl_basis

    def _setup_kl_basis(self, inf_loss=None):
        """Determine the KL psf_basis from stars detected in the field."""
        inf_loss_update = (inf_loss is not None) and (
            self.inf_loss != inf_loss
        )

        if not hasattr(self, "_kl_basis") or inf_loss_update:
            if inf_loss is not None:
                self.inf_loss = inf_loss

            valh, vech = self.eigenv
            power = abs(valh) / np.sum(abs(valh))
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
            if hasattr(self, "_m"):
                logger.debug(vech.shape, self._m.shape)
                self._full_bases = np.dot(self._m, vech)
                self._bases = self._full_bases[:, -cut:]
                psf_basis = [
                    self._bases[:, i].reshape(self.stamp_shape)
                    for i in range(self._bases.shape[1])
                ]
            else:
                for i in range(n_basis):
                    base = np.zeros(self.stamp_shape)
                    for j in range(self.n_sources):
                        pj = self.db.load(j)[0]
                        base += xs[j, i] * pj
                    base = base / np.sum(base)
                    base = base - np.abs(np.min(base))
                    base = base / np.sum(base)
                    psf_basis.append(base)
                    del base
            psf_basis.reverse()
            if self._smooth_autopsf:
                from skimage import filters

                new_psfs = []
                for a_psf in psf_basis:
                    new_psfs.append(
                        filters.gaussian(a_psf, sigma=1.5, preserve_range=True)
                    )
                    new_psfs[-1] = new_psfs[-1] / np.sum(new_psfs[-1])
                psf_basis = new_psfs
            if len(psf_basis) == 1:
                psf_basis[0] = psf_basis[0] / np.sum(psf_basis[0])
            self._kl_basis = psf_basis

    @property
    def kl_afields(self):
        """Get the Karhunen-Loeve coefficients."""
        if not hasattr(self, "_a_fields"):
            self._setup_kl_a_fields()
        return self._a_fields

    def get_afield_domain(self):
        """Get KL coefficient domain grid of coordinates."""
        x, y = np.mgrid[: self.data.data.shape[0], : self.data.data.shape[1]]
        return x, y

    def _setup_kl_a_fields(self, inf_loss=None, updating=False):
        """Calculate coefficients of Karhunen-Loeve expansion."""
        inf_loss_update = (inf_loss is not None) and (
            self.inf_loss != inf_loss
        )

        if not hasattr(self, "_a_fields") or inf_loss_update or updating:
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
            measures = np.flip(self.eigenv[1][:, -n_fields:].T, 0)
            for i in range(n_fields):
                z = measures[i, :]
                a_field_model = models.Polynomial2D(degree=3)
                fitter = fitting.LinearLSQFitter()
                fit = fitter(a_field_model, x, y, z)

                a_fields.append(fit)
            self._a_fields = a_fields

    def get_variable_psf(self, inf_loss=None, shape=None):
        """
        Get the variable PSF KL decomposition.

        Method to obtain the space variant PSF determination,
        according to Lauer 2002 method with Karhunen Loeve transform.

        Parameters
        ----------
        pow_th : float,
            between 0 and 1. It sets the minimum amount of
            information that a PSF-basis of the Karhunen Loeve
            transformation should have in order to be taken into account.
            A high value will return only the most significant components.
            Default is 0.9

        shape : tuple
            the size of the stamps for source extraction.
            This value affects the _best_srcs property, it should be settled in
            the SingleImage instancing step. At this stage it will override the
            value settled in the instancing step only if _best_srcs hasn't been
            called yet, which is the case if you are performing context managed
            image subtraction. (See propersubtract module)

        Returns
        -------
        [a_fields, psf_basis] : list of two lists.
            Basically it returns a sequence of psf-basis elements with its
            associated coefficient.

        Notes
        -----
        The psf_basis elements are numpy arrays of the given fitshape
        (or shape) size.

        The a_fields are astropy.fitting fitted model functions, which need
        arguments to return numpy array fields (for example a_fields[0](x, y)).
        These can be generated using `get_afield_domain`.
        """
        if shape is not None:
            self._shape = shape

        updating = (inf_loss is not None) and (self.inf_loss != inf_loss)

        self._setup_kl_basis(inf_loss)
        self._setup_kl_a_fields(inf_loss, updating=updating)

        a_fields = self.kl_afields
        psf_basis = self.kl_basis

        if a_fields[0] is None:
            psf_basis[0] = psf_basis[0] / np.sum(psf_basis[0])
        return [a_fields, psf_basis]

    @property
    def normal_image(self):
        """Calculate the PSF normalization image given in Lauer 2002.

        It uses the psf-basis elements and coefficients.
        """
        if not hasattr(self, "_normal_image"):
            a_fields, psf_basis = self.get_variable_psf()

            if a_fields[0] is None:
                a = np.ones_like(self.data.data)
                self._normal_image = convolve_fft(a, psf_basis[0])

            else:
                x, y = self.get_afield_domain()
                conv = np.zeros_like(self.data.data)
                for a, psf_i in zip(a_fields, psf_basis):
                    conv += convolve_scp(a(x, y), psf_i)
                self._normal_image = conv
        return self._normal_image

    @property
    def min_sources(self):
        """Get minimum number of sources property value."""
        if not hasattr(self, "_min_sources"):
            return 20
        else:
            return self._min_sources

    @min_sources.setter
    def min_sources(self, n):
        self._min_sources = n

    @property
    def maskthresh(self):
        """Get mask threshold value."""
        if not hasattr(self, "_maskthresh"):
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
        """Get zeropoint property value."""
        if not hasattr(self, "_zp"):
            return 1.0
        else:
            return self._zp

    @zp.setter
    def zp(self, val):
        self._zp = val

    @property
    def var(self):
        """Get globalrms variance value."""
        return self._bkg.globalrms

    @property
    def s_hat_comp(self):
        """Get the fourier transform of S image."""
        if (
            not hasattr(self, "_s_hat_comp")
            or self._s_hat_inf_loss != self.inf_loss
        ):

            a_fields, psf_basis = self.get_variable_psf()

            var = self._bkg.globalrms
            nrm = self.normal_image
            dx, dy = center_of_mass(psf_basis[0])

            if a_fields[0] is None:
                s_hat = (
                    self.interped_hat
                    * _fftwn(
                        psf_basis[0], s=self.data.shape, norm="ortho"
                    ).conj()
                )

                s_hat = fourier_shift(s_hat, (+dx, +dy))
            else:
                s_hat = np.zeros_like(self.data.data, dtype=np.complex128)
                x, y = self.get_afield_domain()
                im_shape = self.data.shape
                for a, psf in zip(a_fields, psf_basis):
                    conv = (
                        _fftwn(self.interped * a(x, y) / nrm, norm="ortho")
                        * _fftwn(psf, s=im_shape, norm="ortho").conj()
                    )
                    conv = fourier_shift(conv, (+dx, +dy))
                    np.add(conv, s_hat, out=s_hat)

            self._s_hat_comp = (self.zp / (var**2)) * s_hat
            self._s_hat_inf_loss = self.inf_loss

        return self._s_hat_comp

    @property
    def s_component(self):
        """Calculate S image component from the image.

        Uses the measured psf to calculate the matched filter S
        (from propercoadd), and is psf space variant capable.
        """
        if not hasattr(self, "_s_component"):
            self._s_component = _ifftwn(self.s_hat_comp, norm="ortho").real
            self._s_inf_loss = self.inf_loss

        if self._s_inf_loss != self.inf_loss:
            self._s_component = _ifftwn(self.s_hat_comp, norm="ortho").real
        return self._s_component

    @property
    def interped(self):
        """Get the interpolated image.

        Used to clean for cosmic rays and NaN pixels across the image.
        """
        if not hasattr(self, "_interped"):
            kernel = Box2DKernel(10)  # Gaussian2DKernel(stddev=2.5) #

            crmask, _ = detect_cosmics(
                indat=np.ascontiguousarray(self.bkg_sub_img.filled(-9999)),
                inmask=self.bkg_sub_img.mask,
                sigclip=6.0,
                cleantype="medmask",
            )
            self.bkg_sub_img.mask = np.ma.mask_or(
                self.bkg_sub_img.mask, crmask
            )
            self.bkg_sub_img.mask = np.ma.mask_or(
                self.bkg_sub_img.mask, np.isnan(self.bkg_sub_img)
            )
            img = self.bkg_sub_img.filled(np.nan)
            img_interp = interpolate_replace_nans(img, kernel, convolve=conv)

            while np.any(np.isnan(img_interp)):
                img_interp = interpolate_replace_nans(
                    img_interp, kernel, convolve=conv
                )
            self._interped = img_interp

        return self._interped

    @property
    def interped_hat(self):
        """Get the fourier transform of the interpolated image."""
        if not hasattr(self, "_interped_hat"):
            self._interped_hat = _fftwn(self.interped, norm="ortho")
        return self._interped_hat

    @property
    def plot(self):
        """Get the plot plugin object property."""
        return self._plot

    def psf_hat_sqnorm(self):
        """Get the squared norm of the PSF fourier transform."""
        psf_basis = self.kl_basis
        a_fields = self.kl_afields
        if a_fields is None:
            p_hat = _fftwn(psf_basis[0], s=self.data.shape, norm="ortho")
            p_hat_sqnorm = p_hat * p_hat.conj()
        else:
            p_hat_sqnorm = np.zeros(self.data.shape, dtype=np.complex128)
            for a_psf in psf_basis:
                psf_hat = _fftwn(a_psf, s=self.data.shape, norm="ortho")
                np.add(
                    psf_hat * psf_hat.conj(), p_hat_sqnorm, out=p_hat_sqnorm
                )

        return p_hat_sqnorm

    def p_sqnorm(self):
        """Get the PSF squared norm."""
        phat = self.psf_hat_sqnorm()
        return _ifftwn(
            fourier_shift(
                phat, (self.stamp_shape[0] / 2, self.stamp_shape[1] / 2)
            ),
            norm="ortho",
        )

    def get_psf_xy(self, x, y):
        """Get the variable PSF coefficient domain coordinate grid."""
        psf_basis = self.kl_basis
        a_fields = self.kl_afields

        if a_fields[0] is not None:
            psf_at_xy = np.zeros_like(psf_basis[0])
            delta = int((psf_at_xy.shape[0] - 1) / 2)
            xp, yp = int(np.round(x)), int(np.round(y))
            xmin = xp - delta
            xmax = xp + delta + 1
            ymin = yp - delta
            ymax = yp + delta + 1
            for apsf, afield in zip(psf_basis, a_fields):
                psf_at_xy += (
                    apsf
                    * afield(*np.mgrid[xmin:xmax, ymin:ymax])
                    / self.normal_image[xmin:xmax, ymin:ymax]
                )

            return psf_at_xy / np.sum(psf_at_xy)
        else:
            return psf_basis[0]


class SingleImageGaussPSF(SingleImage):
    """Atomic processor class for a single image.

    Contains several tools for PSF measures, and different coadding
    building structures.

    It includes the pixel matrix data, as long as some descriptions.
    For statistical values of the pixel matrix the class' methods need to be
    called.

    Inherits from properimage.single_image.SingleImage, but replacing
    every psf determination routine with a multi gaussian fit.


    Parameters
    ----------
    img : `~numpy.ndarray` or :class:`~ccdproc.CCDData`,
                `~astropy.io.fits.HDUList`  or a `str` naming the filename.
        The image object to work with

    mask: `~numpy.ndarray` or a `str` naming the filename.
        The mask image
    """

    def __init__(self, *arg, **kwargs):
        """Initialize class object."""
        super(SingleImageGaussPSF, self).__init__(*arg, **kwargs)

    def get_variable_psf(self, inf_loss=None, shape=None):
        """Obtain a unique Gaussian psf, non variable.

        Returns
        -------
        An astropy model Gaussian2D instance, with the median parameters
        for the fit of every star.
        """
        p_xw = []
        p_yw = []
        p_th = []
        p_am = []
        fitter = utils.fitting.LevMarLSQFitter()
        for i in range(self.n_sources):
            psfi_render = self.db.load(i)[0]
            p = utils.fit_gaussian2d(psfi_render, fitter=fitter)
            #  room for p checking
            gaussian = p[0]
            #  back = p[1]
            p_xw.append(gaussian.x_stddev.value)
            p_yw.append(gaussian.y_stddev.value)
            p_th.append(gaussian.theta.value)
            p_am.append(gaussian.amplitude.value)

        mean_xw, med_xw, std_xw = sigma_clipped_stats(p_xw)
        mean_yw, med_yw, std_yw = sigma_clipped_stats(p_yw)
        mean_th, med_th, std_th = sigma_clipped_stats(p_th)
        mean_am, med_am, std_am = sigma_clipped_stats(p_am)

        mean_model = utils.models.Gaussian2D(
            x_mean=0,
            y_mean=0,
            x_stddev=mean_xw,
            y_stddev=mean_yw,
            theta=mean_th,
            amplitude=1.0,
        )

        return [[None], [mean_model.render(), mean_model]]
