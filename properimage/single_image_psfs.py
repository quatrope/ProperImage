#  single_image_psfs.py
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

from six.moves import range

import numpy as np

from astropy.stats import sigma_clipped_stats

from astropy.modeling import fitting
from astropy.modeling import models

from . import single_image as si

try:
    import pyfftw

    _fftwn = pyfftw.interfaces.numpy_fft.fft2  # noqa
    _ifftwn = pyfftw.interfaces.numpy_fft.ifft2  # noqa
except ImportError:
    _fftwn = np.fft.fft2
    _ifftwn = np.fft.ifft2


class SingleImageGaussPSF(si.SingleImage):
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
        super(SingleImageGaussPSF, self).__init__(*arg, **kwargs)

    def get_variable_psf(self, inf_loss=None, shape=None):
        """Method to obtain a unique Gaussian psf, non variable.
        Returns
        -------
        An astropy model Gaussian2D instance, with the median parameters
        for the fit of every star.


        """

        def fit_gaussian2d(b, fitter=None):
            if fitter is None:
                fitter = fitting.LevMarLSQFitter()

            y2, x2 = np.mgrid[: b.shape[0], : b.shape[1]]
            ampl = b.max() - b.min()
            p = models.Gaussian2D(
                x_mean=b.shape[1] / 2.0,
                y_mean=b.shape[0] / 2.0,
                x_stddev=1.0,
                y_stddev=1.0,
                theta=np.pi / 4.0,
                amplitude=ampl,
            )

            p += models.Const2D(amplitude=b.min())
            out = fitter(p, x2, y2, b, maxiter=1000)
            return out

        p_xw = []
        p_yw = []
        p_th = []
        p_am = []
        fitter = fitting.LevMarLSQFitter()
        for i in range(self.n_sources):
            psfi_render = self.db.load(i)[0]
            p = fit_gaussian2d(psfi_render, fitter=fitter)
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

        # import ipdb; ipdb.set_trace()
        mean_model = models.Gaussian2D(
            x_mean=0,
            y_mean=0,
            x_stddev=mean_xw,
            y_stddev=mean_yw,
            theta=mean_th,
            amplitude=1.0,
        )

        return [[None], [mean_model.render(), mean_model]]
