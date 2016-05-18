"""
sim_coadd_proper translation to Python.

Written by Bruno SANCHEZ

PhD of Astromoy - UNC"""


import numpy as np
import scipy.fftpack as fft
from scipy.stats import stats
from astropy.io import fits
# at this point we assume several images.
# An image is an object with the pixel data and some methods for computing,
#  background and, source detection.
# Psf and a flux scale, are two properties of the image, that can be .

#  M_j = (F_j T) * Psf_j + e_j

# and we have an idea of Var(e_j) = sigma_j (assuming...)

class SingleImage(object):
    def __init__(self, img, imagefile=False, sim=False, meta={}):
        self._attached_to = repr(img)

        if imagefile:
            self.imagedata = fits.getdata(img)
        else:
            self.imagedata = img

        if sim:
            self.meta = meta
        else:
            imgstats = ImageStats(self.imagedata, 'numpy_array')
            imgstats.calc_stats()
            imgstats.summary()
            self.meta = imgstats.full_description

    def __repr__(self):
        print 'SingleImage instance for {}'.format(self._attached_to)



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

        if dataformat not in ('CCDData', 'fits_file', 'numpy_array', 'hdu',\
                                'ndarray', 'str', 'HDUList'):
            raise InputError('Dataformat not recognized, try one of these \
            \n CCDData, fits_file, numpy_array, hdu')

        if dataformat == 'CCDData':
            self.pixmatrix = image_obj.data
            assert isinstance(pixmatrix, np.array)
        elif dataformat == 'fits_file' or dataformat =='str':
            self.pixmatrix = fits.open(image_obj)[0].data
        elif dataformat == 'numpy_array' or dataformat == 'ndarray':
            self.pixmatrix = image_obj
        elif dataformat == 'HDUList':
            self.pixmatrix = image_obj[0].data
        else:
            self.pixmatrix = image_obj[0].data

    def __repr__(self):
        print 'ImageStats instance for {}'.format(self._attached_to)

    def pix_sd(self):
        self.full_description['std'] = self.pixmatrix.std()
        return self.pixmatrix.std()

    def pix_median(self):
        m = np.median(self.pixmatrix)
        self.full_description['median'] = m
        return m

    def count_hist(self):
        h = stats.histogram(self.pixmatrix.flatten(), numbins = 30)
        self.full_description['histogram'] = h
        return h

    def pix_mean(self):
        m = np.mean(self.pixmatrix)
        self.full_description['mean'] = m
        return m

    def to1d(self):
        #self._oneDdata = np.reshape(self.pixmatrix, self.pixmatrix.shape[0]*self.pixmatrix.shape[1])
        self._oneDdata = self.pixmatrix.flatten()
        return

    def calc_stats(self):
        self.sd = self.pix_sd()
        self.median = self.pix_median()
        self.hist = self.count_hist()
        self.mean = self.pix_mean()
        self.to1d()
        return

    def summary(self):
        self.to1d()
        self.summ = stats.describe(self._oneDdata)
        #print self.summ
        return


def match_filter(image, objfilter):
    """
    Function to apply matched filtering to an image
    """
    return None


def psf_extract(image, xy):
    """Function to extract the psf of an image.

    xy should be a list of tuples with positions of the stars
    """
    return None
