"""
sim_coadd_proper translation to Python.

Written by Bruno SANCHEZ

PhD of Astromoy - UNC"""


import numpy as np
import scipy.fftpack as fft

# at this point we assume ten images.
# An image is a background substracted image,
#  with a psf and a flux scale, plus some random gaussian noise.

#  M_j = (F_j T) * Psf_j + e_j

# and we have an idea of Var(e_j) = sigma_j (assuming...)

def match_filter(image, objfilter):
    """
    Function to apply matched filtering to an image
    """
    return None


def psf_extract(image, xy):
    """Function to extract the psf of an image.

    xy should be a list of tuples with
    """

    return None

