"""
sim_coadd_proper translation to Python.

Written by Bruno SANCHEZ

PhD of Astromoy - UNC"""


import numpy as np
import scipy.fftpack as fft
from imageSimulation import big_code

# at this point we assume ten images.
# An image is a background substracted image,
#  with a psf and a flux scale, plus some random gaussian noise.

#  M_j = (F_j T) * Psf_j + e_j

# and we have an idea of Var(e_j) = sigma_j (assuming...)


