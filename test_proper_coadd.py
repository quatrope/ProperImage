# -*- coding: utf-8 -*-
"""
Created on Fri May 13 17:06:14 2016

@author: bruno
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import stats

from imsim import simtools
import propercoadd as pc


N = 128  # side

#x = [np.random.randint(low=10, high=N-10) for j in range(100)]
#y = [np.random.randint(low=10, high=N-10) for j in range(100)]
#xy = [(x[i], y[i]) for i in range(100)]

m = simtools.delta_point(N, center=True)

im = simtools.image(m, N, t_exp=1, FWHM=10, SN=3, bkg_pdf='gaussian')


from astropy.convolution import convolve, convolve_fft

psf = simtools.Psf(31, 5)

im_con = convolve(im, psf)

im_con_fft = convolve_fft(im, psf)

plt.figure()
plt.subplot(121)
plt.imshow(im_con, interpolation='none')
plt.subplot(122)
plt.imshow(im_con_fft, interpolation='None')
plt.title('conv - fft')
plt.show()


