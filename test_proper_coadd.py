# -*- coding: utf-8 -*-
"""
Created on Fri May 13 17:06:14 2016

@author: bruno
"""

from imageSimulation import big_code

N = 512  # side

m = big_code.delta_point(N)

im = big_code.image(m, N, t_exp=1, FWHM=10, SN=100)


