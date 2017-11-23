#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  test_propersubtract.py
#
#  Copyright 2016 Bruno S <bruno@oac.unc.edu.ar>
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
import os
import glob

import matplotlib.pyplot as plt
from astropy.io import fits

from properimage import single_image as s
from properimage import propersubtract as ps

from properimage import utils
from properimage import plot



def main(args):
    imgsdir = '/home/bruno/Data/LIGO_O2/Jan04/newstacks/NGC1341'
    dest_dir = './test/test_images/test_propersubtract'
    imgs = glob.glob(imgsdir+'/ngc1341_1701*.fits')
    #mask = glob.glob(imgsdir+'/*mask*.fits')

    imgs.sort()
    #mask.sort()

    #images = [s.SingleImage(animg, mask=amask) for animg, amask in zip(imgs, mask)]

    images = [s.SingleImage(animg) for animg in imgs]
    for i, animg in enumerate(images[1:]):
        D, P, S_corr = ps.diff(images[0], animg, align=True,
                               iterative=False, shift=False, beta=False)

        fits.writeto(os.path.join(dest_dir,'Diff_{}.fits'.format(i)), D.real, overwrite=True)
        fits.writeto(os.path.join(dest_dir,'P_{}.fits'.format(i)), P.real, overwrite=True)
        fits.writeto(os.path.join(dest_dir,'Scorr_{}.fits'.format(i)), S_corr, overwrite=True)

    for an_img in images:
        an_img._clean()

    return
