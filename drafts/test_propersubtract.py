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
import numpy as np

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from properimage import single_image as s
from properimage import propersubtract as ps

from properimage import utils
from properimage import plot



def main(args):
    imgsdir = '/home/bruno/Data/LIGO_O2/Jan04/newstacks/PGC073926'
    dest_dir = './test/test_images/test_propersubtract'
    imgs = glob.glob(imgsdir+'/pgc073926_17*.fits')
    #mask = glob.glob(imgsdir+'/*mask*.fits')

    imgs.sort()
    #mask.sort()

    #images = [s.SingleImage(animg, mask=amask) for animg, amask in zip(imgs, mask)]
    images = []
    for animg in imgs:
        print(animg)
        try:
            o = s.SingleImage(animg, borders=True, crop=((150,150), (150, 150)))
            images.append(o)
        except:
            pass
        #~ hdu = fits.open(animg)
        #~ data = hdu[0].data

    #~ images = [s.SingleImage(animg, borders=True, crop=((250,250), (250, 250)))
              #~ for animg in imgs]
    fits.writeto(os.path.join(dest_dir,'InterpedRef.fits'),
                     images[0].interped, overwrite=True)

    for i, animg in enumerate(images[1:]):
        #~ ## Erasing stars
        srcs = animg.best_sources
        f60 = np.percentile(srcs['cflux'], q=60)
        jj = np.random.choice(len(srcs), len(srcs)/3, replace=False)
        mea, med, std = sigma_clipped_stats(animg.pixeldata)
        sx, sy = animg.stamp_shape
        st = animg._bkg.globalrms
        for aj in jj:
            star = srcs[aj]
            x = star['x']
            y = star['y']
            if x<sx or animg.pixeldata.shape[0]-x<sx:
                continue
            if y<sy or animg.pixeldata.shape[1]-y<sy:
                continue
            if star['cflux']<f60:
                continue
            print(x,y)
            noise = np.random.normal(loc=med, scale=st, size=animg.stamp_shape)
            animg.pixeldata.data[np.int(x-sx/2.):np.int(x+sx/2.),
                                 np.int(y-sy/2.):np.int(y+sy/2.)] = noise

        ##  Adding stars
        #~ foo = animg.cov_matrix
        #~ srcs = animg.best_sources

        #~ jj = np.random.choice(len(srcs), 12, replace=False)
        #~ for aj in jj:
            #~ star = animg.db.load(aj)[0]
            #~ x, y = np.random.choice(np.min(animg.pixeldata.shape)-np.max(star.shape),
                                    #~ 2, replace=True)
            #~ print star.shape
            #~ animg.pixeldata.data[x:x+star.shape[0], y:y+star.shape[1]] = star
            #~ xc, yc = x+star.shape[0]/2., y+star.shape[1]/2.
            #~ print xc, yc
        fits.writeto(os.path.join(dest_dir,'InterpedNew_{}.fits'.format(i)),
                     animg.interped, overwrite=True)
        try:
            D, P, S_corr, mask = ps.diff(animg, images[0], align=True,
                                   iterative=False, shift=False, beta=True)
            mea, med, std = sigma_clipped_stats(D.real)
            D = np.ma.MaskedArray(D.real, mask).filled(mea)
            fits.writeto(os.path.join(dest_dir,'Diff_{}.fits'.format(i)), D, overwrite=True)
            fits.writeto(os.path.join(dest_dir,'P_{}.fits'.format(i)), P.real, overwrite=True)
            fits.writeto(os.path.join(dest_dir,'Scorr_{}.fits'.format(i)), S_corr, overwrite=True)
        except:
            print('subtraction failed')
            print('ref: ', images[0])
            print('new: ', animg)

    for an_img in images:
        an_img._clean()

    return

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
