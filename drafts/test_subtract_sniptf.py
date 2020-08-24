#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  test_subtract_sniptf.py
#
#  Copyright 2017 Bruno S <bruno@oac.unc.edu.ar>
#
# This file is part of ProperImage (https://github.com/toros-astro/ProperImage)
# License: BSD-3-Clause
# Full Text: https://github.com/toros-astro/ProperImage/blob/master/LICENSE.txt
#
import glob
import os
from properimage import single_image as s
from properimage import propersubtract as ps
from astropy.io import fits
from properimage import utils
from properimage import plot
import astroalign as aa
import numpy as np

def main(args):
    imgsdir = '/home/bruno/Documentos/Data/SNiPTF/imgs'
    imgsdir = '/home/bruno/Data/SNiPTF/imgs'

    #imgsdir = '/home/bruno/Documentos/Data/LIGO_O2/20171116/ESO202-009'
    dest_dir = './test/test_images/test_sub_sniptf'
    imgs = glob.glob(imgsdir+'/*sci*.fits')
    mask = glob.glob(imgsdir+'/*mask*.fits')

    imgs.sort()
    mask.sort()

    images = [s.SingleImage(animg, mask=amask) for animg, amask in zip(imgs, mask)]

    #images = [s.SingleImage(animg) for animg in imgs]
    for i, animg in enumerate(images[1:]):
        #~ t, _ = aa.find_transform(animg.data, images[0].data)

        #~ if abs(t.rotation)>5e-4:
            #~ k = t.__class__
            #~ t = k(np.round(t.params, decimals=5))

        #~ reg = aa.apply_transform(t, animg.data, images[0].data)
        reg = aa.register(animg.data, images[0].data)
        new = s.SingleImage(reg.data, mask=reg.mask)

        fits.writeto('/home/bruno/aligned_{}.fits'.format(i),
                     reg.data, overwrite=True)

        D, P, S_corr, mask = ps.diff(images[0], new, align=True,
                               iterative=False, shift=True, beta=True)

        D = np.ma.MaskedArray(D.real, mask).filled(np.median(D.real))
        fits.writeto(os.path.join(dest_dir,'Diff_{}.fits'.format(i)),
                     D.real, overwrite=True)
        fits.writeto(os.path.join(dest_dir,'P_{}.fits'.format(i)),
                     P.real, overwrite=True)
        fits.writeto(os.path.join(dest_dir,'Scorr_{}.fits'.format(i)),
                     S_corr, overwrite=True)
        new._clean()
    for an_img in images:
        an_img._clean()

    return

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
