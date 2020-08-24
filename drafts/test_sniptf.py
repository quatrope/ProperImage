#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  test_sniptf.py
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
from properimage import propercoadd as pc
from astropy.io import fits
from properimage import utils
from properimage import plot


def main():
    imgsdir = '/home/bruno/Documentos/Data/SNiPTF/imgs'
    imgsdir = '/home/bruno/Data/SNiPTF/imgs'

    imgsdir = '/home/bruno/Data/LIGO_O2/Jan04/20171119/PGC073926/calibrated'
    dest_dir = './test/test_images/test_sniptf'
    imgs = glob.glob(imgsdir+'/aligned_PGC073926-*.fits')
    mask = glob.glob(imgsdir+'/*mask*.fits')

    imgs.sort()
    mask.sort()

    #~ print imgs
    #~ print mask

    #~ for animg in imgs:
        #~ img = fits.open(animg, 'update')
        #~ img[0].data = img[0].data[:495, :495]
        #~ img[0].header['NAXIS1'] = 495
        #~ img[0].header['NAXIS2'] = 495

        #~ img.flush()
        #~ img.close()

    #images = [s.SingleImage(animg, mask=amask) for animg, amask in zip(imgs, mask)]

    images = [s.SingleImage(animg) for animg in imgs]

    for j, an_img in enumerate(images):
        an_img.inf_loss = 0.18
        plot.plot_psfbasis(an_img.kl_basis,
                           path=os.path.join(dest_dir, 'psf_basis_{}.png'.format(j)),
                           nbook=False)
        x, y = an_img.get_afield_domain()
        plot.plot_afields(an_img.kl_afields, x, y,
                          path=os.path.join(dest_dir,'afields_{}.png'.format(j)),
                          nbook=False, size=4)
        fits.writeto(os.path.join(dest_dir,'mask_{}.fits'.format(j)),
                     an_img.mask.astype('uint8'), overwrite=True)
        fits.writeto(os.path.join(dest_dir,'S_comp_{}.fits'.format(j)),
                     an_img.s_component, overwrite=True)
        fits.writeto(os.path.join(dest_dir,'interped_{}.fits'.format(j)),
                     an_img.interped, overwrite=True)
        plot.plt.imshow(an_img.psf_hat_sqnorm().real)
        plot.plt.colorbar()
        plot.plt.savefig(os.path.join(dest_dir,'psf_hat_sqnorm_{}.png'.format(j)))
        plot.plt.close()

    R, P_r = pc.stack_R(images, align=False, n_procs=4, inf_loss=0.25)

    fits.writeto(os.path.join(dest_dir,'R.fits'), R.real, overwrite=True)
    fits.writeto(os.path.join(dest_dir,'P.fits'), P_r.real, overwrite=True)

    for an_img in images:
        an_img._clean()

    return

if __name__ == '__main__':
    import sys
    sys.exit(main())
