#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  imagelist_coadd.py
#
#  Copyright 2017 Bruno S <bruno@oac.unc.edu.ar>
#
# This file is part of ProperImage (https://github.com/toros-astro/ProperImage)
# License: BSD-3-Clause
# Full Text: https://github.com/toros-astro/ProperImage/blob/master/LICENSE.txt
#
import os
from properimage import propercoadd as pc
from properimage import utils as ut



def main(imagelist, rfile=None, sfile=None):

    refpath = imagelist[0]

    newimagelist = [refpath]
    for img in imagelist[1::]:
        try:
            newimagelist.append(ut.align_for_diff(refpath, img))
        except:
            pass

    #with pc.ImageEnsemble(newimagelist, pow_th=0.01) as ensemble:
        #~ R, S = ensemble.calculate_R(n_procs=4, return_S=True)

    #~ ut.encapsule_R(R, path=rfile)
    #~ ut.encapsule_S(S, path=sfile)

if __name__ == '__main__':
    import sys
#    print sys.argv[1::]
    sys.exit(main(sys.argv[1::]))
