#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  imagelist_coadd.py
#
#  Copyright 2017 Bruno S <bruno@oac.unc.edu.ar>
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
