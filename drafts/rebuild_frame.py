#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  rebuild_frame.py
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
import shlex
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt
import sep
from astropy.io import fits

from properimage.tests import simtools
from properimage import propercoadd as pc
from properimage import utils

# =============================================================================
#     PSF measure test by propercoadd
# =============================================================================
N = 512
test_dir = os.path.abspath('./test/test_images/rebuild_psf2')
frame = utils.sim_varpsf(400, test_dir, SN=5.)


with pc.SingleImage(frame) as sim:
    a_fields, psf_basis = sim.get_variable_psf()

utils.plot_afields(a_fields, frame.shape, os.path.join(test_dir, 'a_fields.png'))
utils.plot_psfbasis(psf_basis, os.path.join(test_dir, 'psf_basis.png'), nbook=False)
plt.imshow(np.log10(frame), interpolation='none')
#plt.plot(cat['sources']['x'], cat['sources']['y'], '.k')
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(test_dir, 'test_frame.png'))
plt.close()

cat = sep.extract(frame - sep.Background(frame),
                  thresh=4.5*sep.Background(frame).globalrms)

xy = [(int(row['y']), int(row['x'])) for row in cat]
weights = 100000. * cat['flux']/max(cat['flux'])
m = simtools.delta_point(N*2, center=False, xy=xy)#, weights=weights)
x, y = sim.get_afield_domain()  # np.mgrid[:frame.shape[0], :frame.shape[1]]

rebuild = np.zeros_like(frame)
for i in range(len(psf_basis)):
    psf = psf_basis[i]
    a = a_fields[i]
    rebuild += a(x, y) * simtools.convol_gal_psf_fft(m, psf)

rebuild += 1000.

plt.imshow(np.log10(rebuild), interpolation='none')
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(test_dir, 'frame_rebuild.png'))
plt.close()

f = fits.PrimaryHDU(frame)
f.writeto(os.path.join(test_dir, 'test_frame.fits'), clobber=True)

r = fits.PrimaryHDU(rebuild)
r.writeto(os.path.join(test_dir, 'frame_rebuild.fits'), clobber=True)


scale = np.vdot(frame.flatten(), rebuild.flatten())
scale = scale/np.vdot(rebuild.flatten(), rebuild.flatten())

diff = frame - scale*rebuild

plt.imshow(np.log10(diff), interpolation='none')
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(test_dir, 'diff.png'))
plt.close()

diff = fits.PrimaryHDU(diff)
diff.writeto(os.path.join(test_dir, 'diff.fits'), clobber=True)


