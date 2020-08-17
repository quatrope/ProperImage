#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  test_matching.py
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

import numpy as np
from astropy.table import Table, Column
import properimage.utils as u
import matplotlib.pyplot as plt


# =============================================================================
# matching of sources
# =============================================================================
N = 2000
nstars = 300
x_tstars = np.random.random(size=nstars) * (N-30) + 30.
y_tstars = np.random.random(size=nstars) * (N-30) + 30.

master_x = np.concatenate([x_tstars + np.random.random(size=nstars),
                           np.random.random(size=20) * (N-30) + 30.])
master_y = np.concatenate([y_tstars + np.random.random(size=nstars),
                           np.random.random(size=20) * (N-30) + 30.])

newcat_x = np.concatenate([x_tstars + np.random.random(size=nstars),
                           np.random.random(size=25) * (N-30) + 30.])
newcat_y = np.concatenate([y_tstars + np.random.random(size=nstars),
                           np.random.random(size=25) * (N-30) + 30.])

master = Table([master_x, master_y], names=('x', 'y'))
newcat = Table([newcat_x, newcat_y], names=('x', 'y'))

newcat.sort('x')

Ids, mask = u.matching(master, newcat, radius=2, masked=True)

plt.hist(newcat[mask]['x'] - master[Ids[mask]]['x'])
plt.hist(newcat[mask]['y'] - master[Ids[mask]]['y'])
plt.show()
plt.close()

newcat[mask]['sourceid'] = Ids[mask]
