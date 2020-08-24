#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  test_matching.py
#
#  Copyright 2016 Bruno S <bruno@oac.unc.edu.ar>
#
# This file is part of ProperImage (https://github.com/toros-astro/ProperImage)
# License: BSD-3-Clause
# Full Text: https://github.com/toros-astro/ProperImage/blob/master/LICENSE.txt
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
