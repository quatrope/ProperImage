#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  numpydb.py
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

class NumPyDB:
    def __init__(self, database_name, mode=’store’):
        self.filename = database_name
        self.dn = self.filename + ’.dat’ # NumPy array data
        self.pn = self.filename + ’.map’ # positions & identifiers
        if mode == ’store’:
            # bring files into existence:
            fd = open(self.dn, ’w’); fd.close()
            fm = open(self.pn, ’w’); fm.close()
        elif mode == ’load’:
            # check if files are there:
            if not os.path.isfile(self.dn) or \
               not os.path.isfile(self.pn):
                raise IOError, \
                    "Could not find the files %s and %s" %\
                    (self.dn, self.pn)
            # load mapfile into list of tuples:
            fm = open(self.pn, ’r’)
            lines = fm.readlines()
            self.positions = []
            for line in lines:
                # first column contains file positions in the
                # file .dat for direct access, the rest of the
                # line is an identifier
                c = line.split()
                # append tuple (position, identifier):
                self.positions.append((int(c[0]), ’ ’.join(c[1:]).strip()))
            fm.close()

def mydist(id1, id2):
    """
    Return distance between identifiers id1 and id2.
    The identifiers are of the form ’time=3.1010E+01’.
    """
    t1 = id1[5:]; t2 = id2[5:]
    d = abs(float(t1) - float(t2))
    return d

def locate(self, identifier, bestapprox=None): # base class
    """
    Find position in files where data corresponding
    to identifier are stored.
    bestapprox is a user-defined function for computing
    the distance between two identifiers.
    """
    identifier = identifier.strip()
    # first search for an exact identifier match:
    selected_pos = -1
    selected_id = None
    for pos, id in self.positions:
        if id == identifier:
            selected_pos = pos; selected_id = id; break
    if selected_pos == -1: # ’identifier’ not found?
        if bestapprox is not None:
            # find the best approximation to ’identifier’:
            min_dist = bestapprox(self.positions[0][1], identifier)
            for pos, id in self.positions:
                d = bestapprox(id, identifier)
                if d <= min_dist:
                    selected_pos = pos; selected_id = id
                    min_dist = d
    return selected_pos, selected_id

