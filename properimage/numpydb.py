#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  numpydb.py
#
#  Copyright 2016 Bruno S <bruno@oac.unc.edu.ar>
#
# This file is part of ProperImage (https://github.com/toros-astro/ProperImage)
# License: BSD-3-Clause
# Full Text: https://github.com/toros-astro/ProperImage/blob/master/LICENSE.txt
#
import os

try:
    import cPickle as pickle
except ImportError:
    import pickle


class NumPyDB:
    def __init__(self, database_name, mode="store"):
        self.filename = database_name
        self.dn = self.filename + ".dat"  # NumPy array data
        self.pn = self.filename + ".map"  # positions & identifiers

        if mode == "store":
            # bring files into existence:
            fd = open(self.dn, "w")
            fd.close()

            fm = open(self.pn, "w")
            fm.close()

            self.positions = []

        elif mode == "load":
            # check if files are there:
            if not os.path.isfile(self.dn) or not os.path.isfile(self.pn):
                raise IOError(
                    "Could not find the files {} and {}".format(
                        self.dn, self.pn
                    )
                )
            # load mapfile into list of tuples:
            with open(self.pn, "r") as fm:
                self.positions = []
                for line in fm:
                    # first column contains file positions in the
                    # file .dat for direct access, the rest of the
                    # line is an identifier
                    c = line.split()
                    # append tuple (position, identifier):
                    self.positions.append((int(c[0]), " ".join(c[1:]).strip()))

    def locate(self, identifier):  # base class
        """
        Find position in files where data corresponding
        to identifier are stored.
        bestapprox is a user-defined function for computing
        the distance between two identifiers.
        """
        # first search for an exact identifier match:
        selected_pos = -1
        selected_id = None
        for pos, id in self.positions:
            if id == identifier:
                selected_pos = pos
                selected_id = id
                break
        if selected_pos == -1:  # ’identifier’ not found?
            raise LookupError("Identifier not found")

        return selected_pos, selected_id


class NumPyDB_cPickle(NumPyDB):
    """Use basic cPickle class."""

    def __init__(self, database_name, mode="store"):
        NumPyDB.__init__(self, database_name, mode)

    def dump(self, a, identifier):
        """Dump NumPy array a with identifier."""
        # fd: datafile, fm: mapfile
        with open(self.dn, "ab") as fd:
            with open(self.pn, "a") as fm:
                # fd.tell(): return current position in datafile
                fm.write("%d\t\t %s\n" % (fd.tell(), identifier))
                self.positions.append((fd.tell(), identifier))
                pickle.dump(a, fd, 1)  # 1: binary storage

    def load(self, identifier):
        """Load NumPy array with a given identifier. In case the
        identifier is not found, bestapprox != None means that
        an approximation is sought. The bestapprox argument is
        then taken as a function that can be used for computing
        the distance between two identifiers id1 and id2.
        """
        pos, id = self.locate(identifier)
        if pos < 0:
            return [None, "not found"]
        with open(self.dn, "rb") as fd:
            fd.seek(pos)
            a = pickle.load(fd)
        return [a, id]
