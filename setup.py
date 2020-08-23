#!/usr/bin/env python
# -*- coding: utf-8 -*-


# =============================================================================
# DOCS
# =============================================================================

"""This file is for distribute properimage
"""


# =============================================================================
# IMPORTS
# =============================================================================

import sys
import os
import setuptools

from ez_setup import use_setuptools

use_setuptools()


# =============================================================================
# PATH TO THIS MODULE
# =============================================================================

PATH = os.path.abspath(os.path.dirname(__file__))

# =============================================================================
# Get the version from properimage file itself (not imported)
# =============================================================================

PROPERIMAGE_INIT_PATH = os.path.join(PATH, "properimage", "__init__.py")

with open(PROPERIMAGE_INIT_PATH, "r") as f:
    for line in f:
        if line.startswith("__version__"):
            _, _, PI_VERSION = line.replace('"', "").split()
            break

# =============================================================================
# CONSTANTS
# =============================================================================

REQUIREMENTS = [
    "numpy >= 1.13.2",
    "scipy >= 1.0",
    "astropy >= 2.0",
    "photutils",
    "astroML",
    "sep",
    "astroscrappy>=1.0.5",
    "astroalign>=1.0.3",
    # "pytest>=3.6.2"
    # "pyFFTW>=0.10"
]

# =============================================================================
# DESCRIPTION
# =============================================================================
with open("README.md") as fp:
    LONG_DESCRIPTION = fp.read()

# =============================================================================
# FUNCTIONS
# =============================================================================
print(setuptools.find_packages())  # exclude=['test*']


def do_setup():
    setuptools.setup(
        name="properimage",
        version=PI_VERSION,
        description="Proper Astronomic Image Analysis",
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        author="Bruno Sanchez",
        author_email="bruno@oac.unc.edu.ar",
        url="https://github.com/toros-astro/ProperImage",
        py_modules=["ez_setup"],
        license="BSD 3",
        keywords="astronomy image",
        classifiers=(
            "Development Status :: 4 - Beta",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: Implementation :: CPython",
            "Topic :: Scientific/Engineering",
        ),
        packages=setuptools.find_packages(),  # exclude=['test*']),
        install_requires=REQUIREMENTS,
    )


def do_publish():
    pass


if __name__ == "__main__":
    if sys.argv[-1] == "publish":
        do_publish()
    else:
        do_setup()
