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
import setuptools

from ez_setup import use_setuptools
use_setuptools()


# =============================================================================
# CONSTANTS
# =============================================================================

REQUIREMENTS = ["numpy>=1.6.2",
                "scipy>=0.15",
                "astropy>=1.0",
                "photutils>=0.2",
                "astroML>=0.3",
                "sep>=0.5"#,
                #"pyFFTW>=0.10"
                ]

# =============================================================================
# FUNCTIONS
# =============================================================================
print setuptools.find_packages()


def do_setup():
    setuptools.setup(
        name='properimage',
        version='0.1.0.dev1',
        description='Proper astronomic image analysis',
        author='Bruno Sanchez',
        author_email='bruno@oac.unc.edu.ar',
        url='https://github.com/toros-astro/ProperImage',
        py_modules=['ez_setup'],
        license="BSD 3",
        keywords="astronomy image",
        classifiers=(
            "Development Status :: 4 - Alpha",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: Implementation :: CPython",
            "Topic :: Scientific/Engineering",
        ),
        packages=setuptools.find_packages(exclude=['test*']),
        install_requires=REQUIREMENTS
    )


def do_publish():
    pass


if __name__ == "__main__":
    if sys.argv[-1] == 'publish':
        do_publish()
    else:
        do_setup()
