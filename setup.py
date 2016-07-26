#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from setuptools import setup

setup(name='properimage',
      version='0.1',
      description='Proper astronomic image analysis',
      author='Bruno Sanchez',
      author_email='bruno@oac.unc.edu.ar',
      url='https://github.com/toros-astro/ProperImage',
      py_modules=['properimage', ],
      install_requires=["numpy>=1.6.2",
                        "scipy>=0.15",
                        "astropy>=1.0",
                        ],
      test_suite='tests',
)
