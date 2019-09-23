# Proper image treatments.

[![Build Status](https://travis-ci.org/toros-astro/ProperImage.svg?branch=master)](https://travis-ci.org/toros-astro/ProperImage)
[![codecov](https://codecov.io/gh/toros-astro/ProperImage/branch/master/graph/badge.svg)](https://codecov.io/gh/toros-astro/ProperImage)
[![Documentation Status](https://readthedocs.org/projects/properimage/badge/?version=latest)](http://properimage.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://tldrlegal.com/license/mit-license)
[![Python 3.5+](https://img.shields.io/badge/python-3.5+-blue.svg)](https://badge.fury.io/py/feets)

This code is inspired on Zackay & Ofek 2017 papers *How to coadd images?*

* It can perform a PSF estimation using Karhunen-Löeve expansion, which is based on Lauer 2002 work.

* It can perform the statistical proper coadd of several images.

* It can also perform a proper-subtraction of images.

* Images need to be aligned and registered, or at least github.com/toros-astro/astroalign must be installed.

* Contains a nice plot module for PSF visualization (_needs matplotlib_)


Documentation at https://properimage.readthedocs.io .

## Rerefences

> Zackay, B., & Ofek, E. O. (2017). How to COAAD Images. I. Optimal Source Detection and Photometry of Point Sources Using Ensembles of Images. The Astrophysical Journal, 836(2), 187. http://arxiv.org/abs/1512.06872
>
> Zackay, B., & Ofek, E. O. (2017). How to COAAD Images. II. A Coaddition Image that is Optimal for Any Purpose in the Background-dominated Noise Limit. The Astrophysical Journal, 836(2), 188. http://arxiv.org/abs/1512.06879
>
>Zackay, B., Ofek, E. O., & Gal-Yam, A. (2016). PROPER IMAGE SUBTRACTION—OPTIMAL TRANSIENT DETECTION, PHOTOMETRY, AND HYPOTHESIS TESTING. The Astrophysical Journal, 830(1), 27. 
>
>Lauer, T. (2002, December). Deconvolution with a spatially-variant PSF. In Astronomical Data Analysis II (Vol. 4847, pp. 167-174). International Society for Optics and Photonics. 
