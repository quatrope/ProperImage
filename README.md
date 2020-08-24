# Proper image treatments

[![Build Status](https://travis-ci.org/toros-astro/ProperImage.svg?branch=master)](https://travis-ci.org/toros-astro/ProperImage)
[![codecov](https://codecov.io/gh/toros-astro/ProperImage/branch/master/graph/badge.svg)](https://codecov.io/gh/toros-astro/ProperImage)
[![Documentation Status](https://readthedocs.org/projects/properimage/badge/?version=latest)](http://properimage.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/pypi/l/properimage?color=blue)](https://tldrlegal.com/license/bsd-3-clause-license-(revised))
[![Python 3.5+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://badge.fury.io/py/properimage)

This code is inspired on [Zackay & Ofek 2017](http://arxiv.org/abs/1512.06872)  papers *How to coadd images?* (see References below).

* It can perform a PSF estimation using [Karhunen-Löeve expansion](https://en.wikipedia.org/wiki/Karhunen–Loève_theorem), which is based on [Lauer 2002](https://doi.org/10.1117/12.461035) work.

* It can perform the statistical proper coadd of several images.

* It can also perform a proper-subtraction of images.

* Images need to be aligned and registered, or at least [astroalign](https://github.com/toros-astro/astroalign) must be installed.

* Contains a nice plot module for PSF visualization (_needs matplotlib_)

## Installation

To install from PyPI

```console
$ pip install properimage
```

## Quick usage

### PSF estimation

```python
>>> from properimage import singleimage as si
>>> with si.SingleImage(frame, smooth_psf=False) as sim:
...     a_fields, psf_basis = sim.get_variable_psf(inf_loss=0.15)
```

### Proper-subtraction of images

To create a proper-subtraction of images:

```python
>>> from properimage import propersubtract as ps
>>> D, P, Scorr, mask = ps.diff(ref=ref_path, new=new_path, smooth_psf=False, fitted_psf=True,
...                             align=False, iterative=False, beta=False, shift=False)
```

Where `D`, `P`, `Scorr` refer to the images defined by the same name in [Zackay & Ofek](https://iopscience.iop.org/article/10.3847/0004-637X/830/1/27/meta) paper.

For the full documentation refer to [readthedocs](https://properimage.readthedocs.io).

## Rerefences

> Zackay, B., & Ofek, E. O. (2017). How to Coadd Images. I. Optimal Source Detection and Photometry of Point Sources Using Ensembles of Images. The Astrophysical Journal, 836(2), 187. [Arxiv version](http://arxiv.org/abs/1512.06872)
>
> Zackay, B., & Ofek, E. O. (2017). How to Coadd Images. II. A Coaddition Image that is Optimal for Any Purpose in the Background-dominated Noise Limit. The Astrophysical Journal, 836(2), 188. [Arxiv version](http://arxiv.org/abs/1512.06879)
>
>Zackay, B., Ofek, E. O., & Gal-Yam, A. (2016). Proper Image Subtrraction-Optimal Transient Detection, Photometry, and Hypothesis Testing. [The Astrophysical Journal, 830(1), 27.](https://iopscience.iop.org/article/10.3847/0004-637X/830/1/27/meta)
>
>Lauer, T. (2002, December). Deconvolution with a spatially-variant PSF. In [Astronomical Data Analysis II (Vol. 4847, pp. 167-174). International Society for Optics and Photonics.](https://doi.org/10.1117/12.461035)
