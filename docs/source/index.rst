.. Properimage documentation master file, created by
   sphinx-quickstart on Tue Dec  5 16:43:19 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Properimage documentation
=========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Subtracting a reference image of the "static" sky from a new image for
transient detection and flux measurement is one of the most fundamental
techniques in time domain astronomy.
But due to varying seeing conditions between the reference image and the new
image, the process becomes non-trivial and it requires the adjustment for
different `point-spread functions (PSF) <https://en.wikipedia.org/wiki/Point_spread_function>`_ across the two frames.

**Properimage** is an astronomical image processing code, specially written for
coaddition, and image subtraction under those circumstances.
It is an implementation of the mathematical developement published in the
papers [Zackay2016]_, [Zackay2017a]_, and [Zackay2017b]_ writen by 
Barak Zackay, Eran O. Ofek and Avishay Gal-Yam.

Unlike previous methods to remedy this problem, the method proposed
in [Zackay2016]_ is based on basic statistical principles applied in the Fourier
domain of the image system.
This has several advantages with respect to previous methods, among which are:

* It is numerically stable
* The difference image has uncorrelated white noise
* It is symmetric to the exchange of the new and reference images
* It is at least an order of magnitude faster to compute than some popular methods


Installing Properimage
----------------------

Install the latest release from PyPI

   .. code-block:: console

        $ pip install properimage

License
-------

Properimage is released under `The BSD-3 License <https://raw.githubusercontent.com/toros-astro/Properimage/master/LICENSE.txt>`_.

This license allows unlimited redistribution for any purpose as long as its
copyright notices and the license's disclaimers of warranty are maintained.


Contents:
---------

.. toctree::
    :maxdepth: 2

    tutorial/index.rst
    glossary.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


References
----------
.. [Zackay2016] http://adsabs.harvard.edu/abs/2016ApJ...830...27Z
.. [Zackay2017a] http://adsabs.harvard.edu/abs/2017ApJ...836..187Z
.. [Zackay2017b] http://adsabs.harvard.edu/abs/2017ApJ...836..188Z
