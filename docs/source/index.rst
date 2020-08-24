.. Properimage documentation master file, created by
   sphinx-quickstart on Tue Dec  5 16:43:19 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Properimage's documentation!
=======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Properimage is an astronomical image processing code, specially written for
coaddition, and image subtraction.
It uses the mathematical developement published in the following papers
[Zackay2016]_, [Zackay2017a]_, and [Zackay2017b]_, and a PSF estimation
method published in [Lauer2002]_.

Most of the code is based on a class called *SingleImage*, which provides
methods and properties for image processing such as PSF determination.

.. note::

    A previous version of this code used the concept of *ensembles*, which was a class
    inheriting from python lists, grouping instances of SingleImage,
    providing all the mathematical operations over multiple images such as coadd,
    and subtract methods.
    
    Now to offer more flexibility to user, there is only one class, the SingleImage.
    For coaddition, and subtraction the user must employ the functions provided,
    which take instances of SingleImage as input.


Install Properimage
-------------------

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
.. [Lauer2002] http://adsabs.harvard.edu/abs/2002SPIE.4847..167L
