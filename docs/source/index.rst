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

A previous version of this code used the concept of *ensembles*, which was a class
inheriting from python lists, grouping instances of SingleImage,
providing all the mathematical operations over multiple images such as coadd,
and subtract methods.

Now to offer more flexibility to user, there is only one class, the SingleImage.
For coaddition, and subtraction the user must employ the functions provided,
which take instances of SingleImage as input.


Install Properimage
-------------------

There is no official PyPI release yet, so a developement install procedure
is the current way of obtaining Properimage:

1. Make sure that you have Git_ installed and that you can run its commands
   from a shell. (Enter ``git help`` at a shell prompt to test this.)

2. Check out Properimage's main development branch like so:

   .. code-block:: console

        $ git clone git@github.com:toros-astro/ProperImage.git

   This will create a directory ``ProperImage`` in your current directory.

3. Make sure that the Python interpreter can load ProperImage's code. The most
   convenient way to do this is to use virtualenv_, virtualenvwrapper_, and
   pip_.

4. After setting up and activating the virtualenv, run the following command:

   .. code-block:: console

        $ pip install -e ProperImage/

   This will make ProperImage's code importable, in other words, you're all
   set!

When you want to update your copy of the ProperImage source code, just run the
command ``git pull`` from within the ``ProperImage`` directory. When you do this,
Git will automatically download any changes.


License
-------

Properimage is under `The MIT License <https://raw.githubusercontent.com/toros-astro/Properimage/master/LICENSE.txt>`__

This license allows unlimited redistribution for any purpose as long as its
copyright notices and the license's disclaimers of warranty are maintained.


Contents:
---------

.. toctree::
    :maxdepth: 2

    tutorial/index.rst
    topics/index.rst

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
