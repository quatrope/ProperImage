Tutorial
========

Image subtraction
-----------------

[Add here how the image system is modeled, with the psf, noise, etc]

[Describe what is D, P, Scorr, etc]

To perform a subtraction between a reference image ``ref`` and a new image ``new``,
use the ``diff`` method from ``properimage``.

.. code-block:: python

    from properimage.propersubtract import diff
    D, P, Scorr, mask = diff(ref, new)

Here the input ``ref`` and ``new`` are numpy's ``ndarray`` instances or properimage's ``SingleImage`` instances.

The output will be four different arrays:

D: A difference image 
P: The psf?
Scorr: Subtraction image with correlated noise.
mask: don't know.
