Tutorial
========

Theoretical Background
----------------------

For a reference image :math:`R` and a new image :math:`N`, Zackay et al modeled
them as follows:

.. math::
    R &= F_r T \otimes P_r + \epsilon_r \\
    N &= F_n T \otimes P_n + \epsilon_n + \alpha F_n \delta(q) \otimes P_n

Where :math:`F` is the flux-based zero point of the image, :math:`P` is its PSF,
and :math:`\epsilon` is its noise. :math:`\otimes` denotes convolution.
The last term in :math:`N` is the transient source that may or may not be present on the image.

Under this assumptions, Zack et al derive optimal statistics for transient detection
to distinguish between the hypothesis :math:`\cal{H}|_0` where no transient
is present and :math:`\cal{H}|_1` where a transient is present.
Under these assumptions, Zackay et al derive maximum likelihood optimal expressions for
:math:`S`, :math:`D`, :math:`P_D`, and :math:`S_{corr}`.

:math:`S` is the *optimal statistic for source detection*.

:math:`D` is the *proper subtraction image*.

:math:`P_D` is the PSF of the difference image :math:`D`.

:math:`S_{corr}` is a simple correction to the problem of underestimating noise
level around sources. It is basically :math:`S` as before, divided by a
correction factor that takes into account the local estimated variance of the
extra noise.

.. note::
    :math:`D` and :math:`P_D` are sufficient for any measurement or decision
    on the difference between the images

Subtracting with ProperImage
----------------------------

To perform a subtraction between a reference image ``ref`` and a new image ``new``,
use the ``diff`` method from ``properimage``.

.. code-block:: python

    from properimage.propersubtract import diff
    D, P, Scorr, mask = diff(ref, new)

Here the input ``ref`` and ``new`` are numpy's ``ndarray`` instances,
properimage's ``SingleImage`` instances, astropy's `HDUList` objects,
or a string containing the path to a FITS file.

The output will be four different arrays:
``D``, ``P``, ``Scorr`` as explained above, and ``mask``.

It is recommended that you run your source extraction algorithm on ``D``.
