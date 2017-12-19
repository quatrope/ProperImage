Tutorial - Part #1 - The SingleImage Class
==========================================

In this first tutorial the basics of the methods and properties of the
SingleImage class are explained.

This object class represents the basic unit of data which we will manipulate.
Basically it starts containing pixel data, with its mask.
First lets create an instance of this class with a numpy array.


.. code-block:: python

    import numpy as np
    from properimage import single_image as s

    pixel = np.random.random((128,128))*5.
    # Add some stars to it
    star = [[35, 38, 35],
            [38, 90, 39],
            [35, 39, 34]]
    for i in range(20):
        x, y = np.random.randint(120, size=2)
        pixel[x:x+3,y:y+3] = star

    mask = np.random.randint(2, size=(128,128))
    img  = s.SingleImage(pixel, mask)

We can see that the img object created automatically produces an output
displaying the number of sources found.
This just accounts for sources good enough for PSF estimation, which is
the first step for any processing ProperImage is intended for.

If we try to print the instance, (or obtain the representation output) we find
that the explicit origin of the pixeldata is being displayed

