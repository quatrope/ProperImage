#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  propersubtract.py
#
#  Copyright 2016 Bruno S <bruno@oac.unc.edu.ar>
#
# This file is part of ProperImage (https://github.com/toros-astro/ProperImage)
# License: BSD-3-Clause
# Full Text: https://github.com/toros-astro/ProperImage/blob/master/LICENSE.txt
#

"""propersubtract module from ProperImage,
for coadding astronomical images.

Written by Bruno SANCHEZ

PhD of Astromoy - UNC
bruno@oac.unc.edu.ar

Instituto de Astronomia Teorica y Experimental (IATE) UNC
Cordoba - Argentina

Of 301
"""

import numpy as np

from scipy import optimize

# from scipy import stats
from scipy.ndimage import center_of_mass
from scipy.ndimage.fourier import fourier_shift
from astropy.stats import sigma_clipped_stats
import astroalign as aa
import sep
import time

#  from .single_image import SingleImage as SI
from . import utils as u

try:
    import pyfftw

    _fftwn = pyfftw.interfaces.numpy_fft.fftn  # noqa
    _ifftwn = pyfftw.interfaces.numpy_fft.ifftn  # noqa
    print("using pyfftw interfaces API")
except ImportError:
    _fftwn = np.fft.fft2
    _ifftwn = np.fft.ifft2
    print("using numpy fft API")

aa.PIXEL_TOL = 0.5
eps = np.finfo(np.float64).eps


def diff(
    ref,
    new,
    align=False,
    inf_loss=0.25,
    smooth_psf=False,
    beta=True,
    shift=True,
    iterative=False,
    fitted_psf=True,
):
    """Function that takes a list of SingleImage instances
    and performs a stacking using properimage R estimator
    """
    if fitted_psf:
        try:
            from .single_image_psfs import SingleImageGaussPSF as SI

            print("using single psf, gaussian modeled")
        except ImportError:
            from .single_image import SingleImage as SI
    else:
        from .single_image import SingleImage as SI

    if not isinstance(ref, SI):
        try:
            ref = SI(ref, smooth_psf=smooth_psf)
        except:  # noqa
            try:
                ref = SI(ref.data, smooth_psf=smooth_psf)
            except:  # noqa
                raise

    if not isinstance(new, SI):
        try:
            new = SI(new, smooth_psf=smooth_psf)
        except:  # noqa
            try:
                new = SI(new.data, smooth_psf=smooth_psf)
            except:  # noqa
                raise

    if align:
        registered = aa.register(new.data, ref.data)
        new._clean()
        registered = registered[: ref.data.shape[0], : ref.data.shape[1]]
        new = SI(
            registered.data,
            mask=registered.mask,
            borders=False,
            smooth_psf=smooth_psf,
        )
        # new.data = registered
        # new.data.mask = registered.mask

    # make sure that the alignement has delivered arrays of size
    if new.data.data.shape != ref.data.data.shape:
        import ipdb

        ipdb.set_trace()

    t0 = time.time()
    mix_mask = np.ma.mask_or(new.data.mask, ref.data.mask)

    zps, meanmags = u.transparency([ref, new])
    # print(zps)
    ref.zp = zps[0]
    new.zp = zps[1]
    n_zp = new.zp
    r_zp = ref.zp
    # r_var = ref.var
    # n_var = new.var

    a_ref, psf_ref = ref.get_variable_psf(inf_loss)
    a_new, psf_new = new.get_variable_psf(inf_loss)

    if fitted_psf:
        #  I already know that a_ref and a_new are None, both of them
        #  And each psf is a list, first element a render,
        #  second element a model

        p_r = psf_ref[1]
        p_n = psf_new[1]

        p_r.x_mean = ref.data.data.shape[0] / 2.0
        p_r.y_mean = ref.data.data.shape[1] / 2.0
        p_n.x_mean = new.data.data.shape[0] / 2.0
        p_n.y_mean = new.data.data.shape[1] / 2.0
        p_r.bounding_box = None
        p_n.bounding_box = None

        p_n = p_n.render(np.zeros(new.data.data.shape))
        p_r = p_r.render(np.zeros(ref.data.data.shape))
        #  import ipdb; ipdb.set_trace()

        dx_ref, dy_ref = center_of_mass(p_r)  # [0])
        dx_new, dy_new = center_of_mass(p_n)  # [0])
    else:
        p_r = psf_ref[0]
        p_n = psf_new[0]

        dx_ref, dy_ref = center_of_mass(p_r)  # [0])
        dx_new, dy_new = center_of_mass(p_n)  # [0])
        # print(dx_new, dy_new)
        if dx_new < 0.0 or dy_new < 0.0:
            import ipdb

            ipdb.set_trace()

    # rad_ref_sq = dx_ref*dx_ref + dy_ref*dy_ref
    # rad_new_sq = dx_new*dx_new + dy_new*dy_new

    psf_ref_hat = _fftwn(p_r, s=ref.data.shape, norm="ortho")
    psf_new_hat = _fftwn(p_n, s=new.data.shape, norm="ortho")

    psf_ref_hat[np.where(psf_ref_hat.real == 0)] = eps
    psf_new_hat[np.where(psf_new_hat.real == 0)] = eps

    psf_ref_hat_conj = psf_ref_hat.conj()
    psf_new_hat_conj = psf_new_hat.conj()

    D_hat_r = fourier_shift(psf_new_hat * ref.interped_hat, (-dx_new, -dy_new))
    D_hat_n = fourier_shift(psf_ref_hat * new.interped_hat, (-dx_ref, -dy_ref))
    # D_hat_r = psf_new_hat * ref.interped_hat
    # D_hat_n = psf_ref_hat * new.interped_hat

    norm_b = ref.var ** 2 * psf_new_hat * psf_new_hat_conj
    norm_a = new.var ** 2 * psf_ref_hat * psf_ref_hat_conj

    new_back = sep.Background(new.interped).back()
    ref_back = sep.Background(ref.interped).back()
    gamma = new_back - ref_back
    b = n_zp / r_zp
    norm = np.sqrt(norm_a + norm_b * b ** 2)
    if beta:
        # start with beta=1
        if shift:

            def cost(vec):
                b, dx, dy = vec
                gammap = gamma / np.sqrt(new.var ** 2 + b ** 2 * ref.var ** 2)
                norm = np.sqrt(norm_a + norm_b * b ** 2)
                dhn = D_hat_n / norm
                dhr = D_hat_r / norm
                b_n = (
                    _ifftwn(dhn, norm="ortho")
                    - _ifftwn(fourier_shift(dhr, (dx, dy)), norm="ortho") * b
                    - np.roll(gammap, (int(round(dx)), int(round(dy))))
                )

                cost = b_n.real[100:-100, 100:-100]
                cost = np.sum(np.abs(cost / (cost.shape[0] * cost.shape[1])))

                return cost

            ti = time.time()
            vec0 = [b, 0.0, 0.0]
            bounds = ([0.1, -0.9, -0.9], [10.0, 0.9, 0.9])
            solv_beta = optimize.least_squares(
                cost,
                vec0,
                xtol=1e-5,
                jac="3-point",
                method="trf",
                bounds=bounds,
            )
            tf = time.time()

            if solv_beta.success:
                print(("Found that beta = {}".format(solv_beta.x)))
                print(("Took only {} awesome seconds".format(tf - ti)))
                print(("The solution was with cost {}".format(solv_beta.cost)))
                b, dx, dy = solv_beta.x
            else:
                print("Least squares could not find our beta  :(")
                print("Beta is overriden to be the zp ratio again")
                b = n_zp / r_zp
                dx = 0.0
                dy = 0.0
        elif iterative:
            bi = b

            def F(b):
                gammap = gamma / np.sqrt(new.var ** 2 + b ** 2 * ref.var ** 2)
                norm = np.sqrt(norm_a + norm_b * b ** 2)
                b_n = (
                    _ifftwn(D_hat_n / norm, norm="ortho")
                    - gammap
                    - b * _ifftwn(D_hat_r / norm, norm="ortho")
                )
                # robust_stats = lambda b: sigma_clipped_stats(
                #    b_n(b).real[100:-100, 100:-100])

                return np.sum(np.abs(b_n.real))

            ti = time.time()
            solv_beta = optimize.minimize_scalar(
                F,
                method="bounded",
                bounds=[0.1, 10.0],
                options={"maxiter": 1000},
            )

            tf = time.time()
            if solv_beta.success:
                print(("Found that beta = {}".format(solv_beta.x)))
                print(("Took only {} awesome seconds".format(tf - tf)))
                b = solv_beta.x
            else:
                print("Least squares could not find our beta  :(")
                print("Beta is overriden to be the zp ratio again")
                b = n_zp / r_zp
            dx = dy = 0.0
        else:
            bi = b

            def F(b):
                gammap = gamma / np.sqrt(new.var ** 2 + b ** 2 * ref.var ** 2)
                norm = np.sqrt(norm_a + norm_b * b ** 2)
                b_n = (
                    _ifftwn(D_hat_n / norm, norm="ortho")
                    - gammap
                    - b * _ifftwn(D_hat_r / norm, norm="ortho")
                )
                # robust_stats = lambda b: sigma_clipped_stats(
                #    b_n(b).real[100:-100, 100:-100])

                return np.sum(np.abs(b_n.real))

            ti = time.time()
            solv_beta = optimize.least_squares(
                F, bi, ftol=1e-8, bounds=[0.1, 10.0], jac="2-point"
            )

            tf = time.time()
            if solv_beta.success:
                print(("Found that beta = {}".format(solv_beta.x)))
                print(("Took only {} awesome seconds".format(tf - tf)))
                print(("The solution was with cost {}".format(solv_beta.cost)))
                b = solv_beta.x
            else:
                print("Least squares could not find our beta  :(")
                print("Beta is overriden to be the zp ratio again")
                b = n_zp / r_zp
            dx = dy = 0.0
    else:
        if shift:
            bi = n_zp / r_zp
            gammap = gamma / np.sqrt(new.var ** 2 + b ** 2 * ref.var ** 2)
            norm = np.sqrt(norm_a + norm_b * b ** 2)
            dhn = D_hat_n / norm
            dhr = D_hat_r / norm

            def cost(vec):
                dx, dy = vec
                b_n = (
                    _ifftwn(dhn, norm="ortho")
                    - _ifftwn(fourier_shift(dhr, (dx, dy)), norm="ortho") * b
                    - np.roll(gammap, (int(round(dx)), int(round(dy))))
                )
                cost = b_n.real[100:-100, 100:-100]
                cost = np.sum(np.abs(cost / (cost.shape[0] * cost.shape[1])))
                return cost

            ti = time.time()
            vec0 = [0.0, 0.0]
            bounds = ([-0.9, -0.9], [0.9, 0.9])
            solv_beta = optimize.least_squares(
                cost,
                vec0,
                xtol=1e-5,
                jac="3-point",
                method="trf",
                bounds=bounds,
            )
            tf = time.time()

            if solv_beta.success:
                print(("Found that shift = {}".format(solv_beta.x)))
                print(("Took only {} awesome seconds".format(tf - ti)))
                print(("The solution was with cost {}".format(solv_beta.cost)))
                dx, dy = solv_beta.x
            else:
                print("Least squares could not find our shift  :(")
                dx = 0.0
                dy = 0.0
        else:
            b = new.zp / ref.zp
            dx = 0.0
            dy = 0.0

    norm = norm_a + norm_b * b ** 2

    if dx == 0.0 and dy == 0.0:
        D_hat = (D_hat_n - b * D_hat_r) / np.sqrt(norm)
    else:
        D_hat = (D_hat_n - fourier_shift(b * D_hat_r, (dx, dy))) / np.sqrt(
            norm
        )

    D = _ifftwn(D_hat, norm="ortho")
    if np.any(np.isnan(D.real)):
        pass

    d_zp = b / np.sqrt(ref.var ** 2 * b ** 2 + new.var ** 2)
    P_hat = (psf_ref_hat * psf_new_hat * b) / (np.sqrt(norm) * d_zp)

    P = _ifftwn(P_hat, norm="ortho").real
    dx_p, dy_p = center_of_mass(P)

    S_hat = fourier_shift(d_zp * D_hat * P_hat.conj(), (dx_p, dy_p))

    kr = _ifftwn(
        new.zp * psf_ref_hat_conj * b * psf_new_hat * psf_new_hat_conj / norm,
        norm="ortho",
    )

    kn = _ifftwn(
        new.zp * psf_new_hat_conj * psf_ref_hat * psf_ref_hat_conj / norm,
        norm="ortho",
    )

    V_en = _ifftwn(
        _fftwn(new.data.filled(0) + 1.0, norm="ortho")
        * _fftwn(kn ** 2, s=new.data.shape),
        norm="ortho",
    )

    V_er = _ifftwn(
        _fftwn(ref.data.filled(0) + 1.0, norm="ortho")
        * _fftwn(kr ** 2, s=ref.data.shape),
        norm="ortho",
    )

    S_corr = _ifftwn(S_hat, norm="ortho") / np.sqrt(V_en + V_er)
    print("S_corr sigma_clipped_stats ")
    print(
        (
            "mean = {}, median = {}, std = {}\n".format(
                *sigma_clipped_stats(S_corr.real.flatten(), sigma=4.0)
            )
        )
    )
    print(("Subtraction performed in {} seconds\n\n".format(time.time() - t0)))

    # import ipdb; ipdb.set_trace()
    return D, P, S_corr.real, mix_mask


def get_transients(self, threshold=2.5, neighborhood_size=5.0):
    S = self.subtract()[2]
    threshold = np.std(S) * threshold
    cat = u.find_S_local_maxima(
        S, threshold=threshold, neighborhood_size=neighborhood_size
    )

    return cat
