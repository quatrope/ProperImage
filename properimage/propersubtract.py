#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  propersubtract.py
#
#  Copyright 2016 Bruno S <bruno@oac.unc.edu.ar>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
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
    print('using pyfftw interfaces API')
except ImportError:
    _fftwn = np.fft.fft2
    _ifftwn = np.fft.ifft2
    print('using numpy fft API')

aa.PIXEL_TOL = 0.5
eps = np.finfo(np.float64).eps


def diff(ref, new, align=False, inf_loss=0.25, smooth_psf=False,
         beta=True, shift=True, iterative=False, fitted_psf=True):
    """Function that takes a list of SingleImage instances
    and performs a stacking using properimage R estimator
    """
    if fitted_psf:
        try:
            from .single_image_psfs import SingleImageGaussPSF as SI
            print('using single psf, gaussian modeled')
        except ImportError:
            from .single_image import SingleImage as SI
    else:
        from .single_image import SingleImage as SI

    if not isinstance(ref, SI):
        try:
            ref = SI(ref, smooth_psf=smooth_psf)
        except:  # noqa
            try:
                ref = SI(ref.pixeldata, smooth_psf=smooth_psf)
            except:  # noqa
                raise

    if not isinstance(new, SI):
        try:
            new = SI(new, smooth_psf=smooth_psf)
        except:  # noqa
            try:
                new = SI(new.pixeldata, smooth_psf=smooth_psf)
            except:  # noqa
                raise

    if align:
        registered = aa.register(new.pixeldata, ref.pixeldata)
        new._clean()
        registered = registered[:ref.pixeldata.shape[0],
                                :ref.pixeldata.shape[1]]
        new = SI(registered.data, mask=registered.mask,
                 borders=False, smooth_psf=smooth_psf)
        # new.pixeldata = registered
        # new.pixeldata.mask = registered.mask

    # make sure that the alignement has delivered arrays of size
    if new.pixeldata.data.shape != ref.pixeldata.data.shape:
        import ipdb
        ipdb.set_trace()

    t0 = time.time()
    mix_mask = np.ma.mask_or(new.pixeldata.mask, ref.pixeldata.mask)

    zps, meanmags = u.transparency([ref, new])
    print(zps)
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

        p_r.x_mean = ref.pixeldata.data.shape[0]/2.
        p_r.y_mean = ref.pixeldata.data.shape[1]/2.
        p_n.x_mean = new.pixeldata.data.shape[0]/2.
        p_n.y_mean = new.pixeldata.data.shape[1]/2.
        p_r.bounding_box = None
        p_n.bounding_box = None

        p_n = p_n.render(np.zeros(new.pixeldata.data.shape))
        p_r = p_r.render(np.zeros(ref.pixeldata.data.shape))
        #  import ipdb; ipdb.set_trace()

        dx_ref, dy_ref = center_of_mass(p_r)  # [0])
        dx_new, dy_new = center_of_mass(p_n)  # [0])
    else:
        p_r = psf_ref[0]
        p_n = psf_new[0]

        dx_ref, dy_ref = center_of_mass(p_r)  # [0])
        dx_new, dy_new = center_of_mass(p_n)  # [0])
        # print(dx_new, dy_new)
        if dx_new < 0. or dy_new < 0.:
            import ipdb
            ipdb.set_trace()

    rad_ref_sq = dx_ref*dx_ref + dy_ref*dy_ref
    # rad_new_sq = dx_new*dx_new + dy_new*dy_new

    psf_ref_hat = _fftwn(p_r, s=ref.pixeldata.shape, norm='ortho')
    psf_new_hat = _fftwn(p_n, s=new.pixeldata.shape, norm='ortho')

    psf_ref_hat[np.where(psf_ref_hat.real == 0)] = eps
    psf_new_hat[np.where(psf_new_hat.real == 0)] = eps

    psf_ref_hat_conj = psf_ref_hat.conj()
    psf_new_hat_conj = psf_new_hat.conj()

    D_hat_r = fourier_shift(psf_new_hat * ref.interped_hat, (-dx_new, -dy_new))
    D_hat_n = fourier_shift(psf_ref_hat * new.interped_hat, (-dx_ref, -dy_ref))
    # D_hat_r = psf_new_hat * ref.interped_hat
    # D_hat_n = psf_ref_hat * new.interped_hat

    if beta:
        new_back = sep.Background(new.interped).back()
        ref_back = sep.Background(ref.interped).back()
        gamma = new_back - ref_back
        b = n_zp/r_zp
        # start with beta=1

        if shift:
            def cost_beta(vec, gamma=gamma):
                b, dx, dy = vec[:]

                # gammap = gamma/np.sqrt(new.var**2 + b**2 * ref.var**2)

                norm = b**2 * ref.var**2 * psf_new_hat * psf_new_hat_conj
                norm += new.var**2 * psf_ref_hat * psf_ref_hat_conj

                cost = _ifftwn(D_hat_n/np.sqrt(norm), norm='ortho') - \
                    _ifftwn(fourier_shift((D_hat_r/np.sqrt(norm))*b, (dx, dy)),
                            norm='ortho')  # -\
                # _ifftwn(fourier_shift(_fftwn(gammap), (dx, dy)))
                cost = np.absolute(cost)

                # flux, _, _ = sep.sum_circle(np.ascontiguousarray(cost),
                # ref.best_sources['x'],
                # ref.best_sources['y'],
                # 0.5*np.sqrt(rad_ref_sq))
                # mean_flux = np.mean(flux/(np.pi*rad_ref_sq))

                # cost =np.absolute(cost*cost.conj())[50:-50, 50:-50].flatten()

                # return sigma_clipped_stats(cost, sigma=9.)[2]
                # return np.absolute(mean_flux)
                # return np.std(cost[50:-50, 50:-50].flatten())

                # testing new approach
                hist, edg = np.histogram(cost)
                costs = 0
                costs += np.abs(hist[0]*(edg[1]-edg[0]))
                costs += np.abs(hist[-1]*(edg[-2]-edg[-1]))
                return(costs)

            tbeta0 = time.time()
            vec0 = [b, 0., 0.]
            bounds = ([0.1, -2.9, -2.9], [25., 2.9, 2.9])
            solv_beta = optimize.least_squares(cost_beta,
                                               vec0, ftol=1e-10,
                                               jac='3-point',
                                               bounds=bounds)
            tbeta1 = time.time()

            if solv_beta.success:
                print(('Found that beta = {}'.format(solv_beta.x)))
                print(('Took only {} awesome seconds'.format(tbeta1-tbeta0)))
                print(('The solution was with cost {}'.format(solv_beta.cost)))
                b, dx, dy = solv_beta.x
            else:
                print('Least squares could not find our beta  :(')
                print('Beta is overriden to be the zp ratio again')
                b = n_zp/r_zp
                dx = 0.
                dy = 0.

        elif iterative:
            def beta_next(b, gamma=gamma):
                # gammap = gamma/np.sqrt(new.var**2 + b**2 * ref.var**2)

                norm = b**2 * ref.var**2 * psf_new_hat * psf_new_hat_conj
                norm += new.var**2 * psf_ref_hat * psf_ref_hat_conj

                b_n = _ifftwn(D_hat_n/np.sqrt(norm), norm='ortho') / \
                    _ifftwn(D_hat_r/np.sqrt(norm), norm='ortho')
                # - gammap)

                # b_n = _ifftwn(D_hat_n/np.sqrt(norm)) / \
                # _ifftwn(D_hat_r/np.sqrt(norm))

                ab = b_n.real
                flux, _, _ = sep.sum_circle(np.ascontiguousarray(ab),
                                            ref.best_sources['x'],
                                            ref.best_sources['y'],
                                            0.5*np.sqrt(rad_ref_sq))
                mean_flux = np.mean(flux/(np.pi*rad_ref_sq))
                # ab = ab[(np.percentile(ab, q=97)>ab)*
                # (ab>np.percentile(ab, q=55))]
                mean, med, std = sigma_clipped_stats(ab, iters=3, sigma=3.)

                # print('Sigma clip on beta values')
                # print(mean, med, std)
                print(mean_flux)
                if np.abs(mean_flux-1.) < 1e-3:
                    b_next = b
                elif mean_flux > 1:
                    b_next = b + np.random.random()/10.
                else:
                    b_next = b - np.random.random()/10.
                # b_next = sigma_clipped_stats(ab)[0]
                if b_next == 0.:
                    return b, std
                # b_next = np.mean(b_n)
                return b_next, std

            bi = b  # 1
            print('Start iteration')
            ti = time.time()
            bf, std = beta_next(bi)
            n_iter = 1
            while np.abs(bf-bi) > 0.01 and n_iter < 45:
                bi = bf
                bf, std = beta_next(bi)
                n_iter += 1
            b = bf
            tf = time.time()
            print(('b = {}. Finished on {} iterations, and {} time\n'.format(
                b, n_iter, tf-ti)))
            dx = dy = 0.

        else:
            def cost_beta(vec, gamma=gamma):
                b = vec[0]
                # gammap = gamma/np.sqrt(new.var**2 + b**2 * ref.var**2)

                norm = b**2 * ref.var**2 * psf_new_hat * psf_new_hat_conj
                norm += new.var**2 * psf_ref_hat * psf_ref_hat_conj

                cost = _ifftwn(D_hat_n/np.sqrt(norm), norm='ortho') - \
                    _ifftwn((D_hat_r/np.sqrt(norm))*b, norm='ortho')  # gammap
                cost = np.absolute(cost)

                # this is computationally really expensive

                # flux, _, _ = sep.sum_circle(np.ascontiguousarray(cost),
                # ref.best_sources['x'],
                # ref.best_sources['y'],
                # 0.5*np.sqrt(rad_ref_sq))
                # mean_flux = np.mean(flux/(np.pi*rad_ref_sq))

                # cost =np.absolute(cost*cost.conj())[50:-50, 50:-50].flatten()

                # return sigma_clipped_stats(cost, sigma=9.)[2]
                # return np.absolute(mean_flux)
                # return np.std(cost[50:-50, 50:-50].flatten())

                # testing new approach
                hist, edg = np.histogram(cost)
                costs = 0
                costs += np.abs(hist[0]*(edg[1]-edg[0]))
                costs += np.abs(hist[-1]*(edg[-2]-edg[-1]))
                return(costs)

            dx = 0
            dy = 0
            tbeta0 = time.time()
            vec0 = [new.zp/ref.zp]
            bounds = ([0.01], [25.])
            solv_beta = optimize.least_squares(cost_beta,
                                               vec0, ftol=1e-9,
                                               jac='3-point',
                                               bounds=bounds)
            tbeta1 = time.time()
            if solv_beta.success:
                print(('Found that beta = {}'.format(solv_beta.x)))
                print(('Took only {} awesome seconds'.format(tbeta1-tbeta0)))
                print(('The solution was with cost {}'.format(solv_beta.cost)))
                b = solv_beta.x
            else:
                print('Least squares could not find our beta  :(')
                print('Beta is overriden to be the zp ratio again')
                b = n_zp/r_zp

    else:
        b = new.zp/ref.zp
        dx = 0.
        dy = 0.

    norm = b**2 * ref.var**2 * psf_new_hat * psf_new_hat_conj
    norm += new.var**2 * psf_ref_hat * psf_ref_hat_conj

    if dx == 0. and dy == 0.:
        D_hat = (D_hat_n - b * D_hat_r)/np.sqrt(norm)
    else:
        D_hat = (D_hat_n - fourier_shift(b*D_hat_r, (dx, dy)))/np.sqrt(norm)

    D = _ifftwn(D_hat, norm='ortho')
    if np.any(np.isnan(D.real)):
        pass

    d_zp = new.zp/np.sqrt(ref.var**2 * b**2 + new.var**2)
    P_hat = (psf_ref_hat * psf_new_hat * b)/(np.sqrt(norm)*d_zp)

    P = _ifftwn(P_hat, norm='ortho').real
    dx_p, dy_p = center_of_mass(P)

    S_hat = fourier_shift(d_zp * D_hat * P_hat.conj(), (dx_p, dy_p))

    kr = _ifftwn(b * new.zp * psf_ref_hat_conj *
                 psf_new_hat * psf_new_hat_conj / norm, norm='ortho')

    kn = _ifftwn(b * new.zp * psf_new_hat_conj *
                 psf_ref_hat * psf_ref_hat_conj / norm, norm='ortho')

    V_en = _ifftwn(_fftwn(new.pixeldata.filled(0)+1., norm='ortho') *
                   _fftwn(kn**2, s=new.pixeldata.shape), norm='ortho')

    V_er = _ifftwn(_fftwn(ref.pixeldata.filled(0)+1., norm='ortho') *
                   _fftwn(kr**2, s=ref.pixeldata.shape), norm='ortho')

    S_corr = _ifftwn(S_hat, norm='ortho')/np.sqrt(V_en + V_er)
    print('S_corr sigma_clipped_stats ')
    print(('mean = {}, median = {}, std = {}\n'.format(*sigma_clipped_stats(
        S_corr.real.flatten(), sigma=6.))))
    print(('Subtraction performed in {} seconds\n\n'.format(time.time()-t0)))

    # import ipdb; ipdb.set_trace()
    return D, P, S_corr.real, mix_mask


def get_transients(self, threshold=2.5, neighborhood_size=5.):
    S = self.subtract()[2]
    threshold = np.std(S) * threshold
    cat = u.find_S_local_maxima(S, threshold=threshold,
                                neighborhood_size=neighborhood_size)

    return cat
