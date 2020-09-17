"""plot module from ProperImage,
for coadding astronomical images.

Written by Bruno SANCHEZ

PhD of Astromoy - UNC
bruno@oac.unc.edu.ar

Instituto de Astronomia Teorica y Experimental (IATE) UNC
Cordoba - Argentina

Of 301
"""

import logging
import itertools as it

import numpy as np

from astropy.stats import sigma_clipped_stats

import matplotlib.pyplot as plt

import attr


# =============================================================================
# CONSTANTS
# =============================================================================

FONT = {
    "family": "sans-serif",
    "sans-serif": ["Computer Modern Sans serif"],
    "weight": "regular",
    "size": 12,
}

TEXT = {"usetex": True}

DEFAULT_HEIGHT = 4
DEFAULT_WIDTH = 4


logger = logging.getLogger()


# =============================================================================
# FUNCTIONS
# =============================================================================


def primes(n):
    divisors = [d for d in range(2, n // 2 + 1) if n % d == 0]
    prims = [
        d for d in divisors if all(d % od != 0 for od in divisors if od != d)
    ]
    if len(prims) >= 4:
        return prims[-1] * prims[-2]
    elif len(prims) == 0:
        return n

    return max(prims)


def plot_psfbasis(
    psf_basis, path=None, nbook=False, size=4, iso=False, **kwargs
):
    # psf_basis.reverse()
    xsh, ysh = psf_basis[1].shape
    N = len(psf_basis)
    p = primes(N)
    if N == 2:
        subplots = (2, 1)
    elif p == N:
        subplots = (np.rint(np.sqrt(N)), np.rint(np.sqrt(N) + 1))
    else:
        rows = N // p
        rows += N % p
        subplots = (p, rows)

    plt.figure(figsize=(size * subplots[0], size * subplots[1]))
    for i in range(len(psf_basis)):
        plt.subplot(subplots[1], subplots[0], i + 1)
        plt.imshow(psf_basis[i], interpolation="none", cmap="viridis")
        labels = {"j": i + 1, "sum": np.sum(psf_basis[i])}
        plt.title(r"$\sum p_{j:d} = {sum:4.3e}$".format(**labels))
        # , interpolation='linear')
        plt.colorbar(shrink=0.85)
        if iso:
            plt.contour(
                np.arange(xsh),
                np.arange(ysh),
                psf_basis[i],
                colors="red",
                alpha=0.4,
            )
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
    if not nbook:
        plt.close()


def plot_S(S, path=None, nbook=False):
    if isinstance(S, np.ma.masked_array):
        S = S.filled()
    mean, med, std = sigma_clipped_stats(S)
    plt.imshow(
        S,
        vmax=med + 4 * std,
        vmin=med - 4 * std,
        interpolation="none",
        cmap="viridis",
    )
    plt.tight_layout()
    plt.colorbar()
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
    if not nbook:
        plt.close()
    return


def plot_R(R, path=None, nbook=False):
    if isinstance(R[0, 0], np.complex):
        R = R.real
    if isinstance(R, np.ma.masked_array):
        R = R.filled()
    plt.imshow(np.log10(R), interpolation="none", cmap="viridis")
    plt.tight_layout()
    plt.colorbar()
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
    if not nbook:
        plt.close()
    return


# =============================================================================
# PLOT API
# =============================================================================


@attr.s(frozen=True, repr=False, eq=False, order=False)
class Plot:

    DEFAULT_PLOT = "imshow"

    si = attr.ib()

    def __call__(self, plot=None, **kwargs):
        if plot is not None and (
            plot.startswith("_") or not hasattr(self, plot)
        ):
            raise ValueError(f"Ivalid plot method '{plot}'")
        method = getattr(self, plot or Plot.DEFAULT_PLOT)
        if not callable(method):
            raise ValueError(f"Ivalid plot method '{plot}'")
        return method(**kwargs)

    def imshow(self, ax=None, **kwargs):
        if ax is None:
            fig = plt.gcf()
            ax = plt.gca()
            fig.set_size_inches(h=DEFAULT_HEIGHT, w=DEFAULT_WIDTH)

        kwargs.setdefault("origin", "lower")

        ax.imshow(self.si.data, **kwargs)

        ax.set_title(f"SingleImage {self.si.data.shape}")
        return ax

    def autopsf_coef(
        self, axs=None, inf_loss=None, shape=None, cmap_kw=None, **kwargs
    ):
        a_fields, psf_basis = self.si.get_variable_psf(inf_loss=inf_loss)
        x, y = self.si.get_afield_domain()

        # here we plot
        N = len(a_fields)  # axis needed

        if axs is None:
            p = primes(N)

            fig = plt.gcf()

            if N == 2:
                subplots = (2, 1)
            elif p == N:
                subplots = (round(np.sqrt(N)), round(np.sqrt(N) + 1))
            else:
                rows = int((N // p) + (N % p))
                subplots = (p, rows)

            width = DEFAULT_WIDTH * subplots[0]
            height = DEFAULT_HEIGHT * subplots[1]

            fig.set_size_inches(w=width, h=height)
            axs = fig.subplots(*subplots)

        if N > np.size(axs):
            raise ValueError(
                f"You must provide at least {N} axs. Found {len(axs)}"
            )

        cmap_kw = cmap_kw or {}
        cmap_kw.setdefault("shrink", 0.75)
        cmap_kw.setdefault("aspect", 30)

        title_tpl = r"$a_{j}$,$\sum a_{j}={sum:4.3e}$"
        for idx, a_field, ax in zip(range(N), a_fields, it.chain(*axs)):
            fig = ax.get_figure()

            a = a_field(x, y)

            _, med, std = sigma_clipped_stats(a)

            imshow_kw = kwargs.copy()
            imshow_kw.setdefault("vmax", med + 2 * std)
            imshow_kw.setdefault("vmin", med - 2 * std)

            img = ax.imshow(a, **imshow_kw)
            fig.colorbar(img, ax=ax, **cmap_kw)

            title = title_tpl.format(j=idx + 1, sum=np.sqrt(np.sum(a ** 2)))
            ax.set_title(title)

        return axs
