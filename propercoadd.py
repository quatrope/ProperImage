"""
sim_coadd_proper translation to Python.

Written by Bruno SANCHEZ

PhD of Astromoy - UNC"""

from multiprocessing import Process
from multiprocessing import Queue
from collections import MutableSequence
import numpy as np
from scipy.stats import stats
from scipy import signal as sg
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.modeling import fitting
from astropy.modeling import models
from astropy.convolution import convolve_fft
from astropy.nddata.utils import extract_array
from photutils import psf
import sep
import pickle


class ImageEnsemble(MutableSequence):
    """
    Processor for several images that uses SingleImage as an atomic processing
    unit. It deploys the utilities provided in the mentioned class and combines
    the results, making possible to coadd and subtract astronomical images with
    optimal techniques.


    """
    def __init__(self, imgpaths, *arg, **kwargs):
        super(ImageEnsemble, self).__init__(*arg, **kwargs)
        self.imgl = imgpaths
        self.global_shape = fits.getdata(imgpaths[0]).shape
        print self.global_shape

    def __setitem__(self, i, v):
        self.imgl[i] = v

    def __getitem__(self, i):
        return self.imgl[i]

    def __delitem__(self, i):
        del self.imgl[i]

    def __len__(self):
        return len(self.imgl)

    def insert(self, i, v):
        self.imgl.insert(i, v)

    @property
    def atoms(self):
        if not hasattr(self, '_atoms'):
            self._atoms = [SingleImage(im, imagefile=True) for im in self.imgl]
        elif len(atoms) is not len(self.imgl):
            self._atoms = [SingleImage(im, imagefile=True) for im in self.imgl]
        return self._atoms

    def calculate_S(self, n_procs=2):
        queues = []
        procs  = []
        for chunk in chunk_it(self.atoms, n_procs):
            queue = Queue()
            proc = Combinator(chunk, queue)
            print 'starting new process'
            proc.start()

            queues.append(queue)
            procs.append(proc)
    print 'all chunks started, and procs appended'

        S = np.zeros(self.global_shape)
        for q in queues:
            serialized = q.get()
            print 'loading pickles'
            s_comp = pickle.loads(serialized)
            s_comp = np.ma.masked_array(s_comp, np.isnan(s_comp))
            S = np.add(s_comp, S)
        print 'S calculated, now starting to join processes'

        for proc in procs:
            print 'waiting for procs to join'
            proc.join()
        print 'processes joined, now returning S'
        return S


class Combinator(Process):

    def __init__(self, ensemble, q, *args, **kwargs):
        super(Combinator, self).__init__(*args, **kwargs)
        self.list_to_combine = ensemble
        self.q = q

    def run(self):
        shape = self.list_to_combine[0].imagedata.shape
        S = np.zeros(shape)
        for img in self.list_to_combine:
            s_comp = img.s_component()
            print 'S component obtained, summing arrays'
            S = np.add(s_comp, S)
        print 'chunk processed, now pickling'
        serialized = pickle.dumps(S)
        self.q.put(serialized)



class SingleImage(object):
    """
    Atomic processor class for a single image.
    Contains several tools for PSF measures, and different coadding
    building structures.

    It includes the pixel matrix data, as long as some descriptions.
    For statistical values of the pixel matrix the class' methods need to be
    called.


    Parameters
    ----------
    img : `~numpy.ndarray` or :class:`~ccdproc.CCDData`,
                `~astropy.io.fits.HDUList`  or a `str` naming the filename.
        The image object to work with

    imagefile : `bool`
        optional information regarding if the img is a fits file
        Default to None, and a guessing attempt will be made.
    """
    def __init__(self, img, imagefile=False, sim=False, meta={}):
        self._attached_to = img.__class__.__name__

        if imagefile:
            self.imagedata = fits.getdata(img)
            if not self.imagedata.dtype == 'uint16':
                self.imagedata = self.imagedata.byteswap().newbyteorder()
            else:
                self.imagedata = self.imagedata.astype('float')
        else:
            self.imagedata = img

        if sim:
            self.meta = meta
        else:
            imgstats = ImageStats(self.imagedata, 'numpy_array')
            imgstats.calc_stats()
            # imgstats.summary()
            self.meta = imgstats.full_description

    def __repr__(self):
        return 'SingleImage instance for {}'.format(self._attached_to)

    def sigma_clip_bkg(self):
        if not hasattr(self, 'bkg'):
            self.bkg = sigma_clip(self.imagedata, iters=10)
            self.bkg_mean = self.bkg.mean()
            self.bkg_sd = self.bkg.std()

    @property
    def bkg_sub_img(self):
        if not hasattr(self, 'bkg_sub_img'):
            self.bkg = sep.Background(self.imagedata)
            self._bkg_sub_img = self.imagedata - self.bkg
            self._masked = np.ma.masked_array(self._bkg_sub_img ,
                np.isnan(self._bkg_sub_img))
        return self._bkg_sub_img

    def fit_psf_sep(self, model='astropy-Gaussian2D'):
        """
        Fit and calculate the Psf of an image using sep source detection
        """
        # calculate x, y, flux of stars
        best_srcs = self._best_srcs['sources']
        fitshape = self._best_srcs['fitshape']
        print 'Fitshape = {}'.format(fitshape)

        fitter = fitting.LevMarLSQFitter()
        indices = np.indices(self.bkg_sub_img.shape)
        model_fits = []

        if model=='photutils-IntegratedGaussianPRF':
            prf_model = psf.IntegratedGaussianPRF(x_0=size/2., y_0=size/2.,
                                                    sigma=size/3.)
            prf_model.fixed['flux'] = False
            prf_model.fixed['sigma'] = False
            prf_model.fixed['x_0'] = False
            prf_model.fixed['y_0'] = False

            for row in best_srcs:
                position = (row['y'], row['x'])
                y = extract_array(indices[0], fitshape, position)
                x = extract_array(indices[1], fitshape, position)
                sub_array_data = extract_array(self.bkg_sub_img,
                                                fitshape, position,
                                                fill_value=self.bkg.globalback)
                prf_model.x_0 = position[1]
                prf_model.y_0 = position[0]
                resid = sub_array_data - fit(x,y)
                if np.sum(np.square(resid)) < 5*self.bkg.globalrms*fitshape[0]**2:
                    model_fits.append(fit)
            print 'succesful fits = {}'.format(len(model_fits))

        elif model=='astropy-Gaussian2D':
            prf_model = models.Gaussian2D(x_stddev=1, y_stddev=1)

            for row in best_srcs:
                position = (row['y'], row['x'])
                y = extract_array(indices[0], fitshape, position)
                x = extract_array(indices[1], fitshape, position)
                sub_array_data = extract_array(self.bkg_sub_img,
                                                fitshape, position,
                                                fill_value=self.bkg.globalrms)
                prf_model.x_mean = position[1]
                prf_model.y_mean = position[0]
                fit = fitter(prf_model, x, y, sub_array_data)
                resid = sub_array_data - fit(x,y)
                if np.sum(np.square(resid)) < 5*(self.bkg.globalrms*fitshape[0])**2:
                    model_fits.append(fit)
            print 'succesful fits = {}'.format(len(model_fits))
        return model_fits

    @property
    def _best_srcs(self):
        if not hasattr(self, '_best_sources'):
            try:
                srcs = sep.extract(self.bkg_sub_img, thresh=12*self.bkg.globalrms)
            except Exception:
                sep.set_extract_pixstack(700000)
                srcs = sep.extract(self.bkg_sub_img, thresh=12*self.bkg.globalrms)
            except ValueError:
                srcs = sep.extract(self.bkg_sub_img.byteswap().newbyteorder(),
                    thresh=12*self.bkg.globalrms)


            if len(srcs)<10:
                try:
                    srcs = sep.extract(self.bkg_sub_img, \
                        thresh=2.5*self.bkg.globalrms)
                except Exception:
                    sep.set_extract_pixstack(900000)
                    srcs = sep.extract(self.bkg_sub_img, \
                        thresh=2.5*self.bkg.globalrms)
            if len(srcs)<10:
                print 'No sources detected'
            p_sizes = np.sqrt(np.percentile(srcs['tnpix'], q=[35,55,85]))

            if not p_sizes[1]<11:
                fitshape = (int(p_sizes[1]), int(p_sizes[1]))
            else:
                fitshape = (11,11)

            best_big = srcs['tnpix']>=p_sizes[0]**2.
            best_small = srcs['tnpix']<=p_sizes[2]**2.
            best_flag = srcs['flag']<=1
            fluxes_quartiles = np.percentile(srcs['flux'], q=[30, 60])
            low_flux = srcs['flux'] > fluxes_quartiles[0]
            hig_flux = srcs['flux'] < fluxes_quartiles[1]

            best_srcs = srcs[best_big & best_flag & best_small & hig_flux & low_flux]

            if len(best_srcs) > 100:
                jj = np.random.choice(len(best_srcs), 100, replace=False)
                best_srcs = best_srcs[jj]

            print 'Sources good to calculate = {}'.format(len(best_srcs))
            self._best_sources = {'sources':best_srcs, 'fitshape':fitshape}

            indices = np.indices(self.bkg_sub_img.shape)
            Patch = []
            pos = []
            for row in best_srcs:
                position = [row['y'], row['x']]
                y = extract_array(indices[0], fitshape, position)
                x = extract_array(indices[1], fitshape, position)
                sub_array_data = extract_array(self.bkg_sub_img,
                                                fitshape, position,
                                                fill_value=self.bkg.globalrms)
                Patch.append(sub_array_data)
                pos.append(position)
            self._best_sources['patches'] = np.array(Patch)
            self._best_sources['positions'] = np.array(pos)
            self._best_sources['detected'] = srcs
        return self._best_sources


    def _covMat_from_stars(self):
        """
        Determines the covariance matrix for psf directly from the
        detected stars in the image
        """
        # calculate x, y, flux of stars
        best_srcs = self._best_srcs['sources']
        fitshape = self._best_srcs['fitshape']
        print 'Fitshape = {}'.format(fitshape)

        # best_srcs = best_srcs[best_srcs['flag']<=1]
        renders = self._best_srcs['patches']

        covMat = np.zeros(shape=(len(renders), len(renders)))

        for i in range(len(renders)):
            for j in range(len(renders)):
                if i<=j:
                    psfi_render = renders[i]
                    psfj_render = renders[j]

                    inner = np.vdot(psfi_render.flatten()/np.sum(psfi_render),
                                    psfj_render.flatten()/np.sum(psfj_render))

                    covMat[i, j] = inner
                    covMat[j, i] = inner
        return covMat

    def _covMat_psf(self):
        """
        Determines the covariance matrix for psf gaussian models fitted to
        detected stars in the image
        """
        fitted_models = self.fit_psf_sep()
        covMat = np.zeros(shape=(len(fitted_models), len(fitted_models)))

        renders = [g.render() for g in fitted_models]
        shapes = [np.array(r.shape) for r in renders]
        maxshape = np.array((np.max(shapes), np.max(shapes)))

        for i in range(len(fitted_models)):
            for j in range(len(fitted_models)):
                if i<=j:
                    psfi_render = renders[i]
                    psfj_render = renders[j]
                    shapei = maxshape - np.array(psfi_render.shape)
                    shapej = maxshape - np.array(psfj_render.shape)
                    psfi_render = np.pad(psfi_render,
                                    [[int(shapei[0]/2.), int(round(shapei[0]/2.))],
                                    [int(shapei[1]/2.), int(round(shapei[1]/2.))]],
                                    'edge')

                    psfj_render = np.pad(psfj_render,
                                    [[int(shapej[0]/2.), int(round(shapej[0]/2.))],
                                    [int(shapej[1]/2.), int(round(shapej[1]/2.))]],
                                    'edge')


                    inner = np.vdot(psfi_render.flatten()/np.sum(psfi_render),
                                    psfj_render.flatten()/np.sum(psfj_render))

                    covMat[i, j] = inner
                    covMat[j, i] = inner

                    renders[i] = psfi_render
                    renders[j] = psfj_render

        return [covMat, renders]

    @property
    def _kl_PSF(self, pow_th=0.99):
        """
        Determines the KL psf_basis from PSF gaussian models fitted to
        stars detected in the field.
        """
        if not hasattr(self, 'psf_KL_basis_model'):
            covMat, renders = self._covMat_psf()
            valh, vech = np.linalg.eigh(covMat)

            power = abs(valh)/np.sum(abs(valh))
            cum = 0
            cut = 0
            if not pow_th == 1:
                while cum <= pow_th:
                    cut -= 1
                    cum += power[cut]
            else:
                cut = -len(valh)

            #  Build psf basis
            N_psf_basis = abs(cut)
            lambdas = valh[cut:]
            xs = vech[:, cut:]
            # print cut, lambdas
            psf_basis = []
            for i in range(N_psf_basis):
                psf_basis.append(np.tensordot(xs[:, i], renders, axes=[0, 0]))

            self._psf_KL_basis_model = psf_basis

        return self._psf_KL_basis_model

    @property
    def _kl_from_stars(self, pow_th=0.99):
        """
        Determines the KL psf_basis from stars detected in the field.
        """
        if not hasattr(self, '_psf_KL_basis_stars'):
            covMat = self._covMat_from_stars()
            renders = self._best_srcs['patches']
            valh, vech = np.linalg.eigh(covMat)

            power = abs(valh)/np.sum(abs(valh))
            cum = 0
            cut = 0
            if not pow_th == 1:
                while cum <= pow_th:
                    cut -= 1
                    cum += power[cut]
            else:
                cut = -len(valh)

            #  Build psf basis
            N_psf_basis = abs(cut)
            lambdas = valh[cut:]
            xs = vech[:, cut:]
            # print lambdas
            psf_basis = []
            for i in range(N_psf_basis):
                psf_basis.append(np.tensordot(xs[:, i], renders, axes=[0, 0]))

            self._psf_KL_basis_stars = psf_basis

        return self._psf_KL_basis_stars

    @property
    def _kl_a_fields(self, pow_threshold=0.95, from_stars=True):
        """
        Calculate the coefficients of the expansion in basis of KLoeve.
        """
        if not hasattr(self, '_a_fields'):
            if from_stars:
                psf_basis = self._kl_from_stars

            else:
                psf_basis = self._kl_PSF

            N_fields = len(psf_basis)

            best_srcs = self._best_srcs['sources']
            fitshape = self._best_srcs['fitshape']
            patches = self._best_srcs['patches'][best_srcs['flag']<=1]
            positions = self._best_srcs['positions'][best_srcs['flag']<=1]
            best_srcs = best_srcs[best_srcs['flag']<=1]

            # Each element in patches brings information about the real PSF
            # evaluated -or measured-, giving an interpolation point for a

            a_fields = []
            for i in range(N_fields):
                p_i = psf_basis[i].flatten()
                p_i_sq = np.sum(np.dot(p_i, p_i))

                measures = []
                x = positions[:, 0]
                y = positions[:, 1]
                for a_patch in patches:
                    Pval = a_patch.flatten()
                    measures.append(np.dot(Pval, p_i)/p_i_sq)

                z = np.array(measures)
                # x_domain = [0, self.imagedata.shape[0]]
                # y_domain = [0, self.imagedata.shape[1]]
                a_field_model = models.Polynomial2D(degree=4)
                    # , x_domain=x_domain, y_domain=y_domain)
                fitter = fitting.LinearLSQFitter()
                a_fields.append(fitter(a_field_model, x, y, z))

            self._a_fields = a_fields
        return self._a_fields

    def get_variable_psf(self, delete_patches=False):
        a_fields = self._kl_a_fields
        psf_basis = self._kl_from_stars
        if delete_patches:
            del(self._best_srcs['patches'])
            print 'Patches deleted!'
        return [a_fields, psf_basis]

    @property
    def normal_image(self):
        if not hasattr(self, '_normal_image'):
            a_fields, psf_basis = self.get_variable_psf()
            x, y = np.mgrid[:self.imagedata.shape[0],
                            :self.imagedata.shape[1]]
            conv = np.zeros_like(self.bkg_sub_img)

            for i in range(len(a_fields)):
                a = a_fields[i]
                a = a(x, y)
                psf_i = psf_basis[i]
                conv += sg.fftconvolve(a, psf_i, mode='same')

            self._normal_image = conv
        return self._normal_image

    def s_component(self):
        var = self.meta['std']
        nrm = self.normal_image
        a_fields, psf_basis = self.get_variable_psf(delete_patches=True)
        mfilter = np.zeros_like(self.bkg_sub_img)
        x, y = np.mgrid[:mfilter.shape[0], :mfilter.shape[1]]

        for i in range(len(a_fields)):
            a = a_fields[i]
            psf = psf_basis[i]
            #cross = sg.fftconvolve(self._masked, psf, mode='same')
            cross = convolve_fft(self.bkg_sub_img, psf)
            # import ipdb; ipdb.set_trace()
            conv = np.multiply(a(x, y), cross)
            mfilter += conv

        mfilter = mfilter/nrm
        return mfilter/var**2



class ImageStats(object):
    """
    A class with methods for calculating statistics for an astronomical image.

    It includes the pixel matrix data, as long as some descriptions.
    For statistical values of the pixel matrix the class' methods need to be
    called.
    Parameters
    ----------
    image_obj : `~numpy.ndarray` or :class:`~ccdproc.CCDData`,
                `~astropy.io.fits.HDUList`  or a `str` naming the filename.
        The image object to work with

    dataformat : `str`
        optional dataformat of the image_object.
        Default to None, and a guessing attempt will be made.
    """
    def __init__(self, image_obj, dataformat=None):
        self._attached_to = repr(image_obj)
        self.full_description = {}

        if dataformat is None:
            try:
                dataformat = image_obj.__class__.__name__
            except:
                raise InputError('Dataformat not set nor guessable')

        if dataformat not in ('CCDData', 'fits_file', 'numpy_array', 'hdu',
                            'ndarray', 'str', 'HDUList'):
            raise InputError('Dataformat not recognized, try one of these \
            \n CCDData, fits_file, numpy_array, hdu')

        if dataformat == 'CCDData':
            self.pixmatrix = image_obj.data
            assert isinstance(self.pixmatrix, np.array)
        elif dataformat == 'fits_file' or dataformat == 'str':
            self.pixmatrix = fits.open(image_obj)[0].data
        elif dataformat == 'numpy_array' or dataformat == 'ndarray':
            self.pixmatrix = image_obj
            self.pixmatrix = np.ma.masked_array(self.pixmatrix,
                np.isnan(self.pixmatrix))
        elif dataformat == 'HDUList':
            self.pixmatrix = image_obj[0].data
        else:
            self.pixmatrix = image_obj[0].data

    def __repr__(self):
        return 'ImageStats instance for {}'.format(self._attached_to)

    def pix_sd(self):
        sd = self.pixmatrix.std()
        self.full_description['std'] = sd
        return sd

    def pix_median(self):
        m = np.median(self.pixmatrix)
        self.full_description['median'] = m
        return m

    def count_hist(self):
        h = stats.histogram(self.pixmatrix.flatten(), numbins=30)
        self.full_description['histogram'] = h
        return h

    def pix_mean(self):
        m = np.mean(self.pixmatrix)
        self.full_description['mean'] = m
        return m

    def to1d(self):
        self._oneDdata = self.pixmatrix.flatten()
        return

    def calc_stats(self):
        self.sd = self.pix_sd()
        self.median = self.pix_median()
        #self.hist = self.count_hist()
        self.mean = self.pix_mean()
        # self.to1d()
        return

    def summary(self):
        self.to1d()
        self.summ = stats.describe(self._oneDdata)
        # print self.summ
        return


def chunk_it(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return sorted(out, reverse=True)
