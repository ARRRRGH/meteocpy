# import torch
import numpy as np
from scipy.stats import norm
import warnings
from functools import partial
import itertools
from scipy.sparse import csr
import numba as nb

try:
    from utils import run_jobs, inds_from_slice2d, load_params, BiDict, chunk_list, _AttributeDict
except:
    from meteocpy.utils import run_jobs, inds_from_slice2d, load_params, BiDict, chunk_list, _AttributeDict


@nb.jit(nopython=True)
def convolve_non_aligned(inp, weights, indices):
    # create return array of shape (channels, px)
    conv = np.zeros((inp.shape[0], weights.shape[1]))
    # print(inp.shape, weights.shape, len(indices))
    for i, ind in enumerate(indices):
        s, e = ind
        # allow for broadcasts in px dimension
        j = min(inp.shape[1] - 1, i)
        # print((inp[..., j, :e - s] * weights[:, i, s:e]).shape)
        conv[:, i] = (inp[:, j, :e - s] * weights[:, i, s:e]).reshape(-1)

    return conv


class ApexSensorClass(object):
    DIM_X_AX = 1000

    FWHM_TO_SIGMA = 0.8493218 / 2
    SIGMA_TO_FWHM = 1 / FWHM_TO_SIGMA

    N_SWIR = 199
    N_VNIR_BINNED = 117
    N_VNIR_UNBINNED = 337

    DIM_BANDS_AX_BINNED = N_VNIR_BINNED + N_SWIR
    DIM_BANDS_AX_UNBINNED = N_VNIR_UNBINNED + N_SWIR

    _param_keys = ['cw', 'fwhm', 'rad_coeffs', 'snr_coeffs', 'dc_coeffs', 'adc_coeffs', 'ng4_transmission',
                   'integration_times']

    _model_keys = ['srfs', 'wvls', 'initialized_support', 'srf_support_per_band', 'srf_bands', 'res',
                   'abs_res', 'start_band']

    def __init__(self, cw, fwhm, rad_coeffs, snr_coeffs, dc_coeffs, adc_coeffs,
                 binning_pattern, ng_transmission=None, dt=0.006, vnir_it=117, swir_it=199):

        assert self.check_coeffs(rad_coeffs, ['gain', 'offset'])
        self.rad_coeffs = rad_coeffs

        assert self.check_coeffs(snr_coeffs, ['c_low', 'c_high', 'max_L_low', 'lowest_radiance',
                                              'highest_radiance', 'c_low_frame', 'c_high_frame'])
        self.snr_coeffs = snr_coeffs

        assert self.check_coeffs(dc_coeffs, ['gain', 'offset'])
        self.dc_coeffs = dc_coeffs

        assert self.check_coeffs(adc_coeffs)
        self.adc_coeffs = adc_coeffs

        self.fwhm = fwhm
        self.cw = cw
        self.ng4_transmission = ng_transmission.flatten()

        # TODO: this is a fix!! be sure you understand the binning pattern
        binning_pattern[0] -= 1
        self.binning_pattern = np.r_[binning_pattern.flatten().astype(np.int), np.ones(self.N_SWIR)].astype(np.int)

        # bins lists the bin index for each band, bins.inverse lists the band index for each bin index
        _cumsum = np.cumsum(self.binning_pattern)
        self.bins = BiDict()
        for i in range(self.DIM_BANDS_AX_UNBINNED):
            self.bins[i] = np.where(i < _cumsum)[0][0]

        init_mode = 'binned' if self.is_binned else 'unbinned'

        self.dt = dt
        self.vnir_it = vnir_it / 1000
        self.swir_it = swir_it / 1000

        # throw a warning if unbinned vnir was loaded but meta is binned
        if not self.is_binned and dc_coeffs['gain'].shape[0] == self.DIM_BANDS_AX_BINNED:
            warnings.warn('WARNING: you loaded unbinned calibration files but meta files are binned. Make sure to'
                          'call initialize_srfs with kwarg do_bin=True.')

        self.params = _AttributeDict({'binned': _AttributeDict(), 'unbinned': _AttributeDict()})
        self.params[init_mode] = _AttributeDict({'cw': self.cw, 'fwhm': self.fwhm})

        self.params.binned['integration_times'] = np.r_[np.ones(self.N_VNIR_BINNED) * self.vnir_it,
                                                        np.ones(self.N_SWIR) * self.swir_it]
        self.params.unbinned['integration_times'] = np.r_[np.ones(self.N_VNIR_UNBINNED) * self.vnir_it,
                                                          np.ones(self.N_SWIR) * self.swir_it]

        for key, val in [('rad_coeffs', self.rad_coeffs), ('snr_coeffs', self.snr_coeffs),
                         ('dc_coeffs', self.dc_coeffs), ('adc_coeffs', self.adc_coeffs),
                         ('ng4_transmission', self.ng4_transmission)]:
            bkey = self.get_binned_type(val)
            self.params[bkey][key] = val

        self.model = _AttributeDict({'binned': _AttributeDict(), 'unbinned': _AttributeDict()})

    def get_binned_type(self, obj):
        if hasattr(obj, 'shape') and obj.shape[0] == self.DIM_BANDS_AX_BINNED:
            return 'binned'
        elif hasattr(obj, 'shape'):
            return 'unbinned'
        elif type(obj) is dict:
            return self.get_binned_type(list(obj.values())[0])

        else:
            raise ValueError

    def reset(self):
        self.model = None

    def get(self, key, binned=None):
        binned = True if binned is None else binned
        bkey = self.is_binned_as_str(binned)

        if key in self._param_keys:
            if bkey == 'binned' and bkey not in self.params:
                self.compute_binned_params()
            if bkey == 'unbinned' and bkey not in self.params:
                raise ValueError('Only binned bands are available. Cannot calculate unbinned from binned bands.')
            return self.params[bkey][key]
        elif key in self._model_keys:
            return self.model[bkey][key]
        else:
            raise ValueError('Key must be in _model_keys or _param_keys')

    def is_binned_as_str(self, binned):
        return 'binned' if binned else 'unbinned'

    def get_px_in_bands_ax(self, binned=None):
        if binned is None:
            binned = self.is_binned

        return self.DIM_BANDS_AX_BINNED if binned else self.DIM_BANDS_AX_UNBINNED

    def get_n_vnir(self, binned=None):
        if binned is None:
            binned = self.is_binned

        return self.N_VNIR_BINNED if binned else self.N_VNIR_UNBINNED

    @property
    def is_binned(self):
        return len(self.cw) == self.DIM_BANDS_AX_BINNED

    def check_coeffs(self, coeffs, names=[]):
        for name in names:
            if name not in coeffs:
                return False
        return True

    def shift_bands_to_local(self, bands, binned=True):
        bands = bands.copy()
        if self.get('start_band', binned) is not None:
            bands -= self.get('start_band', binned)
        return bands

    def get_bands(self, ext_bands=None, in_bands=None, binned=True):
        if ext_bands is not None and in_bands is not None:
            return ext_bands, in_bands
        elif ext_bands is not None and in_bands is None:
            return ext_bands, self.shift_bands_to_local(ext_bands, binned=binned)
        elif ext_bands is None and in_bands is not None:
            return self.shift_bands_from_local(in_bands, binned=binned), ext_bands
        else:
            return slice(None, None)

    def shift_bands_from_local(self, bands, binned=True):
        bands = bands.copy()
        if self.get('start_band', binned) is not None:
            bands += self.get('start_band', binned)
        return bands

    def check_srfs_initialized(self, binned=True):
        return self.get('srfs', binned) is not None

    def compute_binned_params(self):
        # binn_patt = np.r_[0, self.binning_pattern]
        # cw = np.mean.reduceat(self.get('cw', False), binn_patt)
        # fwhm = np.max.reduceat(self.get('fwhm', False) + cw, binn_patt) - np.min.reduceat(self.get('fwhm', False)
        #  + cw,binn_patt)
        #
        raise NotImplementedError

    def bin_bands(self, unbinned, ext_bands=None, ufunc=np.add, axis=1):
        if ext_bands is None:
            ext_bands = np.arange(self.DIM_BANDS_AX_UNBINNED)

        # make sure bins aren't only partially covered
        orig_ext_bands = ext_bands.copy()
        ext_bands, bin_index = self.extend_ext_bands(ext_bands, return_bin_index=True)
        # assert np.sum(np.abs(ext_bands - orig_ext_bands)) == 0

        bins = np.cumsum(np.r_[0, self.binning_pattern])

        # if the next index would be larger than unbinned.shape[axis] reduce till end
        end_index = bins[bin_index[-1] + 1]
        if end_index >= unbinned.shape[axis]:
            bins = bins[bin_index] - ext_bands[0]  # shift index to local coordinate system
            ret = ufunc.reduceat(unbinned, bins, axis=axis)

        # need to put a break to reducer at end, exclude the added element
        else:
            bins = np.r_[bins[bin_index], end_index] - ext_bands[0]  # shift index to local coordinate system
            ret = ufunc.reduceat(unbinned, bins, axis=axis)[tuple([slice(None, None)
                                                                   if ax != axis else slice(None, -1)
                                                                   for ax in range(len(unbinned.shape))])]
        return ret

    def check_inp_spectrum_consistency(self, inp_spectrum, inp_wvlens, binned=True, tol=1e-12):
        for inp_wvl, inp_spe in zip(inp_wvlens, inp_spectrum):

            # Check resolution / shape of input
            if self.get('abs_res', binned) is None:
                if not inp_spe.shape[-1] == self.get('srfs', binned).shape[-1]:
                    # raise Exception(('Input Spectrum length invalid. Must supply array of length %d' +
                    #                 ' or supply the input support.') % inp_spectrum.shape[-1])
                    pass

            else:
                if len(inp_wvl) > 1 and np.abs(inp_wvl[1] - inp_wvl[0] - self.get('abs_res', binned)) > tol:
                    raise Exception(('Input spectrum has resolution %f. But you calculated SRFs at ' +
                                     'resolution %f.') % (inp_wvl[1] - inp_wvl[0], self.get('abs_res', binned)))

            # Check support of input is entirely covered by precalculated srfs
            if inp_wvl[0] < self.get('initialized_support', binned)[0] \
                    or inp_wvl[-1] > self.get('initialized_support', binned)[-1]:
                raise Exception('You calculated SRFs for (%f, %f). The input spectrum is out of bounds.'
                                % (self.get('initialized_support', binned)[0],
                                   self.get('initialized_support', binned)[-1]))

        return True

    def get_2d_ranges(self, ranges, res=None, abs_res=None, exact_wvls=None, shape=None, zero_out=True):
        """
        :param zero_out: if calculates with abs_res, whether to zero out the additional space that might be created
        :param ranges: (:, 2)
        :param res:
        :param abs_res:
        :param shape:
        :return: (:, res) or (*shape, res)
        """
        assert res is not None or abs_res is not None or exact_wvls is not None

        lo, hi = zip(ranges.transpose())
        lo, hi = lo[0], hi[0]  # 0 because of zip packing

        def is_in_range(p, mins, maxs):
            return np.logical_and(p > mins.transpose(), p < maxs.transpose())

        if res is not None:
            ranges = np.linspace(lo, hi, res).transpose()
            inds = None
            support_per_band_ext = None

        elif abs_res is not None:
            lens = (hi - lo) // abs_res
            maxlen = np.max(lens).astype(np.int)

            diff_len = ((maxlen - lens) // 2).astype(np.int)
            diff = diff_len * abs_res

            lo_new = lo - diff
            hi_new = hi + diff
            support_per_band_ext = np.stack([lo_new, hi_new], axis=1)

            ranges = np.linspace(lo_new, hi_new, maxlen).transpose()

            if zero_out:
                inds = inds_from_slice2d(diff_len, diff_len, axis=1, end_is_rev=True)

            else:
                inds = None

        # if exact_wvls is supplied
        else:
            @nb.jit(no_python=True)
            def _iterate_range(wvls, lo, hi):
                ranges = []
                for i in range(len(lo)):
                    ranges.append(exact_wvls[np.logical_and(exact_wvls > lo[i], exact_wvls < hi[i])])

                return ranges

            ranges = _iterate_range(exact_wvls, lo, hi)
            inds = None
            support_per_band_ext = None

        if shape is not None:
            ranges = ranges.reshape(*shape, -1)

        return ranges, inds, support_per_band_ext

    def get_support_per_band(self, support):
        """

        :param support : (bands, xtrack, 2)
        :return: (bands, 2)
        """
        lo_support_per_band = np.min(support[..., 0], axis=1)
        hi_support_per_band = np.max(support[..., 1], axis=1)
        return np.stack([lo_support_per_band, hi_support_per_band], axis=1)

    def get_illuminated_bands(self, inp_support, part_covered=False, support_per_band=None, to_local=True, binned=True):
        """
        Get bands that are considered illuminated. If part_covered == False, return bands with mean cw inside inp_support,
        else return all bands that are at least partly covered by inp_support.

        :param part_covered:
        :param inp_support: (2, )
        :param support_per_band : (bands, 2), only needed with part_covered == True
        :return:
        """
        min_, max_ = inp_support[0], inp_support[-1]
        if part_covered:
            touched = np.where(np.logical_and(support_per_band[:, 0] < max_, support_per_band[:, 1] > min_))[0]
        else:
            mean = np.mean(self.get('cw', binned), axis=1)
            touched = np.where(np.logical_and(min_ < mean, mean < max_))[0]

        if to_local:
            touched = self.shift_bands_to_local(touched, binned=binned)

        # there is an overlapping region, in order to prevent non-contiguous ext_bands
        touched = np.arange(np.min(touched), np.max(touched) + 1)

        return touched

    def extend_ext_bands(self, ext_bands, return_bin_index=False):
        bin_index = np.unique([self.bins[i] for i in ext_bands])
        ext_bands = np.sort(np.unique(np.concatenate([self.bins.inverse[i] for i in bin_index])))

        if return_bin_index:
            return ext_bands, bin_index

        return ext_bands

    def compute_srfs(self, support, res=None, abs_res=None, exact_wvls=None, ext_bands=None, zero_out=True, binned=True,
                     do_bin=False):
        """

        :param abs_res:
        :param res:
        :param support: (bands, xtrack)
        :param ext_bands: only the indices in bands are considered
        :return: (bands, xtrack, res)
        """
        assert res is not None or abs_res is not None or exact_wvls is not None
        do_bin = not binned and do_bin

        # TODO: at the moment the function filters ext_bands, this could be extended to pixels
        if ext_bands is None:
            ext_bands = range(self.get_px_in_bands_ax(binned=binned))

        # extend support of bands to cover whole bin
        if do_bin:
            # make sure ext_bands does not cover partially a bin such that we get true srf_support of binned bands
            ext_bands, bin_index = self.extend_ext_bands(ext_bands, return_bin_index=True)

            # calculate support of bands to be binned
            lo_per_px = self.bin_bands(support[ext_bands, :, 0], axis=0, ufunc=np.minimum, ext_bands=ext_bands)
            hi_per_px = self.bin_bands(support[ext_bands, :, 1], axis=0, ufunc=np.maximum, ext_bands=ext_bands)

            for i, ind in enumerate(bin_index):
                support[self.bins.inverse[ind], :, 0] = lo_per_px[i]
                support[self.bins.inverse[ind], :, 1] = hi_per_px[i]

        # select bands
        support_per_px = support[ext_bands].reshape(-1, 2)

        # compute the ranges for the given spacing (abs_res) or with the given sampling (res)
        if abs_res is not None:
            wvls, zero_out_inds, ext_support = self.get_2d_ranges(ranges=support_per_px, abs_res=abs_res,
                                                                  zero_out=zero_out)
        elif res is not None:
            wvls, zero_out_inds, _ = self.get_2d_ranges(ranges=support_per_px, res=res)

        elif exact_wvls is not None:
            wvls, _, _ = self.get_2d_ranges(ranges=support_per_px, exact_wvls=exact_wvls)
        else:
            raise Exception('Must supply abs_res, res or exact_wvls.')

        # compute the model for all pixels in selection
        if abs_res is not None or res is not None:
            srfs = self.srf_model(wvls,
                                  loc=self.get('cw', binned)[ext_bands].reshape(-1, 1),
                                  scale=self.get('fwhm', binned)[ext_bands].reshape(-1, 1) * self.FWHM_TO_SIGMA)

            # reshape pixels to rectangle
            wvls = wvls.reshape(len(ext_bands), self.DIM_X_AX, -1)

            # calculate step size for normalization
            if abs_res is not None:
                step_size = abs_res
            else:
                step_size = (wvls[..., 1] - wvls[..., 0])[..., None]

            # set srfs to zero outside chosen ranges such that we have the same (comparable) coverage everywhere
            if zero_out_inds is not None:
                srfs[zero_out_inds] = 0

            srfs = srfs.reshape(len(ext_bands), self.DIM_X_AX, -1)
            support = support_per_px.reshape(len(ext_bands), self.DIM_X_AX, 2)

            if do_bin:
                srfs = self.bin_bands(srfs, ext_bands=ext_bands, axis=0)

                # wvls is the same for all bands in a bin so minimum chooses one out of many equal
                wvls = self.bin_bands(wvls, ext_bands=ext_bands, axis=0, ufunc=np.minimum)

                # for the support use reduced versions from above
                support = np.stack([lo_per_px, hi_per_px], axis=1).reshape(srfs.shape[0], self.DIM_X_AX, -1)

                if res is not None:
                    warnings.warn('Binning with variable resolution. Provide abs_res not None.')
                    # step_size is the same for all bands in a bin (bc wvls is) so minimum chooses one out of many equal
                    step_size = self.bin_bands(step_size, ext_bands=ext_bands, axis=0, ufunc=np.minimum)

                # return ext_bands in binned indices which are the bin inds
                ext_bands = np.unique([self.bins[i] for i in ext_bands])

            srfs = srfs / np.sum(srfs, axis=-1)[..., None] / step_size
            return srfs, wvls, ext_bands, support

        else:
            srfs = [self.srf_model(wvls[i],
                           loc=self.get('cw', binned)[ext_bands[i]],
                           scale=self.get('fwhm', binned)[ext_bands[i]] * self.FWHM_TO_SIGMA)
            for i in range(len(ext_bands))]

            return srfs, wvls, ext_bands

    def initialize_srf_support(self, sigma, binned=True):
        """

        :param sigma : (int)
        :return: (bands, xtrack, 2)
        """
        range_ = self.get('fwhm', binned) * (self.SIGMA_TO_FWHM / 2) * sigma
        lo_support = self.get('cw', binned) - range_
        hi_support = self.get('cw', binned) + range_

        return np.stack([lo_support, hi_support], axis=2)

    def initialize_srfs(self, inp_support=None, res=None, abs_res=None, exact_wvls=None, srf_support_in_sigma=1.0,
                        part_covered=True, zero_out=True, do_bin=True):
        binned = self.is_binned
        do_bin = not binned and do_bin

        # get +- srf_support_in_sigma for each pix
        srf_support = self.initialize_srf_support(srf_support_in_sigma, binned=binned)

        # get min/max of srf_support for each band
        support_per_band = self.get_support_per_band(srf_support)

        if inp_support is None:
            inp_support = exact_wvls
        # determine which bands are covered by inp_support
        illu_bands = self.get_illuminated_bands(inp_support, part_covered=part_covered,
                                                support_per_band=support_per_band, to_local=False, binned=binned)

        # compute srf for all pixels that are in an illuminated band
        srfs, wvls, illu_bands, srf_support = self.compute_srfs(support=srf_support, ext_bands=illu_bands, res=res,
                                                                abs_res=abs_res, exact_wvls=exact_wvls, zero_out=zero_out,
                                                                binned=binned, do_bin=do_bin)

        support_per_band = self.get_support_per_band(srf_support)

        # save model to correct binned/unbinned
        bkey = self.is_binned_as_str(binned or do_bin)
        self.model[bkey] = _AttributeDict({'srfs': srfs.reshape(1, *srfs.shape),  # add a channel dimension
                                           'wvls': wvls,
                                           'initialized_support': inp_support,
                                           'srf_support_per_band': support_per_band,
                                           'srf_bands': illu_bands,
                                           'res': res,
                                           'abs_res': abs_res,
                                           'start_band': illu_bands[0]})

        # create binned params if we did bin
        if do_bin:
            cw = self.bin_bands(self.params.unbinned.cw, axis=0) / self.binning_pattern[:, None]
            fwhm = self.bin_bands(self.params.unbinned.fwhm, axis=0) / self.binning_pattern[:, None]

            # TODO: how should other parameters be binned?
            self.params.binned.update({'cw': cw, 'fwhm': fwhm})

    # @profile
    def convolve_srfs(self, inp_spectrum, in_bands, inp_wvlens, tol=0.5, check_tol=True, binned=True,
                      conv_mode='numba'):
        """

        :param check_tol: check whether input spectrum fits the SRFs within tolerance tol
        :param tol:
        :param inp_wvlens:
        :param in_bands: (bands, )
        :param inp_spectrum: (channel, pix, res)
        :return: (channel, bands, x_track)
        """
        # flatten 2d band-xtrack structure
        srfs = self.get('srfs', binned)[:, in_bands, ...].reshape(self.get('srfs', binned).shape[0], -1,
                                                                  self.get('srfs', binned).shape[-1])
        wvls = self.get('wvls', binned)[in_bands, ...].reshape(-1, self.get('wvls', binned).shape[-1])

        # find closest start_wvl
        start_ind = np.argmin(np.abs(wvls - inp_wvlens[0]), axis=-1)
        end_ind = np.clip(start_ind + inp_spectrum.shape[-1], a_min=None, a_max=srfs.shape[-1])

        if np.any(end_ind > self.get('srfs', binned).shape[-1]):
            warnings.warn('WARNING: the input spectrum overlaps at least one SRF. Choose larger SRF support.')

        if check_tol:
            min_diffs = np.take_along_axis(wvls, start_ind[:, None], axis=-1)
            if np.any(min_diffs < tol):
                raise Exception(('Difference between input spectrum wave length and SRF wave length is larger than' +
                                 'tol=%d') % tol)

        # THIS IS TOO MEMORY INTENSIVE
        # shape : (channels, xtrack * bands, wvls)
        if conv_mode == 'entire':
            inp_spectrum_arr = np.zeros(tuple([inp_spectrum.shape[0]] + list(srfs.shape[1:])))
            inds = np.array([range(s, e) for s, e in zip(start_ind, end_ind)])[None, :]  # add channel dimension
            np.put_along_axis(inp_spectrum_arr, inds, inp_spectrum, axis=-1)
            return self.convolve(weights=srfs, inp=inp_spectrum_arr).reshape(-1, len(in_bands), self.DIM_X_AX)

        # THIS IS TOO SLOW
        # print(inp_spectrum.shape)
        # for i, (s, e) in enumerate(zip(start_ind, end_ind)):
        #     inp_spectrum_arr[..., s:e] = inp_spectrum[..., :e-s]
        # 
        # return self.convolve(weights=srfs, inp=inp_spectrum_arr).reshape(-1, len(in_bands), self.DIM_X_AX)

        # SO USE NUMBA FOR THE CONVOLUTION
        if conv_mode == 'numba':
            return convolve_non_aligned(inp=inp_spectrum, weights=srfs,
                                        indices=np.asarray(list(zip(start_ind, end_ind))))\
                                        .reshape(-1, len(in_bands), self.DIM_X_AX)

    # @profile
    def convolve(self, inp, weights):
        return np.sum(weights * inp, axis=-1)

    def forward(self, inp_spectrum, inp_wvlens, part_covered=True, tol=0.5, pad=False, ng4=False, invert=True,
                snr=True, dc=True, smear=True, run_with_binned=True, return_binned=False, run_specs=None,
                run_specs_inner=None, conv_mode='numba', *args, **kwargs):
        """

        :param inp_spectrum: (batches, channels, spectrum), computations are threaded along batches, all spectra in a
                             channel are computed in vectorized form. This means they should not differ in their wavelength
                             support, but they may differ in intensity. If only one batch is supplied, it is broadcast to
                             all supplied inp_wvlens.
        :param inp_wvlens: Support for each batch. If only one inp_wvlens is supplied it is broadcast to all batches.

        :param part_covered:
        :param tol:
        :param args:
        :param kwargs:
        :return:
        """
        binned = run_with_binned

        inp_wvlens = np.atleast_2d(inp_wvlens)
        # reshape input spectrum
        if len(inp_spectrum.shape) == 2:
            # we assume (batch, wvl)
            inp_spectrum = inp_spectrum[:, None, None, ...]
        elif len(inp_spectrum.shape) == 3:
            # we assume (batch, channel, wvl)
            inp_spectrum = inp_spectrum[:, :, None, ...]
        elif len(inp_spectrum.shape) == 4:
            # we assume (batch, channel, pix, wvl)
            pass
        elif len(inp_spectrum.shape) == 5:
            # we assume (batch, channel, band, xtrack, wvl)
            inp_spectrum = inp_spectrum.reshape(inp_spectrum.shape[0], inp_spectrum.shape[1], -1,
                                                inp_spectrum.shape[-1])
        else:
            raise Exception('Input spectrum has wrong shape.')

        if self.get('res', binned) is not None:
            warnings.warn('WARNING: calculates convolution at different resolutions.')

        if not return_binned and 'unbinned' not in self.params:
            raise Exception('Unbinned params are not available.')

        assert self.check_srfs_initialized(binned=binned)
        assert self.check_inp_spectrum_consistency(inp_spectrum, inp_wvlens, binned=binned)

        # Determine how many batches per job and prepare run_specs
        if run_specs is None:
            run_specs = {}

        if 'batches_per_job' not in run_specs:
            batches_per_job = 1000
        else:
            batches_per_job = run_specs['batches_per_job']
            run_specs = {k: v for k, v in run_specs.items() if k != 'batches_per_job'}

        if run_specs_inner is None:
            run_specs_inner = dict(joblib=False)

        # broadcast, if only one inp_wvls, assume is same inp_wvls for all inp_spectra in batches
        if len(inp_wvlens) == 1 and len(inp_spectrum) > 1:
            inp_wvlens = [inp_wvlens[0]] * len(inp_spectrum)

        # broadcast, if only one inp_spectrum assume is same for all inp_wvls
        if len(inp_wvlens) > 1 and len(inp_spectrum) == 1:
            inp_spectrum = [inp_spectrum[0]] * len(inp_wvlens)

        # define jobs
        job_inp_spectra = chunk_list(inp_spectrum, batches_per_job)
        job_inp_wvls = chunk_list(inp_wvlens, batches_per_job)
        jobs = [partial(self._forward,
                        inp_spectrum=inp_s, inp_wvlens=inp_w,
                        binned=binned,
                        part_covered=part_covered,
                        tol=tol, pad=pad,
                        ng4=ng4,
                        invert=invert, snr=snr, dc=dc, smear=smear,
                        return_binned=return_binned,
                        conv_mode=conv_mode,
                        run_specs=run_specs_inner, *args, **kwargs)
                for inp_s, inp_w in zip(job_inp_spectra, job_inp_wvls)]

        # flatten out job dimension such that we have (batch, channel, band, xdir)
        res, illu_bands = zip(*run_jobs(jobs, **run_specs))
        res = list(itertools.chain(*res))
        illu_bands = list(itertools.chain(*illu_bands))

        return res, illu_bands

    # @profile
    def _forward(self, inp_spectrum, inp_wvlens, binned, part_covered=True, tol=0.5, pad=False, ng4=False, invert=True,
                 snr=True, dc=True, smear=True, return_binned=False, run_specs={}, conv_mode='numba', *args, **kwargs):

        # ## 0 PREPARATION #############################################################################################
        # determine which bands are illuminated by the input spectrum
        in_illu_bands_per_batch = []
        for support in inp_wvlens:
            # need not put to_local since support_per_band is already local
            in_illu_bands_per_batch.append(self.get_illuminated_bands(inp_support=support,
                                                                      support_per_band=self.get('srf_support_per_band',
                                                                                                binned=binned),
                                                                      part_covered=part_covered,
                                                                      binned=binned,
                                                                      to_local=False))

        # calculate absolute band index
        ext_illu_bands_per_batch = [self.shift_bands_from_local(in_illu_bands, binned=binned)
                                    for in_illu_bands in in_illu_bands_per_batch]

        # ## 1 CONVOLUTION #############################################################################################
        # convolve all illuminated bands, iterate over batches
        jobs = []
        for i, in_illu_bands in enumerate(in_illu_bands_per_batch):
            inp_wvl = inp_wvlens[i] if len(inp_wvlens) > 1 else inp_wvlens[0]
            jobs.append(partial(self.convolve_srfs, inp_spectrum=inp_spectrum[i], inp_wvlens=inp_wvl,
                                in_bands=in_illu_bands, tol=tol, binned=binned, conv_mode=conv_mode))

        res = run_jobs(jobs, **run_specs)

        # ## 2 NG TRANSMISSION #########################################################################################
        if ng4:
            res = [np.einsum('cij, i -> cij', frame, 1 / self.get('ng4_transmission', binned)) for frame in res]

        # ## 3 INVERSION TO DNs ########################################################################################
        if invert:
            # convert to DNs using a custom model defined in self.inversion
            jobs = [partial(self.inversion, frame, ext_bands=ext_illu_bands, binned=binned)
                    for frame, ext_illu_bands in zip(res, ext_illu_bands_per_batch)]

            res = run_jobs(jobs, **run_specs)

            # multiply times integration times
            res = [np.einsum('ci..., i -> ci...', dns, self.get('integration_times', binned)[ext_illu_bands])
                   for dns, ext_illu_bands in zip(res, ext_illu_bands_per_batch)]

        # ## 4 SENSOR NOISE ############################################################################################
        # add SNR noise using a custom model defined in self.snr_model
        if snr:
            jobs = [partial(self.snr_model, dns, ext_bands=ext_illu_bands, binned=binned)
                    for dns, ext_illu_bands in zip(res, ext_illu_bands_per_batch)]

            noise = run_jobs(jobs, **run_specs)
            res = [r + n for r, n in zip(res, noise)]

        # ## 5 DC MODEL ################################################################################################
        # add noisy DC using a custom model defined in self.dc_model
        if dc:
            jobs = [partial(self.dc_model, dns, ext_bands=ext_illu_bands, binned=binned)
                    for dns, ext_illu_bands in zip(res, ext_illu_bands_per_batch)]

            noise = run_jobs(jobs, **run_specs)
            res = [r + n for r, n in zip(res, noise)]

        # ## 6 SMEARING ################################################################################################
        if smear:
            jobs = [partial(self.smear, dns, binned=binned, ext_bands=ext_illu_bands)
                    for dns, ext_illu_bands in zip(res, ext_illu_bands_per_batch)]

            res = run_jobs(jobs, **run_specs)

        # ## 7 BINNING #################################################################################################
        if return_binned and not binned:
            if self.get('res', binned) is not None:
                raise Exception('Binning requires SRFs with absolute resolution (abs_res != None).'
                                'Cannot bin SRFs with different spectral sampling (res != None).')
            else:
                raise NotImplementedError
                # res = [self.bin_bands(r, ext_illu_bands, axis=1)
                #        for r, ext_illu_bands in zip(res, ext_illu_bands_per_batch)]

        # ## 8 PADDING #################################################################################################
        if pad:
            lo_pad = [max(0, ext_illu_bands[0] - 1) for ext_illu_bands in ext_illu_bands_per_batch]
            hi_pad = [self.get_px_in_bands_ax(binned) - ext_illu_bands[-1] for ext_illu_bands in
                      ext_illu_bands_per_batch]
            res = [np.pad(sm, ((0, 0), (lo, hi), (0, 0))) for sm, lo, hi in zip(res, lo_pad, hi_pad)]

        return res, ext_illu_bands_per_batch

    def inversion(self, rad, ext_bands=None, binned=True):
        if ext_bands is None:
            ext_bands = slice(None, None)
        #  - self.get('rad_coeffs', binned)['offset'][ext_bands]
        return (rad - self.get('rad_coeffs', binned)['offset'][ext_bands]) \
               / self.get('rad_coeffs', binned)['gain'][ext_bands]

    def snr_model(self, dn, ext_bands=None, binned=True):
        if ext_bands is None:
            ext_bands = range(self.get_px_in_bands_ax(binned))

        snr = np.zeros_like(dn)

        lo_dn_mask = dn < self.get('snr_coeffs', binned)['max_L_low'][:, ext_bands].reshape(1, -1, 1)
        hi_dn_mask = np.logical_not(lo_dn_mask)

        lo_frame = self.get('snr_coeffs', binned)['c_low_frame'][ext_bands]
        c_low_frame = np.repeat(lo_frame[None, ...], lo_dn_mask.shape[0], axis=0)

        hi_frame = self.get('snr_coeffs', binned)['c_high_frame'][ext_bands]
        c_high_frame = np.repeat(hi_frame[None, ...], hi_dn_mask.shape[0], axis=0)

        if np.any(lo_dn_mask):
            snr[lo_dn_mask] = np.sum([c_low_frame[..., i][lo_dn_mask] * dn[lo_dn_mask] ** (2 - i)
                                      for i in range(3)])

        # check whether there are ext_bands in higher part
        if np.any(hi_dn_mask):
            snr[hi_dn_mask] = np.sum([c_high_frame[..., i][hi_dn_mask] * dn[hi_dn_mask] ** (1 - i)
                                      for i in range(2)])

        snr = np.abs(snr)
        noise = dn / snr * np.random.normal(size=(snr.shape[0], snr.shape[1], self.DIM_X_AX))
        return noise

    def dc_model(self, dn, noise_scale=5, ext_bands=None, binned=True):
        if ext_bands is None:
            ext_bands = range(self.get_px_in_bands_ax(binned))

        ext_bands, in_bands = self.get_bands(ext_bands=ext_bands, binned=binned)

        dc = self.get('dc_coeffs', binned)['gain'][ext_bands] * self.get('integration_times', binned)[in_bands][:, None] \
             + self.get('dc_coeffs', binned)['offset'][ext_bands]
        return dc + np.random.normal(size=(dn.shape[0], len(ext_bands), self.DIM_X_AX)) * noise_scale

    def smear(self, res, binned=True, ext_bands=None):
        # if binned need first to unbin
        if binned:
            # TODO: why is smearing only in VNIR?
            vnir_bands = np.where(ext_bands < self.N_VNIR_BINNED)[0]
            swir_bands = np.where(ext_bands >= self.N_VNIR_BINNED)[0]
            unbinned_vnir, ext_vnir_bands_unb = self.unbin(res[:, vnir_bands], ext_bands=ext_bands[vnir_bands])

            # there are only vnir bands
            if len(swir_bands) == 0 and len(vnir_bands) > 0:
                res = unbinned_vnir
                ext_bands = ext_vnir_bands_unb

            # it's in the overlapping region
            elif len(swir_bands) > 0 and len(vnir_bands) > 0:
                res = np.concatenate([unbinned_vnir, res[:, swir_bands]], axis=1)
                ext_bands = np.concatenate([ext_vnir_bands_unb,
                                            ext_bands[swir_bands] - self.N_VNIR_BINNED + self.N_VNIR_UNBINNED])

            # there are only swir bands
            else:
                pass

        vnir_bands = np.where(ext_bands < self.N_VNIR_UNBINNED)[0]
        if len(vnir_bands) == 0:
            return res

        # TODO: only vnir is smeared?
        # res = integration_time / dt * drad
        d_rad = np.einsum('b, cbx -> cbx', 1 / self.get('integration_times', binned=False)[ext_bands[vnir_bands]],
                          res[:, vnir_bands]) * self.dt

        # forward
        adds = np.cumsum(d_rad[:, vnir_bands][:, ::-1], axis=1) - d_rad[:, [-1]]

        # add smear to DNs
        res[:, vnir_bands] += 2 * adds
        res = self.bin_bands(res, ext_bands=ext_bands, axis=1)
        return res

    def srf_model(self, *args, **kwargs):
        return norm.pdf(*args, **kwargs)

    def unbin(self, res, ext_bands=None):
        new_ext_bands = np.concatenate([self.bins.inverse[band] for band in ext_bands])

        lens = [len(self.bins.inverse[band]) for band in ext_bands]
        edges = np.cumsum(lens)
        ret = np.zeros((res.shape[0], edges[-1], res.shape[2]))

        last_edge = 0
        for i, (edge, l) in enumerate(zip(edges, lens)):
            ret[:, last_edge:edge, :] = res[:, [i], :] / l
            last_edge = edge

        return ret, new_ext_bands


# def load_apex(calibration_path, meta_path, *args, **kwargs):
#     return ApexSensorClass(*args, **kwargs, **load_params(calibration_path, meta_path))


def load_apex(binned_vnir_swir=None, unbinned_vnir_swir=None, unbinned_vnir=None, binned_meta=None, unbinned_meta=None,
              swir=None, unbinned_complete=None, *args, **kwargs):
    # load binned vnir and swir
    if binned_vnir_swir is not None and unbinned_vnir is None:
        return ApexSensorClass(*args, **kwargs, **load_params(calibration_path=binned_vnir_swir,
                                                              meta_path=binned_meta))
    # load unbinned vnir and swir, merge
    elif binned_vnir_swir is not None and unbinned_vnir is not None:

        meta = unbinned_meta if unbinned_meta is not None else binned_meta
        meta = load_params(meta_path=meta)
        params_vnir_swir = load_params(calibration_path=binned_vnir_swir)
        params_vnir = load_params(calibration_path=unbinned_vnir)

        params_vnir['cw'] = np.concatenate([params_vnir['cw'],
                                            params_vnir_swir['cw'][ApexSensorClass.N_VNIR_BINNED:]],
                                           axis=0)

        params_vnir['fwhm'] = np.concatenate([params_vnir['fwhm'],
                                              params_vnir_swir['fwhm'][ApexSensorClass.N_VNIR_BINNED:]],
                                             axis=0)
        params_vnir['rad_coeffs']['gain'] = np.concatenate([params_vnir['rad_coeffs']['gain'],
                                                            meta['rad_coeffs']['gain'][ApexSensorClass.N_VNIR_BINNED:]],
                                                           axis=0)

        params_vnir['rad_coeffs']['offset'] = np.concatenate([params_vnir['rad_coeffs']['offset'],
                                                              meta['rad_coeffs']['offset'][
                                                              ApexSensorClass.N_VNIR_BINNED:]],
                                                             axis=0)
        params_vnir.update(meta)
        return ApexSensorClass(*args, **kwargs, **params_vnir)

    else:
        raise Exception('Invalid files.')


if __name__ == '__main__':
    ap = load_apex(unbinned_vnir='/Users/jim/meteoc/params/unbinned',
                   binned_vnir_swir='/Users/jim/meteoc/params/binned',
                   binned_meta='/Users/jim/meteoc/params/binned_meta', vnir_it=27000, swir_it=15000)

    ap.initialize_srfs([500, 510], abs_res=0.05, srf_support_in_sigma=3, zero_out=True, do_bin=True)
    res, illu_bands = ap.forward(inp_spectrum=np.array([1000000]).reshape(1, 1, -1),
                                 inp_wvlens=np.arange(500, 510, 0.05).reshape(-1, 1), pad=False, part_covered=True,
                                 invert=True, snr=True, dc=True, smear=True, return_binned=False,
                                 run_specs=dict(joblib=False, verbose=False))
