import numpy as np
import scipy as sc


def target_mean(ap, ext_band, inp_spectrum, inp_wvls):
    """
    Radiance in mean wvl

    :param inp_spectrum: (n_realizations, n_wvls)
    :param inp_wvlens: (n_wvls,)
    :param ap:
    :param band:
    :return:
    """
    # cut input to band support
    band = ap.shift_bands_to_local(ext_band)
    support = ap.get('srf_support_per_band')[band]
    mean_wvl = support.mean()

    print(band, support, mean_wvl, inp_wvls)
    # supported_ids = np.where(np.logical_and(inp_wvlens < support[-1], inp_wvlens[0] > support[0]))
    return sc.interpolate.interp1d(inp_wvls, inp_spectrum)(mean_wvl)


def create_inp_spectrum_uniform(config, model='uniform'):
    """
    :param config:
    :return: spectrum (n_realizations, n_wvls), wvls (n_wvls, )
    """
    if model == 'uniform':
        # is in form (n_batches, n_channels) -> (n_channels = n_realizations, n_batches = n_wvls)
        spectrum = config['inp_spectrum'][..., 0].transpose()
        wvls = config['inp_wvlens'].reshape(-1)

        return spectrum, wvls

    else:
        raise NotImplementedError


def gather_simulations(frames, illu_bands, inp_wvls):
    band_dict = {}
    wvl_dict = {}

    # Iterate over all bands such that band_dict[i] = [band_at_wvl1, ..., band_at_wvln]
    # and wvl_dict = [wvl1, ..., wvln]
    for i, (ill_bands, frame) in enumerate(zip(illu_bands, frames)):

        for j, ib in enumerate(ill_bands):

            if ib not in band_dict:
                band_dict[ib] = [frame[:, j, :]]
                wvl_dict[ib] = [inp_wvls[i]]

            else:
                band_dict[ib].append(frame[:, j, :])
                wvl_dict[ib].append(inp_wvls[i])

    return band_dict, wvl_dict


def create_dns(frames, illu_bands, inp_wvls, model='uniform'):
    """
    Create DN vectors for each pixel in bands
    :param ap:
    :param frames:
    :param illu_bands:
    :param band:
    :return:
    """
    band_dict, wvl_dict = gather_simulations(frames, illu_bands, inp_wvlens=inp_wvls)
    # calculate total dns from delta peak results
    # we assume a fully
    if model == 'uniform':
        return [np.stack(frame).sum(axis=0) for band_id, frame in band_dict.items()], wvl_dict
    else:
        raise NotImplementedError
