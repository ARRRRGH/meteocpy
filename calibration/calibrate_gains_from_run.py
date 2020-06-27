import pickle as pkl
import os
import numpy as np
import scipy as sc

simu_path = '/Users/jim/meteocpy/simulations/test'

# load simulation
with open(os.path.join(simu_path, 'frames'), 'rb') as f:
    frames = pkl.load(f)

with open(os.path.join(simu_path, 'band_indices'), 'rb') as f:
    illu_bands = pkl.load(f)

with open(os.path.join(simu_path, 'simulation_config'), 'rb') as f:
    config = pkl.load(f)

with open(os.path.join('/Users/jim/meteocpy', 'saved_apex_models', 'apex_700_800'), 'rb') as f:
    ap = pkl.load(f)


# define target
def target_mean(ap, ext_band, inp_spectrum, inp_wvlens):
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
    return sc.interpolate.interp1d(inp_wvlens, inp_spectrum)(mean_wvl)


# for each simulated band construct an ensemble of input spectra from delta peaks
# e.g. uniform or random intensities
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


def create_dns(ap, frames, illu_bands, inp_wvls, model='uniform'):
    """
    Create DN vectors for each pixel in bands
    :param ap:
    :param frames:
    :param illu_bands:
    :param band:
    :return:
    """
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

    # calculate total dns from delta peak results
    # we assume a fully
    if model == 'uniform':
        return [np.stack(frame).sum(axis=0) for band_id, frame in band_dict.items()], wvl_dict
    else:
        raise NotImplementedError


# create input spectrum
inp_spectra, inp_wvls = create_inp_spectrum_uniform(config, model='uniform')

# create dn output from delta peak results
dns, wvls_per_band = create_dns(ap, frames=frames, illu_bands=illu_bands, inp_wvls=inp_wvls)

# create target, TODO: THIS IS FUNDAMENTALLY WRONG AS IT ASSUMES LINEARITY
target = target_mean
target_per_band = [target(ap, band, inp_spectra, inp_wvls) for band in wvls_per_band.keys()]

print(len(dns), len(target_per_band))

# fit gain over all input spectra in the ensemble
p = np.asarray([[np.polyfit(dns[band][:, x], target_per_band[band], deg=1, cov=False) for x in range(ap.DIM_X_AX)]
                for band in range(len(dns))])

print(p.shape)
