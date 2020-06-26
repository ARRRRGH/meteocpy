import pickle as pkl
import os
import numpy as np
import scipy as sc

simu_path = '/Users/jim/meteocpy/simulations/test'

# load simulation
with open(os.path.join(simu_path, 'frames'), 'rb') as f:
    res = pkl.load(f)

with open(os.path.join(simu_path, 'band_indices'), 'rb') as f:
    illu_bands = pkl.load(f)

with open(os.path.join(simu_path, 'simulation_config'), 'rb') as f:
    config = pkl.load(f)

# with open(config['apex_save_path'], 'rb') as f:
#     ap = pkl.load(f)

with open(os.path.join('/Users/jim/meteocpy', 'saved_apex_models', 'apex_700_1000'), 'rb') as f:
    ap = pkl.load(f)


# define target
def target_mean(ap, band, inp_spectrum, inp_wvlens):
    """
    Radiance in mean wvl

    :param inp_spectrum: (n_realizations, n_wvls)
    :param inp_wvlens: (n_wvls,)
    :param ap:
    :param band:
    :return:
    """
    # cut input to band support
    support = ap.get('srf_support_per_band')[band]
    mean_wvl = support.mean()

    # supported_ids = np.where(np.logical_and(inp_wvlens < support[-1], inp_wvlens[0] > support[0]))
    return sc.interpolate.interp1d(inp_wvlens, inp_spectrum)(mean_wvl)


# for each simulated band construct an ensemble of input spectra from delta peaks
# e.g. uniform or random intensities
def create_inp_spectrum_uniform(config):
    """
    :param config:
    :return:
    """
    spectrum = config['inp_spectrum'][..., 0]
    wvls = config['inp_wvlens'].reshape(-1)

    return spectrum, wvls


# calc target for each band
band = 10
target = target_mean(ap, band, *create_inp_spectrum_uniform(config))

# fit gain over all input spectra in the ensemble



create_inp_spectrum_uniform(config)