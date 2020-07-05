import pickle as pkl
import os
import numpy as np

from calibration.utils_calibrate import target_mean, create_inp_spectrum_uniform, create_dns

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
target = target_mean

# for each simulated band construct an ensemble of input spectra from delta peaks
# e.g. uniform or random intensities


# create input spectrum
inp_spectra, inp_wvls = create_inp_spectrum_uniform(config, model='uniform')

# create dn output from delta peak results
dns, wvls_per_band = create_dns(frames=frames, illu_bands=illu_bands, inp_wvls=inp_wvls)

# create target, TODO: THIS IS FUNDAMENTALLY WRONG AS IT ASSUMES LINEARITY
target = target_mean
target_per_band = [target(ap, band, inp_spectra, inp_wvls) for band in wvls_per_band.keys()]

print(len(dns), len(target_per_band))

# fit gain over all input spectra in the ensemble
p = np.asarray([[np.polyfit(dns[band][:, x], target_per_band[band], deg=1, cov=False) for x in range(ap.DIM_X_AX)]
                for band in range(len(dns))])

print(p.shape)
