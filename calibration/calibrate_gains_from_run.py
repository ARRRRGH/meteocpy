import pickle as pkl
import os
import numpy as np
import sys
sys.path.append('/home/jim/meteocpy')
import matplotlib.pyplot as plt

from calibration.utils_calibrate import target_mean, create_inp_spectrum, create_dns

simu_path = '/Users/jim/meteocpy/simulations/test'
target = target_mean
is_mono = False
plot = False



# load simulation
with open(os.path.join(simu_path, 'frames'), 'rb') as f:
    frames = pkl.load(f)

with open(os.path.join(simu_path, 'band_indices'), 'rb') as f:
    illu_bands = pkl.load(f)

with open(os.path.join(simu_path, 'simulation_config'), 'rb') as f:
    config = pkl.load(f)

with open(os.path.join('/Users/jim/meteocpy', 'saved_apex_models', 'apex_400_2000'), 'rb') as f:
    ap = pkl.load(f)


if is_mono:
    # for each simulated band construct an ensemble of input spectra from delta peaks
    # e.g. uniform or random intensities
    
    # create input spectrum
    inp_spectra, inp_wvls = create_inp_spectrum(config, model='multi')
    

else:
    inp_spectra, inp_wvls = config['inp_spectrum'].squeeze(0), config['inp_wvlens'][0]
    
# create dn output from delta peak results, if there is only one frame as in
# is_mono == False this just reorders data
dns, wvls_per_band = create_dns(frames=frames, illu_bands=illu_bands, inp_wvls=inp_wvls)
 
# create target, TODO: THIS IS FUNDAMENTALLY WRONG AS IT ASSUMES LINEARITY
target_per_band = [target(ap, band, inp_spectra, inp_wvls) for band in wvls_per_band.keys()]


# fit gain over all input spectra in the ensemble
p = np.asarray([[np.polyfit(dns[band][:, x], target_per_band[band], deg=1, cov=False) 
                 if target_per_band[band] is not None else np.array([np.nan, np.nan])
                 for x in range(ap.DIM_X_AX)]
                 for band in range(len(dns)) ])


if plot:
    band = 10
    x = 500
    xs = np.linspace(dns[band][:, x][0], dns[band][:, x][-1], 3)
    fit = np.poly1d(np.polyfit(dns[band][:, x], target_per_band[band], deg=1, cov=False))
    plt.scatter(dns[band][:, x], target_per_band[band]); plt.plot(xs, fit(xs), 'red')
    plt.show()