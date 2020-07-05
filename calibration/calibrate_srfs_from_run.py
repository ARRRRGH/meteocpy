import pickle as pkl
import os
import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize as opt
from scipy.stats import norm

from calibration.utils_calibrate import gather_simulations
from utils import chunk_list

from functools import partial


# load simulation
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


# gather_simulations
band_dict, wvl_dict = gather_simulations(frames, illu_bands, config['inp_wvlens'].reshape(-1))


# define srf model
def gaussian_w_off(xdata, *params, size=10):
    # split params
    a = np.array(params[:size]).reshape(-1, 1)
    mu = np.array(params[size:2*size]).reshape(-1, 1)
    sigma = np.array(params[2*size:3*size]).reshape(-1, 1)
    off = np.array(params[3*size:]).reshape(-1, 1)

    ret = a * norm.pdf(xdata, mu, sigma) + off
    print(ret.shape)
    return ret.flatten()


# fit srf model for each simulated pixel
srf_model = gaussian_w_off
channel = -1
for band in band_dict.keys():

    # run in a loop to reduce number of simultaneously fitted params
    # -> circumvent scipy's limitation on 1000 params
    size = 10
    xtrack_inds = list(np.arange(1000))
    for chunk in chunk_list(xtrack_inds, size):
        chunk = [chunk]
        sigma0 = np.sqrt(ap.get('fwhm'))[band].reshape(-1, 1)[chunk]
        mu0 = ap.get('cw')[band].reshape(-1, 1)[chunk]
        init = np.concatenate([np.ones(mu0.shape), mu0, sigma0, np.zeros(mu0.shape)], axis=0)

        xdata = np.array(wvl_dict[band])  # .reshape(1, -1)
        ydata = np.stack(band_dict[band])[:, channel, chunk].squeeze().transpose()
        #plt.plot(ydata[0])
        #plt.show()
        opt.curve_fit(partial(srf_model, size=size), p0=init, xdata=xdata, ydata=ydata.flatten(), maxfev=int(2e4))
        # srf_model(xdata, *init).shape
