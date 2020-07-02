import pickle as pkl
import os
import numpy as np
import pandas as pd
from scipy.interpolate import BarycentricInterpolator

try:
    from meteocpy.forward import apex
except ModuleNotFoundError:
    from forward import apex

home = '/Users/'
simulation_name = 'test'
recompute = True
rang = [700, 800]

here_path = os.path.dirname(os.path.dirname(__file__))
save_path = os.path.join(here_path, 'saved_apex_models/apex_%d_%d' % (rang[0], rang[1]))

out_path = os.path.join(here_path, 'simulations', simulation_name)
os.makedirs(out_path, exist_ok=True)

if not os.path.exists(save_path) or recompute:
    ap = apex.load_apex(unbinned_vnir=home+'jim/meteoc/params/unbinned', binned_vnir_swir=home+'/jim/meteoc/params/binned',
                        binned_meta=home+'jim/meteoc/params/binned_meta', vnir_it=27000, swir_it=15000)

    ap.initialize_srfs(rang, abs_res=0.1, srf_support_in_sigma=1, zero_out=True, do_bin=True)

    with open(save_path, 'wb') as f:
        pkl.dump(ap, f)
else:
    with open(save_path, 'rb') as f:
        ap = pkl.load(f)

calibr = pd.read_csv(home+'jim/meteoc/data/OGSE_Large sphere radiance.csv')
calibr = calibr.iloc[:-1, :3].iloc[np.where(np.logical_and(calibr.iloc[:, 0] > rang[0],
                                                           calibr.iloc[:, 0] < rang[-1]))][:3]
inp_spectrum = calibr.iloc[:, 1].values
wvls = calibr.iloc[:, 0].values

wvls_new = np.linspace(wvls[0], wvls[-1], 100)
inp_spectrum = BarycentricInterpolator(wvls, inp_spectrum)(wvls_new)
wvls = wvls_new

intensity_var = np.arange(0.5, 4, 0.1)
# intensity_var = np.array([1])

# create input_spectrum, dirac peak for all wvls in calibr at intensities in intensity_var
inp_spectrum = np.stack([inp_spectrum * var for var in intensity_var], axis=1) * 5e6
inp_spectrum = inp_spectrum.reshape(len(inp_spectrum), len(intensity_var), 1)

# Simulate forward
config = dict(inp_spectrum=inp_spectrum,
              inp_wvlens=wvls.reshape(-1, 1), pad=False, part_covered=True,
              invert=True, snr=True, dc=True, smear=True, return_binned=False, conv_mode='numba',
              run_specs=dict(joblib=False, verbose=False, batches_per_job=100, n_jobs=6))

res, illu_bands = ap.forward(**config)


with open(os.path.join(out_path, 'frames'), 'wb') as f:
    pkl.dump(res, f)

with open(os.path.join(out_path, 'band_indices'), 'wb') as f:
    pkl.dump(illu_bands, f)

with open(os.path.join(out_path, 'simulation_config'), 'wb') as f:
    pkl.dump(config, f)







