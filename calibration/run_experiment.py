import pickle as pkl
import os
import numpy as np
import pandas as pd

try:
    from meteocpy.forward import apex
except ModuleNotFoundError:
    from forward import apex


rang = [700, 1000]
save_path = '/Users/jim/meteocpy/saved_apex_models/apex_%d_%d' % (rang[0], rang[1])

if not os.path.exists(save_path):
    ap = apex.load_apex(unbinned_vnir='/Users/jim/meteoc/params/unbinned', binned_vnir_swir='/Users/jim/meteoc/params/binned',
                        binned_meta='/Users/jim/meteoc/params/binned_meta', vnir_it=27000, swir_it=15000)

    ap.initialize_srfs(rang, abs_res=0.1, srf_support_in_sigma=3, zero_out=True, do_bin=True)

    with open(save_path, 'wb') as f:
        pkl.dump(ap, f)
else:
    with open(save_path, 'rb') as f:
        ap = pkl.load(f)

calibr = pd.read_csv('/Users/jim/meteoc/data/OGSE_Large sphere radiance.csv')
calibr = calibr.iloc[:-1, :3].iloc[np.where(np.logical_and(calibr.iloc[:, 0] > rang[0],
                                                           calibr.iloc[:, 0] < rang[-1]))]
inp_spectrum = calibr.iloc[:, 1]
wvls = calibr.iloc[:, 0]

intensity_var = np.arange(0.1, 2.5, 0.1)

print(wvls)

# create input_spectrum, dirac peak for all wvls in calibr at intensities in intensity_var
inp_spectrum = pd.concat([inp_spectrum * var for var in intensity_var], 1)
inp_spectrum = inp_spectrum.values.reshape(len(inp_spectrum), len(intensity_var), 1)

# Simulate forward
res, illu_bands = ap.forward(inp_spectrum=inp_spectrum,
                             inp_wvlens=wvls.values.reshape(-1, 1), pad=False, part_covered=True,
                             invert=True, snr=True, dc=True, smear=True, return_binned=False,
                             run_specs=dict(joblib=False, verbose=False, batches_per_job=100))


