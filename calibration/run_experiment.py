import sys
sys.path.append('/home/jim')

import pickle as pkl
import os
import numpy as np
import pandas as pd
from scipy.interpolate import BarycentricInterpolator
import scipy.signal as sgn

try:
    from meteocpy.forward import apex
except ModuleNotFoundError:
    from forward import apex


home = '/home/'


def run_experiment(simulation_name, recompute, rang, n, intensity_var, batches_per_job, n_jobs):
    ##### LOAD INPUT SPECTRUM #####################################################
    calibr = pd.read_csv(home+'jim/meteoc/data/OGSE_Large sphere radiance.csv')
    calibr = calibr.iloc[:-1, :3].iloc[np.where(np.logical_and(calibr.iloc[:, 0] > rang[0],
                                                               calibr.iloc[:, 0] < rang[-1]))]#[:30]
    
    
    inp_spectrum = calibr.iloc[:, 1].values
    wvls = calibr.iloc[:, 0].values
    
    # resample
    wvls_ = np.linspace(wvls[0], wvls[-1], int(wvls[-1] - wvls[0]) * n)  # get n samples per nm
    inp_spectrum = BarycentricInterpolator(wvls, inp_spectrum)(wvls_)
    wvls = wvls_
    
    
    # create input_spectrum, dirac peak for all wvls in calibr at intensities in intensity_var
    inp_spectrum = np.stack([inp_spectrum * var for var in intensity_var], axis=1) * 5e6
    inp_spectrum = inp_spectrum.reshape(len(inp_spectrum), len(intensity_var), 1)
    
    
    
    
    
    
    ##### DEFINE APEX INSTANCE ###################################################
    if not os.path.exists(save_path) or recompute:
        ap = apex.load_apex(unbinned_vnir=home+'jim/meteoc/params/unbinned', 
                            binned_vnir_swir=home+'/jim/meteoc/params/binned',
                            binned_meta=home+'jim/meteoc/params/binned_meta', 
                            vnir_it=27000, swir_it=15000)
    
        # ap.initialize_srfs(rang, abs_res=abs_res, srf_support_in_sigma=3, zero_out=True, do_bin=True)
        ap.initialize_srfs(exact_wvls=wvls, srf_support_in_sigma=3, zero_out=True, 
                           do_bin=True)
    
        with open(save_path, 'wb') as f:
            pkl.dump(ap, f)
    else:
        with open(save_path, 'rb') as f:
            ap = pkl.load(f)
    
    
    
    
    
    ##### RUN #####################################################################
    # Simulate forward
    config = dict(inp_spectrum=inp_spectrum,
                  inp_wvlens=wvls.reshape(-1, 1), pad=False, part_covered=True,
                  invert=True, snr=True, dc=True, smear=False, return_binned=False, 
                  run_specs=dict(joblib=True, verbose=False, 
                                 batches_per_job=batches_per_job, n_jobs=n_jobs))
    
    res, illu_bands = ap.forward(**config)
    
    
    ##### WRITE ###################################################################
    with open(os.path.join(out_path, 'frames'), 'wb') as f:
        pkl.dump(res, f)
    
    with open(os.path.join(out_path, 'band_indices'), 'wb') as f:
        pkl.dump(illu_bands, f)
    
    with open(os.path.join(out_path, 'simulation_config'), 'wb') as f:
        pkl.dump(config, f) 


if __name__ == '__main__':
    
    ##### SETTINGS ################################################################
    simulation_name = 'test'
    recompute = False
    rang = [900, 1200]
    abs_res = 1
    n = 10  # n samples per nm
    # intensity_var = np.arange(0.6, 2.5, 1)
    intensity_var = np.array([1])  # multiplicative variation of intensity
    batches_per_job = 100
    n_jobs = 10
    
    here_path = os.path.dirname(os.path.dirname(__file__))
    save_path = os.path.join(here_path, 'saved_apex_models/apex_%d_%d' % (rang[0], rang[1]))
    
    out_path = os.path.join(here_path, 'simulations', simulation_name)
    os.makedirs(out_path, exist_ok=True)
    
    run_experiment(simulation_name, recompute, rang, n, intensity_var, 
                   batches_per_job, n_jobs)

    





