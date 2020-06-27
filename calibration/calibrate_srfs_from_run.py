import pickle as pkl
import os
import numpy as np
import scipy as sc

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

# define srf model
srf_model = ap.sr
# fit srf model for each simulated pixel