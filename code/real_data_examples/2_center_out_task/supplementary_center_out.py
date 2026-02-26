# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 17:07:11 2025

@author: anton
"""

import mne
import numpy as np

# %%
fpath = "C:/Users/ansbel/Documents/2Git/TriCo/data/external/sub2_center_out_epochs.fif"

# %%
epochs = mne.read_epochs(fpath,preload=True)

# %% 
epochs.plot()

# %%
mask = (
    (epochs.metadata['pp'] == 2) &
    (epochs.metadata['correct_trials'] == 1)
)

idx = np.where(mask)[0]
idx

# %%
epochs = epochs[idx]

# %%
epochs.save(fpath,overwrite=True)

# %%

