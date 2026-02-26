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

# import mne
# import numpy as np
# from scipy.io import savemat

# # ----------------------------
# # Load data
# # ----------------------------
# fpath = "C:/Users/ansbel/Documents/2Git/TriCo/data/external/sub2_center_out_epochs.fif"
# epochs = mne.read_epochs(fpath, preload=True)
# epochs = epochs.pick_channels(epochs.ch_names[:38])

# info = epochs.info
# montage = info.get_montage()

# if montage is None:
#     raise RuntimeError("No montage found in the file.")

# # ----------------------------
# # Extract channel info
# # ----------------------------
# labels = epochs.ch_names
# pos_dict = montage.get_positions()['ch_pos']

# # positions in meters (MNE default)
# chanpos = np.array([pos_dict[ch] for ch in labels])

# # convert to millimeters (как у тебя в MATLAB)
# chanpos_mm = chanpos * 1000

# n_ch = len(labels)

# # ----------------------------
# # Build FieldTrip structure
# # ----------------------------
# elec = {
#     'label': np.array(labels, dtype=object).reshape(-1,1),
#     'chanpos': chanpos_mm,
#     'elecpos': chanpos_mm,
#     'tra': np.eye(n_ch),          # identity transformation
#     'unit': 'mm',
#     'type': 'eeg',
#     'fid': {}                     # пустой struct
# }

# savemat("electrodes_data.mat", {'electrodes_data': elec})

