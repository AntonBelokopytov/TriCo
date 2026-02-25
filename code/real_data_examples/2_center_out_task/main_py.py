# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 17:07:11 2025

@author: anton
"""

import mne

# %%
fpath = "data/Patient1_OFF_2-35Hz_pp4s_epochs.fif"

# %%
epochs = mne.read_epochs(fpath,preload=True)

# %% 
epochs.plot()

# %%

