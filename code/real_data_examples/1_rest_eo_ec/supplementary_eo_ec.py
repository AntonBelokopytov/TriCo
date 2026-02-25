# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 17:07:11 2025

@author: anton
"""
import mne

# %%
fpath = "C:/Users/ansbel/Documents/2Git/TriCo/data/external/sub1_rest_ec_eo_raw.fif"

# %%
raw = mne.io.read_raw(fpath,preload=True)

# %%
raw.plot()

# %%

