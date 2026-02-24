# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 17:07:11 2025

@author: anton
"""
import mne

# %%
fpath = "data/sub1_rest_ec_eo.fif"

# %%
raw = mne.io.read_raw_fif(fpath,preload=True)

# %%
raw.plot()

# %%

