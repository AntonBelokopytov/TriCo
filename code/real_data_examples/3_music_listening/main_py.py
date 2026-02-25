# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 17:07:11 2025

@author: anton
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from scipy.signal import hilbert
from scipy.io import savemat

# %%
# fpath = "data/sub1_clear.fif"
fpath = "D:/OS(CURRENT)/data/music/exp2/21.03_g1/DmiAna_2200_3-clear.fif"

raw = mne.io.read_raw_fif(fpath,preload=True)
sfreq = raw.info['sfreq']

# %% 
raw.plot()

# %%
raw.filter(l_freq=15,h_freq=25)

# %%
ica = ICA(
    n_components=38,     # объяснить 99% дисперсии
    method='fastica',      # можно 'infomax'
    random_state=42,
    max_iter='auto'
)

ica.fit(raw)

# %%
filters = (ica.unmixing_matrix_ @ ica.pca_components_) * ica.pre_whitener_.T
filters = filters.T
patterns = ica.get_components()

mat_dict = {
    "filters": filters,
    "patterns": patterns,
    "sfreq": sfreq,
    "ch_names": np.array(raw.ch_names, dtype=object)
}

savemat("ica_filters_patterns.mat", mat_dict)

# %%
i = 3  # номер компоненты

mne.viz.plot_topomap(
    patterns[:, i],
    raw.info,
    cmap='RdBu_r',
    contours=6
)
# %%
ica.plot_components()

# %%
sources = ica.get_sources(raw)
data = sources.get_data()   # shape: (n_components, n_times)

# %%
analytic = hilbert(data, axis=1)
envelopes = np.abs(analytic)

# %%
# ==== Визуализация ====
times = np.arange(data.shape[1]) / sfreq

plt.figure(figsize=(12, 8))

i = 0
plt.plot(times, envelopes[i], label=f'IC {i}')

plt.xlabel('Time (s)')
plt.title('ICA component envelopes (15–25 Hz)')
plt.show()


# %%
# All the conditions:
raw.annotations.description

# %%
conditions = ['RS_EC1', 
              'RS_EO1', 
              '2Hz', 
              '05Hz', 
              '4Hz', 
              '1Hz', 
              '3Hz', 
              'NoRy_1',
              'Waltz_1', 
              'Waltz_2', 
              'NoRy_2', 
              'NoRy_3', 
              'Waltz_3', 
              'NoRy_4',
              'Waltz_4', 
              'NoRy_5', 
              'Waltz_5', 
              'RS_EC2', 
              'RS_EO2', 
              'Waltz_6',
              'Waltz_7', 
              'Waltz_8']

# %%
raw.plot_sensors()

# %%
# 1. Извлекаем монтаж
montage = raw.get_montage()
pos = montage.get_positions()['ch_pos']

# 2. Смещаем все координаты на 2 см назад по оси Y
for ch in pos:
    pos[ch][1] -= 0.02  # Сдвиг по Y на -20мм

# 3. Пересоздаем монтаж с новыми координатами
new_montage = mne.channels.make_dig_montage(ch_pos=pos, coord_frame='head')

# 4. Применяем его к данным
raw.set_montage(new_montage)

# Теперь рисуется корректно по умолчанию
raw.plot_sensors(show_names=True)


# %%
epochs = mne.make_fixed_length_epochs(raw,
                                      reject_by_annotation=False,
                                      duration=120)

# %%
epochs.plot_sensors()

# %%
epochs.save("data/sub2_clear_epochs.fif", 
            overwrite=True)

# %%

