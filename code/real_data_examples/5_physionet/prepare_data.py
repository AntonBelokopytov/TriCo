# -*- coding: utf-8 -*-
"""
Конкатенация ВСЕХ сессий S001 с детальными метками:
- Rest глаза открыты / закрыты
- Воображение левой/правой руки
- Воображение обеих кистей / обеих ступней.
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
import umap
import scipy.io
from pyriemann.utils.mean import mean_riemann
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
import os

# %% Параметры
subject = 'S042'
base_dir = "E:/REVE/MNE-eegbci-data/files/eegmmidb/1.0.0/"
file_numbers = [4, 6, 8, 10, 12, 14]

# %% Очистка имён каналов
def to_standard_1005(ch_name):
    name = ch_name.replace('.', '').strip().upper()
    if name.endswith('Z') and len(name) > 1:
        name = name[:-1] + 'z'
    if name.startswith('FP'):
        name = 'Fp' + name[2:]
    return name

# %% Загрузка, фильтрация и ПРАВИЛЬНАЯ замена меток
raw_ica_list = []
montage_standard = mne.channels.make_standard_montage('standard_1005')

for num in file_numbers:
    fname = f"{subject}/{subject}R{num:02d}.edf"
    fpath = os.path.join(base_dir, fname)
    print(f"\n=== {fpath} ===")

    raw = mne.io.read_raw_edf(fpath, preload=True)

    # Каналы
    new_names = {ch: to_standard_1005(ch) for ch in raw.ch_names}
    raw.rename_channels(new_names)
    raw.set_montage(montage_standard)

    # Фильтрация
    raw_filt = raw.copy().notch_filter(60).filter(l_freq=1.0, h_freq=40.0, fir_design='firwin')
    raw_filt.crop(tmin=2, tmax=raw_filt.times[-1] - 2)

    if raw_filt.info['bads']:
        raw_filt.interpolate_bads(reset_bads=True)

    # ===== НАЗНАЧЕНИЕ МЕТОК =====
    if num == 1:                     # Baseline, eyes open
        t0_label = 'Rest_EO'
        t1_label = None
        t2_label = None
    elif num == 2:                   # Baseline, eyes closed
        t0_label = 'Rest_EC'
        t1_label = None
        t2_label = None
    elif num in [4, 8, 12]:          # Воображение левой/правой руки (глаза открыты)
        t0_label = 'Rest_EO'
        t1_label = 'Hand_Left'
        t2_label = 'Hand_Right'
    elif num in [6, 10, 14]:         # Воображение обеих кистей/ступней (глаза открыты)
        t0_label = 'Rest_EO'
        t1_label = 'Both_Fists'
        t2_label = 'Both_Feet'
    else:
        raise ValueError(f"Неизвестный номер сессии: {num}")

    # Замена аннотаций
    new_descriptions = []
    for desc in raw_filt.annotations.description:
        if desc == 'T0':
            new_descriptions.append(t0_label)
        elif desc == 'T1' and t1_label is not None:
            new_descriptions.append(t1_label)
        elif desc == 'T2' and t2_label is not None:
            new_descriptions.append(t2_label)
        else:
            new_descriptions.append(desc)   # BAD_, EDGE и т.п.

    new_annot = mne.Annotations(
        onset=raw_filt.annotations.onset,
        duration=raw_filt.annotations.duration,
        description=new_descriptions,
        orig_time=raw_filt.annotations.orig_time
    )
    raw_filt.set_annotations(new_annot)
    raw_ica_list.append(raw_filt)

# %% Конкатенация
raw_all = mne.concatenate_raws(raw_ica_list)
print(f"Объединённая запись: {raw_all.n_times / raw_all.info['sfreq']:.1f} с")

# %% SSD (сигнал: все условия кроме Rest_EO/EC, шум: Rest_EO+Rest_EC)
sfreq = raw_all.info['sfreq']
b_signal, a_signal = butter(3, np.array([15, 25]) / (sfreq / 2), btype='band')
b_broad, a_broad = butter(3, np.array([14, 27]) / (sfreq / 2), btype='band')
b_stop, a_stop = butter(3, np.array([14.5, 25.5]) / (sfreq / 2), btype='stop')
crop_samples = int(0.5 * sfreq)

label_mapping = {
    'Rest_EO': 'Rest_EO', 'Rest_EC': 'Rest_EC',
    'Hand_Left': 'Hand_Left', 'Hand_Right': 'Hand_Right',
    'Both_Fists': 'Both_Fists', 'Both_Feet': 'Both_Feet'
}

signal_pieces, noise_pieces = [], []
data = raw_all.get_data()

for annot in raw_all.annotations:
    desc = annot['description']
    if desc not in label_mapping:
        continue
    onset = annot['onset']
    dur = annot['duration']
    if dur <= 0:
        continue

    tmax = onset + dur
    if tmax > raw_all.times[-1]:
        tmax = raw_all.times[-1]
    if tmax - onset < 2.0:
        continue

    start_idx = int(onset * sfreq)
    end_idx = int(tmax * sfreq)
    seg = data[:, start_idx:end_idx]

    seg_signal = filtfilt(b_signal, a_signal, seg, axis=1)
    seg_noise_broad = filtfilt(b_broad, a_broad, seg, axis=1)
    seg_noise = filtfilt(b_stop, a_stop, seg_noise_broad, axis=1)

    if seg_signal.shape[1] > 2 * crop_samples:
        seg_signal = seg_signal[:, crop_samples:-crop_samples]
        seg_noise = seg_noise[:, crop_samples:-crop_samples]

    signal_pieces.append(seg_signal)
    noise_pieces.append(seg_noise)

if not signal_pieces:
    raise ValueError("Нет блоков для SSD!")

signal_conc = np.concatenate(signal_pieces, axis=1)
noise_conc = np.concatenate(noise_pieces, axis=1)

C_signal = np.cov(signal_conc)
C_noise = np.cov(noise_conc) + 1e-5 * np.trace(np.cov(noise_conc)) * np.eye(noise_conc.shape[0])

eigvals, eigvecs = eigh(C_signal, C_noise)
idx = np.argsort(eigvals)[::-1]
n_comp = 50
W_ssd = eigvecs[:, idx[:n_comp]]
A_ssd = C_signal @ W_ssd
print(f"SSD: {n_comp} компонент")

# %% Нарезка окон (2 с окно, шаг 0.1 с)
Wsize = 2
Ssize = 0.5
overlap = Wsize - Ssize

X_band, X_unfilt = [], []
labels, trials = [], []
trial_id = 1

raw_band = raw_all.copy().filter(l_freq=15, h_freq=25, picks='eeg')
raw_unfilt = raw_all.copy().pick_types(eeg=True)

for annot in raw_all.annotations:
    desc = annot['description']
    if desc not in label_mapping:
        continue
    onset = annot['onset']
    dur = annot['duration']
    if dur <= 0:
        continue
    tmax = onset + dur
    if tmax > raw_all.times[-1]:
        tmax = raw_all.times[-1]
    if tmax - onset < Wsize:
        continue

    raw_crop = raw_band.copy().crop(tmin=onset, tmax=tmax)
    raw_crop_unf = raw_unfilt.copy().crop(tmin=onset, tmax=tmax)

    epochs_band = mne.make_fixed_length_epochs(
        raw_crop, duration=Wsize, overlap=overlap,
        preload=True, reject_by_annotation=False, verbose=False)
    epochs_unf = mne.make_fixed_length_epochs(
        raw_crop_unf, duration=Wsize, overlap=overlap,
        preload=True, reject_by_annotation=False, verbose=False)

    epochs_band.drop_bad(verbose=False)
    epochs_unf.drop_bad(verbose=False)

    if len(epochs_band) > 0 and len(epochs_band) == len(epochs_unf):
        X_band.append(epochs_band.get_data(copy=False))
        X_unfilt.append(epochs_unf.get_data(copy=False))
        labels.extend([desc] * len(epochs_band))
        trials.extend([trial_id] * len(epochs_band))
        trial_id += 1

X_band = np.concatenate(X_band, axis=0)
X_unfilt = np.concatenate(X_unfilt, axis=0)
labels = np.array(labels)
trials = np.array(trials)

print(f"Окон: {len(labels)}")
print(np.unique(labels, return_counts=True))

# %% Проекция в SSD
n_win, n_ch, n_times = X_band.shape
X_ssd_proj = np.zeros((n_win, n_comp, n_times))
for i in range(n_win):
    X_ssd_proj[i] = W_ssd.T @ X_band[i]

# %% Ковариации и Tangent Space
cov_band = Covariances(estimator='oas').fit_transform(X_band)
cov_ssd = Covariances(estimator='oas').fit_transform(X_ssd_proj)
ts_ssd = TangentSpace(metric='riemann').fit_transform(cov_ssd)
Cmean_ssd = mean_riemann(cov_ssd)

# %% UMAP
n_feat = ts_ssd.shape[1]
ts_wz = np.vstack([ts_ssd, np.zeros((1, n_feat))])

red2 = umap.UMAP(n_neighbors=20, n_components=2, min_dist=0.1, metric='euclidean').fit_transform(ts_wz)
R2d = red2[:-1, :] - red2[-1, :]

red3 = umap.UMAP(n_neighbors=20, n_components=3, min_dist=0.1, metric='euclidean').fit_transform(ts_wz)
R3d = red3[:-1, :] - red3[-1, :]

red10 = umap.UMAP(n_neighbors=20, n_components=10, min_dist=0.1,
                  metric='euclidean').fit_transform(ts_wz)
R10d = red10[:-1, :] - red10[-1, :]

# %% Визуализация (6 цветов)
color_map = {
    'Rest_EO':    'gray',
    'Rest_EC':    'black',
    'Hand_Left':  'red',
    'Hand_Right': 'blue',
    'Both_Fists': 'orange',
    'Both_Feet':  'green'
}
point_colors = [color_map[lbl] for lbl in labels]

plt.figure(figsize=(8,6))
plt.scatter(R2d[:,0], R2d[:,1], c=point_colors, alpha=0.6)
plt.title('2D UMAP (мю-ритм, все типы)')
for lbl, col in color_map.items():
    plt.scatter([], [], c=col, label=lbl)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(projection='3d')
ax.scatter(R3d[:,0], R3d[:,1], R3d[:,2], c=point_colors, alpha=0.6)
ax.set_title('3D UMAP')
for lbl, col in color_map.items():
    ax.scatter([], [], [], c=col, label=lbl)
ax.legend()
ax.grid(True)
plt.show()

# %% Сохранение в MATLAB
mdic = {
    'X_ssd': np.transpose(X_ssd_proj, (2,1,0)),
    'X_raw': np.transpose(X_unfilt, (2,1,0)),
    'Rmean2d': R2d,
    'Rmean3d': R3d,
    'labels': labels,
    'trials': trials,
    'sfreq': sfreq,
    'ch_names': np.array(raw_all.ch_names, dtype=object),
    'Cmean_ssd': Cmean_ssd,
    'covmats_ssd': np.transpose(cov_ssd, (1,2,0)),
    'covmats_band': np.transpose(cov_band, (1,2,0)),
    'ts_data_ssd': ts_ssd,
    'W_ssd': W_ssd,
    'A_ssd': A_ssd,
    'Rmean10d': R10d
}
montage = raw_all.get_montage()
if montage is not None:
    pos = montage.get_positions()['ch_pos']
    ch_pos = np.array([pos[ch] for ch in raw_all.ch_names if ch in pos])
    mdic['ch_pos'] = ch_pos

scipy.io.savemat('results_matlab_mu_all_6classes.mat', mdic)
print("Сохранено в results_matlab_mu_all_6classes.mat")