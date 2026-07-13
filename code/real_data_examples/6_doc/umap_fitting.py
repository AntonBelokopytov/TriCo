# -*- coding: utf-8 -*-
"""
Подготовка данных DoC (ХНС) для mSPoC + UMAP.
Включает парсинг .ezm, ICA, SSD и Риманову геометрию.
"""

import os
import mne
import re
import numpy as np
import scipy.io as sio
import umap
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# %% === 0. ГЛОБАЛЬНЫЕ ПУТИ И НАСТРОЙКИ ===
input_edf = 'C:/Users/ansbel/Documents/GitHub/DoC/Sorted/Андреев парадигмы/Андреев.edf'

base_dir = os.path.dirname(input_edf)
file_name = os.path.splitext(os.path.basename(input_edf))[0]

ezm_fpath = os.path.join(base_dir, file_name + '.ezm')
fif_fpath = os.path.join(base_dir, file_name + '_raw.fif')
mat_fpath = os.path.join(base_dir, file_name + '_eeg_riemann_data.mat')

print(f"Рабочая директория: {base_dir}")

# %% === 1. ЗАГРУЗКА И ПАРСИНГ EZM ===
raw = mne.io.read_raw_edf(input_edf, preload=True)

onsets, durations, descriptions = [], [], []
with open(ezm_fpath, 'r', encoding='cp1251') as f:
    lines = f.readlines()
    
in_annotations = False
for line in lines:
    line = line.strip()
    if line == '[Annotations]':
        in_annotations = True
        continue
    elif line.startswith('[') and in_annotations:
        break
        
    if in_annotations and '=' in line:
        data_parts = line.split('=')[1].split('|')
        if len(data_parts) >= 14:
            try:
                onsets.append(float(data_parts[3].replace(',', '.')))
                durations.append(float(data_parts[13].replace(',', '.')))
                descriptions.append(data_parts[5])
            except ValueError:
                pass

meas_date = raw.annotations.orig_time if raw.annotations is not None else raw.info['meas_date']
ezm_annots = mne.Annotations(onset=onsets, duration=durations, description=descriptions, orig_time=meas_date)

if raw.annotations is not None:
    raw.set_annotations(raw.annotations + ezm_annots)
else:
    raw.set_annotations(ezm_annots)

# %% === 2. КАНАЛЫ, РЕФЕРЕНС И ФИЛЬТРАЦИЯ ===
montage = mne.channels.make_standard_montage('standard_1020')
standard_chs_lower = [ch.lower() for ch in montage.ch_names]

picks_original, clean_ch_names = [], []
for orig_ch in raw.ch_names:
    cleaned = re.sub(r'(?i)eeg[\s_-]?', '', orig_ch)
    cleaned = re.sub(r'(?i)[\s_-]?(ref|le)', '', cleaned).strip()
    if cleaned.lower() in standard_chs_lower:
        idx = standard_chs_lower.index(cleaned.lower())
        picks_original.append(orig_ch)
        clean_ch_names.append(montage.ch_names[idx])

data = raw.get_data(picks=picks_original)
sfreq = raw.info['sfreq']
info = mne.create_info(ch_names=clean_ch_names, sfreq=sfreq, ch_types='eeg')

raw_fif = mne.io.RawArray(data, info)
raw_fif.set_meas_date(raw.info['meas_date'])
if raw.annotations is not None:
    raw_fif.set_annotations(raw.annotations)

raw_fif.set_montage(montage)

if 'A1' in raw_fif.ch_names and 'A2' in raw_fif.ch_names:
    raw_fif.set_eeg_reference(ref_channels=['A1', 'A2'], projection=False)

raw_fif_dropped = raw_fif.copy().drop_channels(['A1','A2']).notch_filter(50).filter(l_freq=1, h_freq=70)

# %%
raw_fif_interp = raw_fif_dropped.interpolate_bads(reset_bads=True)

# %% === 3. ICA ===
# Делаем фиктивные эпохи только для ОБУЧЕНИЯ ICA (чтобы отбросить артефакты)
epochs_for_ica = mne.make_fixed_length_epochs(raw_fif_interp, duration=1.0, preload=True, reject_by_annotation=True, verbose=False)

ica = mne.preprocessing.ICA(n_components=0.999, method='fastica', random_state=42)
ica.fit(epochs_for_ica)

# ПРИМЕНЯЕМ ICA к непрерывному сигналу один раз!
raw_ica = ica.apply(raw_fif_interp.copy())

# %%
raw_ica = raw_fif_interp

# %% === 4. SPATIO-SPECTRAL DECOMPOSITION (SSD) ===
print("Выполнение SSD...")
# Для бета-ритма 15-30 Гц
b_signal, a_signal = butter(3, np.array([15, 30]) / (sfreq / 2), btype='band')
b_broad, a_broad = butter(3, np.array([13, 32]) / (sfreq / 2), btype='band')
b_stop, a_stop = butter(3, np.array([14.5, 30.5]) / (sfreq / 2), btype='stop')

crop_samples = int(0.5 * sfreq)
data_ica = raw_ica.get_data(picks='eeg')
signal_pieces, noise_pieces = [], []

for annot in raw_ica.annotations:
    desc = annot['description']
    if 'BAD_' in desc or annot['duration'] < 2.0:
        continue
    
    tmin, tmax = annot['onset'], annot['onset'] + annot['duration']
    start_idx, end_idx = int(tmin * sfreq), int(min(tmax * sfreq, data_ica.shape[1]))
    seg_data = data_ica[:, start_idx:end_idx]
    
    seg_signal = filtfilt(b_signal, a_signal, seg_data, axis=1)
    seg_noise_broad = filtfilt(b_broad, a_broad, seg_data, axis=1)
    seg_noise = filtfilt(b_stop, a_stop, seg_noise_broad, axis=1)
    
    if seg_signal.shape[1] > 2 * crop_samples:
        seg_signal = seg_signal[:, crop_samples:-crop_samples]
        seg_noise = seg_noise[:, crop_samples:-crop_samples]
        
    signal_pieces.append(seg_signal)
    noise_pieces.append(seg_noise)

signal_conc = np.concatenate(signal_pieces, axis=1)
noise_conc = np.concatenate(noise_pieces, axis=1)

C_signal = np.cov(signal_conc)
C_noise = np.cov(noise_conc)
C_noise_reg = C_noise + 1e-5 * np.trace(C_noise) * np.eye(C_noise.shape[0])

eigvals, eigvecs = eigh(C_signal, C_noise_reg)
idx_sorted = np.argsort(eigvals)[::-1]
n_components_ssd = min(30, C_signal.shape[0])  # Защита от малого кол-ва каналов
W_ssd = eigvecs[:, idx_sorted[:n_components_ssd]]
A_ssd = C_signal @ W_ssd

# %% === 5. СКОЛЬЗЯЩЕЕ ОКНО ===
Wsize = 1.0
Ssize = 0.5
overlap = Wsize - Ssize 

raw_band = raw_ica.copy().filter(l_freq=15, h_freq=30, picks='eeg')
raw_unfilt = raw_ica.copy().pick_types(eeg=True)

X_windows_band, X_windows_unfilt = [], []
labels, trials = [], []
trial_id = 1

for annot in raw_ica.annotations:
    desc = annot['description']
    if 'BAD_' in desc or annot['duration'] < Wsize:
        continue
        
    base_cond = re.sub(r'_\d+$', '', desc)
    tmin, tmax = annot['onset'], annot['onset'] + annot['duration']
    
    raw_crop_band = raw_band.copy().crop(tmin=tmin, tmax=tmax)
    raw_crop_unfilt = raw_unfilt.copy().crop(tmin=tmin, tmax=tmax)
    
    ep_band = mne.make_fixed_length_epochs(raw_crop_band, duration=Wsize, overlap=overlap, preload=True, reject_by_annotation=True, verbose=False)
    ep_unfilt = mne.make_fixed_length_epochs(raw_crop_unfilt, duration=Wsize, overlap=overlap, preload=True, reject_by_annotation=True, verbose=False)
    
    if len(ep_band) > 0 and len(ep_band) == len(ep_unfilt):
        X_windows_band.append(ep_band.get_data(copy=False))
        X_windows_unfilt.append(ep_unfilt.get_data(copy=False))
        labels.extend([base_cond] * len(ep_band))
        trials.extend([trial_id] * len(ep_band))
        trial_id += 1

X_windows_band = np.concatenate(X_windows_band, axis=0)
X_windows_unfilt = np.concatenate(X_windows_unfilt, axis=0)
labels, trials = np.array(labels), np.array(trials)

# ПРОЕКЦИЯ В SSD
print("Проекция в SSD...")
X_windows_ssd = np.zeros((X_windows_band.shape[0], n_components_ssd, X_windows_band.shape[2]))
for i in range(X_windows_band.shape[0]):
    X_windows_ssd[i] = W_ssd.T @ X_windows_band[i]

# %% === 6. РИМАНОВА ГЕОМЕТРИЯ И UMAP ===
covmats_ssd = Covariances(estimator='oas').fit_transform(X_windows_ssd)
ts_data_ssd = TangentSpace(metric='riemann').fit_transform(covmats_ssd)
Cmean_ssd = mean_riemann(covmats_ssd)

print("Добавление нулевой точки и UMAP...")
zero_point = np.zeros((1, ts_data_ssd.shape[1]))
ts_data_with_zero = np.vstack([ts_data_ssd, zero_point])

red5d = umap.UMAP(n_neighbors=20, n_components=20, min_dist=0.1, metric='euclidean', output_metric='euclidean')
R5d_all = red5d.fit_transform(ts_data_with_zero)

red3d = umap.UMAP(n_neighbors=20, n_components=3, min_dist=0.1, metric='euclidean', output_metric='euclidean')
R3d_all = red3d.fit_transform(ts_data_with_zero)

red2d = umap.UMAP(n_neighbors=20, n_components=2, min_dist=0.1, metric='euclidean', output_metric='euclidean')
R2d_all = red2d.fit_transform(ts_data_with_zero)

Rmean5d = R5d_all[:-1, :] - R5d_all[-1, :]
Rmean3d = R3d_all[:-1, :] - R3d_all[-1, :]
Rmean2d = R2d_all[:-1, :] - R2d_all[-1, :]

# %%
import matplotlib.pyplot as plt

# 1. Создаем фигуру
fig = plt.figure()

# 2. Добавляем 3D-оси на фигуру
ax = fig.add_subplot(projection='3d')

# 3. Строим график, используя обычный метод .scatter()
ax.scatter(Rmean3d[:, 0], Rmean3d[:, 1], Rmean3d[:, 2])

# 4. (Опционально) Добавляем подписи осей
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
   
# %% === 7. ЭКСПОРТ ДЛЯ MATLAB ===
X_ssd_mat = np.transpose(X_windows_ssd, (2, 1, 0))
X_raw_mat = np.transpose(X_windows_unfilt, (2, 1, 0)) 
covmats_ssd_mat = np.transpose(covmats_ssd, (1, 2, 0))

montage_pos = np.array([montage.get_positions()['ch_pos'].get(ch, [np.nan]*3) for ch in raw_ica.ch_names])

mdic = {
    'X_ssd': X_ssd_mat,
    'X_raw': X_raw_mat,
    'Rmean2d': Rmean2d,
    'Rmean3d': Rmean3d,
    'Rmean5d': Rmean5d,
    'labels': labels.astype(object),
    'trials': trials.astype(np.int32),
    'sfreq': sfreq,
    'ch_names': np.array(raw_ica.ch_names, dtype=object),
    'ch_pos': montage_pos,
    'Cmean_ssd': Cmean_ssd,
    'covmats_ssd': covmats_ssd_mat,
    'ts_data_ssd': ts_data_ssd,
    'W_ssd': W_ssd,
    'A_ssd': A_ssd,
    'Wsize': Wsize,
    'Ssize': Ssize
}

sio.savemat(mat_fpath, mdic)
print(f'Данные успешно сохранены для MATLAB в:\n{mat_fpath}')