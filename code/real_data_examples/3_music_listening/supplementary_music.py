# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 17:07:11 2025

@author: anton
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
import umap
import scipy.io
from pyriemann.utils.mean import mean_riemann
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh

# %%
import sys
sys.path.append("C:/Users/ansbel/Documents/GitHub/site-packages/umap_meeg")
from topological_spatial_filter import fit_filters
import torch

# %%
# fpath = "C:/Users/ansbel/Documents/GitHub/TriCo/data/external/music_listening/part1/eeg/10_07_g1_2223_raw.fif"
fpath = "C:/Users/ansbel/Documents/GitHub/TriCo/data/external/music_listening/part2/eeg/TumAle_raw.fif"

raw = mne.io.read_raw_fif(fpath,preload=True)
sfreq = raw.info['sfreq']

# %%
# raw_filt = raw.copy().notch_filter(50).filter(l_freq=0.1,h_freq=70)
raw_filt = raw

# %% 
raw_filt.plot(n_channels=38)

# %%
raw_interpolated = raw_filt.copy().interpolate_bads(reset_bads=True)

# %%
raw_interpolated.plot(n_channels=38)

# %%
raw_interpolated.save()

# %%
ica = ICA(
    n_components=0.999,     
    method='fastica',      
    random_state=42,
    max_iter='auto'
)

ica.fit(raw_interpolated)

# %%
ica.plot_components()
ica.plot_sources(raw_filt)

# %%
raw_ica = ica.apply(raw_interpolated)

# %%
raw_ica.plot(n_channels=38)

# %%
raw_ica = raw

# %%
new_durations = [
    ann['duration'] if ann['description'].startswith('BAD_') else 120.0
    for ann in raw.annotations
]

# Создаем новые аннотации и применяем их к объекту
new_ann = mne.Annotations(
    onset=raw.annotations.onset,
    duration=new_durations,
    description=raw.annotations.description,
    orig_time=raw.annotations.orig_time  # важно сохранить исходную временную привязку
)

raw_ica.set_annotations(new_ann)

print("Длительности аннотаций успешно обновлены!")

# %%
new_descriptions = [
    'RS_EC_1', 'RS_EO_1', '2Hz', '05Hz', '4Hz', '1Hz', '3Hz',
    'NoRy_1', 'Waltz_1', 'Waltz_2', 'NoRy_2', 'NoRy_3', 'Waltz_3',
    'NoRy_4', 'Waltz_4', 'NoRy_5', 'Waltz_5', 'RS_EC_2', 'RS_EO_2',
    'Waltz_6', 'Waltz_7', 'Waltz_8'
]
# new_descriptions = [
#     'RS_EC_1', 'RS_EO_1', '2Hz', '05Hz', '4Hz', '1Hz', '3Hz',
#     'NoRy_1', 'Waltz_1', 'Waltz_2', 'NoRy_2', 'NoRy_3', 'Waltz_3',
#     'NoRy_4', 'Waltz_4', 'NoRy_5', 'Waltz_5', 'RS_EC_2', 'RS_EO_2',
# ]

descriptions = raw_ica.annotations.description

# Индексы значимых аннотаций (не BAD, не EDGE)
significant_mask = np.array([('BAD' not in d and 'EDGE' not in d) for d in descriptions])
significant_indices = np.where(significant_mask)[0]

# Проверка соответствия
if len(significant_indices) != len(new_descriptions):
    raise ValueError(
        f"Несоответствие: {len(significant_indices)} значимых, ожидается {len(new_descriptions)}"
    )

# Создаём новый список описаний
new_desc = list(descriptions)
for idx, label in zip(significant_indices, new_descriptions):
    new_desc[idx] = label

# Создаём новый объект Annotations с обновлёнными описаниями
old_annot = raw_ica.annotations
new_annot = mne.Annotations(
    onset=old_annot.onset,
    duration=old_annot.duration,
    description=np.array(new_desc, dtype='U20'),  # тип с запасом по длине
    orig_time=old_annot.orig_time
)

# Применяем новые аннотации к raw_ica
raw_ica.set_annotations(new_annot)

# Проверяем изменение
print(raw_ica.annotations.description)

# %% 
# Получаем данные после ICA (все каналы EEG)
raw_clean = raw_ica.copy().pick_types(eeg=True)
data = raw_ica.get_data()  

# Списки для хранения отфильтрованных и обрезанных кусков
signal_pieces = []
noise_pieces = []
times_pieces = []

# Параметры фильтров
b_signal, a_signal = butter(3, np.array([15, 25]) / (int(sfreq) / 2), btype='band')
b_broad, a_broad = butter(3, np.array([13, 30]) / (int(sfreq) / 2), btype='band')
b_stop, a_stop = butter(3, np.array([14.5, 25.5]) / (int(sfreq) / 2), btype='stop')
# b_signal, a_signal = butter(3, np.array([8, 12]) / (int(sfreq) / 2), btype='band')
# b_broad, a_broad = butter(3, np.array([6, 14]) / (int(sfreq) / 2), btype='band')
# b_stop, a_stop = butter(3, np.array([7.5, 12.5]) / (int(sfreq) / 2), btype='stop')

# Длительность для обрезания краёв (в секундах)
crop_duration = 0.5  # 0.5 секунды с каждой стороны
crop_samples = int(crop_duration * sfreq)

# Проходим по аннотациям
for annot in raw_ica.annotations:
    desc = annot['description']
    if desc == 'BAD_':
    # if desc == 'BAD_' or desc == 'RS_EC_1' or desc == 'RS_EC_2':
        continue  
    duration = annot['duration']
    if duration < 2.0:  # слишком короткий блок - пропускаем
        continue
    print(desc)
    
    tmin = annot['onset']
    tmax = tmin + duration
    if tmax > raw_ica.times[-1]:
        tmax = raw_ica.times[-1]
    
    # Вырезаем кусок данных
    start_idx = int(tmin * sfreq)
    end_idx = int(tmax * sfreq)
    seg_data = data[:, start_idx:end_idx]
    
    # Фильтрация сигнала (15–25 Гц)
    seg_signal = filtfilt(b_signal, a_signal, seg_data, axis=1)
    
    # Фильтрация шума: широкополосный 13–27 Гц, затем режекция
    seg_noise_broad = filtfilt(b_broad, a_broad, seg_data, axis=1)
    seg_noise = filtfilt(b_stop, a_stop, seg_noise_broad, axis=1)
    
    # Обрезаем края для удаления переходных процессов
    if seg_signal.shape[1] > 2 * crop_samples:
        seg_signal_cropped = seg_signal[:, crop_samples:-crop_samples]
        seg_noise_cropped = seg_noise[:, crop_samples:-crop_samples]
    else:
        # Если блок слишком короткий, не обрезаем (или пропускаем)
        seg_signal_cropped = seg_signal
        seg_noise_cropped = seg_noise
    
    # Добавляем в общий список
    signal_pieces.append(seg_signal_cropped)
    noise_pieces.append(seg_noise_cropped)

# Сшиваем все куски в один длинный массив
if signal_pieces:
    signal_concatenated = np.concatenate(signal_pieces, axis=1)
    noise_concatenated = np.concatenate(noise_pieces, axis=1)
    print(f"Сшито {len(signal_pieces)} блоков, общая длина: {signal_concatenated.shape[1]} отсчётов.")
else:
    raise ValueError("Нет подходящих блоков для SSD!")

# Вычисляем ковариации по сшитому сигналу
C_signal = np.cov(signal_concatenated)
C_noise = np.cov(noise_concatenated)

# Регуляризация шумовой ковариации
reg_coeff = 1e-5
C_noise_reg = C_noise + reg_coeff * np.trace(C_noise) * np.eye(C_noise.shape[0])

# Обобщённая проблема собственных значений
eigvals, eigvecs = eigh(C_signal, C_noise_reg)
idx_sorted = np.argsort(eigvals)[::-1]
n_components_ssd = 30
W_ssd = eigvecs[:, idx_sorted[:n_components_ssd]]
A_ssd = C_signal @ W_ssd

print(f"SSD выполнено на сшитых блоках, получено {n_components_ssd} компонент.")

# %% 
Wsize = 2
Ssize = 0.5
overlap = Wsize - Ssize 

X_windows_band = []
X_windows_ssd = []
X_windows_unfilt = []
 
labels = []
trials = [] 
trial_id = 1

raw_band = raw_ica.copy().filter(l_freq=15, h_freq=25).pick_types(eeg=True)
raw_unfilt = raw_ica.copy().pick_types(eeg=True)

for annot in raw_ica.annotations:
    desc = annot['description']
        
    base_cond = desc

    if annot['duration'] < Wsize or desc == 'BAD_':
    # if annot['duration'] < Wsize or desc == 'BAD_' or desc == 'RS_EC_1' or desc == 'RS_EC_2':
        continue
    print(desc)

    tmin = annot['onset']
    tmax = tmin + annot['duration']
    if tmax > raw_ica.times[-1]:
        tmax = raw_ica.times[-1]
    
    raw_crop = raw_band.copy().crop(tmin=tmin, tmax=tmax)
    raw_crop_unfilt = raw_unfilt.copy().crop(tmin=tmin, tmax=tmax)
    
    epochs_band = mne.make_fixed_length_epochs(
        raw_crop, 
        duration=Wsize, 
        overlap=overlap, 
        preload=True, 
        reject_by_annotation=True, 
        verbose=False
    )
    epochs_band.drop_bad(verbose=False)

    epochs_ssd = mne.make_fixed_length_epochs(
        raw_crop, 
        duration=Wsize, 
        overlap=overlap, 
        preload=True, 
        reject_by_annotation=True, 
        verbose=False
    )
    epochs_ssd.drop_bad(verbose=False)

    epochs_unfilt = mne.make_fixed_length_epochs(
        raw_crop_unfilt, 
        duration=Wsize, 
        overlap=overlap, 
        preload=True, 
        reject_by_annotation=True, 
        verbose=False
    )
    epochs_unfilt.drop_bad(verbose=False)
    
    if epochs_band:            
        if len(epochs_band) > 0 and len(epochs_band) == len(epochs_unfilt):
            X_windows_band.append(epochs_band.get_data(copy=False))
            X_windows_unfilt.append(epochs_unfilt.get_data(copy=False)) 
            labels.extend([base_cond] * len(epochs_band))
            trials.extend([trial_id] * len(epochs_band))
            trial_id += 1 

if len(X_windows_band) > 0:
    X_windows_band = np.concatenate(X_windows_band, axis=0) 
    X_windows_unfilt = np.concatenate(X_windows_unfilt, axis=0)
    labels = np.array(labels)
    trials = np.array(trials)
else:
    raise ValueError("После удаления артефактов не осталось ни одного чистого окна!")

# %%
# =====================================================================
# ПРОЕКЦИЯ ЭПОХ В SSD-ПРОСТРАНСТВО
# =====================================================================
# X_windows_band имеет размер (n_windows, n_channels, n_times)
n_windows, n_ch, n_times = X_windows_band.shape
X_windows_ssd_proj = np.zeros((n_windows, n_components_ssd, n_times))
for i in range(n_windows):
    X_windows_ssd_proj[i] = W_ssd.T @ X_windows_band[i]   # (n_comp, time)

print(f"Эпохи спроецированы в SSD-пространство, размер: {X_windows_ssd_proj.shape}")

# %%
covmats_band = Covariances(estimator='oas').fit_transform(X_windows_band)

print("Вычисление ковариаций на SSD-эпохах...")
covmats_ssd = Covariances(estimator='oas').fit_transform(X_windows_ssd_proj)

print("Проекция в касательное пространство...")
ts_data_ssd = TangentSpace(metric='riemann').fit_transform(covmats_ssd)

Cmean_ssd = mean_riemann(covmats_ssd)

# %%
w_opt, final_losses = fit_filters(
    C=torch.from_numpy(covmats_ssd).float(),          
    T_features=torch.from_numpy(ts_data_ssd).float(), 
    K=1,
    n_neighbors=15,
    epochs=100,
    lr=0.05
)

# %%
Env = []
# Переносим веса на CPU и приводим к типу Double (float64)
w_vector = w_opt[0, :, 0].double()

for i in range(covmats_ssd.shape[0]):
    # Превращаем матрицу ковариации в тензор PyTorch того же типа
    cov_tensor = torch.from_numpy(covmats_ssd[i, :, :]).double()
    
    # Вычисляем значение (транспонирование .T для 1D-вектора в PyTorch не требуется)
    val = w_vector @ cov_tensor @ w_vector
    
    # Добавляем в список как обычное число Python
    Env.append(val.item())

# %%
# UMAP на SSD-данных (с добавлением нулевой точки)
n_features = ts_data_ssd.shape[1]
zero_point = np.zeros((1, n_features))
ts_data_with_zero = np.vstack([ts_data_ssd, zero_point])

reducer10d = umap.UMAP(n_neighbors=20, n_components=10, min_dist=0.1, metric='euclidean', output_metric='euclidean')
R10d_all = reducer10d.fit_transform(ts_data_with_zero)

reducer3d = umap.UMAP(n_neighbors=20, n_components=3, min_dist=0.1, metric='euclidean', output_metric='euclidean')
R3d_all = reducer3d.fit_transform(ts_data_with_zero)

reducer2d = umap.UMAP(n_neighbors=20, n_components=2, min_dist=0.1, metric='euclidean', output_metric='euclidean')
R2d_all = reducer2d.fit_transform(ts_data_with_zero)

Rmean10d = R10d_all[:-1, :] - R10d_all[-1, :]
Rmean3d = R3d_all[:-1, :] - R3d_all[-1, :]
Rmean2d = R2d_all[:-1, :] - R2d_all[-1, :]

print("UMAP на SSD-данных завершён.")

# %%
fig = plt.figure()
ax = fig.add_subplot(projection='3d') 

img = ax.scatter(Rmean3d[:,0],Rmean3d[:,1],Rmean3d[:,2])

# %%
plt.scatter(Rmean2d[:,0], Rmean2d[:,1])

# %%
# =====================================================================
# ПОДГОТОВКА ДАННЫХ ДЛЯ MATLAB
# =====================================================================
# SSD-эпохи: (time, comp, windows)
X_ssd_mat = np.transpose(X_windows_ssd_proj, (2, 1, 0))

# Сырые эпохи: (time, ch, windows) – уже есть X_windows_unfilt
X_raw_mat = np.transpose(X_windows_unfilt, (2, 1, 0))

# Ковариации SSD-эпох: (comp, comp, windows)
covmats_ssd_mat = np.transpose(covmats_ssd, (1, 2, 0))
covmats_band_mat = np.transpose(covmats_band, (1, 2, 0))

labels_arr = np.array(labels, dtype=object)
ch_names_arr = np.array(raw_ica.ch_names[:38], dtype=object)

mdic = {
    'X_ssd': X_ssd_mat,               # для mSPoC
    'X_raw': X_raw_mat,               # для TF-карт
    'Rmean2d': Rmean2d,
    'Rmean3d': Rmean3d,
    'Rmean10d': Rmean10d,
    'labels': labels_arr,
    'trials': trials,
    'sfreq': sfreq,
    'ch_names': ch_names_arr,
    'Cmean_ssd': Cmean_ssd,
    'covmats_ssd': covmats_ssd_mat,
    'covmats_band': covmats_band_mat,
    'ts_data_ssd': ts_data_ssd,
    'W_ssd': W_ssd,                   # фильтры (ch, n_comp)
    'A_ssd': A_ssd,                   # паттерны (ch, n_comp)
}

# Координаты электродов (если есть)
montage = raw_ica.get_montage()
if montage is not None:
    pos = montage.get_positions()['ch_pos']
    ch_pos = np.array([pos[ch] for ch in raw_ica.ch_names[:38]])
    mdic['ch_pos'] = ch_pos

scipy.io.savemat('results_for_matlab.mat', mdic)
print("Файл results_for_matlab.mat сохранён с SSD-результатами.")

# %%






