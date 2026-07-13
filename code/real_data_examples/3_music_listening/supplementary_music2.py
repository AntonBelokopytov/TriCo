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
fpath = "C:/Users/ansbel/Documents/GitHub/TriCo/data/external/music_listening/part1/eeg/10_07_g1_2223_raw.fif"
# fpath = "C:/Users/ansbel/Documents/GitHub/TriCo/data/external/music_listening/part2/eeg/TumAle_raw.fif"

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
# new_descriptions = [
#     'RS_EC_1', 'RS_EO_1', '2Hz', '05Hz', '4Hz', '1Hz', '3Hz',
#     'NoRy_1', 'Waltz_1', 'Waltz_2', 'NoRy_2', 'NoRy_3', 'Waltz_3',
#     'NoRy_4', 'Waltz_4', 'NoRy_5', 'Waltz_5', 'RS_EC_2', 'RS_EO_2',
#     'Waltz_6', 'Waltz_7', 'Waltz_8'
# ]
new_descriptions = [
    'RS_EC_1', 'RS_EO_1', '2Hz', '05Hz', '4Hz', '1Hz', '3Hz',
    'NoRy_1', 'Waltz_1', 'Waltz_2', 'NoRy_2', 'NoRy_3', 'Waltz_3',
    'NoRy_4', 'Waltz_4', 'NoRy_5', 'Waltz_5', 'RS_EC_2', 'RS_EO_2',
]

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
ddd = pairwise_distance(ts_data_ssd, metric='euclid')

# %%
import time
from pyriemann.geometry.distance import distance, pairwise_distance

# Замер для Варианта 1 (две матрицы)
start_time = time.perf_counter()
dist_0_1 = distance(covmats_ssd[0], covmats_ssd[1], metric='riemann')
end_time = time.perf_counter()

print(f"Время выполнения distance: {end_time - start_time:.6f} секунд")


# Замер для Варианта 2 (все эпохи)
start_time = time.perf_counter()
dist_matrix_ssd = pairwise_distance(covmats_ssd, metric='riemann')
end_time = time.perf_counter()

print(f"Время выполнения pairwise_distance: {end_time - start_time:.6f} секунд")

# %%
N = 8
current_dim = covmats_ssd.shape[1]          # исходная размерность после SSD
covmats_current = covmats_ssd.copy()
ts_current = TangentSpace(metric='riemann').fit_transform(covmats_current)

filters_full = []        # итоговые фильтры в полном пространстве
losses = []              # история потерь для каждого компонента (массив по эпохам)
umap_coords_history = [] # UMAP-вложения на каждом этапе
power_history = []       # мощность компонента (на исходных covmats_ssd) для окраски
dims_history = []        # размерность пространства на каждом шаге

# Кумулятивная матрица преобразования из полного пространства в текущее редуцированное
B_cumulative = np.eye(current_dim)

for comp in range(N):
    print(f"\n=== Компонент {comp+1} (текущая размерность {current_dim}) ===")

    # 1. Поиск фильтра в текущем редуцированном пространстве
    w_opt, final_losses, loss_history = fit_filters(
        C=torch.from_numpy(covmats_current).float(),
        # T_features=torch.from_numpy(ts_current).float(),
        D_matrix=dist_matrix_ssd,
        N_dim=2,
        K_restarts=1,
        n_neighbors=20,
        epochs=50,
        lr=0.05
    )
    losses.append(loss_history)         # сохраняем кривую обучения
    v_np = w_opt[0, 0, :]               # фильтр в текущем (редуцированном) базисе

    # Проекция фильтра обратно в исходное полное пространство (SSD-компоненты)
    w_full = B_cumulative @ v_np
    filters_full.append(w_full)

    # Мощность найденного источника на исходных (нередуцированных) ковариационных матрицах
    p_comp = np.array([np.log(w_full.T @ covmats_ssd[i] @ w_full) for i in range(covmats_ssd.shape[0])])
    power_history.append(p_comp)

    # UMAP для текущего касательного пространства (до дефляции)
    if ts_current.shape[1] >= 2:
        umap_step = umap.UMAP(n_components=2, random_state=42)
        coords = umap_step.fit_transform(ts_current)
    else:
        coords = None  # одномерное пространство, нельзя вложить в 2D
    umap_coords_history.append(coords)
    dims_history.append(current_dim)

    # 2. Дефляция (удаление найденного направления) – кроме последнего шага
    Cmean = np.mean(covmats_current,0)
    if comp < N - 1:
        P = np.eye(current_dim) - np.outer(Cmean @ v_np, v_np) / (v_np.T @ Cmean @ v_np)
        C_def = P @ Cmean @ P.T
        S, U = np.linalg.eigh(C_def)
        Pu = U[:, 1:]       
        
        covmats_new = np.zeros((covmats_current.shape[0], current_dim-1, current_dim-1))
        for i in range(covmats_current.shape[0]):
            covmats_new[i] = Pu.T @ covmats_current[i] @ Pu        
        
        B_cumulative = B_cumulative @ Pu
        current_dim -= 1
        covmats_current = covmats_new
        ts_current = TangentSpace(metric='riemann').fit_transform(covmats_current)

    print(f"  Потери: {final_losses[-1]:.4f}")

# %%
# ========== График 1: кривые обучения ==========
plt.figure(figsize=(10, 6))
for idx, loss in enumerate(losses):
    plt.plot(loss, label=f'Компонент {idx+1}')
plt.xlabel('Эпоха оптимизации')
plt.ylabel('Loss')
plt.title('Эволюция функции потерь для каждого фильтра')
plt.legend()
plt.grid(True)
plt.show()

# ========== График 2: UMAP на каждом шаге дефляции ==========
n_steps = len(umap_coords_history)
cols = int(np.ceil(n_steps / 2))
fig, axes = plt.subplots(2, cols, figsize=(4*cols, 8))
axes = axes.flatten()
for step, (coords, dim) in enumerate(zip(umap_coords_history, dims_history)):
    ax = axes[step]
    if coords is None:
        ax.text(0.5, 0.5, f'Шаг {step}\n(одномерное пространство)', ha='center', va='center')
        ax.set_title(f'Шаг {step}')
    else:
        p_vals = power_history[step]
        p_z = (p_vals - p_vals.mean()) / p_vals.std()
        ax.scatter(coords[:, 0], coords[:, 1], c=p_z, cmap='plasma', s=10)
        ax.set_title(f'Шаг {step} (размерность {dim})')
    ax.set_xticks([])
    ax.set_yticks([])
# Скрываем лишние панели, если есть
for ax in axes[n_steps:]:
    ax.set_visible(False)
plt.suptitle('Эволюция топологии касательного пространства по шагам дефляции', fontsize=14)
plt.tight_layout()
plt.show()


# %%
reducer = umap.UMAP(n_components=2)
umap_coords = reducer.fit_transform(ts_data_ssd)

# %%
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

comp_idx = 0 # выберите нужный компонент

w_comp = filters_full[comp_idx]
A_pattern = A_ssd @ w_comp                   # паттерн в сенсорном пространстве
W_sensor = W_ssd @ w_comp                    # фильтр в сенсорном пространстве
p_vals = power_history[comp_idx]
p_z = (p_vals - p_vals.mean()) / p_vals.std()

# UMAP дефлированного пространства для данного этапа
umap_defl = umap_coords_history[comp_idx]   # может быть None, если dim < 2

# Порядок условий: сохраняем порядок появления, не сортируем
unique_labels_ordered = []
for lab in labels:
    if lab not in unique_labels_ordered:
        unique_labels_ordered.append(lab)

# Цветовая палитра
cmap = plt.cm.get_cmap('tab10', len(unique_labels_ordered))
label_to_color = {lab: cmap(i) for i, lab in enumerate(unique_labels_ordered)}

fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1], height_ratios=[1, 1])

# ---- Верхний ряд ----
# 1. UMAP исходного недефлированного пространства
ax_umap_orig = fig.add_subplot(gs[0, 0])
sc1 = ax_umap_orig.scatter(umap_coords[:, 0], umap_coords[:, 1],
                           c=p_z, cmap='plasma', s=15)
plt.colorbar(sc1, ax=ax_umap_orig, label='z-score log-power')
ax_umap_orig.set_title(f'Компонент {comp_idx+1}\nUMAP исходного пр-ва')
ax_umap_orig.set_xticks([])
ax_umap_orig.set_yticks([])

# 2. Топопаттерн
ax_topo = fig.add_subplot(gs[0, 1])
mne.viz.plot_topomap(A_pattern, raw_ica.info, axes=ax_topo, show=False)
ax_topo.set_title('Топографический паттерн')

# 3. UMAP дефлированного пространства (если доступен)
ax_umap_defl = fig.add_subplot(gs[0, 2])
if umap_defl is not None:
    sc2 = ax_umap_defl.scatter(umap_defl[:, 0], umap_defl[:, 1],
                               c=p_z, cmap='plasma', s=15)
    plt.colorbar(sc2, ax=ax_umap_defl, label='z-score log-power')
    ax_umap_defl.set_title(f'UMAP дефлированного пр-ва\n(шаг {comp_idx})')
else:
    ax_umap_defl.text(0.5, 0.5, 'Размерность < 2\n(нет вложения)',
                      ha='center', va='center', transform=ax_umap_defl.transAxes)
    ax_umap_defl.set_title(f'Дефлированное пр-во (шаг {comp_idx})')
ax_umap_defl.set_xticks([])
ax_umap_defl.set_yticks([])

# ---- Нижний ряд ----
# 4. Пространственный фильтр (сенсорное пространство)
ax_filt = fig.add_subplot(gs[1, 0])
mne.viz.plot_topomap(W_sensor, raw_ica.info, axes=ax_filt, show=False)
ax_filt.set_title('Топография фильтра')

# 5. Огибающая мощности с подписями условий на оси Ox
ax_env = fig.add_subplot(gs[1, 1])
window_idx = np.arange(len(p_z))
ax_env.plot(window_idx, p_z, color='black', lw=0.8)

# Цветная заливка по условиям и сбор меток для оси X
xtick_positions = []
xtick_labels = []
for lab in unique_labels_ordered:
    mask = (labels == lab)
    if not np.any(mask):
        continue
    changes = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    for s, e in zip(starts, ends):
        ax_env.axvspan(s, e-1, facecolor=label_to_color[lab], alpha=0.2, zorder=0)
        # центр интервала
        mid = (s + e - 1) / 2
        xtick_positions.append(mid)
        xtick_labels.append(lab)

# Устанавливаем метки на оси X с поворотом
ax_env.set_xticks(xtick_positions)
ax_env.set_xticklabels(xtick_labels, rotation=90, ha='center', fontsize=8)
ax_env.set_xlabel('Номер окна')
ax_env.set_ylabel('z-score log-power')
ax_env.set_title('Огибающая мощности источника')

# 6. Правая нижняя панель оставляем пустой (можно добавить что-то своё)
ax_extra = fig.add_subplot(gs[1, 2])
ax_extra.axis('off')

plt.suptitle(f'Компонент {comp_idx+1} — исходное и дефлированное вложения', fontsize=16)
plt.tight_layout()
plt.show()

# %%s
# Предполагаем, что у вас уже есть:
# raw_ica — исходный Raw после ICA (или любой другой)
# W_ssd — матрица SSD (n_channels, n_components_ssd)
# filters_full — массив (n_filters, n_components_ssd)

# 1. Берём только EEG-каналы (чтобы размерность совпадала с W_ssd)
raw_eeg = raw_ica.copy().pick_types(eeg=True)
data = raw_eeg.get_data()          # (n_channels, n_samples)
sfreq = raw_eeg.info['sfreq']

# 2. Матрица проекции из сенсорного пространства в компоненты
P = W_ssd @ np.array(filters_full).T         # (n_channels, n_filters)
components = P.T @ data            # (n_filters, n_samples)

# 3. Создаём info для новых каналов
n_filters = components.shape[0]
ch_names = [f'Comp_{i+1:02d}' for i in range(n_filters)]
ch_types = ['eeg'] * n_filters 
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

# 4. Создаём RawArray
raw_components = mne.io.RawArray(components, info)

# 5. Переносим аннотации (если нужно)
new_ann = mne.Annotations(raw_eeg.annotations.onset,raw_eeg.annotations.duration,raw_eeg.annotations.description)
raw_components.set_annotations(new_ann)

# 6. Визуализация
raw_components.copy().filter(l_freq=15, h_freq=25).plot(
    picks='all',  # <--- Ключевое исправление
    n_channels=n_filters, 
    scalings='auto', 
    title='Выделенные компоненты'
)
# Или по отдельности:
# raw_components.plot_psd()
