# -*- coding: utf-8 -*-
"""
Оценка временной динамики (десинхронизации) в задаче Center-Out.
Использование скользящего окна внутри длинных 7-секундных эпох.
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
import umap
import seaborn as sns

# %% 1. Загрузка и подготовка данных
fpath = "C:/Users/ansbel/Documents/GitHub/TriCo/data/external/center_out/sub2_center_out_epochs.fif"
epochs = mne.read_epochs(fpath, preload=True)
epochs = epochs.pick_types(eeg=True)
epochs.drop_bad()

# Метки условий
inv_event_id = {v: k for k, v in epochs.event_id.items()}
labels = np.array([inv_event_id[event_id] for event_id in epochs.events[:, 2]])

# %% 2. Фильтрация данных (Бета-ритм 15-25 Гц)
epochs_filtered = epochs.copy().filter(l_freq=9, h_freq=14)
data = epochs_filtered.get_data(copy=False)  # (n_epochs, n_channels, n_times)
times = epochs_filtered.times                # Временная ось (от -1 до 6 сек)
sfreq = epochs_filtered.info['sfreq']

# %% 3. Нарезка на скользящие окна
Wsize_sec = 2  
Ssize_sec = 0.5 

Wsize_samp = int(Wsize_sec * sfreq)
Ssize_samp = int(Ssize_sec * sfreq)

n_epochs, n_channels, n_times = data.shape
start_indices = np.arange(0, n_times - Wsize_samp + 1, Ssize_samp)

X_windows = []
time_labels = []  # Время центра окна
cond_labels = []  # Условие (s1d2, s3d2 и т.д.)
trial_labels = [] # Номер конкретной попытки (эпохи)

print(f"Нарезка {n_epochs} эпох окном {Wsize_sec}с с шагом {Ssize_sec}с...")

for i in range(n_epochs):
    cond = labels[i]
    for start_idx in start_indices:
        end_idx = start_idx + Wsize_samp
        window_data = data[i, :, start_idx:end_idx]
        
        # Центр окна в секундах (например, от -0.5 до 5.5)
        center_time = times[start_idx + Wsize_samp // 2]
        
        X_windows.append(window_data)
        time_labels.append(center_time)
        cond_labels.append(cond)
        trial_labels.append(i)

X_windows = np.array(X_windows)
time_labels = np.array(time_labels)
cond_labels = np.array(cond_labels)
trial_labels = np.array(trial_labels)

print(f"Получено {len(X_windows)} маленьких окон.")

# %% 4. Вычисление ковариаций и проекция в Tangent Space
print("Вычисление ковариационных матриц...")
covmats = Covariances(estimator='oas').fit_transform(X_windows)

print("Проекция в касательное пространство...")
ts_data = TangentSpace(metric='riemann').fit_transform(covmats)

# %% 5. Обучение UMAP
print("Обучение UMAP...")
reducer = umap.UMAP(
    n_neighbors=20, 
    n_components=2, 
    min_dist=0.1, 
    metric='euclidean',
    random_state=42
)
embedding = reducer.fit_transform(ts_data)

# %% 6. Визуализация: Раскраска по времени
plt.figure(figsize=(12, 8))

# Рисуем все точки, цвет зависит от времени окна (time_labels)
scatter = plt.scatter(
    embedding[:, 0], 
    embedding[:, 1], 
    c=time_labels,       # Раскрашиваем по времени!
    cmap='viridis',      # От фиолетового (начало) к желтому (конец)
    s=15, 
    alpha=0.7,
    edgecolors='none'
)

# Добавляем цветовую шкалу
cbar = plt.colorbar(scatter)
cbar.set_label('Время относительно начала эпохи (секунды)', fontsize=12)

# Опционально: Нарисуем траекторию ОДНОЙ эпохи (например, первой), чтобы было видно движение
first_trial_idx = np.where(trial_labels == 0)[0]
plt.plot(
    embedding[first_trial_idx, 0], 
    embedding[first_trial_idx, 1], 
    color='red', 
    linewidth=1.5, 
    linestyle='-',
    label='Пример траектории одной эпохи'
)
# Отметим старт траектории
plt.scatter(embedding[first_trial_idx[0], 0], embedding[first_trial_idx[0], 1], color='red', marker='x', s=100, label='Start')

plt.title('Траектория десинхронизации (UMAP), раскраска по времени', fontsize=14)
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %% 7. Опционально: Сохранение для MATLAB
import scipy.io as sio
# sio.savemat('center_out_umap_time.mat', {
#     'ts_data': ts_data,
#     'embedding': embedding,
#     'time_labels': time_labels,
#     'cond_labels': cond_labels,
#     'trial_labels': trial_labels
# })