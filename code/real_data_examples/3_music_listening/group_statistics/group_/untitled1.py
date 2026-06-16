# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 03:39:24 2026

@author: ansbel
"""

import numpy as np
import umap
import scipy.io as sio
import matplotlib.pyplot as plt

# 1. Загрузка данных из MATLAB
# Убедись, что Mean_Dist_Matrix сохранена в .mat файл
data = sio.loadmat('code/real_data_examples/3_music_listening/group_statistics/Mean_Dist_Matrix.mat')
dist_matrix = data['Mean_Dist_Matrix']

# 2. Инициализация UMAP для гиперболического пространства
# В библиотеке UMAP для Python вложение в гиперболоид 
# включается через метрику 'hyperboloid'
reducer = umap.UMAP(
    n_components=2,           # Для Пуанкаре диска нужно 2 компоненты
    metric='precomputed',     # Так как у нас матрица расстояний
    output_metric='hyperboloid', # Ключевой момент!
    n_neighbors=30,
    min_dist=0.1,
    random_state=42
)

# 3. Расчет вложения
embedding = reducer.fit_transform(dist_matrix)

# 4. Проекция из гиперболоида в диск Пуанкаре (Poincare disk)
# UMAP возвращает координаты в модели гиперболоида (x, y, z),
# где z^2 - x^2 - y^2 = 1.
# Проекция на диск: x_disk = x / (1 + z), y_disk = y / (1 + z)
z = np.sqrt(1 + np.sum(embedding**2, axis=1))
disk_x = embedding[:, 0] / (1 + z) 
disk_y = embedding[:, 1] / (1 + z)

# 5. Визуализация
plt.figure(figsize=(10, 10))
# Рисуем границу диска
circle = plt.Circle((0, 0), 1, color='k', fill=False, lw=2)
plt.gca().add_patch(circle)

# %%
# --- Определение параметров (как в MATLAB) ---
N_common = 19
max_wins_per_cond = 217

# Создаем метки: [1, 1, ..., 1 (217 раз), 2, 2, ..., 19, 19]
cond_labels = np.repeat(np.arange(1, N_common + 1), max_wins_per_cond)

# Если размерность dist_matrix меньше, чем Total_Epochs (из-за NaN в матрице расстояний),
# нужно обрезать cond_labels, чтобы они совпадали по длине
if len(cond_labels) > dist_matrix.shape[0]:
    cond_labels = cond_labels[:dist_matrix.shape[0]]
# Рисуем точки
# Здесь можешь использовать свои метки условий (cond_labels)
plt.scatter(embedding[:, 0], embedding[:, 1], c=cond_labels, cmap='jet', s=5, alpha=0.5)

plt.axis('equal')
plt.axis('off')
plt.title('UMAP Hyperbolic Embedding (Poincare Disk)')
plt.show()