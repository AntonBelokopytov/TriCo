%% =====================================================================
% PERMUTATION NULL MODEL: CIRCULAR SHIFT EPOCH-BY-EPOCH UMAP
% =====================================================================
close all; clear; clc;

% Инициализируем генератор случайных чисел для воспроизводимости (опционально)
rng('shuffle'); 

% --- 1. НАСТРОЙКИ ---
base_dirs = {
    'C:\Users\ansbel\Documents\GitHub\TriCo\data\external\music_listening\part1', ...
    'C:\Users\ansbel\Documents\GitHub\TriCo\data\external\music_listening\part2'
};

freq_name = 'beta';
N_common = 19;          
Wsize = 2;              
Ssize = 0.5;            

max_wins_per_cond = 217; 
Total_Epochs = N_common * max_wins_per_cond;

target_conditions = {
    'EC1', 'EO1', '2Hz', '05Hz', '4Hz', '1Hz', '3Hz', ...
    'NoRy 1', 'Waltz 1', 'Waltz 2', 'NoRy 2', 'NoRy 3', ...
    'Waltz 3', 'NoRy 4', 'Waltz 4', 'NoRy 5', 'Waltz 5', ...
    'EC2', 'EO2'
};

% --- 2. СБОР ДАННЫХ И РАСЧЕТ ИНДИВИДУАЛЬНЫХ МАТРИЦ РАССТОЯНИЙ ---
disp('>>> НАЧАЛО РАСЧЕТА ПЕРМУТИРОВАННЫХ МАТРИЦ (CIRCULAR SHIFT) <<<');

% Используем оптимизированное накопление памяти
Sum_Dist_Matrix = zeros(Total_Epochs, Total_Epochs);
Count_Dist_Matrix = zeros(Total_Epochs, Total_Epochs);
subj_count = 0;

for p = 1:length(base_dirs)
    b_dir = base_dirs{p};
    emb_dir = fullfile(b_dir, 'embeddings', freq_name);
    mat_files = dir(fullfile(emb_dir, ['UMAP_*_', freq_name, '.mat']));
    
    if contains(b_dir, 'part1')
        cond_len = 110;
    else
        cond_len = 120;
    end
    
    for f = 1:length(mat_files)
        file_path = fullfile(emb_dir, mat_files(f).name);
        try
            data = load(file_path, 'Tcovs_valid', 'valid_windows');
            
            % 1. ВОССТАНОВЛЕНИЕ АБСОЛЮТНОЙ ВРЕМЕННОЙ ШКАЛЫ
            total_file_epochs = length(data.valid_windows);
            cond_idx_epochs = zeros(1, total_file_epochs);
            valid_boundaries = false(1, total_file_epochs);
            
            for k = 1:total_file_epochs
                t_start_trunc = (k-1) * Ssize;
                t_end_trunc = t_start_trunc + Wsize;
                
                t_start_orig = t_start_trunc + 0.5;
                t_end_orig = t_end_trunc + 0.5;
                
                cond_start = floor(t_start_orig / cond_len) + 1;
                cond_end = floor((t_end_orig - 1e-5) / cond_len) + 1;
                
                cond_idx_epochs(k) = cond_start;
                valid_boundaries(k) = (cond_start == cond_end);
            end
            
            % 2. ЗАПОЛНЕНИЕ МАТРИЦЫ ПРИЗНАКОВ T_subj
            n_features = size(data.Tcovs_valid, 1);
            T_subj = NaN(n_features, Total_Epochs);
            
            for c = 1:N_common
                epochs_in_c = find(cond_idx_epochs == c & valid_boundaries);
                epochs_in_c = epochs_in_c(1:min(length(epochs_in_c), max_wins_per_cond));
                
                for k = 1:length(epochs_in_c)
                    orig_epoch_idx = epochs_in_c(k);
                    if data.valid_windows(orig_epoch_idx)
                        valid_idx = sum(data.valid_windows(1:orig_epoch_idx));
                        global_idx = (c - 1) * max_wins_per_cond + k;
                        T_subj(:, global_idx) = data.Tcovs_valid(:, valid_idx);
                    end
                end
            end
            
            % =========================================================
            % МАГИЯ ПЕРМУТАЦИИ: ЦИКЛИЧЕСКИЙ СДВИГ ЭПОХ ДЛЯ ИСПЫТУЕМОГО
            % =========================================================
            % Генерируем случайный сдвиг от 1 до Total_Epochs-1
            shift_val = randi([1, Total_Epochs - 1]);
            
            % Сдвигаем столбцы (эпохи) по кругу. 
            % ВАЖНО: NaN (bads) тоже корректно сдвигаются вместе с данными!
            T_subj = circshift(T_subj, shift_val, 2);
            
            disp(['[~] Испытуемый ', mat_files(f).name, ' сдвинут на ', num2str(shift_val), ' эпох.']);
            % =========================================================

            % 3. РАСЧЕТ И НОРМИРОВКА МАТРИЦЫ РАССТОЯНИЙ
            D_vec = pdist(T_subj', 'euclidean'); 
            D_subj = squareform(D_vec);
            
            std_val = std(D_subj(:), 'omitnan');
            D_subj_norm = D_subj / std_val;
            
            % Накопление суммы (Экономит ОЗУ)
            valid_mask = ~isnan(D_subj_norm);
            Sum_Dist_Matrix(valid_mask) = Sum_Dist_Matrix(valid_mask) + D_subj_norm(valid_mask);
            Count_Dist_Matrix(valid_mask) = Count_Dist_Matrix(valid_mask) + 1;
            
            subj_count = subj_count + 1;
        catch ME
            disp(['[-] Ошибка в файле ', mat_files(f).name, ': ', ME.message]);
        end
    end
end
disp(['ИТОГО: Успешно собрано ', num2str(subj_count), ' сдвинутых матриц.']);

%% --- 3. УСРЕДНЕНИЕ МАТРИЦЫ РАССТОЯНИЙ ---
disp('>>> ШАГ 2: УСРЕДНЕНИЕ ГРУППОВОЙ МАТРИЦЫ <<<');

Mean_Dist_Matrix = Sum_Dist_Matrix ./ Count_Dist_Matrix;

if any(isnan(Mean_Dist_Matrix(:)))
    disp('[!] В матрице остались NaN (тотальный артефакт у всех). Заполняю максимальным...');
    max_d = max(Mean_Dist_Matrix(:));
    Mean_Dist_Matrix(isnan(Mean_Dist_Matrix)) = max_d;
end

Mean_Dist_Matrix(1:size(Mean_Dist_Matrix,1)+1:end) = 0;

%% --- 4. НЕЛИНЕЙНОЕ ВЛОЖЕНИЕ (UMAP И MDS) ---
disp('>>> ШАГ 3: РАСЧЕТ ВЛОЖЕНИЙ (UMAP & MDS) <<<');

[Y_mds, ~] = cmdscale(Mean_Dist_Matrix);
Y_mds = Y_mds(:, 1:3);

try
    [Y_umap, ~] = run_umap(Mean_Dist_Matrix, 'metric', 'precomputed', ...
                           'n_neighbors', 30, 'n_components', 3, 'min_dist', 0.1);
catch
    disp('run_umap не найден, используем объект UMAP...');
    u = UMAP('n_neighbors', 30, 'n_components', 3, 'min_dist', 0.1);
    u.metric = 'precomputed';
    Y_umap = u.fit_transform(Mean_Dist_Matrix);
end

%% --- 5. ВИЗУАЛИЗАЦИЯ ТРАЕКТОРИЙ ---
disp('>>> ШАГ 4: ОТРИСОВКА <<<');

cond_labels = repelem(1:N_common, max_wins_per_cond);
cmap = jet(N_common);

figure('Name', 'NULL MODEL: Epoch-by-Epoch UMAP', 'Color', 'w', 'Position', [100, 100, 1000, 800]);
hold on; grid on;

% Отрисовываем пермутированные данные
for c = 1:N_common
    idx = (cond_labels == c);
    scatter3(Y_umap(idx, 1), Y_umap(idx, 2), Y_umap(idx, 3), 15, cmap(c,:), 'filled', 'MarkerFaceAlpha', 0.6);
end

for c = 1:N_common
    idx = (cond_labels == c);
    cx = mean(Y_umap(idx, 1));
    cy = mean(Y_umap(idx, 2));
    cz = mean(Y_umap(idx, 3));
    
    scatter3(cx, cy, cz, 250, cmap(c,:), 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    text(cx, cy, cz+0.3, target_conditions{c}, 'FontSize', 12, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.95 0.95 0.95], 'HorizontalAlignment', 'center');
end

% Добавляем красный предупреждающий заголовок, чтобы не перепутать графики
title(sprintf('NULL MODEL (RANDOM CIRCULAR SHIFT)\nUMAP-Вложение (N=%d испытуемых)', subj_count), ...
      'FontSize', 16, 'FontWeight', 'bold', 'Color', 'r');
xlabel('UMAP 1'); ylabel('UMAP 2'); zlabel('UMAP 3');
view(-45, 30);

disp('>>> ПЕРМУТАЦИОННЫЙ ПАЙПЛАЙН ЗАВЕРШЕН <<<');