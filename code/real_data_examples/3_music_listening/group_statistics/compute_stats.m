%% =====================================================================
% BATCH STATS PIPELINE (eSPoC SAMPLE-SHIFT PERMUTATION TEST)
% =====================================================================
close all; clear; clc;

% --- 1. Настройка путей и параметров ---
data_dir       = 'C:\Users\ansbel\Documents\GitHub\TriCo\data\external\music_listening\part2\eeg\';
base_emb_dir   = 'C:\Users\ansbel\Documents\GitHub\TriCo\data\external\music_listening\part2\embeddings\';
base_stats_dir = 'C:\Users\ansbel\Documents\GitHub\TriCo\data\external\music_listening\part2\stats\';
ft_path        = 'C:\Users\ansbel\Documents\GitHub\site-packages\fieldtrip';

freq_name = 'beta';         
freq_band = [15, 25];       
Wsize     = 2;              
Ssize     = 0.5;            
cond_len  = 120;
nMC       = 1000;            % Количество итераций
alpha_lvl = 0.05;           % Уровень значимости

% --- АВТОМАТИЧЕСКАЯ МАРШРУТИЗАЦИЯ ПАПОК ПО ДИАПАЗОНУ ---
emb_dir = fullfile(base_emb_dir, freq_name);
stats_dir = fullfile(base_stats_dir, freq_name);

if ~exist(stats_dir, 'dir')
    mkdir(stats_dir);
end

if ~exist('ft_defaults','file'), addpath(ft_path); end
ft_defaults;

% Загрузка матрицы BADS (нужна для пересоздания маски)
load(fullfile(data_dir, 'BADS.mat'));

% Ищем все UMAP файлы для заданного диапазона в соответствующей подпапке
mat_files = dir(fullfile(emb_dir, ['UMAP_*_', freq_name, '.mat']));
disp(['Найдено файлов для статистики: ', num2str(length(mat_files))]);

%% --- 2. Главный цикл по файлам ---
for f = 1:length(mat_files)
    
    mat_name = mat_files(f).name;
    base_name = strrep(mat_name, 'UMAP_', '');
    base_name = strrep(base_name, ['_', freq_name, '.mat'], '');
    raw_fname = [base_name, '_raw.fif'];
    
    disp('==================================================');
    disp(['Запуск статистики для: ', base_name, ' [', freq_name, ']']);
    
    % 1. Загрузка UMAP (нам нужны R, U и valid_windows)
    load(fullfile(emb_dir, mat_name), 'R', 'U', 'valid_windows', 'nEpochs');
    
    % 2. Загрузка и подготовка сырых данных
    cfg = [];
    cfg.dataset = fullfile(data_dir, raw_fname);
    cfg.continuous = 'yes';
    Xinf = ft_preprocessing(cfg);
    Fs = Xinf.fsample;
    
    Xraw = Xinf.trial{1}(1:size(U,1), :); 
    total_samples = size(Xraw, 2);
    
    % Воссоздание исходных масок
    mask_ts = true(1, total_samples);
    cond_ts = zeros(1, total_samples);
    samples_per_cond = round(cond_len * Fs);
    for k = 1:nEpochs
        idx_st = (k-1) * samples_per_cond + 1;
        idx_en = min(k * samples_per_cond, total_samples);
        cond_ts(idx_st:idx_en) = k;
    end
    
    mat_field_name = ['s_', base_name];
    if exist(mat_field_name, 'var')
        bad_intervals = eval(mat_field_name);
        for i = 1:size(bad_intervals, 1)
            bad_st_samp = max(1, round(bad_intervals(i, 1) * Fs) + 1);
            bad_en_samp = min(total_samples, round((bad_intervals(i, 1) + bad_intervals(i, 2)) * Fs));
            mask_ts(bad_st_samp : bad_en_samp) = false;
        end
    end
    
    % Фильтрация и применение сохраненного SVD (матрица U)
    [b, a] = butter(3, freq_band / (Fs/2));   
    Xfilt = filtfilt(b, a, Xraw')'; 
    Xfilt = Xfilt(:, round(Fs/2) : end - round(Fs/2));
    mask_ts = mask_ts(:, round(Fs/2) : end - round(Fs/2));
    cond_ts = cond_ts(:, round(Fs/2) : end - round(Fs/2));
    
    Xfiltpca = U' * Xfilt;
    
    % Нарезка истинных данных
    ep_wins = epoch_data(Xfiltpca', Fs, Wsize, Ssize);
    X_true = ep_wins(:, :, valid_windows);
    Z_true = R'; 
    
    %% --- 3. ИСТИННЫЙ eSPoC ---
    disp('Расчет eSPoC на истинных данных...');
    [W_true, A_true, Vf_true, Vz_true, corrs_true, eigenvalues_true, cca_corrs_true] = espoc(X_true, Z_true);
    
    %% --- 4. ПЕРЕСТАНОВОЧНЫЙ ТЕСТ (СДВИГ СЭМПЛОВ) ---
    disp(['Запуск Permutation Test (', num2str(nMC), ' итераций)...']);
    
    corrmax = zeros(3, nMC);
    corrmin = zeros(3, nMC);
    
    parfor (i = 1:nMC, 4)
        % 1. Случайный сдвиг сэмплов
        r_idx = randi([1, size(Xfiltpca, 2)]);
        XCirc = circshift(Xfiltpca, r_idx, 2);
        mask_shifted = circshift(mask_ts, r_idx, 2);
        cond_shifted = circshift(cond_ts, r_idx, 2);
        
        % 2. Нарезка сдвинутых данных
        ep_wins_shift = epoch_data(XCirc', Fs, Wsize, Ssize);
        
        % Добавляем squeeze, чтобы убрать лишнее измерение [W x 1 x E] -> [W x E]
        m_wins_shift = squeeze(epoch_data(double(mask_shifted'), Fs, Wsize, Ssize));
        c_wins_shift = squeeze(epoch_data(cond_shifted', Fs, Wsize, Ssize));
        
        valid_artifacts = all(m_wins_shift, 1);
        valid_boundaries = (c_wins_shift(1, :) == c_wins_shift(end, :));
        valid_shifted = (valid_artifacts & valid_boundaries)';
        
        X_test = ep_wins_shift(:, :, valid_shifted);
        
        % 3. Выравнивание по минимальной длине (усечение хвоста)
        neps = min(size(X_test, 3), size(R, 1));
        
        if neps > 10
            % Отключаем CCA warnings внутри цикла
            warning('off', 'all');
            [~,~,~,~,corrs_perm] = espoc(X_test(:, :, 1:neps), Z_true(:, 1:neps)); 
            warning('on', 'all');
            
            if ~isempty(corrs_perm)
                corrmax(:, i) = max(corrs_perm, [], 2);
                corrmin(:, i) = min(corrs_perm, [], 2);
            end
        end
    end
    
    %% --- 5. ОЦЕНКА ПОРОГОВ ---
    valid_iters = max(corrmax, [], 1) ~= 0;
    actual_nMC = sum(valid_iters);
    disp(['Фактически выполнено итераций: ', num2str(actual_nMC)]);
    
    if actual_nMC > 20
        corrmax_valid = corrmax(:, valid_iters);
        corrmin_valid = corrmin(:, valid_iters);
        
        corrmax_sorted = sort(max(corrmax_valid, [], 1), 'descend');
        corrmin_sorted = sort(min(corrmin_valid, [], 1), 'ascend');
        
        % Считаем индексы для альфа-уровня
        idx_max = max(1, floor(alpha_lvl * actual_nMC));
        idx_min = max(1, floor(alpha_lvl * actual_nMC));
        
        max_val = corrmax_sorted(idx_max);
        min_val = corrmin_sorted(idx_min);
    else
        max_val = NaN; min_val = NaN;
        disp('Недостаточно итераций для расчета порогов.');
    end
    
    disp(['Пороги: Max = ', num2str(max_val, '%.3f'), ' | Min = ', num2str(min_val, '%.3f')]);
    
    %% --- 6. СОХРАНЕНИЕ ВИЗУАЛИЗАЦИИ ---
    disp('Сохранение графиков и матриц...');
    
    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [200, 200, 800, 500]);
    
    % Рисуем истинные корреляции
    stem(corrs_true', 'LineWidth', 1.5); hold on; grid on;
    
    % Рисуем линии порогов
    if ~isnan(max_val)
        yline(max_val, 'r', ['Alpha ', num2str(alpha_lvl), ' Max (', num2str(max_val, '%.2f'), ')'], 'LineWidth', 1.5); 
        yline(min_val, 'b', ['Alpha ', num2str(alpha_lvl), ' Min (', num2str(min_val, '%.2f'), ')'], 'LineWidth', 1.5);
    end
    
    xlabel('Local component index'); ylabel('Correlation');
    title(sprintf('Significance Thresholds: %s [%s]', strrep(base_name, '_', '\_'), freq_name));
    
    % Создаем аккуратную легенду для Global компонент
    leg_labels = arrayfun(@(x) sprintf('Global %d', x), 1:size(corrs_true, 1), 'UniformOutput', false);
    legend(leg_labels, 'Location', 'best');
    xlim([0.5 size(corrs_true, 2)+0.5]);
    
    % 1. Сохраняем JPG
    img_name = sprintf('STATS_%s_%s.jpg', base_name, freq_name);
    saveas(fig, fullfile(stats_dir, img_name));
    
    % 2. Делаем график видимым и сохраняем интерактивный FIG
    set(fig, 'Visible', 'on');
    fig_name = sprintf('STATS_%s_%s.fig', base_name, freq_name);
    saveas(fig, fullfile(stats_dir, fig_name), 'fig');
    
    close(fig);
    
    %% --- 7. СОХРАНЕНИЕ ДАННЫХ В .MAT ---
    stat_out_name = sprintf('STATS_%s_%s.mat', base_name, freq_name);
    
    % Сохраняем все истинные векторы eSPoC + статистику
    save(fullfile(stats_dir, stat_out_name), ...
        'W_true', 'A_true', 'Vf_true', 'Vz_true', 'corrs_true', 'eigenvalues_true', 'cca_corrs_true', ...
        'corrmax', 'corrmin', 'max_val', 'min_val', 'actual_nMC', 'alpha_lvl');
        
    disp(['Данные сохранены в: ', stat_out_name]);
end
disp('==================================================');
disp('РАСЧЕТ СТАТИСТИКИ ЗАВЕРШЕН!');