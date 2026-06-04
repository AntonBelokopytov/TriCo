%% =====================================================================
% DISCOVERING CANONICAL NETWORKS BY MAXIMUM INTER-SUBJECT CONSISTENCY (ISC)
% =====================================================================
close all; clear; clc;

% --- 1. НАСТРОЙКИ И ПУТИ ---
base_dirs = {
    'C:\Users\ansbel\Documents\GitHub\TriCo\data\external\music_listening\part1\', ...
    'C:\Users\ansbel\Documents\GitHub\TriCo\data\external\music_listening\part2\'
};
freq_name = 'beta';

% Параметры поиска
N_spatial_clusters = 10; % На сколько исходных кусков бьем пространственные паттерны
Min_Subj_Required = 5;   % Минимальное число людей в кластере, чтобы считать его сетью
Top_N_to_Plot = 10;       % Сколько лучших сетей по ISC показать в итоге

target_conditions = {
    'EC1', 'EO1', '2Hz', '05Hz', '4Hz', '1Hz', '3Hz', ...
    'NoRy 1', 'Waltz 1', 'Waltz 2', 'NoRy 2', 'NoRy 3', ...
    'Waltz 3', 'NoRy 4', 'Waltz 4', 'NoRy 5', 'Waltz 5', ...
    'EC2', 'EO2'
};
N_common = length(target_conditions);

%% --- 2. ПАСС 1: ПОИСК МАКСИМАЛЬНОЙ ДЛИНЫ ТАЙМЛАЙНА ---
disp('>>> ШАГ 1: ВЫРАВНИВАНИЕ АБСОЛЮТНОЙ ШКАЛЫ ВРЕМЕНИ <<<');
Max_Windows = 0;
valid_files = {}; valid_bases = {};

for p = 1:length(base_dirs)
    emb_dir = fullfile(base_dirs{p}, 'embeddings', freq_name);
    mat_files = dir(fullfile(emb_dir, ['UMAP_*_', freq_name, '.mat']));
    for f = 1:length(mat_files)
        fname = mat_files(f).name;
        base = strrep(fname, 'UMAP_', ''); base = strrep(base, ['_', freq_name, '.mat'], '');
        load(fullfile(emb_dir, fname), 'valid_windows');
        if length(valid_windows) > Max_Windows, Max_Windows = length(valid_windows); end
        valid_files{end+1} = fullfile(emb_dir, fname);
        valid_bases{end+1} = base;
    end
end
N_subj = length(valid_files);
fprintf('Найдено %d испытуемых. Максимальный таймлайн: %d окон.\n', N_subj, Max_Windows);

%% --- 3. ПАСС 2: СБОР ПАТТЕРНОВ И ОГИБАЮЩИХ (С УЧЕТОМ NaN) ---
disp('>>> ШАГ 2: СБОР ЗНАЧИМЫХ КОМПОНЕНТ <<<');

All_A = [];
All_Envelopes_Abs = {};
All_Corr_Values = [];
All_Subj_IDs = {};

for s = 1:N_subj
    stats_dir = strrep(fileparts(valid_files{s}), 'embeddings', 'stats');
    stat_fname = fullfile(stats_dir, sprintf('STATS_%s_%s.mat', valid_bases{s}, freq_name));
    
    if exist(stat_fname, 'file')
        load(valid_files{s}, 'Covs', 'valid_windows', 'U');
        st = load(stat_fname);
        if isvector(st.corrs_true), st.corrs_true = st.corrs_true(:)'; end
        [sig_gl, sig_lcl] = find((st.corrs_true > st.max_val) | (st.corrs_true < st.min_val));
        
        Covs_valid = Covs(:, :, valid_windows);
        
        for k = 1:length(sig_gl)
            gl_idx = sig_gl(k); lcl_idx = sig_lcl(k);
            if ndims(st.A_true) == 3, a_pca = squeeze(st.A_true(gl_idx, :, lcl_idx))'; w_pca = squeeze(st.W_true(gl_idx, :, lcl_idx))';
            else, a_pca = st.A_true(:, lcl_idx); w_pca = st.W_true(:, lcl_idx); end
            
            % Пространственный паттерн
            a_sens = U * a_pca;
            [~, m_idx] = max(abs(a_sens)); a_sens = a_sens * sign(a_sens(m_idx));
            All_A(:, end+1) = a_sens;
            
            % Огибающая (Z-score)
            S_raw = zeros(1, size(Covs_valid, 3));
            for i = 1:size(Covs_valid, 3), S_raw(i) = w_pca' * Covs_valid(:, :, i) * w_pca; end
            S_raw = (S_raw - mean(S_raw)) / std(S_raw); 
            
            % Вставляем огибающую на АБСОЛЮТНЫЙ таймлайн с NaN-ами
            S_abs = NaN(1, Max_Windows);
            S_abs(valid_windows) = S_raw;
            
            All_Envelopes_Abs{end+1} = S_abs;
            All_Corr_Values(end+1) = st.corrs_true(gl_idx, lcl_idx);
            All_Subj_IDs{end+1} = valid_bases{s};
        end
    end
end
fprintf('Собрано %d пространственных паттернов.\n', size(All_A, 2));

%% --- 4. ПРОСТРАНСТВЕННАЯ КЛАСТЕРИЗАЦИЯ И РАСЧЕТ ISC ---
disp('>>> ШАГ 3: ПОИСК СЕТЕЙ С МАКСИМАЛЬНОЙ СИНХРОНИЕЙ (ISC) <<<');

% Кластеризуем паттерны (модуль корреляции, инвариантно к знаку)
C_spat = corr(All_A);
D_spat = squareform(1 - abs(C_spat) - diag(diag(1 - abs(C_spat))));
spat_cluster_idx = cluster(linkage(D_spat, 'average'), 'maxclust', N_spatial_clusters);

Cluster_Mean_ISC = zeros(N_spatial_clusters, 1);
Cluster_Pruned_Indices = cell(N_spatial_clusters, 1);
Cluster_ISC_Matrices = cell(N_spatial_clusters, 1);

for c = 1:N_spatial_clusters
    raw_idx = find(spat_cluster_idx == c);
    subjs_in_raw = All_Subj_IDs(raw_idx);
    unique_subj_cluster = unique(subjs_in_raw);
    
    % Пропускаем, если мало людей для оценки синхронии
    if length(unique_subj_cluster) < Min_Subj_Required
        Cluster_Mean_ISC(c) = NaN; continue; 
    end
    
    % Внутрисубъектный прунинг: 1 человек = 1 лучшая огибающая
    idx_pruned = [];
    for us = 1:length(unique_subj_cluster)
        subj_mask = strcmp(subjs_in_raw, unique_subj_cluster{us});
        subj_comp_indices = raw_idx(subj_mask);
        [~, best_sub_idx] = max(abs(All_Corr_Values(subj_comp_indices)));
        idx_pruned(end+1) = subj_comp_indices(best_sub_idx);
    end
    Cluster_Pruned_Indices{c} = idx_pruned;
    
    % Сбор матрицы огибающих
    Env_Matrix = zeros(length(idx_pruned), Max_Windows);
    for i = 1:length(idx_pruned), Env_Matrix(i, :) = All_Envelopes_Abs{idx_pruned(i)}; end
    
    % Считаем ISC (игнорируем NaN артефакты)
    ISC_Matrix = corr(Env_Matrix', 'rows', 'pairwise');
    Cluster_ISC_Matrices{c} = ISC_Matrix;
    
    % Средняя синхрония (только верхний треугольник)
    Cluster_Mean_ISC(c) = mean(ISC_Matrix(triu(true(size(ISC_Matrix)), 1)));
end

%% --- 5. РАНЖИРОВАНИЕ И ВЫБОР ТОП-ПАТТЕРНОВ ---
[sorted_ISC, sort_idx] = sort(Cluster_Mean_ISC, 'descend', 'MissingPlacement', 'last');

disp('--- РЕЙТИНГ КЛЮЧЕВЫХ НЕЙРОСЕТЕЙ ПО СИНХРОНИИ ---');
for i = 1:N_spatial_clusters
    if ~isnan(sorted_ISC(i))
        N_ppl = length(Cluster_Pruned_Indices{sort_idx(i)});
        fprintf('%d. Кластер %d | ISC = %.3f | Найдено у %d испытуемых\n', i, sort_idx(i), sorted_ISC(i), N_ppl);
    end
end

%% --- 6. ВИЗУАЛИЗАЦИЯ ТОП-ПАТТЕРНОВ (ПАСПОРТА СЕТЕЙ) ---
disp('>>> ШАГ 4: ОТРИСОВКА ПАСПОРТОВ СЕТЕЙ-ПОБЕДИТЕЛЕЙ <<<');

% Подготовка FieldTrip
cfg_dum = []; cfg_dum.dataset = fullfile(base_dirs{1}, 'eeg', '10_07_g1_2200_raw.fif'); cfg_dum.continuous = 'yes';
Xinf = ft_preprocessing(cfg_dum); laycfg = []; laycfg.elec = Xinf.elec; lay = ft_prepare_layout(laycfg);     
topo = []; topo.dimord = 'chan_time'; topo.label = Xinf.elec.label(1:38); topo.time = 0;
cfg_topo = []; cfg_topo.marker = 'labels'; cfg_topo.layout = lay; cfg_topo.comment = 'no'; cfg_topo.style = 'fill'; cfg_topo.colorbar = 'yes';

valid_top = sum(~isnan(sorted_ISC));
num_to_plot = min(Top_N_to_Plot, valid_top);

figure('Name', 'Top ISC Canonical Networks', 'Color', 'w', 'Position', [100, 100, 1400, 300 * num_to_plot]);
t = tiledlayout(num_to_plot, 3, 'TileSpacing', 'compact');
sgtitle('Canonical Functional Networks Ranked by Inter-Subject Consistency (ISC)', 'FontWeight', 'bold', 'FontSize', 16);

for i = 1:num_to_plot
    c = sort(sort_idx(i)); % Исходный номер кластера
    c = sort_idx(i);
    idx = Cluster_Pruned_Indices{c};
    isc_mat = Cluster_ISC_Matrices{c};
    mean_isc = sorted_ISC(i);
    
    % Выравниваем паттерны перед усреднением
    c_A = All_A(:, idx);
    ref_A = c_A(:, 1); 
    for j = 1:size(c_A, 2)
        if corr(ref_A, c_A(:, j)) < 0, c_A(:, j) = -c_A(:, j); end
    end
    mean_A = mean(c_A, 2);
    
    % Усредняем огибающую (Mean OmitNaN)
    Env_Matrix = zeros(length(idx), Max_Windows);
    for j = 1:length(idx), Env_Matrix(j, :) = All_Envelopes_Abs{idx(j)}; end
    mean_S = mean(Env_Matrix, 1, 'omitnan');
    
    % --- График 1: Усредненный Топоплот (Анатомия) ---
    ax_topo = nexttile(t, (i-1)*3 + 1);
    topo.avg = mean_A; cfg_topo.figure = ax_topo; ft_topoplotER(cfg_topo, topo);
    title(sprintf('Rank #%d: ISC = %.3f\nFound in %d subjects', i, mean_isc, length(idx)), 'Color', [0.8 0.2 0.2]);

    % --- График 2: Матрица ISC ---
    ax_isc = nexttile(t, (i-1)*3 + 2);
    imagesc(ax_isc, isc_mat, [-0.1 0.7]); colormap(ax_isc, 'parula'); colorbar(ax_isc);
    title('Subject-by-Subject Synchrony'); xlabel('Subject'); ylabel('Subject'); axis(ax_isc, 'square');

    % --- График 3: Групповая динамика мощности ---
    ax_env = nexttile(t, (i-1)*3 + 3); hold on; grid on;
    % Поскольку 5000 точек рисовать густо, сгладим слегка для красоты
    smooth_S = smoothdata(mean_S(~isnan(mean_S)), 'gaussian', 10);
    plot(ax_env, smooth_S, 'LineWidth', 1.5, 'Color', [0.2 0.5 0.8]);
    ylabel('Mean Z-Power'); title('Group Mean Temporal Envelope (NaNs omitted)');
    xlim([1 length(smooth_S)]);
end

disp('>>> АНАЛИЗ ЗАВЕРШЕН УСПЕШНО <<<');