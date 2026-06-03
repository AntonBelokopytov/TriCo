%% =====================================================================
% HIGH-RESOLUTION EPOCH-BY-EPOCH TRAJECTORY & SYNCHRONY ANALYSIS
% =====================================================================
close all; clear; clc;

% --- 1. НАСТРОЙКИ И ПУТИ ---
base_dirs = {
    'C:\Users\ansbel\Documents\GitHub\TriCo\data\external\music_listening\part1\', ...
    'C:\Users\ansbel\Documents\GitHub\TriCo\data\external\music_listening\part2\'
};
freq_name = 'beta';

target_conditions = {
    'EC1', 'EO1', '2Hz', '05Hz', '4Hz', '1Hz', '3Hz', ...
    'NoRy 1', 'Waltz 1', 'Waltz 2', 'NoRy 2', 'NoRy 3', ...
    'Waltz 3', 'NoRy 4', 'Waltz 4', 'NoRy 5', 'Waltz 5', ...
    'EC2', 'EO2'
};

cond_groups = [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1];
group_colors = [0.3 0.3 0.3; 0.8 0.2 0.2; 0.2 0.5 0.9]; 
N_common = length(target_conditions);

%% --- 2. ПАСС 1: ПОИСК МАКСИМАЛЬНОЙ ДЛИНЫ ТАЙМЛАЙНА И СБОР ЦЕНТРОИДОВ ---
disp('>>> ШАГ 1: ВЫРАВНИВАНИЕ АБСОЛЮТНОЙ ШКАЛЫ ВРЕМЕНИ И СБОР БАЗИСА <<<');
Max_Windows = 0;
valid_files = {};
valid_bases = {};

% Сначала просто собираем имена файлов и ищем максимальную длину эксперимента
for p = 1:length(base_dirs)
    emb_dir = fullfile(base_dirs{p}, 'embeddings', freq_name);
    mat_files = dir(fullfile(emb_dir, ['UMAP_*_', freq_name, '.mat']));
    
    for f = 1:length(mat_files)
        fname = mat_files(f).name;
        base = strrep(fname, 'UMAP_', ''); base = strrep(base, ['_', freq_name, '.mat'], '');
        load(fullfile(emb_dir, fname), 'valid_windows');
        
        L = length(valid_windows);
        if L > Max_Windows, Max_Windows = L; end
        valid_files{end+1} = fullfile(emb_dir, fname);
        valid_bases{end+1} = base;
    end
end
N_subj = length(valid_files);
fprintf('Найдено файлов: %d. Максимальное число окон: %d\n', N_subj, Max_Windows);

% Теперь собираем 19 центроидов для каждого, чтобы вычислить Эталон
cc_all = NaN(N_common, 3, N_subj);
for s = 1:N_subj
    load(valid_files{s}, 'R', 'valid_cond_idx');
    for c = 1:N_common
        idx = find(valid_cond_idx == c);
        if ~isempty(idx), cc_all(c, :, s) = mean(R(idx, 1:3), 1); end
    end
end
Ref_Shape = mean(cc_all, 3, 'omitnan'); % Эталонная геометрия центроидов

%% --- 3. ПАСС 2: ПРОКРУСТОВО ВЫРАВНИВАНИЕ ТРАЕКТОРИЙ И СБОР ОГИБАЮЩИХ ---
disp('>>> ШАГ 2: ВЫРАВНИВАНИЕ ВСЕХ ЭПОХ И СБОР ОГИБАЮЩИХ <<<');

All_R_abs = NaN(Max_Windows, 3, N_subj);
All_C_abs = NaN(Max_Windows, N_subj);

All_Spatial_Patterns = [];
All_Envelopes_Abs = {};
All_Corr_Values = [];
All_Subj_IDs = {};

for s = 1:N_subj
    load(valid_files{s}, 'R', 'valid_windows', 'valid_cond_idx', 'Covs', 'U');
    
    % --- 1. ВЫРАВНИВАНИЕ UMAP-ТРАЕКТОРИИ ---
    cc_subj = cc_all(:,:,s);
    % Получаем трансформацию центроидов субъекта к эталону
    [~, ~, transform] = procrustes(Ref_Shape, cc_subj);
    
    % Применяем эту трансформацию ко ВСЕМ эпохам (R) этого субъекта
    % tr.b - масштаб, tr.T - вращение, tr.c - сдвиг
    R_aligned = transform.b * R(:,1:3) * transform.T + repmat(transform.c(1,:), size(R,1), 1);
    
    % Ставим на абсолютный таймлайн
    All_R_abs(valid_windows, :, s) = R_aligned;
    All_C_abs(valid_windows, s) = valid_cond_idx;
    
    % --- 2. СБОР ПАТТЕРНОВ И ОГИБАЮЩИХ ---
    stats_dir = strrep(fileparts(valid_files{s}), 'embeddings', 'stats');
    stat_fname = fullfile(stats_dir, sprintf('STATS_%s_%s.mat', valid_bases{s}, freq_name));
    
    if exist(stat_fname, 'file')
        st = load(stat_fname);
        if isvector(st.corrs_true), st.corrs_true = st.corrs_true(:)'; end
        [sig_gl, sig_lcl] = find((st.corrs_true > st.max_val) | (st.corrs_true < st.min_val));
        
        Covs_valid = Covs(:, :, valid_windows);
        added_comps = 0;
        
        for k = 1:length(sig_gl)
            gl_idx = sig_gl(k); lcl_idx = sig_lcl(k);
            if ndims(st.A_true) == 3, a_pca = squeeze(st.A_true(gl_idx, :, lcl_idx))'; w_pca = squeeze(st.W_true(gl_idx, :, lcl_idx))';
            else, a_pca = st.A_true(:, lcl_idx); w_pca = st.W_true(:, lcl_idx); end
            
            a_sens = U * a_pca;
            [~, m_idx] = max(abs(a_sens)); a_sens = a_sens * sign(a_sens(m_idx));
            All_Spatial_Patterns(:, end+1) = a_sens;
            
            S_raw = zeros(1, size(Covs_valid, 3));
            for i = 1:size(Covs_valid, 3), S_raw(i) = w_pca' * Covs_valid(:, :, i) * w_pca; end
            S_raw = (S_raw - mean(S_raw)) / std(S_raw); 
            
            S_abs = NaN(1, Max_Windows);
            S_abs(valid_windows) = S_raw;
            
            All_Envelopes_Abs{end+1} = S_abs;
            All_Corr_Values(end+1) = st.corrs_true(gl_idx, lcl_idx);
            All_Subj_IDs{end+1} = valid_bases{s};
            added_comps = added_comps + 1;
        end
        fprintf('  [%s] выровнен. Добавлено %d eSPoC компонент.\n', valid_bases{s}, added_comps);
    end
end
fprintf('ИТОГО: собрано %d значимых eSPoC компонент со всей группы.\n', size(All_Spatial_Patterns, 2));

%% --- 4. ВЫЧИСЛЕНИЕ ГРУППОВОЙ ТРАЕКТОРИИ ---
disp('>>> ШАГ 3: УСРЕДНЕНИЕ ЭПОХ С ИГНОРИРОВАНИЕМ АРТЕФАКТОВ <<<');
% Усредняем 3D-координаты. 'omitnan' игнорирует моргания у отдельных людей!
Group_R_abs = mean(All_R_abs, 3, 'omitnan');
Group_C_abs = round(median(All_C_abs, 2, 'omitnan'));

% Находим кадры, где выжил хотя бы один человек
valid_global_frames = ~all(isnan(Group_R_abs), 2) & ~isnan(Group_C_abs);

Y_epochs = Group_R_abs(valid_global_frames, :);
Group_C_Clean = Group_C_abs(valid_global_frames);
fprintf('Финальная длина сплошной групповой траектории: %d окон.\n', size(Y_epochs, 1));

%% --- 5. ВИЗУАЛИЗАЦИЯ HIGH-RES ТРАЕКТОРИИ ---
disp('>>> ШАГ 4: ОТРИСОВКА ВЫСОКОРАЗРЕШЕННОЙ ТРАЕКТОРИИ <<<');

figure('Name', 'High-Res Trajectory', 'Color', 'w', 'Position', [100, 100, 1000, 800]);
hold on; grid on;

% Линия траектории
plot3(Y_epochs(:,1), Y_epochs(:,2), Y_epochs(:,3), 'Color', [0.8 0.8 0.8], 'LineWidth', 0.5);

% Эпохи по группам (Rest, Metronome, Music)
for g = 1:3
    idx = find(ismember(Group_C_Clean, find(cond_groups == g)));
    scatter3(Y_epochs(idx,1), Y_epochs(idx,2), Y_epochs(idx,3), 20, group_colors(g,:), 'filled', 'MarkerFaceAlpha', 0.6);
end

% Звезды - центроиды условий
for c = 1:19
    idx = find(Group_C_Clean == c);
    if ~isempty(idx)
        mc = mean(Y_epochs(idx, :), 1);
        scatter3(mc(1), mc(2), mc(3), 300, group_colors(cond_groups(c),:), 'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
        text(mc(1), mc(2), mc(3)+0.1, target_conditions{c}, 'FontSize', 12, 'FontWeight', 'bold');
    end
end

title(sprintf('High-Resolution Brain State Trajectory (Procrustes Averaged, N=%d)', N_subj), 'FontSize', 14);
xlabel('Aligned UMAP 1'); ylabel('Aligned UMAP 2'); zlabel('Aligned UMAP 3'); view(-45, 30);

%% --- 6. ОЦЕНКА МЕЖСУБЪЕКТНОЙ СИНХРОНИИ (ISC) ---
disp('>>> ШАГ 5: АНАЛИЗ СИНХРОНИИ ВНУТРИ ПРОСТРАНСТВЕННЫХ СЕТЕЙ <<<');

C_spat = corr(All_Spatial_Patterns);
D_spat = squareform(1 - abs(C_spat) - diag(diag(1 - abs(C_spat))));
spat_cluster_idx = cluster(linkage(D_spat, 'average'), 'maxclust', 4);

figure('Name', 'Envelope Synchrony (ISC)', 'Color', 'w', 'Position', [150, 150, 1400, 800]);
t = tiledlayout(2, 2, 'TileSpacing', 'compact');
sgtitle('Inter-Subject Synchrony (Epoch-by-Epoch corr, NaNs ignored)', 'FontWeight', 'bold');

for c = 1:4
    raw_idx = find(spat_cluster_idx == c);
    if isempty(raw_idx), continue; end
    
    subjs_in_raw = All_Subj_IDs(raw_idx);
    unique_subj_cluster = unique(subjs_in_raw);
    
    idx_pruned = [];
    for us = 1:length(unique_subj_cluster)
        subj_mask = strcmp(subjs_in_raw, unique_subj_cluster{us});
        subj_comp_indices = raw_idx(subj_mask);
        [~, best_sub_idx] = max(abs(All_Corr_Values(subj_comp_indices)));
        idx_pruned(end+1) = subj_comp_indices(best_sub_idx);
    end
    
    Env_Matrix = zeros(length(idx_pruned), Max_Windows);
    for i = 1:length(idx_pruned), Env_Matrix(i, :) = All_Envelopes_Abs{idx_pruned(i)}; end
    
    % Игнорируем NaN (артефакты) при расчете синхронии
    ISC_Matrix = corr(Env_Matrix', 'rows', 'pairwise');
    
    ax = nexttile(t);
    imagesc(ax, ISC_Matrix, [-0.2 0.8]); colormap(ax, 'parula'); colorbar(ax);
    title(ax, sprintf('Network %d ISC Matrix\n(N=%d unique subjects)', c, length(idx_pruned)));
    
    mean_isc = mean(ISC_Matrix(triu(true(size(ISC_Matrix)), 1)));
    fprintf('Сеть %d: субъектов = %d, Средний уровень синхронии (ISC) = %.3f\n', c, length(idx_pruned), mean_isc);
    
    xlabel(ax, 'Subject'); ylabel(ax, 'Subject'); axis(ax, 'square');
end
disp('>>> АНАЛИЗ ВЫСОКОГО РАЗРЕШЕНИЯ ЗАВЕРШЕН <<<');