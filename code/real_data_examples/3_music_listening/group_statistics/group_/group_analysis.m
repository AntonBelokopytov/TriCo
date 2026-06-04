%% =====================================================================
% ULTIMATE INTER-SUBJECT GROUP ANALYSIS FOR eSPoC SPATIAL PATTERNS
% =====================================================================
close all; clear; clc;

% --- 1. НАСТРОЙКИ ---
base_dirs = {
    'C:\Users\ansbel\Documents\GitHub\TriCo\data\external\music_listening\part1\', ...
    'C:\Users\ansbel\Documents\GitHub\TriCo\data\external\music_listening\part2\'
};

ft_path   = 'C:\Users\ansbel\Documents\GitHub\site-packages\fieldtrip';
freq_name = 'beta';

% Настройки синхронизации экспериментов
N_common_conds = 19; % Берем только 19 общих условий
max_wins_per_cond = 217; % 110 сек при окне 2с и шаге 0.5с -> 217 окон
N_clusters = 4; % Число выделяемых нейросетей (можно менять: 3, 4, 5...)

target_conditions = {
    'EC1', 'EO1', '2Hz', '05Hz', '4Hz', '1Hz', '3Hz', ...
    'NoRy 1', 'Waltz 1', 'Waltz 2', 'NoRy 2', 'NoRy 3', ...
    'Waltz 3', 'NoRy 4', 'Waltz 4', 'NoRy 5', 'Waltz 5', ...
    'EC2', 'EO2'
};

if ~exist('ft_defaults','file'), addpath(ft_path); end
ft_defaults;

%% --- 2. СБОР ДАННЫХ СО ВСЕХ ИСПЫТУЕМЫХ ---
all_A = [];          % Значимые паттерны
all_S_cond = [];     % Огибающие (усредненные по условиям)
all_subject_ids = {};% ID субъекта
all_corrs = [];      % Сила корреляции компоненты
all_dist_matrices = []; % Для матриц UMAP

disp('>>> НАЧАЛО СБОРА ДАННЫХ ИЗ PART 1 И PART 2 <<<');

for p = 1:length(base_dirs)
    b_dir = base_dirs{p};
    emb_dir   = fullfile(b_dir, 'embeddings', freq_name);
    stats_dir = fullfile(b_dir, 'stats', freq_name);
    
    mat_files = dir(fullfile(stats_dir, ['STATS_*_', freq_name, '.mat']));
    
    for f = 1:length(mat_files)
        stat_name = mat_files(f).name;
        base_name = strrep(stat_name, 'STATS_', '');
        base_name = strrep(base_name, ['_', freq_name, '.mat'], '');
        
        emb_file = fullfile(emb_dir, ['UMAP_', base_name, '_', freq_name, '.mat']);
        if ~exist(emb_file, 'file'), continue; end
        
        stats = load(fullfile(stats_dir, stat_name));
        embs = load(emb_file);
        
        if isvector(stats.corrs_true), stats.corrs_true = stats.corrs_true(:)'; end
        [sig_gl, sig_lcl] = find((stats.corrs_true > stats.max_val) | (stats.corrs_true < stats.min_val));
        
        if isempty(sig_gl)
            disp(['[Пропуск] ', base_name, ' — нет значимых компонент.']);
            continue;
        end
        
        % 1. Собираем матрицу расстояний UMAP (только 19 условий)
        ccx = NaN(1, N_common_conds); ccy = NaN(1, N_common_conds); ccz = NaN(1, N_common_conds);
        for c = 1:N_common_conds
            idx = find(embs.valid_cond_idx == c);
            if length(idx) > max_wins_per_cond, idx = idx(1:max_wins_per_cond); end % Обрезка Part2 до 110 сек
            if ~isempty(idx)
                ccx(c) = mean(embs.R(idx, 1)); ccy(c) = mean(embs.R(idx, 2)); ccz(c) = mean(embs.R(idx, 3));
            end
        end
        if ~any(isnan(ccx))
            D = pdist([ccx; ccy; ccz]');
            all_dist_matrices(:, :, end+1) = squareform(D);
        end
        
        % 2. Собираем Паттерны и Мощности
        Covs_valid = embs.Covs(:, :, embs.valid_windows);
        
        for k = 1:length(sig_gl)
            gl_idx = sig_gl(k); lcl_idx = sig_lcl(k);
            
            % Паттерн
            if ndims(stats.A_true) == 3
                a_pca = squeeze(stats.A_true(gl_idx, :, lcl_idx))';
                w_pca = squeeze(stats.W_true(gl_idx, :, lcl_idx))';
            else
                a_pca = stats.A_true(:, lcl_idx);
                w_pca = stats.W_true(:, lcl_idx);
            end
            a_sens = embs.U * a_pca;
            [~, m_idx] = max(abs(a_sens)); a_sens = a_sens * sign(a_sens(m_idx)); % Базовое выравнивание
            
            all_A(:, end+1) = a_sens;
            all_subject_ids{end+1} = base_name;
            all_corrs(end+1) = stats.corrs_true(gl_idx, lcl_idx);
            
            % Огибающая мощности
            S_raw = zeros(1, size(Covs_valid, 3));
            for i = 1:size(Covs_valid, 3)
                S_raw(i) = w_pca' * Covs_valid(:, :, i) * w_pca;
            end
            
            % Усредняем мощность с учетом BADS и ограничения в 110 сек
            S_cond = NaN(1, N_common_conds);
            for c = 1:N_common_conds
                idx = find(embs.valid_cond_idx == c);
                if length(idx) > max_wins_per_cond, idx = idx(1:max_wins_per_cond); end % Обрезка Part2
                if ~isempty(idx)
                    S_cond(c) = mean(S_raw(idx)); % mean() игнорирует вырезанные BADS!
                end
            end
            
            % Z-score для межсубъектного сравнения
            S_cond = (S_cond - mean(S_cond, 'omitnan')) ./ std(S_cond, 'omitnan');
            all_S_cond(end+1, :) = S_cond;
        end
        disp(['[+] ', base_name, ' — добавлено ', num2str(length(sig_gl)), ' компонент.']);
    end
end
disp(['ИТОГО: ', num2str(size(all_A, 2)), ' значимых компонент от ', num2str(length(unique(all_subject_ids))), ' испытуемых.']);
unique_subjs_total = unique(all_subject_ids);

% Dummy layout для топоплотов FieldTrip
cfg_dum = []; cfg_dum.dataset = fullfile(base_dirs{1}, 'eeg', '10_07_g1_2200_raw.fif'); cfg_dum.continuous = 'yes';
Xinf = ft_preprocessing(cfg_dum); laycfg = []; laycfg.elec = Xinf.elec; lay = ft_prepare_layout(laycfg);     
topo = []; topo.dimord = 'chan_time'; topo.label = Xinf.elec.label(1:38); topo.time = 0;
cfg_topo = []; cfg_topo.marker = 'labels'; cfg_topo.layout = lay; cfg_topo.comment = 'no'; cfg_topo.style = 'fill'; cfg_topo.colorbar = 'no';

%% =====================================================================
% КЛАСТЕРИЗАЦИЯ И РАЗРЕШЕНИЕ КОНФЛИКТОВ (PRUNING)
% =====================================================================
C = corr(all_A);
D_vec = squareform(1 - abs(C) - diag(diag(1 - abs(C)))); % Магия: 1 - модуль корреляции
Z = linkage(D_vec, 'average'); % ИЗМЕНЕНО: UPGMA вместо Ward для корреляций
cluster_idx = cluster(Z, 'maxclust', N_clusters);

% --- АЛГОРИТМ ПРОПОЛКИ (PRUNING) ---
% Правило: 1 испытуемый = максимум 1 паттерн в кластере.
is_pruned = false(length(cluster_idx), 1); % СТРОГО вектор-столбец
for c = 1:N_clusters
    raw_idx = find(cluster_idx == c);
    subjs_in_raw = all_subject_ids(raw_idx);
    unique_subj_cluster = unique(subjs_in_raw);
    
    for us = 1:length(unique_subj_cluster)
        subj_mask = strcmp(subjs_in_raw, unique_subj_cluster{us});
        subj_comp_indices = raw_idx(subj_mask);
        
        if length(subj_comp_indices) == 1
            is_pruned(subj_comp_indices) = true; % Берем единственную компоненту
        else
            % КОНФЛИКТ! Выбираем ту, у которой eSPoC корреляция сильнее
            conflicting_corrs = all_corrs(subj_comp_indices);
            [~, best_sub_idx] = max(abs(conflicting_corrs));
            is_pruned(subj_comp_indices(best_sub_idx)) = true;
        end
    end
end
disp(['После разрешения внутрисубъектных конфликтов осталось ', num2str(sum(is_pruned)), ' независимых паттернов.']);
cmap_clust = lines(N_clusters);

%% =====================================================================
% РИСУНОК 1: ДЕНДРОГРАММА И ЛАНДШАФТ ПАТТЕРНОВ (MDS)
% =====================================================================
figure('Name', 'Clustering Landscape', 'Color', 'w', 'Position', [100, 100, 1400, 500]);
tiledlayout(1, 2, 'TileSpacing', 'compact');

nexttile;
dendrogram(Z, 0, 'ColorThreshold', 'default');
title('Hierarchical Clustering (Sign-Invariant)'); ylabel('1 - |Correlation|'); xlabel('Component Index');

nexttile;
[Y, ~] = cmdscale(D_vec);
hold on; grid on;
for c = 1:N_clusters
    idx = (cluster_idx == c);
    scatter(Y(idx,1), Y(idx,2), 60, cmap_clust(c,:), 'filled', 'MarkerEdgeColor', 'k', 'DisplayName', sprintf('Cluster %d', c));
end
title('MDS Landscape of ALL Spatial Patterns'); legend('Location', 'best');

%% =====================================================================
% РИСУНОК 2: СОРТИРОВАННАЯ МАТРИЦА КОРРЕЛЯЦИЙ ПАТТЕРНОВ
% =====================================================================
[sorted_clust, sort_idx] = sort(cluster_idx);
sorted_C = abs(C(sort_idx, sort_idx)); % Берем модуль для оценки коллинеарности

figure('Name', 'Sorted Correlation Matrix', 'Color', 'w', 'Position', [150, 150, 600, 600]);
imagesc(sorted_C); colormap(hot); colorbar; hold on;
title('Absolute Spatial Correlation Between Components');
xlabel('Sorted Components'); ylabel('Sorted Components');
bound = 0.5;
for c = 1:N_clusters
    clust_size = sum(sorted_clust == c);
    rectangle('Position', [bound, bound, clust_size, clust_size], 'EdgeColor', cmap_clust(c,:), 'LineWidth', 3);
    bound = bound + clust_size;
end
axis square;

%% =====================================================================
% РИСУНОК 3: ГРАФОВАЯ СЕТЬ ПРОСТРАНСТВЕННОЙ СХОЖЕСТИ
% =====================================================================
adj_matrix = abs(C);
adj_matrix(adj_matrix < 0.75) = 0; % Рисуем связи только для R > 0.75
adj_matrix = adj_matrix - diag(diag(adj_matrix)); 

figure('Name', 'Network Graph', 'Color', 'w', 'Position', [200, 200, 700, 700]);
G = graph(adj_matrix);
p = plot(G, 'Layout', 'force', 'NodeLabel', {}, 'MarkerSize', 6);
title('Force-Directed Graph of Pattern Similarities (|R| > 0.75)');
for c = 1:N_clusters, highlight(p, find(cluster_idx == c), 'NodeColor', cmap_clust(c,:)); end
set(gca, 'XTick', [], 'YTick', [], 'XColor', 'none', 'YColor', 'none');

%% =====================================================================
% РИСУНОК 4: EIGEN-NETWORKS (ГЛАВНЫЕ ПРОСТРАНСТВЕННЫЕ КОМПОНЕНТЫ)
% =====================================================================
[U_patt, S_patt, ~] = svd(all_A - mean(all_A, 2), 'econ');
var_exp_patt = diag(S_patt).^2 / sum(diag(S_patt).^2) * 100;

figure('Name', 'Eigen-Networks', 'Color', 'w', 'Position', [250, 250, 1200, 400]);
t_eigen = tiledlayout(1, 4, 'TileSpacing', 'compact');
sgtitle('Eigen-Networks (PCA of all significant Spatial Patterns)', 'FontWeight', 'bold', 'FontSize', 14);

ax_var = nexttile(t_eigen);
plot(1:min(10, length(var_exp_patt)), var_exp_patt(1:min(10, length(var_exp_patt))), '-ko', 'LineWidth', 2, 'MarkerFaceColor', 'k');
title('Variance Explained'); xlabel('PC Index'); ylabel('% Variance'); grid on;

cfg_topo.colorbar = 'yes';
for i = 1:3
    ax_t = nexttile(t_eigen);
    topo.avg = U_patt(:, i);
    [~, m_idx] = max(abs(topo.avg)); topo.avg = topo.avg * sign(topo.avg(m_idx)); 
    ft_topoplotER(cfg_topo, topo);
    title(sprintf('Eigen-Pattern %d\n(%.1f%% var)', i, var_exp_patt(i)));
end
cfg_topo.colorbar = 'no';

%% =====================================================================
% РИСУНОК 5: ГЛУБОКИЙ ПРОФИЛЬ КЛАСТЕРОВ (С ЖЕСТКИМ ПРУНИНГОМ)
% =====================================================================
figure('Name', 'Group Clusters Deep Profile', 'Color', 'w', 'Position', [300, 300, 1600, 900]);
t = tiledlayout(N_clusters, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

cluster_mean_A = zeros(38, N_clusters); 

for c = 1:N_clusters
    % БЕРЕМ ТОЛЬКО ПРУНЕННЫЕ (НЕЗАВИСИМЫЕ) ИНДЕКСЫ ДЛЯ ЭТОГО КЛАСТЕРА
    % Исправлено: жесткое использование векторов-столбцов (:), чтобы избежать неявного расширения матрицы
    idx = find((cluster_idx(:) == c) & is_pruned(:)); 
    
    unique_subj_cluster = unique(all_subject_ids(idx));
    pct_common = (length(unique_subj_cluster) / length(unique_subjs_total)) * 100;
    
    % Выравнивание знаков внутри кластера по эталону
    c_A = all_A(:, idx);
    ref_A = c_A(:, 1); 
    c_S = all_S_cond(idx, :);
    for i = 1:size(c_A, 2)
        if corr(ref_A, c_A(:, i)) < 0, c_A(:, i) = -c_A(:, i); end
    end
    
    mean_A = mean(c_A, 2);
    cluster_mean_A(:, c) = mean_A;
    std_A  = std(c_A, 0, 2);
    T_map  = mean_A ./ (std_A / sqrt(size(c_A, 2))); % T-статистика
    
    mean_S = mean(c_S, 1, 'omitnan');
    std_S  = std(c_S, 0, 1, 'omitnan') / sqrt(length(idx)); 
    
    % График 5.1: Усредненный Топоплот
    ax_topo = nexttile(t, (c-1)*3 + 1);
    topo.avg = mean_A; cfg_topo.figure = ax_topo; ft_topoplotER(cfg_topo, topo);
    title(sprintf('Cluster %d (N=%d unique subjs)\nFound in %.0f%% of group', c, length(idx), pct_common));
        
    % График 5.2: T-Map (Консистентность паттерна)
    ax_tmap = nexttile(t, (c-1)*3 + 2);
    topo.avg = T_map; cfg_topo.figure = ax_tmap; ft_topoplotER(cfg_topo, topo);
    title('T-Statistic Map (Consistency)');
    
    % График 5.3: Поведенческий профиль (Огибающие)
    ax_env = nexttile(t, (c-1)*3 + 3); hold on; grid on;
    bar(ax_env, 1:N_common_conds, mean_S, 'FaceColor', cmap_clust(c,:), 'EdgeColor', 'k', 'FaceAlpha', 0.8);
    errorbar(ax_env, 1:N_common_conds, mean_S, std_S, 'k', 'LineStyle', 'none', 'LineWidth', 1.5);
    
    xlim([0.5 N_common_conds+0.5]); xticks(1:N_common_conds);
    if c == N_clusters, xticklabels(target_conditions); xtickangle(45); else, xticklabels([]); end
    ylabel('Z-Power'); title('Task-Evoked Response');
end

%% =====================================================================
% РИСУНОК 6: ТОП-5 СЕНСОРОВ ДЛЯ КАЖДОГО КЛАСТЕРА
% =====================================================================
figure('Name', 'Top Sensors Feature Importance', 'Color', 'w', 'Position', [350, 350, 1200, 600]);
t2 = tiledlayout(2, ceil(N_clusters/2), 'TileSpacing', 'compact');
for c = 1:N_clusters
    ax = nexttile(t2);
    [sorted_vals, sorted_idx] = sort(abs(cluster_mean_A(:, c)), 'descend');
    top_vals = cluster_mean_A(sorted_idx(1:5), c); 
    top_labels = topo.label(sorted_idx(1:5));
    
    bar(ax, top_vals, 'FaceColor', cmap_clust(c,:));
    xticks(ax, 1:5); xticklabels(ax, top_labels);
    title(sprintf('Cluster %d Top Sensors', c)); ylabel('Mean Pattern Weight'); grid on;
end

%% =====================================================================
% РИСУНОК 7: ИНДЕКС ЛАТЕРАЛИЗАЦИИ (HEMISPHERIC ASYMMETRY)
% =====================================================================
left_labels  = {'Fp1','F7','F3','FT7','FC3','T3','C3','TP7','CP3','T5','P3','P5','PO3','PO7','O1'};
right_labels = {'Fp2','F8','F4','FT8','FC4','T4','C4','TP8','CP4','T6','P4','P6','PO4','PO8','O2'};
[~, left_idx]  = ismember(left_labels, topo.label);
[~, right_idx] = ismember(right_labels, topo.label);
left_idx(left_idx==0) = []; right_idx(right_idx==0) = [];

% Считаем LI: (Right - Left) / (Right + Left) на ОЧИЩЕННЫХ (pruned) данных
LI = zeros(1, size(all_A, 2));
for i = 1:size(all_A, 2)
    patt_power = abs(all_A(:, i)); 
    pow_L = sum(patt_power(left_idx)); pow_R = sum(patt_power(right_idx));
    LI(i) = (pow_R - pow_L) / (pow_R + pow_L);
end

figure('Name', 'Lateralization Index', 'Color', 'w', 'Position', [400, 400, 600, 500]);
hold on; grid on;
for c = 1:N_clusters
    clust_LI = LI((cluster_idx(:) == c) & is_pruned(:)); % Исправлено
    scatter(c + 0.1*randn(1, length(clust_LI)), clust_LI, 40, cmap_clust(c,:), 'filled', 'MarkerFaceAlpha', 0.6);
    errorbar(c, mean(clust_LI), std(clust_LI), 'k', 'LineWidth', 2, 'Marker', 'o', 'MarkerFaceColor', 'k', 'MarkerSize', 8);
end
yline(0, '--k', 'Symmetric', 'LineWidth', 1.5);
xticks(1:N_clusters); xticklabels(arrayfun(@(x) sprintf('Clust %d', x), 1:N_clusters, 'UniformOutput', false));
ylabel('Lateralization Index: <0 (Left) vs >0 (Right)');
title('Hemispheric Asymmetry of Spatial Networks (Pruned)');

%% =====================================================================
% РИСУНОК 8: МАТРИЦА СУБЪЕКТ-СУБЪЕКТ (MAX SPATIAL OVERLAP)
% =====================================================================
N_subj = length(unique_subjs_total);
Subj_Sim_Matrix = zeros(N_subj, N_subj);

for i = 1:N_subj
    for j = 1:N_subj
        idx_i = find(strcmp(all_subject_ids, unique_subjs_total{i}));
        idx_j = find(strcmp(all_subject_ids, unique_subjs_total{j}));
        
        if isempty(idx_i) || isempty(idx_j)
            Subj_Sim_Matrix(i,j) = NaN; continue;
        end
        % Берем все паттерны субъекта i и j
        A_i = all_A(:, idx_i); A_j = all_A(:, idx_j);
        
        % Находим максимальную абсолютную корреляцию между любыми их паттернами
        sub_C = abs(corr(A_i, A_j));
        Subj_Sim_Matrix(i, j) = max(sub_C(:));
    end
end

figure('Name', 'Subject x Subject Similarity', 'Color', 'w', 'Position', [450, 450, 650, 550]);
imagesc(Subj_Sim_Matrix, [0 1]); colormap(hot); colorbar;
title('Max Spatial Similarity Between Subjects');
xticks(1:N_subj); yticks(1:N_subj);
xticklabels(strrep(unique_subjs_total, '_', '\_')); yticklabels(strrep(unique_subjs_total, '_', '\_'));
xtickangle(45); axis square;

%% =====================================================================
% РИСУНОК 9 & 10: СРЕДНИЙ UMAP И COMMONALITY МАТРИЦА
% =====================================================================
figure('Name', 'Subject Commonality', 'Color', 'w', 'Position', [500, 500, 500, 700]);
C_matrix = zeros(length(unique_subjs_total), N_clusters);
for i = 1:length(unique_subjs_total)
    for c = 1:N_clusters
        % Считаем только уникальные компоненты после Pruning (Исправлено)
        mask = (cluster_idx(:) == c) & is_pruned(:);
        C_matrix(i, c) = sum(strcmp(all_subject_ids(mask), unique_subjs_total{i}));
    end
end
imagesc(C_matrix); colormap('parula'); colorbar;
title('Subject Contribution to Clusters (Pruned)');
yticks(1:length(unique_subjs_total)); yticklabels(strrep(unique_subjs_total, '_', '\_'));
xticks(1:N_clusters); xticklabels(arrayfun(@(x) sprintf('Clust %d', x), 1:N_clusters, 'UniformOutput', false));

if ~isempty(all_dist_matrices)
    figure('Name', 'Group Mean UMAP Distance', 'Color', 'w', 'Position', [550, 550, 600, 500]);
    imagesc(mean(all_dist_matrices, 3)); colormap(flipud(hot)); colorbar;
    title('Group Mean UMAP Distances');
    xticks(1:N_common_conds); yticks(1:N_common_conds);
    xticklabels(target_conditions); yticklabels(target_conditions); xtickangle(45); axis square;
end

%% =====================================================================
% РИСУНОК 11: ПРОСТРАНСТВЕННЫЙ ЦЕНТР МАСС (ДЛЯ ВСЕХ КОМПОНЕНТ)
% =====================================================================
chan_X = zeros(38, 1); chan_Y = zeros(38, 1);
for i = 1:38
    l_idx = strcmp(lay.label, topo.label{i});
    chan_X(i) = lay.pos(l_idx, 1); chan_Y(i) = lay.pos(l_idx, 2);
end

figure('Name', 'Scalp Center of Mass', 'Color', 'w', 'Position', [600, 600, 600, 600]);
hold on; axis equal; axis off;
rectangle('Position',[-0.5 -0.5 1 1], 'Curvature', [1 1], 'EdgeColor', 'k', 'LineWidth', 2);
plot(0, 0.55, '^k', 'MarkerSize', 15, 'MarkerFaceColor', 'k'); % Нос

for i = 1:size(all_A, 2)
    patt = abs(all_A(:, i)); patt = patt.^3;          
    cm_x = sum(patt .* chan_X) / sum(patt);
    cm_y = sum(patt .* chan_Y) / sum(patt);
    
    % Если компонента была отсеяна прунингом - рисуем ее полупрозрачной
    alpha_val = 0.9; if ~is_pruned(i), alpha_val = 0.2; end
    scatter(cm_x, cm_y, 100, cmap_clust(cluster_idx(i),:), 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 0.5, 'MarkerFaceAlpha', alpha_val);
end
title('Center of Mass of Spatial Patterns (Faded = Pruned Intra-Subject Conflicts)');

disp('>>> ГРУППОВОЙ АНАЛИЗ ЗАВЕРШЕН УСПЕШНО <<<');