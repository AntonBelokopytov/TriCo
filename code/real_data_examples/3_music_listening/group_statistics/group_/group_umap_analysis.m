%% =====================================================================
% DATA-DRIVEN UMAP TOPOLOGY ANALYSIS (HIERARCHICAL EXTENDED)
% =====================================================================
close all; clear; clc;

% --- 1. НАСТРОЙКИ ---
base_dirs = {
    'C:\Users\ansbel\Documents\GitHub\TriCo\data\external\music_listening\part1\', ...
    'C:\Users\ansbel\Documents\GitHub\TriCo\data\external\music_listening\part2\'
};

freq_name = 'beta';
N_common = 19; 
max_wins = 217;

target_conditions = {
    'EC1', 'EO1', '2Hz', '05Hz', '4Hz', '1Hz', '3Hz', ...
    'NoRy 1', 'Waltz 1', 'Waltz 2', 'NoRy 2', 'NoRy 3', ...
    'Waltz 3', 'NoRy 4', 'Waltz 4', 'NoRy 5', 'Waltz 5', ...
    'EC2', 'EO2'
};

% Детальная истинная разметка (Ground Truth) на 5 категорий
idx_ec    = [1, 18];
idx_eo    = [2, 19];
idx_metro = 3:7;
idx_waltz = [9, 10, 13, 15, 17];
idx_nory  = [8, 11, 12, 14, 16];

true_labels = cell(N_common, 1);
for i = 1:N_common
    if ismember(i, idx_ec),          true_labels{i} = 'EC (Eyes Closed)';
    elseif ismember(i, idx_eo),      true_labels{i} = 'EO (Eyes Open)';
    elseif ismember(i, idx_metro),   true_labels{i} = 'Metronome';
    elseif ismember(i, idx_waltz),   true_labels{i} = 'Waltz';
    elseif ismember(i, idx_nory),    true_labels{i} = 'No Rhythm';
    end
end
unique_true = {'EC (Eyes Closed)', 'EO (Eyes Open)', 'Metronome', 'Waltz', 'No Rhythm'};

%% --- 2. СБОР ДАННЫХ ---
disp('>>> ШАГ 1: СБОР ЦЕНТРОИДОВ UMAP <<<');
All_Centroids = []; 
subj_names = {};

for p = 1:length(base_dirs)
    emb_dir = fullfile(base_dirs{p}, 'embeddings', freq_name);
    mat_files = dir(fullfile(emb_dir, ['UMAP_*_', freq_name, '.mat']));
    
    for f = 1:length(mat_files)
        load(fullfile(emb_dir, mat_files(f).name), 'R', 'valid_cond_idx');
        
        cc = NaN(N_common, 3);
        for c = 1:N_common
            idx = find(valid_cond_idx == c);
            if length(idx) > max_wins, idx = idx(1:max_wins); end
            if ~isempty(idx), cc(c, :) = mean(R(idx, 1:3), 1); end
        end
        
        if ~any(isnan(cc(:)))
            All_Centroids(:, :, end+1) = cc;
            subj_names{end+1} = strrep(mat_files(f).name, 'UMAP_', '');
        end
    end
end
N_subj = size(All_Centroids, 3);
disp(['Собраны 3D-центроиды от ', num2str(N_subj), ' испытуемых.']);

%% --- 3. ГЛОБАЛЬНОЕ ВЫРАВНИВАНИЕ (PROCRUSTES) ---
disp('>>> ШАГ 2: СОЗДАНИЕ УНИВЕРСАЛЬНОГО ПРОСТРАНСТВА <<<');
Reference_Shape = mean(All_Centroids, 3);
Aligned_Centroids = zeros(size(All_Centroids));
for s = 1:N_subj
    [~, Aligned_Centroids(:,:,s)] = procrustes(Reference_Shape, All_Centroids(:,:,s));
end

X_all = reshape(permute(Aligned_Centroids, [1 3 2]), [], 3);
Y_true_cond = repmat((1:N_common)', N_subj, 1); 

%% --- 4. ПЕРСПЕКТИВА 1: ИЕРАРХИЧЕСКИЙ ПОИСК КЛАСТЕРОВ (GLOBAL vs TASK-ONLY) ---
disp('>>> ШАГ 3: DATA-DRIVEN ПОИСК ЧИСЛА СОСТОЯНИЙ (NESTED APPROACH) <<<');
K_test = 2:8;

% А) Глобальный поиск (Покажет K=2 из-за каньона EC/EO)
eva_global = evalclusters(X_all, 'kmeans', 'Silhouette', 'KList', K_test);
K_global = eva_global.OptimalK;

% Б) ЗУМ В АУДИО (Исключаем EC/EO)
idx_audio = [idx_metro, idx_waltz, idx_nory];
X_audio = X_all(ismember(Y_true_cond, idx_audio), :);
eva_audio = evalclusters(X_audio, 'kmeans', 'Silhouette', 'KList', K_test);
K_audio = eva_audio.OptimalK;

figure('Name', 'Optimal K Evaluation', 'Color', 'w', 'Position', [100, 100, 1000, 400]);
tiledlayout(1, 2);

nexttile;
plot(K_test, eva_global.CriterionValues, '-ko', 'LineWidth', 2, 'MarkerFaceColor', 'b');
xline(K_global, '--b', ['Global K = ', num2str(K_global)], 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
title('Global Search (All Conditions)'); xlabel('Clusters (K)'); ylabel('Silhouette Value'); grid on;

nexttile;
plot(K_test, eva_audio.CriterionValues, '-ko', 'LineWidth', 2, 'MarkerFaceColor', 'r');
xline(K_audio, '--r', ['Audio-Only K = ', num2str(K_audio)], 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
title('Microscope Search (Audio Only)'); xlabel('Clusters (K)'); ylabel('Silhouette Value'); grid on;

% Итоговое K = Глобальное K + Аудио K - 1 (перекрывающийся ствол)
% Но для упрощения консенсусной матрицы, если мы ищем детальные кластеры, мы форсируем K = 4 или 5
K_opt = max(4, K_global + K_audio - 1); 
disp(['Применяем детальное разрешение: K_opt = ', num2str(K_opt)]);

%% --- 5. КОНСЕНСУСНАЯ КЛАСТЕРИЗАЦИЯ И МАТРИЦА ---
disp('>>> ШАГ 4: ВЫЧИСЛЕНИЕ МАТРИЦЫ КОНСЕНСУСА <<<');
Cooc_Matrix = zeros(N_common, N_common); 
for s = 1:N_subj
    D = pdist(All_Centroids(:, :, s));
    labels = cluster(linkage(D, 'ward'), 'maxclust', K_opt);
    
    Adj = zeros(N_common, N_common);
    for i = 1:N_common
        for j = 1:N_common
            if labels(i) == labels(j), Adj(i,j) = 1; end
        end
    end
    Cooc_Matrix = Cooc_Matrix + Adj;
end
Cooc_Matrix = (Cooc_Matrix / N_subj) * 100;

%% --- 6. ПЕРСПЕКТИВА 2: ЛАНДШАФТ УСЛОВИЙ (CONSENSUS MDS) ---
disp('>>> ШАГ 5: ПОСТРОЕНИЕ 2D КАРТЫ УСЛОВИЙ (MDS) <<<');
% Переводим % совпадения в дистанцию
D_cooc = squareform(100 - Cooc_Matrix - diag(diag(100 - Cooc_Matrix))); 
[Y_mds, ~] = cmdscale(D_cooc);

% Кластеризуем сами условия для раскраски
Z_consensus = linkage(D_cooc, 'average');
Empirical_Labels = cluster(Z_consensus, 'maxclust', K_opt);

figure('Name', 'Condition Landscape (MDS)', 'Color', 'w', 'Position', [150, 150, 800, 600]);
hold on; grid on;
cmap_emp = lines(K_opt);

% Рисуем 19 условий как точки на плоскости (чем ближе, тем чаще они слипаются)
for c = 1:N_common
    scatter(Y_mds(c,1), Y_mds(c,2), 150, cmap_emp(Empirical_Labels(c),:), 'filled', 'MarkerEdgeColor', 'k');
    text(Y_mds(c,1), Y_mds(c,2) + 2, target_conditions{c}, 'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
end
title('Consensus Landscape of Conditions (MDS)');
xlabel('Dimension 1 (Topological Separation)'); ylabel('Dimension 2');

%% --- 7. ПЕРСПЕКТИВА 3: ГРАФ ТОПОЛОГИИ УСЛОВИЙ ---
disp('>>> ШАГ 6: ПОСТРОЕНИЕ ГРАФА СВЯЗЕЙ <<<');
% Строим граф: связываем условия, если они кластеризуются вместе > 50% раз
Adj_Graph = Cooc_Matrix;
Adj_Graph(Adj_Graph < 50) = 0; 
Adj_Graph = Adj_Graph - diag(diag(Adj_Graph));

figure('Name', 'Condition Topology Graph', 'Color', 'w', 'Position', [200, 200, 700, 700]);
G = graph(Adj_Graph, target_conditions);
p = plot(G, 'Layout', 'force', 'MarkerSize', 10, 'LineWidth', G.Edges.Weight / 20);
title('Topological Graph (Edges = Co-clustering > 50%)');
for k = 1:K_opt
    highlight(p, find(Empirical_Labels == k), 'NodeColor', cmap_emp(k,:));
end
set(gca, 'XTick', [], 'YTick', [], 'XColor', 'none', 'YColor', 'none');

%% --- 8. ВАЛИДАЦИЯ (CONFUSION MATRIX: ЭМПИРИКА vs РЕАЛЬНОСТЬ) ---
disp('>>> ШАГ 7: ВНЕШНЯЯ ВАЛИДАЦИЯ (СВЕРКА С 5-Ю КАТЕГОРИЯМИ) <<<');
Conf_Mat = zeros(5, K_opt);
for i = 1:N_common
    true_idx = find(strcmp(unique_true, true_labels{i}));
    emp_idx = Empirical_Labels(i);
    Conf_Mat(true_idx, emp_idx) = Conf_Mat(true_idx, emp_idx) + 1;
end
Conf_Mat_Pct = (Conf_Mat ./ sum(Conf_Mat, 1)) * 100;

figure('Name', 'External Validation', 'Color', 'w', 'Position', [250, 250, 700, 500]);
imagesc(Conf_Mat_Pct); colormap('parula'); cb = colorbar; cb.Label.String = '% of Cluster Composition';
title(sprintf('Cluster Composition (Purity)\nData-Driven vs 5 Ground Truth States'));
yticks(1:5); yticklabels(unique_true);
xticks(1:K_opt); xticklabels(arrayfun(@(x) sprintf('Empirical\nCluster %d', x), 1:K_opt, 'UniformOutput', false));
for c = 1:K_opt
    for r = 1:5
        val = Conf_Mat_Pct(r, c);
        txt_color = 'k'; if val > 50, txt_color = 'w'; end
        text(c, r, sprintf('%.0f%%', val), 'HorizontalAlignment', 'center', 'Color', txt_color, 'FontWeight', 'bold');
    end
end

%% --- 9. ОЦЕНКА ЧИСТОТЫ (PURITY) ---
total_points = sum(Conf_Mat(:));
majority_class_sum = sum(max(Conf_Mat, [], 1));
Purity = (majority_class_sum / total_points) * 100;
disp(['--------------------------------------------------']);
disp(['Итоговая чистота кластеризации (Purity) по 5-ти детальным классам: ', num2str(Purity, '%.2f'), '%']);
disp('>>> ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН <<<');