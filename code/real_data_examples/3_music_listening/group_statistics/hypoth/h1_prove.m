%% =====================================================================
% ГИПОТЕЗА 1: СУЩЕСТВОВАНИЕ И НЕСЛУЧАЙНОСТЬ МЕТА-СОСТОЯНИЙ (PERMUTATION TEST)
% =====================================================================
close all; clear; clc;

% --- 1. НАСТРОЙКИ И ЗАГРУЗКА ---
base_dirs = {
    'C:\Users\ansbel\Documents\GitHub\TriCo\data\external\music_listening\part1\', ...
    'C:\Users\ansbel\Documents\GitHub\TriCo\data\external\music_listening\part2\'
};
freq_name = 'beta';
N_common = 19; 

target_conditions = {
    'EC1', 'EO1', '2Hz', '05Hz', '4Hz', '1Hz', '3Hz', ...
    'NoRy 1', 'Waltz 1', 'Waltz 2', 'NoRy 2', 'NoRy 3', ...
    'Waltz 3', 'NoRy 4', 'Waltz 4', 'NoRy 5', 'Waltz 5', ...
    'EC2', 'EO2'
};

% Собираем данные
MAX_SUBJECTS = 50; All_Centroids_temp = NaN(N_common, 3, MAX_SUBJECTS); subj_count = 0;
for p = 1:length(base_dirs)
    emb_dir = fullfile(base_dirs{p}, 'emb eddings', freq_name);
    mat_files = dir(fullfile(emb_dir, ['UMAP_*_', freq_name, '.mat']));
    for f = 1:length(mat_files)
        load(fullfile(emb_dir, mat_files(f).name), 'R', 'valid_cond_idx');
        cc = NaN(N_common, 3);
        for c = 1:N_common
            idx = find(valid_cond_idx == c);
            if ~isempty(idx), cc(c, :) = mean(R(idx, 1:3), 1); end
        end
        if ~any(isnan(cc(:)))
            subj_count = subj_count + 1; All_Centroids_temp(:, :, subj_count) = cc;
        end
    end
end
All_Centroids = All_Centroids_temp(:, :, 1:subj_count);
N_subj = subj_count;

% --- 2. ПРОКРУСТОВО ВЫРАВНИВАНИЕ (GPA) ---
disp('>>> ИТЕРАТИВНОЕ ВЫРАВНИВАНИЕ (GPA) <<<');
Reference_Shape = All_Centroids(:,:,1);
Reference_Shape = Reference_Shape - mean(Reference_Shape, 1);
Reference_Shape = Reference_Shape / norm(Reference_Shape, 'fro');
Aligned_Centroids = zeros(size(All_Centroids));
for iter = 1:50
    for s = 1:N_subj
        [~, Z_aligned] = procrustes(Reference_Shape, All_Centroids(:,:,s));
        Aligned_Centroids(:,:,s) = Z_aligned;
    end
    New_Mean_Shape = mean(Aligned_Centroids, 3);
    New_Mean_Shape = New_Mean_Shape - mean(New_Mean_Shape, 1);
    New_Mean_Shape = New_Mean_Shape / norm(New_Mean_Shape, 'fro');
    if norm(New_Mean_Shape - Reference_Shape, 'fro') < 1e-6, break; end
    Reference_Shape = New_Mean_Shape;
end
Grand_Mean_Centroids = Reference_Shape;

%% --- 3. ПЕРЕСТАНОВОЧНЫЙ ТЕСТ (PERMUTATION TEST) ---
disp(' '); disp('================================================================');
disp('>>> ПРОВЕРКА ГИПОТЕЗЫ 1: НЕСЛУЧАЙНОСТЬ МЕТА-СОСТОЯНИЙ <<<');
disp('================================================================');

% Истинная разметка (1=EC, 2=EO, 3=Metro, 4=Waltz, 5=NoRy)
true_labels = zeros(N_common, 1);
true_labels([1, 18]) = 1; 
true_labels([2, 19]) = 2; 
true_labels(3:7) = 3;     
true_labels([9, 10, 13, 15, 17]) = 4; 
true_labels([8, 11, 12, 14, 16]) = 5; 

% Цвета для визуализации промежуточных шагов
cmap = [0.2 0.2 0.2;  0.6 0.6 0.6;  0.9 0.2 0.2;  0.2 0.4 0.9;  0.4 0.8 0.9];

% Функция 1: Псевдо-F статистика (Отношение дисперсий)
calc_pseudo_F = @(pts, labels) ...
    (sum(pdist(grpstats(pts, labels, 'mean')).^2) / max(1, length(unique(labels))-1)) / ...
    (sum(arrayfun(@(k) sum(pdist(pts(labels==k, :)).^2), unique(labels))) / max(1, size(pts,1)-length(unique(labels))));

% Функция 2: Средний индекс Силуэта (Плотность и изолированность кластеров)
calc_silhouette = @(pts, labels) mean(silhouette(pts, labels));

% --- Расчет реальных биологических метрик ---
Real_F_Stat = calc_pseudo_F(Grand_Mean_Centroids, true_labels);
Real_Sil_Stat = calc_silhouette(Grand_Mean_Centroids, true_labels);

% --- Генерация нулевых распределений ---
N_permutations = 10000;
Null_F_Dist = zeros(N_permutations, 1);
Null_Sil_Dist = zeros(N_permutations, 1);

disp('Генерация нулевого распределения (10 000 перестановок)...');
% Сохраним одну случайную перестановку для визуализации
demo_shuffled_labels = true_labels(randperm(N_common)); 

for p = 1:N_permutations
    shuffled_labels = true_labels(randperm(N_common));
    Null_F_Dist(p) = calc_pseudo_F(Grand_Mean_Centroids, shuffled_labels);
    Null_Sil_Dist(p) = calc_silhouette(Grand_Mean_Centroids, shuffled_labels);
end

% Подсчет точных p-values
p_val_F = sum(Null_F_Dist >= Real_F_Stat) / N_permutations;
p_val_Sil = sum(Null_Sil_Dist >= Real_Sil_Stat) / N_permutations;

p_str_F = sprintf('= %.4f', p_val_F); if p_val_F == 0, p_str_F = sprintf('< 1e-%d', log10(N_permutations)); end
p_str_Sil = sprintf('= %.4f', p_val_Sil); if p_val_Sil == 0, p_str_Sil = sprintf('< 1e-%d', log10(N_permutations)); end

%% --- 4. КОНСОЛЬНЫЙ ОТЧЕТ ---
fprintf('\n--- РЕЗУЛЬТАТЫ МУЛЬТИ-МЕТРИЧЕСКОГО ТЕСТА ---\n');
fprintf('МЕТРИКА 1: Pseudo-F Statistic (ANOVA-подобная кучность)\n');
fprintf('  Реальная структура: %.2f\n', Real_F_Stat);
fprintf('  Максимальная случайность: %.2f\n', max(Null_F_Dist));
fprintf('  Достоверность (p-value): %s\n\n', p_str_F);

fprintf('МЕТРИКА 2: Silhouette Score (Плотность ближайших соседей)\n');
fprintf('  Реальная структура: %.3f\n', Real_Sil_Stat);
fprintf('  Максимальная случайность: %.3f\n', max(Null_Sil_Dist));
fprintf('  Достоверность (p-value): %s\n', p_str_Sil);

disp('================================================================');

%% --- 5. КОМПЛЕКСНАЯ ВИЗУАЛИЗАЦИЯ (4 ПАНЕЛИ) ---
figure('Name', 'Hypothesis 1: Comprehensive Permutation Proof', 'Color', 'w', 'Position', [100, 100, 1400, 900]);
tiledlayout(2, 2, 'TileSpacing', 'compact');

% ПАНЕЛЬ 1: Реальные мета-состояния (Биология)
ax1 = nexttile; hold(ax1, 'on'); grid(ax1, 'on');
for i = 1:N_common
    scatter3(ax1, Grand_Mean_Centroids(i,1), Grand_Mean_Centroids(i,2), Grand_Mean_Centroids(i,3), ...
        250, cmap(true_labels(i), :), 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    % ДОБАВЛЕНЫ ПОДПИСИ
    text(ax1, Grand_Mean_Centroids(i,1), Grand_Mean_Centroids(i,2), Grand_Mean_Centroids(i,3) + 0.05, ...
        target_conditions{i}, 'FontSize', 10, 'FontWeight', 'bold');
end
title(ax1, '1. True Biological Grouping', 'FontSize', 14, 'Color', [0.1 0.5 0.2]);
subtitle(ax1, 'Highly structured, well-separated meta-states');
view(ax1, -45, 30);

% ПАНЕЛЬ 2: Случайные мета-состояния (Механизм перестановки)
ax2 = nexttile; hold(ax2, 'on'); grid(ax2, 'on');
for i = 1:N_common
    scatter3(ax2, Grand_Mean_Centroids(i,1), Grand_Mean_Centroids(i,2), Grand_Mean_Centroids(i,3), ...
        250, cmap(demo_shuffled_labels(i), :), 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    % ДОБАВЛЕНЫ ПОДПИСИ
    text(ax2, Grand_Mean_Centroids(i,1), Grand_Mean_Centroids(i,2), Grand_Mean_Centroids(i,3) + 0.05, ...
        target_conditions{i}, 'FontSize', 10, 'FontWeight', 'bold');
end
title(ax2, '2. Example of Shuffled Labels (Null Hypothesis)', 'FontSize', 14, 'Color', [0.7 0.2 0.2]);
subtitle(ax2, 'Colors are randomly assigned to spatial coordinates');
view(ax2, -45, 30);

% ПАНЕЛЬ 3: Доказательство №1 (Pseudo-F)
ax3 = nexttile; hold(ax3, 'on'); grid(ax3, 'on');
histogram(ax3, Null_F_Dist, 50, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'w');
xline(ax3, Real_F_Stat, '-r', 'REAL DATA', 'LineWidth', 3, 'LabelVerticalAlignment', 'top', 'FontSize', 11, 'FontWeight', 'bold');
title(ax3, '3. Permutation Test: Pseudo-F Statistic');
xlabel(ax3, 'F-Statistic (Higher = Better clustered)');
ylabel(ax3, 'Frequency (10,000 permutations)');
text(ax3, Real_F_Stat * 0.7, max(ylim(ax3))*0.8, sprintf('p %s', p_str_F), 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'r');

% ПАНЕЛЬ 4: Доказательство №2 (Silhouette)
ax4 = nexttile; hold(ax4, 'on'); grid(ax4, 'on');
histogram(ax4, Null_Sil_Dist, 50, 'FaceColor', [0.5 0.7 0.9], 'EdgeColor', 'w');
xline(ax4, Real_Sil_Stat, '-r', 'REAL DATA', 'LineWidth', 3, 'LabelVerticalAlignment', 'top', 'FontSize', 11, 'FontWeight', 'bold');
title(ax4, '4. Alternative Proof: Silhouette Score');
xlabel(ax4, 'Silhouette Index (Density and Separation)');
ylabel(ax4, 'Frequency (10,000 permutations)');
text(ax4, Real_Sil_Stat * 0.7, max(ylim(ax4))*0.8, sprintf('p %s', p_str_Sil), 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'r');

disp('>>> АНАЛИЗ ГИПОТЕЗЫ 1 УСПЕШНО ЗАВЕРШЕН <<<');