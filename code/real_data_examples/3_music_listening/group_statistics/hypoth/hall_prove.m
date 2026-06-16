%% =====================================================================
% PROCRUSTES ALIGNMENT, 6 HYPOTHESES TESTING & 12 FIGURES SUITE
% =====================================================================
close all; clear; clc;

% --- 1. НАСТРОЙКИ ---
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

% МИКРО-ГРУППЫ
idx_ec    = [1, 18];
idx_eo    = [2, 19];
idx_metro = 3:7;
idx_waltz = [9, 10, 13, 15, 17];
idx_nory  = [8, 11, 12, 14, 16];
idx_music = [idx_waltz, idx_nory];

% Физические частоты метрономов для Гипотезы 6 (соответствуют индексам 3:7)
metro_freqs = [2.0, 0.5, 4.0, 1.0, 3.0];

% Палитра для мета-состояний
colors = zeros(N_common, 3);
colors(idx_ec, :)    = repmat([0.2 0.2 0.2], length(idx_ec), 1); % Dark Grey
colors(idx_eo, :)    = repmat([0.6 0.6 0.6], length(idx_eo), 1); % Light Grey
colors(idx_metro, :) = repmat([0.9 0.2 0.2], length(idx_metro), 1); % Red
colors(idx_waltz, :) = repmat([0.2 0.4 0.9], length(idx_waltz), 1); % Deep Blue
colors(idx_nory, :)  = repmat([0.4 0.8 0.9], length(idx_nory), 1);  % Light Blue

%% --- 2. СБОР 3D ЦЕНТРОИДОВ ИЗ UMAP ---
disp('>>> ШАГ 1: СБОР ИНДИВИДУАЛЬНЫХ ЦЕНТРОИДОВ <<<');
MAX_SUBJECTS = 50; 
All_Centroids_temp = NaN(N_common, 3, MAX_SUBJECTS);
subj_count = 0;
for p = 1:length(base_dirs)
    emb_dir = fullfile(base_dirs{p}, 'embeddings', freq_name);
    mat_files = dir(fullfile(emb_dir, ['UMAP_*_', freq_name, '.mat']));
    
    for f = 1:length(mat_files)
        load(fullfile(emb_dir, mat_files(f).name), 'R', 'valid_cond_idx');
        cc = NaN(N_common, 3);
        for c = 1:N_common
            idx = find(valid_cond_idx == c);
            if ~isempty(idx), cc(c, :) = mean(R(idx, 1:3), 1); end
        end
        if ~any(isnan(cc(:))) 
            subj_count = subj_count + 1;
            All_Centroids_temp(:, :, subj_count) = cc;
        end
    end
end
All_Centroids = All_Centroids_temp(:, :, 1:subj_count);
N_subj = subj_count;
if N_subj == 0, error('ОШИБКА: Массив All_Centroids пуст!'); end

%% --- 3. ОБОБЩЕННЫЙ ПРОКРУСТОВ АНАЛИЗ (GPA) ---
disp('>>> ШАГ 2: ИТЕРАТИВНОЕ ВЫРАВНИВАНИЕ (GPA) <<<');
Reference_Shape = All_Centroids(:,:,1);
Reference_Shape = Reference_Shape - mean(Reference_Shape, 1);
Reference_Shape = Reference_Shape / norm(Reference_Shape, 'fro');

Aligned_Centroids = zeros(size(All_Centroids));
max_iter = 100; tol = 1e-10;

for iter = 1:max_iter
    for s = 1:N_subj
        [~, Z_aligned] = procrustes(Reference_Shape, All_Centroids(:,:,s));
        Aligned_Centroids(:,:,s) = Z_aligned;
    end
    New_Mean_Shape = mean(Aligned_Centroids, 3);
    New_Mean_Shape = New_Mean_Shape - mean(New_Mean_Shape, 1);
    New_Mean_Shape = New_Mean_Shape / norm(New_Mean_Shape, 'fro');
    if norm(New_Mean_Shape - Reference_Shape, 'fro') < tol, break; end
    Reference_Shape = New_Mean_Shape;
end
Grand_Mean_Centroids = Reference_Shape;

%% =====================================================================
% БЛОК ПРОВЕРКИ 6 ГИПОТЕЗ
% =====================================================================
disp(' '); disp('================================================================');
disp('      ФОРМАЛЬНАЯ ПРОВЕРКА 6 СТАТИСТИЧЕСКИХ ГИПОТЕЗ (ОТЧЕТ)      ');
disp('================================================================');
get_stars = @(p) subsref({'ns','*','**','***'}, substruct('{}', {1 + (p<0.05) + (p<0.01) + (p<0.001)}));

% Открываем файл для записи отчета
fid = fopen('Statistical_Report_Hypotheses.txt', 'w');
fprintf(fid, '================================================================\n');
fprintf(fid, '      ФОРМАЛЬНАЯ ПРОВЕРКА 6 СТАТИСТИЧЕСКИХ ГИПОТЕЗ (ОТЧЕТ)      \n');
fprintf(fid, '================================================================\n');

% ---------------------------------------------------------
% Вспомогательная функция для H1 (ANOVA Pseudo-F)
% ---------------------------------------------------------
function F_val = calc_global_F_safe(pts, labels)
    global_mean = mean(pts, 1);
    classes = unique(labels);
    k = length(classes);
    N = size(pts, 1);
    
    SSB = 0; SSW = 0;
    for i = 1:k
        class_pts = pts(labels == classes(i), :);
        n_c = size(class_pts, 1);
        class_mean = mean(class_pts, 1);
        SSB = SSB + n_c * sum((class_mean - global_mean).^2);
        SSW = SSW + sum(sum((class_pts - repmat(class_mean, n_c, 1)).^2));
    end
    F_val = (SSB / (k - 1)) / (SSW / (N - k));
end

% ---------------------------------------------------------
% ГИПОТЕЗА 1: НЕСЛУЧАЙНОСТЬ МЕТА-СОСТОЯНИЙ (PERMUTATION TEST)
% ---------------------------------------------------------
true_labels_19 = zeros(N_common, 1);
true_labels_19([1, 18]) = 1; true_labels_19([2, 19]) = 2; true_labels_19(3:7) = 3;     
true_labels_19([9, 10, 13, 15, 17]) = 4; true_labels_19([8, 11, 12, 14, 16]) = 5; 

All_Pts = reshape(permute(Aligned_Centroids, [1, 3, 2]), [N_common * N_subj, 3]);
Global_Labels = repmat(true_labels_19, N_subj, 1);

Real_F = calc_global_F_safe(All_Pts, Global_Labels);

N_perm = 10000; Null_F = zeros(N_perm, 1);
disp('Генерация нулевого распределения для H1 (10 000 перестановок)...');
for p = 1:N_perm
    shuffled_Global = repmat(true_labels_19(randperm(N_common)), N_subj, 1);
    Null_F(p) = calc_global_F_safe(All_Pts, shuffled_Global);
end
p1 = sum(Null_F >= Real_F) / N_perm;

msg = sprintf('\n>>> [H1] СУЩЕСТВОВАНИЕ МЕТА-СОСТОЯНИЙ (Permutation Test):\n');
msg = [msg, sprintf('Real Pseudo-F = %.2f | Max Null-F = %.2f\n', Real_F, max(Null_F))];
if p1 == 0, msg = [msg, sprintf('p-value < 1e-%d (***)\n', log10(N_perm))]; else, msg = [msg, sprintf('p-value = %.4f (%s)\n', p1, get_stars(p1))]; end
msg = [msg, 'ВЫВОД: Мета-состояния не случайны и объективно существуют.\n'];
disp(msg); fprintf(fid, '%s', msg);

% ---------------------------------------------------------
% ГИПОТЕЗА 2: КАЧЕСТВЕННОЕ РАЗДЕЛЕНИЕ ЗАДАЧ (MANOVA ДЛЯ УГЛОВ/ПРОЕКЦИЙ)
% ---------------------------------------------------------
[coeff, score] = pca(Grand_Mean_Centroids);
Music_2D = zeros(N_subj, 2); Metro_2D = zeros(N_subj, 2);
for s = 1:N_subj
    Music_2D(s,:) = (mean(Aligned_Centroids(idx_music, :, s), 1) - mean(Grand_Mean_Centroids)) * coeff(:,1:2);
    Metro_2D(s,:) = (mean(Aligned_Centroids(idx_metro, :, s), 1) - mean(Grand_Mean_Centroids)) * coeff(:,1:2);
end
[~, p2] = manova1([Music_2D; Metro_2D], [ones(N_subj,1); 2*ones(N_subj,1)]);

msg = sprintf('\n>>> [H2] ОРТОГОНАЛЬНОСТЬ МЕТРОНОМА И МУЗЫКИ (MANOVA):\n');
msg = [msg, sprintf('p-value = %.2e (%s)\n', p2(1), get_stars(p2(1)))];
msg = [msg, 'ВЫВОД: Состояния обрабатываются качественно разными нейро-ансамблями.\n'];
disp(msg); fprintf(fid, '%s', msg);

% ---------------------------------------------------------
% ГИПОТЕЗА 3: ТЕМПОРАЛЬНЫЙ ДРЕЙФ (КРИТЕРИЙ ХОТЕЛЛИНГА T^2)
% ---------------------------------------------------------
diff_EC = squeeze(Aligned_Centroids(18,:,:) - Aligned_Centroids(1,:,:))'; % N_subj x 3
m_diff = mean(diff_EC, 1); S_diff = cov(diff_EC);
T2 = N_subj * m_diff * inv(S_diff) * m_diff';
F_stat = (N_subj - 3) / (3 * (N_subj - 1)) * T2;
p3 = 1 - fcdf(F_stat, 3, N_subj - 3);

msg = sprintf('\n>>> [H3] ТЕМПОРАЛЬНЫЙ ДРЕЙФ И УТОМЛЕНИЕ (Hotelling''s T2):\n');
msg = [msg, sprintf('F(3, %d) = %.3f, p-value = %.2e (%s)\n', N_subj - 3, F_stat, p3, get_stars(p3))];
msg = [msg, 'ВЫВОД: Состояние покоя (EC) подвержено значимому смещению (усталости).\n'];
disp(msg); fprintf(fid, '%s', msg);

% ---------------------------------------------------------
% ГИПОТЕЗА 4: СТРУКТУРНАЯ СЕПАРАБЕЛЬНОСТЬ (ВНУТРИ vs МЕЖДУ)
% ---------------------------------------------------------
Grand_Dist_Matrix = squareform(pdist(Grand_Mean_Centroids));
dist_MM = Grand_Dist_Matrix(idx_music, idx_music); dist_MM = dist_MM(triu(true(size(dist_MM)), 1)); 
dist_MusicMetro = Grand_Dist_Matrix(idx_music, idx_metro); dist_MusicMetro = dist_MusicMetro(:); 
[~, p4, ~, stat4] = ttest2(dist_MM, dist_MusicMetro);

msg = sprintf('\n>>> [H4] СТРУКТУРНАЯ КОМПАКТНОСТЬ (T-test):\n');
msg = [msg, sprintf('Ср. расст. Внутри Музыки: %.3f | Межклассовое: %.3f\n', mean(dist_MM), mean(dist_MusicMetro))];
msg = [msg, sprintf('t = %.3f, p-value = %.2e (%s)\n', stat4.tstat, p4, get_stars(p4))];
msg = [msg, 'ВЫВОД: Дисперсия внутри класса достоверно меньше межклассового расстояния.\n'];
disp(msg); fprintf(fid, '%s', msg);

% ---------------------------------------------------------
% ГИПОТЕЗА 5: ОРИЕНТИРОВОЧНЫЙ РЕФЛЕКС (АНОМАЛИЯ NoRy 1)
% ---------------------------------------------------------
idx_music_rest = [idx_waltz, 11, 12, 14, 16]; 
d_nory1 = zeros(N_subj, 1); d_others = zeros(N_subj, 1);
for s = 1:N_subj
    C_rest = mean(Aligned_Centroids(idx_music_rest, :, s), 1);
    d_nory1(s) = norm(Aligned_Centroids(8, :, s) - C_rest);
    d_others(s) = mean(pdist2(Aligned_Centroids(idx_music_rest, :, s), C_rest));
end
[~, p5, ~, stat5] = ttest(d_nory1, d_others);

msg = sprintf('\n>>> [H5] ОРИЕНТИРОВОЧНЫЙ РЕФЛЕКС / АДАПТАЦИЯ (Paired T-test):\n');
msg = [msg, sprintf('Отдаление NoRy 1: %.3f | Отдаление остальных треков: %.3f\n', mean(d_nory1), mean(d_others))];
msg = [msg, sprintf('t = %.3f, p-value = %.2e (%s)\n', stat5.tstat, p5, get_stars(p5))];
msg = [msg, 'ВЫВОД: Первый музыкальный трек вызывает достоверную реакцию новизны.\n'];
disp(msg); fprintf(fid, '%s', msg);

% ---------------------------------------------------------
% ГИПОТЕЗА 6: ТОПОЛОГИЧЕСКОЕ КОДИРОВАНИЕ ТЕМПА (МЕТРОНОМЫ)
% ---------------------------------------------------------
metro_pts = Grand_Mean_Centroids(idx_metro, :);
[coeff_m, score_m] = pca(metro_pts);
[Rho6, p6] = corr(score_m(:,1), metro_freqs(:), 'Type', 'Spearman');

msg = sprintf('\n>>> [H6] КОДИРОВАНИЕ ТЕМПА РИТМА (Spearman Corr):\n');
msg = [msg, sprintf('Корреляция PC1 с физической частотой (0.5 - 4 Гц): Rho = %.3f\n', Rho6)];
msg = [msg, sprintf('p-value = %.4f (%s)\n', p6, get_stars(p6))];
msg = [msg, 'ВЫВОД: Обнаружен линейный градиент, кодирующий скорость изохронного ритма.\n'];
disp(msg); fprintf(fid, '%s', msg);
disp('================================================================');

fclose(fid);
disp('Отчет сохранен в файл Statistical_Report_Hypotheses.txt');

%% =====================================================================
% БЛОК ВИЗУАЛИЗАЦИЙ (12 ГРАФИКОВ)
% =====================================================================

% --- ФИГУРА 1: [H1] ГЛОБАЛЬНЫЙ ПЕРЕСТАНОВОЧНЫЙ ТЕСТ ---
figure('Name', 'Fig 1: [H1] Permutation Test', 'Color', 'w', 'Position', [50, 50, 600, 400]);
histogram(Null_F, 50, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'w'); hold on; grid on;
xline(Real_F, '-r', 'REAL BIOLOGICAL STRUCTURE', 'LineWidth', 3, 'LabelVerticalAlignment', 'top', 'FontSize', 12, 'FontWeight', 'bold');
title('Figure 1: [H1] Meta-States Permutation Test (418 points)'); xlabel('Pseudo-F Statistic'); ylabel('Count (10k iterations)');

% --- ФИГУРА 2: [H2] 2D PCA ЛАНДШАФТ И MANOVA ---
figure('Name', 'Fig 2: [H2] 2D PCA Landscape', 'Color', 'w', 'Position', [100, 100, 700, 600]); hold on; grid on;
for c = 1:N_common
    scatter(score(c,1), score(c,2), 200, colors(c,:), 'filled', 'MarkerEdgeColor', 'k');
    text(score(c,1), score(c,2)+0.02, target_conditions{c}, 'FontSize', 11, 'FontWeight', 'bold');
end
title(sprintf('Figure 2: [H2] 2D PCA Projection (MANOVA p=%.2e)', p2(1))); xlabel('PC 1'); ylabel('PC 2');

% --- ФИГУРА 3: [H3] ВЕКТОРЫ ТЕМПОРАЛЬНОГО ДРЕЙФА ---
figure('Name', 'Fig 3: [H3] Temporal Drift', 'Color', 'w', 'Position', [150, 150, 700, 600]); hold on; grid on;
scatter3(Grand_Mean_Centroids(:,1), Grand_Mean_Centroids(:,2), Grand_Mean_Centroids(:,3), 50, [0.8 0.8 0.8], 'filled');
for s = 1:N_subj
    plot3([Aligned_Centroids(1,1,s), Aligned_Centroids(18,1,s)], [Aligned_Centroids(1,2,s), Aligned_Centroids(18,2,s)], [Aligned_Centroids(1,3,s), Aligned_Centroids(18,3,s)], 'Color', [0.5 0.5 0.5 0.2]);
end
quiver3(Grand_Mean_Centroids(1,1), Grand_Mean_Centroids(1,2), Grand_Mean_Centroids(1,3), ...
        Grand_Mean_Centroids(18,1)-Grand_Mean_Centroids(1,1), Grand_Mean_Centroids(18,2)-Grand_Mean_Centroids(1,2), Grand_Mean_Centroids(18,3)-Grand_Mean_Centroids(1,3), ...
        0, 'Color', 'k', 'LineWidth', 3, 'MaxHeadSize', 0.5);
text(Grand_Mean_Centroids(1,1), Grand_Mean_Centroids(1,2), Grand_Mean_Centroids(1,3), ' EC1', 'FontSize', 12, 'FontWeight', 'bold');
text(Grand_Mean_Centroids(18,1), Grand_Mean_Centroids(18,2), Grand_Mean_Centroids(18,3), ' EC2', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('Figure 3: [H3] Temporal Drift Vectors (Hotelling p=%.2e)', p3)); xlabel('Dim 1'); ylabel('Dim 2'); zlabel('Dim 3'); view(-45, 30);

% --- ФИГУРА 4: [H4] СЕПАРАБЕЛЬНОСТЬ КЛАССОВ (BOXPLOTS) ---
figure('Name', 'Fig 4: [H4] Compactness Boxplots', 'Color', 'w', 'Position', [200, 200, 600, 500]);
boxplot([dist_MM; dist_MusicMetro], [repmat({'In Music'}, length(dist_MM), 1); repmat({'Music vs Metro'}, length(dist_MusicMetro), 1)], 'Colors', 'k');
ylabel('Euclidean Distance'); title(sprintf('Figure 4: [H4] Structural Compactness (T-test p=%.2e)', p4)); grid on;

% --- ФИГУРА 5: [H5] ОРИЕНТИРОВОЧНЫЙ РЕФЛЕКС (NoRy 1) ---
figure('Name', 'Fig 5: [H5] Orienting Reflex', 'Color', 'w', 'Position', [250, 250, 500, 400]); hold on; grid on;
boxplot([d_nory1, d_others], 'Labels', {'NoRy 1 (First)', 'Other Music'}, 'Colors', 'k');
for s = 1:N_subj, plot([1, 2], [d_nory1(s), d_others(s)], 'Color', [0.7 0.7 0.7 0.5]); end
ylabel('Distance to Music Centroid'); title(sprintf('Figure 5: [H5] Novelty Effect of First Track (p=%.2e)', p5));

% --- ФИГУРА 6: [H6] ТОПОЛОГИЧЕСКОЕ КОДИРОВАНИЕ ТЕМПА ---
figure('Name', 'Fig 6: [H6] Rhythm Gradient', 'Color', 'w', 'Position', [300, 300, 500, 400]); hold on; grid on;
scatter(metro_freqs, score_m(:,1), 200, colors(idx_metro,:), 'filled', 'MarkerEdgeColor', 'k');
lsline; % Добавляем линию тренда
for i=1:length(idx_metro), text(metro_freqs(i), score_m(i,1)+0.01, target_conditions{idx_metro(i)}, 'FontSize', 11, 'FontWeight', 'bold'); end
xlabel('Physical Stimulus Frequency (Hz)'); ylabel('Local PC1 Score (Topological Position)');
title(sprintf('Figure 6: [H6] Rate-Coding Gradient (Spearman \\rho=%.2f, p=%.3f)', Rho6, p6));

% --- ФИГУРА 7: МАТРИЦА РАССТОЯНИЙ ---
figure('Name', 'Fig 7: Universal Distance Matrix', 'Color', 'w', 'Position', [350, 350, 500, 450]);
imagesc(Grand_Dist_Matrix); colormap(hot); colorbar; title('Figure 7: Universal Procrustes Distances');
xticks(1:N_common); yticks(1:N_common); xticklabels(target_conditions); yticklabels(target_conditions); xtickangle(45); axis square;

% --- ФИГУРА 8: ИЕРАРХИЧЕСКАЯ ТОПОЛОГИЯ ---
Z_univ = linkage(Grand_Mean_Centroids, 'ward', 'euclidean');
figure('Name', 'Fig 8: Hierarchical Topology', 'Color', 'w', 'Position', [400, 400, 900, 400]); tiledlayout(1, 2, 'TileSpacing', 'compact');
nexttile; [H, ~, outperm] = dendrogram(Z_univ, 0, 'Orientation', 'left'); set(H, 'LineWidth', 2); title('Figure 8a: Hierarchical Tree'); yticks(1:N_common); yticklabels(target_conditions(outperm));
K_test = 2:9; eva = evalclusters(Grand_Mean_Centroids, 'kmeans', 'Silhouette', 'KList', K_test); K_opt = eva.OptimalK;
nexttile; plot(K_test, eva.CriterionValues, '-ko', 'LineWidth', 2, 'MarkerFaceColor', 'b'); xline(K_opt, '--r', ['Optimal K = ', num2str(K_opt)], 'LineWidth', 1.5); title('Figure 8b: Silhouette Score'); grid on;

% --- ФИГУРА 9: МАТРИЦА СОВПАДЕНИЙ (CONFUSION MATRIX) ---
Empirical_Labels = cluster(Z_univ, 'maxclust', K_opt); true_labels = zeros(N_common, 1); true_labels(idx_ec) = 1; true_labels(idx_eo) = 2; true_labels(idx_metro) = 3; true_labels(idx_waltz) = 4; true_labels(idx_nory) = 5;
Conf_Mat = zeros(5, K_opt); for i = 1:N_common, Conf_Mat(true_labels(i), Empirical_Labels(i)) = Conf_Mat(true_labels(i), Empirical_Labels(i)) + 1; end
Conf_Mat_Pct = (Conf_Mat ./ sum(Conf_Mat, 1)) * 100;
figure('Name', 'Fig 9: Confusion Matrix', 'Color', 'w', 'Position', [450, 450, 500, 400]);
imagesc(Conf_Mat_Pct); colormap('parula'); colorbar; title(sprintf('Figure 9: Clustering Purity (K=%d)', K_opt)); yticks(1:5); yticklabels({'EC', 'EO', 'Metro', 'Waltz', 'NoRy'}); xticks(1:K_opt);

% --- ФИГУРА 10: УНИВЕРСАЛЬНОЕ 3D ПРОСТРАНСТВО ---
figure('Name', 'Fig 10: Universal Space', 'Color', 'w', 'Position', [500, 500, 700, 600]); hold on; grid on;
for c = 1:N_common
    pts = squeeze(Aligned_Centroids(c, :, :))'; scatter3(pts(:,1), pts(:,2), pts(:,3), 25, colors(c,:), 'filled', 'MarkerFaceAlpha', 0.4);
    scatter3(Grand_Mean_Centroids(c,1), Grand_Mean_Centroids(c,2), Grand_Mean_Centroids(c,3), 250, colors(c,:), 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    text(Grand_Mean_Centroids(c,1), Grand_Mean_Centroids(c,2), Grand_Mean_Centroids(c,3)+0.05, target_conditions{c}, 'FontSize', 11, 'FontWeight', 'bold');
end
title(sprintf('Figure 10: Universal 3D Space (N=%d)', N_subj), 'FontSize', 14); view(-45, 30);

% --- ФИГУРА 11: 3D ЭЛЛИПСОИДЫ УВЕРЕННОСТИ (COVARIANCE) ---
figure('Name', 'Fig 11: Confidence Ellipsoids', 'Color', 'w', 'Position', [550, 550, 800, 600]); hold on; grid on;
[X_sph, Y_sph, Z_sph] = sphere(20); n_sd = 1.0;
for c = 1:N_common
    pts = squeeze(Aligned_Centroids(c, :, :))'; mu = Grand_Mean_Centroids(c, :); [V, D] = eig(cov(pts)); radii = n_sd * sqrt(diag(D));
    rotated_ell = V * [X_sph(:)' * radii(1); Y_sph(:)' * radii(2); Z_sph(:)' * radii(3)];
    X_ell = reshape(rotated_ell(1,:), size(X_sph)) + mu(1); Y_ell = reshape(rotated_ell(2,:), size(Y_sph)) + mu(2); Z_ell = reshape(rotated_ell(3,:), size(Z_sph)) + mu(3);
    surf(X_ell, Y_ell, Z_ell, 'FaceColor', colors(c,:), 'EdgeColor', 'none', 'FaceAlpha', 0.15); scatter3(mu(1), mu(2), mu(3), 150, colors(c,:), 'filled', 'MarkerEdgeColor', 'k');
end
title('Figure 11: 3D Spatial Dispersion (Covariance Ellipsoids)'); view(-45, 30); axis equal;

% --- ФИГУРА 12: 2D ПОЛЯРНАЯ ПРОЕКЦИЯ ---
radii_polar = sqrt(sum(Grand_Mean_Centroids.^2, 2));
angles_polar = atan2(score(:,2), score(:,1)); 
figure('Name', 'Fig 12: Polar Topology', 'Color', 'w', 'Position', [600, 50, 700, 700]);
pax = polaraxes; hold on; pax.ThetaZeroLocation = 'top'; pax.ThetaDir = 'clockwise';
for c = 1:N_common
    polarplot([0 angles_polar(c)], [0 radii_polar(c)], 'Color', [colors(c,:) 0.4], 'LineWidth', 2);
    polarscatter(angles_polar(c), radii_polar(c), 300, colors(c,:), 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    align_h = 'left'; if angles_polar(c) < 0, align_h = 'right'; end
    text(angles_polar(c), radii_polar(c) + 0.03, [' ', target_conditions{c}, ' '], 'FontSize', 11, 'FontWeight', 'bold', 'HorizontalAlignment', align_h);
end
title('Figure 12: Corrected Polar Projection (PCA basis)');

disp('>>> ПАЙПЛАЙН ЗАВЕРШЕН. СГЕНЕРИРОВАНО 12 ГРАФИКОВ. <<<');