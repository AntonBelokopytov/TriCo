%% =====================================================================
% PROCRUSTES ALIGNMENT, MICRO-STATES & FULL VISUALIZATION SUITE
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

% МИКРО-ГРУППЫ ДЛЯ ДЕТАЛЬНОЙ СТАТИСТИКИ
idx_ec    = [1, 18];
idx_eo    = [2, 19];
idx_metro = 3:7;
idx_waltz = [9, 10, 13, 15, 17];
idx_nory  = [8, 11, 12, 14, 16];
idx_music = [idx_waltz, idx_nory];

% Палитра для классов
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
subj_names = {};
subj_count = 0;

for p = 1:length(base_dirs)
    emb_dir = fullfile(base_dirs{p}, 'embeddings', freq_name);
    mat_files = dir(fullfile(emb_dir, ['UMAP_*_', freq_name, '.mat']));
    
    for f = 1:length(mat_files)
        fname = mat_files(f).name;
        base_name = strrep(fname, 'UMAP_', '');
        base_name = strrep(base_name, ['_', freq_name, '.mat'], '');
        
        load(fullfile(emb_dir, fname), 'R', 'valid_cond_idx');
        
        cc = NaN(N_common, 3);
        for c = 1:N_common
            idx = find(valid_cond_idx == c);
            if ~isempty(idx), cc(c, :) = mean(R(idx, 1:3), 1); end
        end
        
        if ~any(isnan(cc(:))) 
            subj_count = subj_count + 1;
            All_Centroids_temp(:, :, subj_count) = cc;
            subj_names{subj_count} = base_name;
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
max_iter = 100; tol = 1e-12;

for iter = 1:max_iter
    for s = 1:N_subj
        [~, Z_aligned] = procrustes(Reference_Shape, All_Centroids(:,:,s));
        Aligned_Centroids(:,:,s) = Z_aligned;
    end
    New_Mean_Shape = mean(Aligned_Centroids, 3);
    New_Mean_Shape = New_Mean_Shape - mean(New_Mean_Shape, 1);
    New_Mean_Shape = New_Mean_Shape / norm(New_Mean_Shape, 'fro');
    
    if norm(New_Mean_Shape - Reference_Shape, 'fro') < tol
        Reference_Shape = New_Mean_Shape; break;
    end
    Reference_Shape = New_Mean_Shape;
end
Grand_Mean_Centroids = Reference_Shape;

%% =====================================================================
% ОТЧЕТ И СТАТИСТИКА (ВЫВОД В КОНСОЛЬ)
% =====================================================================
disp(' '); disp('================================================================');
disp('            ДЕТАЛЬНЫЙ СТАТИСТИЧЕСКИЙ ОТЧЕТ И ГРАФИКИ            ');
disp('================================================================');

%% --- ФИГУРА 1: ДИСПЕРСИЯ УСЛОВИЙ ---
Cond_Variances = zeros(N_common, 1);
for c = 1:N_common
    pts = squeeze(Aligned_Centroids(c, :, :))'; 
    Cond_Variances(c) = mean(pdist2(pts, Grand_Mean_Centroids(c, :)));
end
[sort_var, var_idx] = sort(Cond_Variances, 'ascend');

figure('Name', 'Fig 1: Cross-Subject Variance', 'Color', 'w', 'Position', [100, 100, 800, 400]);
bar(Cond_Variances, 'FaceColor', [0.2 0.6 0.7]); hold on; grid on;
xticks(1:N_common); xticklabels(target_conditions); xtickangle(45);
ylabel('Mean Distance to Group Centroid'); title('Figure 1: Cross-Subject Variance (Stability)');

%% --- ФИГУРА 2: МАТРИЦА РАССТОЯНИЙ (RSA) ---
Grand_Dist_Matrix = squareform(pdist(Grand_Mean_Centroids));

figure('Name', 'Fig 2: Universal Distance Matrix', 'Color', 'w', 'Position', [150, 150, 600, 550]);
imagesc(Grand_Dist_Matrix); colormap(hot); colorbar;
title('Figure 2: Distances in Universal Procrustes Space');
xticks(1:N_common); yticks(1:N_common);
xticklabels(target_conditions); yticklabels(target_conditions); xtickangle(45); axis square;

%% --- ФИГУРА 3: ВЕКТОРЫ ТЕМПОРАЛЬНОГО ДРЕЙФА ---
fprintf('\n--- 1. АНАЛИЗ ТЕМПОРАЛЬНОГО ДРЕЙФА ---\n');
d_EC = norm(Grand_Mean_Centroids(1, :) - Grand_Mean_Centroids(18, :)); 
d_EO = norm(Grand_Mean_Centroids(2, :) - Grand_Mean_Centroids(19, :)); 
fprintf('Сдвиг Закрытых глаз (EC1 -> EC2): %.4f\n', d_EC);
fprintf('Сдвиг Открытых глаз (EO1 -> EO2): %.4f\n', d_EO);

figure('Name', 'Fig 3: Temporal Drift Vectors', 'Color', 'w', 'Position', [200, 200, 700, 500]);
hold on; grid on;
scatter3(Grand_Mean_Centroids(:,1), Grand_Mean_Centroids(:,2), Grand_Mean_Centroids(:,3), 50, [0.8 0.8 0.8], 'filled');
% Рисуем стрелку EC1 -> EC2
quiver3(Grand_Mean_Centroids(1,1), Grand_Mean_Centroids(1,2), Grand_Mean_Centroids(1,3), ...
        Grand_Mean_Centroids(18,1)-Grand_Mean_Centroids(1,1), Grand_Mean_Centroids(18,2)-Grand_Mean_Centroids(1,2), Grand_Mean_Centroids(18,3)-Grand_Mean_Centroids(1,3), ...
        0, 'Color', 'k', 'LineWidth', 2, 'MaxHeadSize', 0.5);
text(Grand_Mean_Centroids(1,1), Grand_Mean_Centroids(1,2), Grand_Mean_Centroids(1,3), ' EC1', 'FontSize', 12, 'FontWeight', 'bold');
text(Grand_Mean_Centroids(18,1), Grand_Mean_Centroids(18,2), Grand_Mean_Centroids(18,3), ' EC2', 'FontSize', 12, 'FontWeight', 'bold');
% Рисуем стрелку EO1 -> EO2
quiver3(Grand_Mean_Centroids(2,1), Grand_Mean_Centroids(2,2), Grand_Mean_Centroids(2,3), ...
        Grand_Mean_Centroids(19,1)-Grand_Mean_Centroids(2,1), Grand_Mean_Centroids(19,2)-Grand_Mean_Centroids(2,2), Grand_Mean_Centroids(19,3)-Grand_Mean_Centroids(2,3), ...
        0, 'Color', 'b', 'LineWidth', 2, 'MaxHeadSize', 0.5);
text(Grand_Mean_Centroids(2,1), Grand_Mean_Centroids(2,2), Grand_Mean_Centroids(2,3), ' EO1', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'b');
text(Grand_Mean_Centroids(19,1), Grand_Mean_Centroids(19,2), Grand_Mean_Centroids(19,3), ' EO2', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'b');
title('Figure 3: Temporal Drift (Start vs End of Experiment)'); xlabel('Dim 1'); ylabel('Dim 2'); zlabel('Dim 3'); view(-45, 30);

%% --- ФИГУРА 4: ПОЛНАЯ МАТРИЦА СЕПАРАБЕЛЬНОСТИ МИКРО-СОСТОЯНИЙ ---
disp('>>> ШАГ 4: ПОЛНЫЙ АНАЛИЗ ВСЕХ ПАР МИКРО-СОСТОЯНИЙ <<<');

% ОПРЕДЕЛЯЕМ 5 МИКРО-СОСТОЯНИЙ
groups = {idx_ec, idx_eo, idx_metro, idx_waltz, idx_nory};
group_names = {'EC', 'EO', 'Metro', 'Waltz', 'NoRy'};
N_g = length(groups);

Meta_Dist_Matrix = zeros(N_g, N_g);
all_dists = [];
all_labels = {};

% Перебираем все возможные комбинации (Внутри и Между)
for i = 1:N_g
    for j = i:N_g
        % Извлекаем расстояния между группой i и группой j
        d = Grand_Dist_Matrix(groups{i}, groups{j});
        
        if i == j
            % Внутри одной группы (берем только верхний треугольник без диагонали)
            d = d(triu(true(size(d)), 1));
            label = sprintf('In %s', group_names{i});
        else
            % Между разными группами (берем всю подматрицу перекрестных расстояний)
            d = d(:);
            label = sprintf('%s vs %s', group_names{i}, group_names{j});
        end
        
        if ~isempty(d)
            % Сохраняем среднее расстояние для тепловой карты
            Meta_Dist_Matrix(i, j) = mean(d);
            Meta_Dist_Matrix(j, i) = mean(d); % Делаем матрицу симметричной
            
            % Сохраняем сырые данные для Boxplot
            all_dists = [all_dists; d];
            all_labels = [all_labels; repmat({label}, length(d), 1)];
        end
    end
end

figure('Name', 'Fig 4: Comprehensive Separability', 'Color', 'w', 'Position', [150, 150, 1400, 600]);
tiledlayout(1, 2, 'TileSpacing', 'compact');

% График 4A: Тепловая матрица расстояний между 5 классами
ax1 = nexttile;
imagesc(ax1, Meta_Dist_Matrix); colormap(ax1, 'hot'); cb = colorbar(ax1);
cb.Label.String = 'Mean Euclidean Distance';
title(ax1, 'Figure 4a: Distance Between Meta-States');
xticks(ax1, 1:N_g); xticklabels(ax1, group_names);
yticks(ax1, 1:N_g); yticklabels(ax1, group_names);
axis(ax1, 'square');

% Добавляем значения прямо в ячейки матрицы для наглядности
for i = 1:N_g
    for j = 1:N_g
        text_color = 'w'; if Meta_Dist_Matrix(i,j) > max(Meta_Dist_Matrix(:))*0.6, text_color = 'k'; end
        text(ax1, j, i, sprintf('%.2f', Meta_Dist_Matrix(i,j)), ...
            'HorizontalAlignment', 'center', 'Color', text_color, 'FontWeight', 'bold');
    end
end

% График 4B: Большой Boxplot для всех 15 пар
ax2 = nexttile;
boxplot(ax2, all_dists, all_labels, 'LabelOrientation', 'inline', 'Colors', 'k');
ylabel(ax2, 'Euclidean Distance in UMAP Space');
title(ax2, 'Figure 4b: All Pairwise Comparisons (Within and Between States)');
grid(ax2, 'on');

% T-test для парочки самых важных гипотез (чтобы вывести в консоль)
fprintf('\n--- КЛЮЧЕВЫЕ СТАТИСТИЧЕСКИЕ ПРОВЕРКИ ---\n');
d_in_metro = all_dists(strcmp(all_labels, 'In Metro'));
d_in_waltz = all_dists(strcmp(all_labels, 'In Waltz'));
d_metro_waltz = all_dists(strcmp(all_labels, 'Metro vs Waltz'));
d_waltz_nory = all_dists(strcmp(all_labels, 'Waltz vs NoRy'));

[~, p_mw] = ttest2(d_in_waltz, d_metro_waltz);
[~, p_wn] = ttest2(d_in_waltz, d_waltz_nory);

fprintf('Отличает ли мозг Вальс от Метронома? p-value = %.2e\n', p_mw);
fprintf('Отличает ли мозг Вальс от NoRy? p-value = %.2e\n', p_wn);

%% --- ФИГУРА 5: ИЕРАРХИЧЕСКАЯ КЛАСТЕРИЗАЦИЯ И СИЛУЭТ ---
Z_univ = linkage(Grand_Mean_Centroids, 'ward', 'euclidean');
coph = cophenet(Z_univ, pdist(Grand_Mean_Centroids));
fprintf('\n--- 3. ИЕРАРХИЧЕСКАЯ ТОПОЛОГИЯ ---\n');
fprintf('Кофенетическая корреляция: %.3f\n', coph);

figure('Name', 'Fig 5: Hierarchical Topology', 'Color', 'w', 'Position', [300, 300, 1000, 500]);
tiledlayout(1, 2, 'TileSpacing', 'compact');
nexttile;
[H, ~, outperm] = dendrogram(Z_univ, 0, 'Orientation', 'left', 'ColorThreshold', 'default');
set(H, 'LineWidth', 2); title('Figure 5a: Hierarchical Tree'); xlabel('Distance'); yticks(1:N_common); yticklabels(target_conditions(outperm));

K_test = 2:9;
eva = evalclusters(Grand_Mean_Centroids, 'kmeans', 'Silhouette', 'KList', K_test);
K_opt = eva.OptimalK;
fprintf('Оптимальное K (по Силуэту) = %d\n', K_opt);

nexttile;
plot(K_test, eva.CriterionValues, '-ko', 'LineWidth', 2, 'MarkerFaceColor', 'b');
xline(K_opt, '--r', ['Optimal K = ', num2str(K_opt)], 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
title('Figure 5b: Silhouette Score'); xlabel('Number of Clusters (K)'); ylabel('Silhouette Value'); grid on;

%% --- ФИГУРА 6: МАТРИЦА СОВПАДЕНИЙ (CONFUSION MATRIX) ---
Empirical_Labels = cluster(Z_univ, 'maxclust', K_opt);
true_labels = zeros(N_common, 1);
true_labels(idx_ec) = 1; true_labels(idx_eo) = 2; true_labels(idx_metro) = 3; true_labels(idx_waltz) = 4; true_labels(idx_nory) = 5;
true_names = {'EC', 'EO', 'Metronome', 'Waltz', 'NoRy'};

Conf_Mat = zeros(5, K_opt);
for i = 1:N_common, Conf_Mat(true_labels(i), Empirical_Labels(i)) = Conf_Mat(true_labels(i), Empirical_Labels(i)) + 1; end
Conf_Mat_Pct = (Conf_Mat ./ sum(Conf_Mat, 1)) * 100;

figure('Name', 'Fig 6: Empirical vs Hypothesis', 'Color', 'w', 'Position', [350, 350, 600, 500]);
imagesc(Conf_Mat_Pct); colormap('parula'); cb = colorbar; cb.Label.String = '% of Cluster Composition';
title(sprintf('Figure 6: Data-Driven Clusters vs 5 Micro-States (K=%d)', K_opt));
yticks(1:5); yticklabels(true_names); xticks(1:K_opt); xticklabels(arrayfun(@(x) sprintf('Clust %d', x), 1:K_opt, 'UniformOutput', false));
for c = 1:K_opt
    for r = 1:5
        val = Conf_Mat_Pct(r, c); txt_color = 'k'; if val > 50, txt_color = 'w'; end
        text(c, r, sprintf('%.0f%%', val), 'HorizontalAlignment', 'center', 'Color', txt_color, 'FontWeight', 'bold');
    end
end

%% --- ФИГУРА 7: УНИВЕРСАЛЬНОЕ 3D ПРОСТРАНСТВО ---
figure('Name', 'Fig 7: Universal Micro-State Space', 'Color', 'w', 'Position', [400, 400, 900, 700]);
hold on; grid on;
for c = 1:N_common
    pts = squeeze(Aligned_Centroids(c, :, :))'; 
    scatter3(pts(:,1), pts(:,2), pts(:,3), 25, colors(c,:), 'filled', 'MarkerFaceAlpha', 0.4);
    scatter3(Grand_Mean_Centroids(c,1), Grand_Mean_Centroids(c,2), Grand_Mean_Centroids(c,3), 250, colors(c,:), 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    text(Grand_Mean_Centroids(c,1), Grand_Mean_Centroids(c,2), Grand_Mean_Centroids(c,3)+0.05, target_conditions{c}, 'FontSize', 11, 'FontWeight', 'bold');
end
title(sprintf('Figure 7: Universal 3D Space (N=%d)', N_subj), 'FontSize', 14); view(-45, 30);
plot3(Grand_Mean_Centroids(idx_metro,1), Grand_Mean_Centroids(idx_metro,2), Grand_Mean_Centroids(idx_metro,3), 'Color', [0.9 0.2 0.2], 'LineWidth', 2);
plot3(Grand_Mean_Centroids(idx_music,1), Grand_Mean_Centroids(idx_music,2), Grand_Mean_Centroids(idx_music,3), 'Color', [0.2 0.6 0.9], 'LineWidth', 2);

disp('>>> ПАЙПЛАЙН ЗАВЕРШЕН. СГЕНЕРИРОВАНО 7 ГРАФИКОВ. <<<');

%% --- ФИГУРА 8: 3D ДИСПЕРСИЯ И ЭЛЛИПСОИДЫ УВЕРЕННОСТИ (COVARIANCE) ---
disp('>>> ШАГ 5: РАСЧЕТ И ВИЗУАЛИЗАЦИЯ 3D-ЭЛЛИПСОИДОВ ДИСПЕРСИИ <<<');

figure('Name', 'Fig 8: 3D Confidence Ellipsoids', 'Color', 'w', 'Position', [450, 450, 1000, 800]);
hold on; grid on;

% Параметр: сколько стандартных отклонений охватывает эллипсоид
% n_sd = 1 (охватывает ~68% плотности), n_sd = 2 (~95% плотности)
n_sd = 1.0; 

% Генерация базовой сферы (шаблон для деформации)
[X_sph, Y_sph, Z_sph] = sphere(20);

for c = 1:N_common
    % Вытягиваем 3D координаты всех испытуемых для данного условия
    pts = squeeze(Aligned_Centroids(c, :, :))'; % Матрица [22 испытуемых x 3 координаты]
    mu = Grand_Mean_Centroids(c, :);            % Истинный центр масс
    
    % 1. Считаем матрицу ковариации 3x3
    Cov_Mat = cov(pts);
    
    % 2. Спектральное разложение (Находим оси эллипсоида)
    % V - собственные векторы (направления осей)
    % D - собственные значения (дисперсия вдоль этих осей)
    [V, D] = eig(Cov_Mat);
    
    % Радиусы эллипсоида = (SD) * корень из собственных значений
    radii = n_sd * sqrt(diag(D));
    
    % 3. Трансформируем сферу в эллипсоид нужного размера
    X_ell = X_sph * radii(1);
    Y_ell = Y_sph * radii(2);
    Z_ell = Z_sph * radii(3);
    
    % 4. Поворачиваем эллипсоид в пространстве (согласно собственным векторам)
    rotated_ell = V * [X_ell(:)'; Y_ell(:)'; Z_ell(:)'];
    
    % Возвращаем форму матрицы и сдвигаем в центр масс условия (mu)
    X_ell = reshape(rotated_ell(1,:), size(X_sph)) + mu(1);
    Y_ell = reshape(rotated_ell(2,:), size(Y_sph)) + mu(2);
    Z_ell = reshape(rotated_ell(3,:), size(Z_sph)) + mu(3);
    
    % --- ОТРИСОВКА ---
    % Рисуем полупрозрачный эллипсоид
    surf(X_ell, Y_ell, Z_ell, 'FaceColor', colors(c,:), 'EdgeColor', 'none', 'FaceAlpha', 0.15);
    
    % Рисуем точки отдельных испытуемых внутри эллипсоида
    scatter3(pts(:,1), pts(:,2), pts(:,3), 15, colors(c,:), 'filled', 'MarkerFaceAlpha', 0.5);
    
    % Рисуем жирный центроид
    scatter3(mu(1), mu(2), mu(3), 150, colors(c,:), 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.0);
    
    % Подпись условия
    text(mu(1), mu(2), mu(3)+0.15, target_conditions{c}, 'FontSize', 10, 'FontWeight', 'bold', 'Color', [0.3 0.3 0.3]);
end

title(sprintf('Figure 8: 3D Spatial Dispersion (Covariance Ellipsoids at %.1f SD)', n_sd), 'FontSize', 14);
xlabel('Dim 1'); ylabel('Dim 2'); zlabel('Dim 3'); 
view(-45, 30);
axis equal; % КРИТИЧЕСКИ ВАЖНО: чтобы оси были в одном масштабе и эллипсы не искажались визуально!

% Легкие линии макро-связей
plot3(Grand_Mean_Centroids(idx_metro,1), Grand_Mean_Centroids(idx_metro,2), Grand_Mean_Centroids(idx_metro,3), 'Color', [0.9 0.2 0.2 0.3], 'LineWidth', 2);
plot3(Grand_Mean_Centroids(idx_music,1), Grand_Mean_Centroids(idx_music,2), Grand_Mean_Centroids(idx_music,3), 'Color', [0.2 0.6 0.9 0.3], 'LineWidth', 2);