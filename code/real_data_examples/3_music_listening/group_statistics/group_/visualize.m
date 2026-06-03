%% =====================================================================
% OFFLINE VISUALIZATION TOOL FOR eSPoC & UMAP RESULTS
% =====================================================================
close all; clear; clc;

% --- 1. НАСТРОЙКИ (ЧТО СМОТРИМ) ---
% Выбери базовые пути (part1 или part2)
data_dir       = 'C:\Users\ansbel\Documents\GitHub\TriCo\data\external\music_listening\part2\eeg\';
base_emb_dir   = 'C:\Users\ansbel\Documents\GitHub\TriCo\data\external\music_listening\part2\embeddings\';
base_stats_dir = 'C:\Users\ansbel\Documents\GitHub\TriCo\data\external\music_listening\part2\stats\';
ft_path        = 'C:\Users\ansbel\Documents\GitHub\site-packages\fieldtrip';

% Выбери файл и диапазон для визуализации
base_name = 'TumAle';  % Впиши нужный ID
freq_name = 'beta';           % Выбери: theta, alpha, beta, gamma

% Условия эксперимента (скрипт сам возьмет первые nEpochs)
target_conditions = {
    'EC1', 'EO1', '2Hz', '05Hz', '4Hz', '1Hz', '3Hz', ...
    'NoRy 1', 'Waltz 1', 'Waltz 2', 'NoRy 2', 'NoRy 3', ...
    'Waltz 3', 'NoRy 4', 'Waltz 4', 'NoRy 5', 'Waltz 5', ...
    'EC2', 'EO2', 'Waltz 6', 'Waltz 7', 'Waltz 8'
};

%% --- 2. ЗАГРУЗКА ДАННЫХ ---
if ~exist('ft_defaults','file'), addpath(ft_path); end
ft_defaults;

disp(['Загрузка данных для: ', base_name, ' [', freq_name, ']']);
emb_file   = fullfile(base_emb_dir, freq_name, sprintf('UMAP_%s_%s.mat', base_name, freq_name));
stats_file = fullfile(base_stats_dir, freq_name, sprintf('STATS_%s_%s.mat', base_name, freq_name));

if ~exist(emb_file, 'file') || ~exist(stats_file, 'file')
    error('Файлы не найдены! Проверьте правильность путей.');
end

load(emb_file);   
load(stats_file); 

% Защита размерности, если найдена только 1 глобальная компонента
if isvector(corrs_true), corrs_true = corrs_true(:)'; end
if isvector(eigenvalues_true), eigenvalues_true = eigenvalues_true(:)'; end

% Загружаем шапку сырого файла только ради лейблов электродов (для топоплотов)
cfg = []; cfg.dataset = fullfile(data_dir, [base_name, '_raw.fif']); cfg.continuous = 'yes';
Xinf = ft_preprocessing(cfg);
laycfg = []; laycfg.elec = Xinf.elec; lay = ft_prepare_layout(laycfg);     
topo = []; topo.dimord = 'chan_time'; topo.label = Xinf.elec.label(1:38); topo.time = 0;
cfg_topo = []; cfg_topo.marker = 'labels'; cfg_topo.layout = lay; 
cfg_topo.comment = 'no'; cfg_topo.style = 'fill'; cfg_topo.colorbar = 'no';

%% --- 3. ПРЕДВАРИТЕЛЬНАЯ ПОДГОТОВКА ПЕРЕМЕННЫХ ---
Rmean = R - mean(R, 1);
Covs_valid = Covs(:, :, valid_windows);

% Динамическое воссоздание вектора отбеленных ковариационных признаков (Feat) 
% для расчета "истинного" глобального сигнала (gl_src)
n_chan = size(Covs_valid, 1);
n_feat = (n_chan^2 - n_chan)/2 + n_chan;
F = zeros(n_feat, size(Covs_valid, 3));
for i = 1:size(Covs_valid, 3)
    C = Covs_valid(:,:,i);
    upper_mask = triu(true(size(C)));
    upper_triu_mask = triu(true(size(C)), 1);
    C(upper_triu_mask) = C(upper_triu_mask) * sqrt(2);
    v = C(upper_mask);
    F(:, i) = v(:);
end
F = F - mean(F, 2);

% --- Расчет канонических проекций ---
gl_src = Vf_true' * F;              % Проекция ковариаций
emb_can_pr = Vz_true' * Rmean';     % Проекция UMAP

% Таймлайн условия
boundaries = find(diff(valid_cond_idx) ~= 0);
ticks = [0; boundaries; length(valid_cond_idx)];
cmap = jet(nEpochs);

% Центроиды в UMAP пространстве
ccx = NaN(1, nEpochs); ccy = NaN(1, nEpochs); ccz = NaN(1, nEpochs);
for i = 1:nEpochs
    idx = (valid_cond_idx == i);
    if any(idx)
        ccx(i) = mean(R(idx, 1)); ccy(i) = mean(R(idx, 2)); ccz(i) = mean(R(idx, 3));
    end
end

% Центроиды в каноническом пространстве eSPoC
x = emb_can_pr(1,:); y = emb_can_pr(2,:); z = emb_can_pr(3,:);
cx_can = NaN(1, nEpochs); cy_can = NaN(1, nEpochs); cz_can = NaN(1, nEpochs);
for i = 1:nEpochs
    idx = (valid_cond_idx == i);
    if any(idx)
        cx_can(i) = mean(x(idx)); cy_can(i) = mean(y(idx)); cz_can(i) = mean(z(idx));
    end
end

%% =====================================================================
% РИСУНОК 1: ВРЕМЕННАЯ ДИНАМИКА UMAP КОМПОНЕНТ
% =====================================================================
figure('Name', 'UMAP Temporal Dynamics', 'Color', 'w', 'Position', [50, 50, 900, 400]);
plot(R, 'LineWidth', 1.2); hold on; grid on;
for k = 1:length(ticks), xline(ticks(k), '--', 'Color', [0.7 0.7 0.7]); end
xlim([0, size(R,1)]); xticks(ticks(1:end-1)); 
xticklabels(arrayfun(@(val) ['(', num2str(val), ')'], 1:nEpochs, 'UniformOutput', false));
ylabel('UMAP Coordinate'); xlabel('Conditions');
legend({'Dim 1', 'Dim 2', 'Dim 3'}, 'Location', 'best');
title(['Temporal Evolution of UMAP | ', strrep(base_name, '_', '\_')]);

%% =====================================================================
% РИСУНОК 2: eSPoC ПОРОГИ ЗНАЧИМОСТИ (STEM)
% =====================================================================
figure('Name', 'eSPoC Significance', 'Color', 'w', 'Position', [100, 100, 800, 400]);
stem(corrs_true', 'LineWidth', 1.5); hold on; grid on;
if ~isnan(max_val), yline(max_val, 'r', ['Max (', num2str(max_val, '%.2f'), ')'], 'LineWidth', 1.5); end
if ~isnan(min_val), yline(min_val, 'b', ['Min (', num2str(min_val, '%.2f'), ')'], 'LineWidth', 1.5); end
xlabel('Local component index'); ylabel('Correlation');
title(['eSPoC Correlations & Significance | ', strrep(base_name, '_', '\_')]);
leg_labels = arrayfun(@(x) sprintf('Global %d', x), 1:size(corrs_true, 1), 'UniformOutput', false);
legend(leg_labels, 'Location', 'best');
xlim([0.5 size(corrs_true, 2)+0.5]);

%% =====================================================================
% РИСУНОК 3: СРАВНЕНИЕ КАНОНИЧЕСКИХ ПРОЕКЦИЙ (CCA FIT) - ВОССТАНОВЛЕНО
% =====================================================================
n_plot_comps = min(3, size(gl_src, 1)); % Защита, если размерность < 3
figure('Name', 'Canonical Projections Match', 'Color', 'w', 'Position', [150, 150, 1200, 400]);
t_cca = tiledlayout(n_plot_comps, 1, 'TileSpacing', 'compact');
sgtitle('Global Source Signal vs UMAP Canonical Target', 'FontSize', 14, 'FontWeight', 'bold');

for i = 1:n_plot_comps
    nexttile; hold on; grid on;
    gl_n = (gl_src(i,:) - mean(gl_src(i,:))) / std(gl_src(i,:));
    can_n = (emb_can_pr(i,:) - mean(emb_can_pr(i,:))) / std(emb_can_pr(i,:));
    
    plot(gl_n, 'LineWidth', 1.2, 'Color', [0.2 0.4 0.8]); 
    plot(can_n, 'LineWidth', 1.2, 'Color', [0.9 0.3 0.3]);
    
    for k = 1:length(ticks), xline(ticks(k), '--', 'Color', [0.8 0.8 0.8]); end
    title(sprintf('Global %d | Corr = %.2f', i, corr(gl_n', can_n')));
    xticks(ticks(1:end-1)); xticklabels(arrayfun(@(val) num2str(val), 1:nEpochs, 'UniformOutput', false));
    xlim([0 ticks(end)]);
    if i == 1, legend({'Unconstrained EEG Covariance', 'UMAP Canonical'}, 'Location', 'best'); end
end

%% =====================================================================
% РИСУНОК 4: КАНОНИЧЕСКОЕ ПРОСТРАНСТВО 3D (С ЦЕНТРОИДАМИ)
% =====================================================================
figure('Name', 'Canonical Space', 'Color', 'w', 'Position', [200, 200, 800, 600]);
plot3(cx_can(~isnan(cx_can)), cy_can(~isnan(cy_can)), cz_can(~isnan(cz_can)), 'k', 'LineWidth', 1.5); hold on; grid on;
for i = 1:nEpochs
    idx = (valid_cond_idx == i);
    if any(idx)
        scatter3(x(idx), y(idx), z(idx), 10, repmat(cmap(i,:), sum(idx), 1), 'filled', 'MarkerFaceAlpha', 0.3);
        scatter3(cx_can(i), cy_can(i), cz_can(i), 150, cmap(i,:), 'filled', 'MarkerEdgeColor', 'k');
        text(cx_can(i), cy_can(i), cz_can(i), num2str(i), 'FontSize', 12, 'FontWeight', 'bold', ...
            'BackgroundColor', [0.95 0.95 0.95], 'Margin', 1, 'HorizontalAlignment', 'center');
    end
end
xlabel('Canonical Axis 1'); ylabel('Canonical Axis 2'); zlabel('Canonical Axis 3');
title(['eSPoC Canonical Space | ', strrep(base_name, '_', '\_')]);
view(-45, 30);

%% =====================================================================
% РИСУНОК 5: СПЕКТР СОБСТВЕННЫХ ЗНАЧЕНИЙ (НОВЫЙ)
% =====================================================================
figure('Name', 'Eigenvalue Spectrum', 'Color', 'w', 'Position', [250, 250, 600, 400]);
plot(eigenvalues_true', '-o', 'LineWidth', 2, 'MarkerSize', 6); hold on; grid on;
xlabel('Local Component Index'); ylabel('\lambda value');
title('Eigenvalue Spectrum of the matrix W (Rank of the reaction)');
legend(leg_labels, 'Location', 'best');
xlim([0.5, size(eigenvalues_true, 2)+0.5]);

%% =====================================================================
% РИСУНОК 6: МАТРИЦА РАССТОЯНИЙ МЕЖДУ УСЛОВИЯМИ (НОВЫЙ)
% =====================================================================
figure('Name', 'Condition Distance Matrix', 'Color', 'w', 'Position', [300, 300, 600, 500]);
% Считаем попарные расстояния между центроидами в 3D пространстве eSPoC
valid_c = ~isnan(cx_can);
D = pdist([cx_can(valid_c); cy_can(valid_c); cz_can(valid_c)]');
D_sq = squareform(D);

imagesc(D_sq); colormap(flipud(hot)); colorbar;
title('Euclidean Distances Between Conditions (Canonical Space)');
valid_labels = target_conditions(valid_c);
xticks(1:sum(valid_c)); yticks(1:sum(valid_c));
xticklabels(valid_labels); yticklabels(valid_labels);
xtickangle(45); axis square;

%% =====================================================================
% РИСУНОК 7: ДЕТАЛЬНЫЙ АНАЛИЗ ОДНОЙ КОМПОНЕНТЫ (2x2)
% =====================================================================
gl_src_idx  = 1;
lcl_src_idx = 1;

if size(W_true, 1) >= gl_src_idx && size(W_true, 3) >= lcl_src_idx
    % Вытягиваем нужную матрицу с учетом возможной 1D размерности
    if ndims(W_true) == 3
        wx_pca = squeeze(W_true(gl_src_idx, :, lcl_src_idx))';
        ax_pca = squeeze(A_true(gl_src_idx, :, lcl_src_idx))';
    else
        wx_pca = W_true(:, lcl_src_idx);
        ax_pca = A_true(:, lcl_src_idx);
    end
    
    wx = U * wx_pca; 
    ax = U * ax_pca;
    [~, idx_max] = max(abs(wx)); wx = wx .* sign(wx(idx_max));
    [~, idx_max] = max(abs(ax)); ax = ax .* sign(ax(idx_max));

    figure('Name', 'Single Component Layout', 'Color', 'w', 'Position', [350, 350, 1000, 700]);
    t = tiledlayout(2,2, 'TileSpacing', 'compact', 'Padding', 'compact');
    sgtitle(sprintf('Global %d | Local %d | Corr: %.3f', gl_src_idx, lcl_src_idx, corrs_true(gl_src_idx, lcl_src_idx)), ...
            'FontSize', 14, 'FontWeight', 'bold');

    ax1 = nexttile(t,1); title(ax1, 'Spatial Filter (W)');
    topo.avg = wx; cfg_topo.figure = ax1; ft_topoplotER(cfg_topo, topo); 
    ax2 = nexttile(t,2); title(ax2, 'Spatial Pattern (A)');
    topo.avg = ax; cfg_topo.figure = ax2; ft_topoplotER(cfg_topo, topo); 

    ax3 = nexttile(t,3,[1,2]); hold on; grid on;
    title(ax3, 'Source Power Envelope vs Canonical Target');

    S_plot = zeros(1, size(Covs_valid, 3));
    for i = 1:size(Covs_valid, 3)
        S_plot(i) = wx_pca' * Covs_valid(:, :, i) * wx_pca;
    end
    S_plot = (S_plot - mean(S_plot)) / std(S_plot);

    zz = emb_can_pr(gl_src_idx, :);
    zz = (zz - mean(zz)) / std(zz) * sign(corrs_true(gl_src_idx, lcl_src_idx)); 

    plot(S_plot, 'LineWidth', 1.5, 'Color', [0.2 0.6 0.8]); 
    plot(zz, 'LineWidth', 1.5, 'Color', [0.9 0.3 0.3]);
    for k = 1:length(ticks), xline(ax3, ticks(k), '--', 'Color', [0.8 0.8 0.8]); end
    xticks(ax3, ticks(1:end-1)); xlim([0 ticks(end)]);
    xticklabels(ax3, target_conditions(1:nEpochs)); xtickangle(45);
    ylabel('Normalized Amplitude');
    legend('Source Envelope (eSPoC)', 'Canonical Target (UMAP)', 'Location', 'best');
end

%% =====================================================================
% РИСУНОК 8: МУЛЬТИ-КОМПОНЕНТНАЯ ПАНЕЛЬ (4 компоненты в ряд)
% =====================================================================
comp_indices = [1, 2, 3, 4];  
gl_src_idx_mult = 1; 

figure('Name', 'Multiple Components', 'Color', 'w', 'Position', [400, 400, 1400, 800]);
t2 = tiledlayout(4, 5, 'TileSpacing', 'compact', 'Padding', 'compact');
all_right_axes = [];       

for i = 1:length(comp_indices)
    lcl_idx = comp_indices(i);   
    if lcl_idx > size(corrs_true, 2), continue; end
    
    if ndims(W_true) == 3
        wx_pca = squeeze(W_true(gl_src_idx_mult, :, lcl_idx))';
        ax_pca = squeeze(A_true(gl_src_idx_mult, :, lcl_idx))';
    else
        wx_pca = W_true(:, lcl_idx);
        ax_pca = A_true(:, lcl_idx);
    end
    
    wx = U * wx_pca; ax = U * ax_pca;
    [~, m_idx] = max(abs(wx)); wx = wx .* sign(wx(m_idx));
    [~, m_idx] = max(abs(ax)); ax = ax .* sign(ax(m_idx));
    row = (i-1)*5;
    
    ax_wx = nexttile(t2, row + 1); topo.avg = wx; cfg_topo.figure = ax_wx; ft_topoplotER(cfg_topo, topo);
    if i == 1, title('Filter'); end
    ax_ax = nexttile(t2, row + 2); topo.avg = ax; cfg_topo.figure = ax_ax; ft_topoplotER(cfg_topo, topo);
    if i == 1, title('Pattern'); end
    
    ax_plot = nexttile(t2, row + 3, [1 3]); all_right_axes = [all_right_axes ax_plot];
    S_plot = zeros(1, size(Covs_valid, 3));
    for j = 1:size(Covs_valid, 3), S_plot(j) = wx_pca' * Covs_valid(:, :, j) * wx_pca; end
    S_plot = (S_plot - mean(S_plot)) / std(S_plot);
    
    zz = emb_can_pr(gl_src_idx_mult, :);
    zz = (zz - mean(zz)) / std(zz) * sign(corrs_true(gl_src_idx_mult, lcl_idx)); 
    
    plot(ax_plot, S_plot, 'LineWidth', 1.2, 'Color', [0.2 0.6 0.8]); hold on;
    plot(ax_plot, zz, 'LineWidth', 1, 'Color', [0.9 0.3 0.3]); grid on;
    
    c_val = corrs_true(gl_src_idx_mult, lcl_idx);
    t_color = 'k'; if c_val > max_val || c_val < min_val, t_color = [0 0.5 0]; end 
    title(ax_plot, sprintf('Comp %d | Corr = %.2f', lcl_idx, c_val), 'Color', t_color);
    xticks(ax_plot, ticks(1:end-1)); xticklabels(ax_plot, []); xlim(ax_plot, [0, ticks(end)]);
    for k = 1:length(ticks), xline(ax_plot, ticks(k), '--', 'Color', [0.8 0.8 0.8]); end
end

if ~isempty(all_right_axes)
    xticklabels(all_right_axes(end), target_conditions(1:nEpochs)); 
    xtickangle(all_right_axes(end), 45); xlabel(all_right_axes(end), 'Experimental Conditions');
    legend(all_right_axes(1), {'Source Envelope', 'Canonical Target'}, 'Location', 'northeastoutside');
end
disp('Все 8 графиков успешно построены!');