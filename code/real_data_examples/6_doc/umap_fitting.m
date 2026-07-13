close all;
clear;
clc;

% =========================================================================
% 1. НАСТРОЙКА ПУТЕЙ И ПОДГРУЗКА FIELDTRIP
% =========================================================================
ft_path = 'C:\Users\ansbel\Documents\GitHub\site-packages\fieldtrip'; % ПУТЬ К FIELDTRIP
if ~exist('ft_defaults','file')
    addpath(ft_path);
end
ft_defaults;

%% ========================================================================
% 2. ЗАГРУЗКА ДАННЫХ ИЗ .mat 
% ========================================================================
% Можно раскомментировать uigetfile, если хотите выбирать файл вручную:
% [filename, pathname] = uigetfile('*.mat', 'Выберите файл eeg_riemann_data.mat');
% load(fullfile(pathname, filename));

mat_file = 'C:/Users/ansbel/Documents/GitHub/DoC/Sorted/Андреев парадигмы/Андреев_eeg_riemann_data.mat';
load(mat_file);   

Fs = double(sfreq);
labels_cell = cellstr(labels);
trials = double(trials);

% --- Подготовка данных ---
X_epo = double(X_ssd);            % SSD данные для mSPoC
X_raw_epo = double(X_raw);        % сырые данные для TFR
N_epoch_trial = size(X_epo, 3);
Covs = double(covmats_ssd);       % Ковариации в SSD

% --- Целевые вложения ---
Rmean2d = double(Rmean2d);
Rmean3d = double(Rmean3d);
Rmean5d = double(Rmean5d);
Z_target = Rmean5d; % mSPoC цель

%% ========================================================================
% 3. ИНТЕЛЛЕКТУАЛЬНЫЙ ПАРСИНГ УСЛОВИЙ
% ========================================================================
n_epochs = length(labels_cell);
base_conds = cell(n_epochs, 1);
state_flags = NaN(n_epochs, 1); % NaN-Pause, 1-Active, 0-Rest, 2-Instruction

for i = 1:n_epochs
    lbl = labels_cell{i};
    % Разделяем по подчеркиванию ИЛИ пробелу (для 'Pause 1')
    parts = strsplit(lbl, {'_', ' '}); 
    base_conds{i} = parts{1}; % Получаем 'Pain', 'Letter', 'Pause' и т.д.
    
    % Ищем состояние
    if contains(lbl, 'active', 'IgnoreCase', true)
        state_flags(i) = 1;
    elseif contains(lbl, 'rest', 'IgnoreCase', true) || contains(lbl, 'passive', 'IgnoreCase', true)
        state_flags(i) = 0;
    elseif contains(lbl, 'instruct', 'IgnoreCase', true)
        state_flags(i) = 2;
    end
end

% Находим границы смены состояний (для цветных патчей)
state_bounds = [1];
for i = 2:n_epochs
    % Если состояние изменилось (и это не переход между двумя разными паузами)
    if state_flags(i) ~= state_flags(i-1) && ~(isnan(state_flags(i)) && isnan(state_flags(i-1)))
        state_bounds(end+1) = i;
    end
end
state_bounds(end+1) = n_epochs + 1;

% Находим границы базовых условий (для разделителей оси X)
cond_bounds = [1];
cond_names = {base_conds{1}};
for i = 2:n_epochs
    if ~strcmp(base_conds{i}, base_conds{i-1})
        cond_bounds(end+1) = i;
        cond_names{end+1} = base_conds{i};
    end
end
cond_bounds(end+1) = n_epochs + 1;

% Вычисляем центры блоков (для подписей X)
cond_centers = zeros(1, length(cond_names));
for i = 1:length(cond_names)
    cond_centers(i) = (cond_bounds(i) + cond_bounds(i+1) - 1) / 2;
end

% Для 2D карты оставляем оригинальные 45 кластеров
[conditions_exact, ~, condition_idx] = unique(labels_cell, 'stable');
num_exact_clusters = length(conditions_exact);

%% ========================================================================
% 4. ПОДГОТОВКА LAYOUT ДЛЯ ТОПОМАП (защита от NaN каналов)
% ========================================================================
ch_names_cell = cellstr(ch_names);
valid_channels = ~isnan(ch_pos(:,1)); 

topo = []; topo.dimord = 'chan_time'; topo.label = ch_names_cell; topo.time = 0;

elec = [];
elec.label = ch_names_cell(valid_channels);
elec.elecpos = ch_pos(valid_channels, :);
elec.chanpos = ch_pos(valid_channels, :);

laycfg = []; laycfg.elec = elec;
lay = ft_prepare_layout(laycfg);     
topo_cfg = []; topo_cfg.marker = 'labels'; topo_cfg.layout = lay; 
topo_cfg.comment = 'no'; topo_cfg.style = 'fill'; 
topo_cfg.markersymbol = '.'; topo_cfg.colorbar = 'yes'; 

%% ========================================================================
% 5. mSPoC (в SSD-пространстве)
% ========================================================================
[W, Vz, ~, A, Az, out] = my_mspoc(X_epo, Z_target');

corrs = zeros(size(W,2), 1);
for k = 1:size(W,2)
    Zpr = Z_target * Vz(:,k);
    Env = zeros(1, N_epoch_trial);
    for ep_idx = 1:N_epoch_trial
        Env(ep_idx) = log(W(:, k)' * Covs(:,:,ep_idx) * W(:, k));
    end
    corrs(k) = corr(Env', Zpr);
end

[corrs_sorted_abs, sort_idx] = sort(abs(corrs), 'descend');
corrs_sorted = corrs(sort_idx);
W_sorted  = W(:, sort_idx);
A_sorted  = A(:, sort_idx);
Vz_sorted = Vz(:, sort_idx) .* sign(corrs_sorted)';
Az_sorted = Az(:, sort_idx) .* sign(corrs_sorted)';

disp('Correlations (sorted):');
disp(corrs_sorted');

%% ========================================================================
% 6. ГЛОБАЛЬНЫЙ 3D UMAP (Умная группировка)
% ========================================================================
unique_bases = unique(base_conds, 'stable');
num_bases = length(unique_bases);
cmap_bases = turbo(num_bases); % Палитра для базовых условий

figure('Name', 'Global 3D UMAP (Grouped)', 'Color', 'w', 'Position', [100, 100, 1000, 700]);
hold on; grid on;

% Траектория времени (бледная линия)
plot3(Rmean3d(:,1), Rmean3d(:,2), Rmean3d(:,3), 'Color', [0.7 0.7 0.7 0.5]);

% Рисуем точки по состояниям
for b = 1:num_bases
    idx_b = strcmp(base_conds, unique_bases{b});
    col = cmap_bases(b,:);
    
    idx_act = idx_b & (state_flags == 1);
    idx_pas = idx_b & (state_flags == 0);
    idx_ins = idx_b & (state_flags == 2);
    idx_pau = idx_b & isnan(state_flags);
    
    if any(idx_act), scatter3(Rmean3d(idx_act,1), Rmean3d(idx_act,2), Rmean3d(idx_act,3), 40, col, '^', 'filled', 'MarkerEdgeColor', 'k'); end
    if any(idx_pas), scatter3(Rmean3d(idx_pas,1), Rmean3d(idx_pas,2), Rmean3d(idx_pas,3), 40, col, 'o', 'filled', 'MarkerEdgeColor', 'k'); end
    if any(idx_ins), scatter3(Rmean3d(idx_ins,1), Rmean3d(idx_ins,2), Rmean3d(idx_ins,3), 30, col, 's', 'filled'); end
    if any(idx_pau), scatter3(Rmean3d(idx_pau,1), Rmean3d(idx_pau,2), Rmean3d(idx_pau,3), 15, col, 'filled'); end
end

xlabel('UMAP 1'); ylabel('UMAP 2'); zlabel('UMAP 3');
title('3D UMAP Embedding (Color=Task, Shape=State)', 'FontSize', 14);
view(-45, 30);

% Искусственная легенда
h_leg = []; leg_names = {};
for b = 1:num_bases
    h_leg(end+1) = plot3(NaN, NaN, NaN, 'o', 'MarkerFaceColor', cmap_bases(b,:), 'MarkerEdgeColor', 'k', 'MarkerSize', 8);
    leg_names{end+1} = unique_bases{b};
end
h_leg(end+1) = plot3(NaN, NaN, NaN, '^k', 'MarkerFaceColor', 'k', 'MarkerSize', 8); leg_names{end+1} = 'Active (\Delta)';
h_leg(end+1) = plot3(NaN, NaN, NaN, 'ok', 'MarkerFaceColor', 'k', 'MarkerSize', 8); leg_names{end+1} = 'Rest (O)';
h_leg(end+1) = plot3(NaN, NaN, NaN, 'sk', 'MarkerFaceColor', 'k', 'MarkerSize', 8); leg_names{end+1} = 'Instruction (\square)';
legend(h_leg, leg_names, 'Location', 'bestoutside', 'Interpreter', 'none');
hold off;

%% ========================================================================
% 7. КОМПЛЕКСНАЯ ВИЗУАЛИЗАЦИЯ ТОП-КОМПОНЕНТ
% ========================================================================
n_to_plot = min(10, length(sort_idx));
scale_f = max(sqrt(Rmean2d(:,1).^2 + Rmean2d(:,2).^2));

ccx = zeros(1, num_exact_clusters); ccy = zeros(1, num_exact_clusters);
for c = 1:num_exact_clusters
    idx_c = (condition_idx == c);
    ccx(c) = median(Rmean2d(idx_c,1));
    ccy(c) = median(Rmean2d(idx_c,2));
end

for i = 1:n_to_plot
    r_val = corrs_sorted(i);
    
    % Проекция фильтра и паттерна в сенсоры
    wx = W_ssd * W_sorted(:, i);      
    ax = A_ssd * A_sorted(:, i);       
    
    [~, max_idx] = max(abs(wx)); wx = wx .* sign(wx(max_idx));
    [~, max_idx] = max(abs(ax)); ax = ax .* sign(ax(max_idx));
    
    % Мощность источника
    S_pow = zeros(1, N_epoch_trial);
    for j = 1:N_epoch_trial
        w_curr = W_sorted(:, i);
        S_pow(j) = log(w_curr' * Covs(:,:,j) * w_curr);
    end
    S_raw = S_pow;
    S_pow = (S_pow - mean(S_pow)) / std(S_pow);
    
    % Цель UMAP
    zz = Z_target * Vz_sorted(:, i);
    zz = (zz - mean(zz)) / std(zz);
    
    vz_2d_eff = pinv(Rmean2d) * zz;
    az_2d_eff = (Rmean2d' * zz) / (zz' * zz);
    vz_2d = vz_2d_eff / norm(vz_2d_eff) * scale_f * 0.4;
    az_2d = az_2d_eff / norm(az_2d_eff) * scale_f * 0.4;
    
    % -----------------------------------------------------------------
    % ФИГУРА И СЕТКА
    % -----------------------------------------------------------------
    fig_name = sprintf('Component Rank %d | Corr = %.3f', i, r_val);
    figure('Color', 'w', 'Position', [50+i*30, 50+i*30, 1600, 700], 'Name', fig_name);
    t = tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    % --- [1] 3D UMAP (простая версия для панели) ---
    ax_umap = nexttile(t, 1); hold(ax_umap, 'on'); grid(ax_umap, 'on');
    scatter3(ax_umap, Rmean3d(:,1), Rmean3d(:,2), Rmean3d(:,3), 15, S_pow, 'filled', 'MarkerFaceAlpha', 0.8);
    colormap(ax_umap, 'jet');
    view(ax_umap, -45, 30);
    xlabel(ax_umap, 'UMAP 1'); ylabel(ax_umap, 'UMAP 2'); zlabel(ax_umap, 'UMAP 3');
    title(ax_umap, '3D UMAP (Colored by Power)', 'FontSize', 12);
    
    % --- [2] 2D ИЗОЛИНИИ ---
    ax_iso = nexttile(t, 2); hold(ax_iso, 'on');
    MWS_raw = movmean(S_raw', 5); 
    
    x_min = min(Rmean2d(:,1)); x_max = max(Rmean2d(:,1)); dx = x_max - x_min;
    y_min = min(Rmean2d(:,2)); y_max = max(Rmean2d(:,2)); dy = y_max - y_min;
    x_min_pad = x_min - 0.1*dx; x_max_pad = x_max + 0.1*dx;
    y_min_pad = y_min - 0.1*dy; y_max_pad = y_max + 0.1*dy;
    
    x_edge = [linspace(x_min_pad, x_max_pad, 10)'; linspace(x_min_pad, x_max_pad, 10)'; repmat(x_min_pad, 10, 1); repmat(x_max_pad, 10, 1)];
    y_edge = [repmat(y_min_pad, 10, 1); repmat(y_max_pad, 10, 1); linspace(y_min_pad, y_max_pad, 10)'; linspace(y_min_pad, y_max_pad, 10)'];
    X_all = [Rmean2d(:,1); x_edge]; Y_all = [Rmean2d(:,2); y_edge]; MWS_all = [MWS_raw; ones(size(x_edge)) * min(MWS_raw)];
    
    F = scatteredInterpolant(X_all, Y_all, MWS_all, 'natural', 'linear');
    [Xg, Yg] = meshgrid(linspace(x_min_pad, x_max_pad, 100), linspace(y_min_pad, y_max_pad, 100));
    ProjGrid = F(Xg, Yg); ProjGrid(ProjGrid > max(MWS_raw)) = max(MWS_raw);
    
    pcolor(ax_iso, Xg, Yg, ProjGrid); shading(ax_iso, 'interp');
    contour(ax_iso, Xg, Yg, ProjGrid, 10, 'k', 'LineWidth', 0.05);
    
    % Подписи номеров кластеров (не всех 45, а только центров базовых задач)
    for c = 1:length(cond_centers)
        b_idx = round(cond_centers(c));
        text(ax_iso, Rmean2d(b_idx,1), Rmean2d(b_idx,2), num2str(c), 'FontWeight','bold', 'BackgroundColor', [1 1 1 0.7], 'Margin', 1);
    end
    
    quiver(ax_iso, 0, 0, vz_2d(1), vz_2d(2), 'Color', 'k', 'LineWidth', 2, 'MaxHeadSize', 0.5, 'AutoScale', 'off');
    quiver(ax_iso, 0, 0, az_2d(1), az_2d(2), 'Color', 'r', 'LineWidth', 3, 'MaxHeadSize', 0.5, 'AutoScale', 'off');
    xlabel(ax_iso, 'UMAP 1'); ylabel(ax_iso, 'UMAP 2'); title(ax_iso, 'Isolines', 'FontSize', 12); axis(ax_iso, 'tight');
    
    % --- [3] ФИЛЬТР ---
    ax_w = nexttile(t, 3); topo.avg = wx; topo_cfg.figure = ax_w;
    ft_topoplotER(topo_cfg, topo); title(ax_w, 'Spatial Filter', 'FontSize', 12);
    
    % --- [4] ПАТТЕРН ---
    ax_p = nexttile(t, 4); topo.avg = ax; topo_cfg.figure = ax_p;
    ft_topoplotER(topo_cfg, topo); title(ax_p, 'Spatial Pattern', 'FontSize', 12);
    
    % --- [5] ДИНАМИКА ---
    ax_dyn = nexttile(t, 5, [1 4]);
    hold(ax_dyn, 'on'); grid(ax_dyn, 'on');
    ylims = [min(S_pow)*1.1, max(S_pow)*1.1];
    
    % Отрисовка патчей Active/Rest/Instruction
    for b = 1:(length(state_bounds)-1)
        x_start = state_bounds(b) - 0.5;
        x_end   = state_bounds(b+1) - 0.5;
        st = state_flags(state_bounds(b));
        if st == 1
            patch(ax_dyn, [x_start x_end x_end x_start], [ylims(1) ylims(1) ylims(2) ylims(2)], 'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
        elseif st == 0
            patch(ax_dyn, [x_start x_end x_end x_start], [ylims(1) ylims(1) ylims(2) ylims(2)], 'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
        elseif st == 2
            patch(ax_dyn, [x_start x_end x_end x_start], [ylims(1) ylims(1) ylims(2) ylims(2)], 'y', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
        end
    end
    
    h_pow = plot(ax_dyn, S_pow, 'LineWidth', 1.5, 'Color', [0 0.447 0.741]);
    h_tgt = plot(ax_dyn, zz, 'LineWidth', 1.5, 'Color', [0.850 0.325 0.098]);
    
    % Вертикальные разделители условий
    for k = 2:length(cond_bounds)-1
        xline(ax_dyn, cond_bounds(k) - 0.5, '-k', 'LineWidth', 1.5);
    end
    
    xlim(ax_dyn, [0.5 N_epoch_trial+0.5]); ylim(ax_dyn, ylims);
    xticks(ax_dyn, cond_centers);
    xticklabels(ax_dyn, cond_names);
    xtickangle(ax_dyn, 30);
    set(ax_dyn, 'TickLabelInterpreter', 'none');
    
    h_act = plot(ax_dyn, NaN, NaN, 'r', 'LineWidth', 8); 
    h_pas = plot(ax_dyn, NaN, NaN, 'b', 'LineWidth', 8);
    h_ins = plot(ax_dyn, NaN, NaN, 'y', 'LineWidth', 8);
    legend(ax_dyn, [h_pow, h_tgt, h_act, h_pas, h_ins], ...
        {'Source Power (Z)', 'Projected UMAP Target', 'Active', 'Rest/Passive', 'Instruction'}, ...
        'Location', 'bestoutside');
    
    title(ax_dyn, sprintf('Component Dynamics (Correlation = %.3f)', r_val), 'FontSize', 14);
    ylabel(ax_dyn, 'Amplitude (Z-score)');
    
    % --- 8. TIME-FREQUENCY КАРТА (отдельное окно) ---
    f_target = 1:0.5:40;
    window_length = floor(Fs * Wsize);
    TF_map = zeros(length(f_target), N_epoch_trial);
    
    for ep = 1:N_epoch_trial
        s_ep_unfilt = wx' * squeeze(X_raw_epo(:,:,ep))';
        [pxx, ~] = pwelch(s_ep_unfilt, hamming(window_length), [], f_target, Fs);
        TF_map(:, ep) = pxx;
    end
    
    % Бейзлайн: ищем эпохи, где в метке есть 'rest' или 'pause'
    baseline_idx = contains(labels_cell, 'rest', 'IgnoreCase', true) | contains(labels_cell, 'pause', 'IgnoreCase', true);
    if ~any(baseline_idx), baseline_idx = 1:N_epoch_trial; end
    
    baseline_power = mean(TF_map(:, baseline_idx), 2);
    TF_map_db = 10 * log10(TF_map ./ repmat(baseline_power, 1, size(TF_map, 2)));    
    
    fig_tf_name = sprintf('TF Dynamics | Component Rank %d', i);
    figure('Color', 'w', 'Position', [100+i*30, 150+i*30, 1200, 300], 'Name', fig_tf_name);
    
    clim_max = max(abs(prctile(TF_map_db(:), [5 95])));
    imagesc(1:N_epoch_trial, f_target, TF_map_db);
    axis('xy'); colormap('jet'); caxis([-clim_max clim_max]);
    cbar_tf = colorbar; cbar_tf.Label.String = 'Power Change (dB vs Rest)';
    title(sprintf('Broadband Time-Frequency Dynamics (C_rank=%d, r=%.2f)', i, r_val), 'FontSize', 12);
    ylabel('Frequency (Hz)', 'FontWeight', 'bold');
    
    hold on;
    for k = 2:length(cond_bounds)-1
        xline(cond_bounds(k) - 0.5, '--w', 'LineWidth', 1.5);
    end
    hold off;
    
    xlim([0.5 N_epoch_trial+0.5]);
    xticks(cond_centers);
    xticklabels(cond_names);
    xtickangle(30);
    set(gca, 'TickLabelInterpreter', 'none');
end