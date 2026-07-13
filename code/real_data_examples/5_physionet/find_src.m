close all
clear
clc
ft_path = 'C:\Users\ansbel\Documents\GitHub\site-packages\fieldtrip';
if ~exist('ft_defaults','file')
    addpath(ft_path);
end
ft_defaults;

%% ========================================================================
% 1. ЗАГРУЗКА ДАННЫХ (результаты из Python)
% ========================================================================
load('results_matlab_mu_all_6classes.mat');

Fs = sfreq;
n_channels_original = size(W_ssd, 1);
n_comp_ssd = size(X_ssd, 2);

%% ========================================================================
% 2. СТРУКТУРА ЭЛЕКТРОДОВ ДЛЯ FIELDTRIP
% ========================================================================
elec = [];
elec.label = cellstr(ch_names);
elec.chanpos = ch_pos;
elec.elecpos = ch_pos;
elec.unit = 'm';

cfg_lay = [];
cfg_lay.elec = elec;
lay = ft_prepare_layout(cfg_lay);

cfg_topo = [];
cfg_topo.marker       = 'labels';
cfg_topo.layout       = lay;
cfg_topo.comment      = 'no';
cfg_topo.style        = 'fill';
cfg_topo.markersymbol = 'o';
cfg_topo.colorbar     = 'EastOutside';
cfg_topo.elec         = elec;

topo = [];
topo.dimord = 'chan_time';
topo.label  = elec.label;
topo.time   = 0;
topo.elec   = elec;

%% ========================================================================
% 3. ПОДГОТОВКА ЭПОХ
% ========================================================================
X_epo = X_ssd;          % (time, n_comp, windows)
X_raw_epo = X_raw;      % (time, channels, windows)
N_epoch_trial = size(X_epo, 3);

%% ========================================================================
% 4. ПРЕОБРАЗОВАНИЕ МЕТОК
% ========================================================================
if ischar(labels) || isstring(labels)
    labels = cellstr(labels);
end
if iscell(labels)
    labels = strtrim(labels);
end
conditions = unique(labels, 'stable');
num_clusters = length(conditions);
disp('Условия:'); disp(conditions);

%% ========================================================================
% 5. КОВАРИАЦИИ и UMAP-цели
% ========================================================================
Covs = covmats_ssd;
Rmean2d = double(Rmean2d);
Rmean3d = double(Rmean3d);
Rmean10d = double(Rmean10d);
Z_target = Rmean10d;

%% ========================================================================
% 6. mSPoC
% ========================================================================
[W, Vz, ~, A, Az, out] = my_mspoc(X_epo, Z_target');

corrs = zeros(size(W,2), 1);
for k = 1:size(W,2)
    Zpr = Z_target * Vz(:,k);
    Env = zeros(1, N_epoch_trial);
    for ep_idx = 1:N_epoch_trial
        Env(ep_idx) = log(W(:, k)' * W_ssd' * covmats_band(:,:,ep_idx) * W_ssd * W(:, k));
    end
    corrs(k) = corr(Env', Zpr);
end

[~, sort_idx] = sort(abs(corrs), 'descend');
corrs_sorted = corrs(sort_idx);
W_sorted  = W(:, sort_idx);
A_sorted  = A(:, sort_idx);
Vz_sorted = Vz(:, sort_idx);
Az_sorted = Az(:, sort_idx);

disp('Correlations (sorted):'); disp(corrs_sorted');

%% ========================================================================
% 7. ПОДГОТОВКА ГРАНИЦ БЛОКОВ И МЕТОК
% ========================================================================
ccx = zeros(1, num_clusters);
ccy = zeros(1, num_clusters);
for c = 1:num_clusters
    idx = strcmp(labels, conditions{c});
    ccx(c) = median(Rmean2d(idx,1));
    ccy(c) = median(Rmean2d(idx,2));
end
scale_f = max(sqrt(Rmean2d(:,1).^2 + Rmean2d(:,2).^2));

% Границы блоков – гарантируем строку
change_idx = find(diff(trials) ~= 0) + 1;
change_idx = change_idx(:)';               % <-- строка
ticks = [0, change_idx, N_epoch_trial];    % теперь все элементы строки

block_starts = [1, change_idx];            % тоже строка
block_labels = labels(block_starts);       % cell array с меткой каждого блока

%% ========================================================================
% 8. ГЛОБАЛЬНЫЙ 3D UMAP
% ========================================================================
cmap = jet(num_clusters);
figure('Name', 'Global 3D UMAP (SSD-based)', 'Color', 'w', 'Position', [100, 100, 900, 700]);
hold on; grid on;
for c = 1:num_clusters
    idx = strcmp(labels, conditions{c});
    scatter3(Rmean3d(idx,1), Rmean3d(idx,2), Rmean3d(idx,3), ...
        25, cmap(c,:), 'filled', 'MarkerFaceAlpha', 0.7);
end
xlabel('UMAP 1'); ylabel('UMAP 2'); zlabel('UMAP 3');
title('3D UMAP Embedding (SSD, centered)', 'FontSize', 14);
view(-45, 30);
legend(conditions, 'Location', 'bestoutside', 'Interpreter', 'none');
hold off;

%% ========================================================================
% 9. КОМПЛЕКСНАЯ ВИЗУАЛИЗАЦИЯ ТОП-КОМПОНЕНТ
% ========================================================================
n_to_plot = min(10, length(sort_idx));

for i = 1:n_to_plot
    r_val = corrs_sorted(i);

    % Сенсорные фильтр/паттерн
    wx = W_ssd * W_sorted(:, i);
    ax = A_ssd * W_sorted(:, i);

    [~, max_idx] = max(abs(wx)); wx = wx .* sign(wx(max_idx));
    [~, max_idx] = max(abs(ax)); ax = ax .* sign(ax(max_idx));

    % Мощность источника
    S_pow = zeros(1, N_epoch_trial);
    for j = 1:N_epoch_trial
        S_pow(j) = log(wx' * covmats_band(:,:,j) * wx);
    end
    S_raw = S_pow;
    S_pow = (S_pow - mean(S_pow)) / std(S_pow);

    % Проекция целевой переменной
    zz = Z_target * Vz_sorted(:, i);
    zz = (zz - mean(zz)) / std(zz) * sign(r_val);

    % 2D векторы для стрелок
    zz_c = zz;
    R2_c = Rmean2d;
    vz_2d_eff = pinv(R2_c) * zz_c;
    az_2d_eff = (R2_c' * zz_c) / (zz_c' * zz_c);
    vz_2d = vz_2d_eff / norm(vz_2d_eff) * scale_f * 0.4;
    az_2d = az_2d_eff / norm(az_2d_eff) * scale_f * 0.4;

    % ---------- ФИГУРА ----------
    fig_name = sprintf('Component %d (Rank %d) | Corr = %.3f', i, i, r_val);
    figure('Color', 'w', 'Position', [50+i*30, 50+i*30, 1600, 700], 'Name', fig_name);
    t = tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

    % [1] 3D UMAP
    ax_umap = nexttile(t, 1);
    hold(ax_umap, 'on'); grid(ax_umap, 'on');
    for c = 1:num_clusters
        idx = strcmp(labels, conditions{c});
        scatter3(ax_umap, Rmean3d(idx,1), Rmean3d(idx,2), Rmean3d(idx,3), ...
            15, cmap(c,:), 'filled', 'MarkerFaceAlpha', 0.4);
    end
    view(ax_umap, -45, 30);
    xlabel('UMAP 1'); ylabel('UMAP 2'); zlabel('UMAP 3');
    title('3D UMAP Space', 'FontSize', 14);

    % [2] 2D ИЗОЛИНИИ
    ax_iso = nexttile(t, 2);
    hold(ax_iso, 'on');
    MWS_raw = movmean(S_raw', 10);
    x_min = min(Rmean2d(:,1)); x_max = max(Rmean2d(:,1));
    y_min = min(Rmean2d(:,2)); y_max = max(Rmean2d(:,2));
    dx = x_max - x_min; dy = y_max - y_min;
    pad_factor = 0.1;
    x_min_pad = x_min - pad_factor*dx; x_max_pad = x_max + pad_factor*dx;
    y_min_pad = y_min - pad_factor*dy; y_max_pad = y_max + pad_factor*dy;

    n_edge = 10;
    x_edge = [linspace(x_min_pad, x_max_pad, n_edge)'; linspace(x_min_pad, x_max_pad, n_edge)'; repmat(x_min_pad, n_edge, 1); repmat(x_max_pad, n_edge, 1)];
    y_edge = [repmat(y_min_pad, n_edge, 1); repmat(y_max_pad, n_edge, 1); linspace(y_min_pad, y_max_pad, n_edge)'; linspace(y_min_pad, y_max_pad, n_edge)'];
    x_corn = [x_min_pad; x_min_pad; x_max_pad; x_max_pad];
    y_corn = [y_min_pad; y_max_pad; y_min_pad; y_max_pad];
    x_edge = [x_edge; x_corn]; y_edge = [y_edge; y_corn];

    MWS_edge = ones(size(x_edge)) * min(MWS_raw);
    X_all = [Rmean2d(:,1); x_edge]; Y_all = [Rmean2d(:,2); y_edge]; MWS_all = [MWS_raw; MWS_edge];

    F = scatteredInterpolant(X_all, Y_all, MWS_all, 'natural', 'linear');
    [Xg, Yg] = meshgrid(linspace(x_min_pad, x_max_pad, 100), linspace(y_min_pad, y_max_pad, 100));
    ProjGrid = F(Xg, Yg);
    ProjGrid(ProjGrid > max(MWS_raw)) = max(MWS_raw);

    pcolor(ax_iso, Xg, Yg, ProjGrid); shading(ax_iso, 'interp');
    cbar = colorbar(ax_iso); cbar.Label.String = 'Source power';
    contour(ax_iso, Xg, Yg, ProjGrid, 10, 'k', 'LineWidth', 0.05);

    for c = 1:num_clusters
        text(ax_iso, ccx(c), ccy(c), conditions{c}, 'FontWeight','bold', ...
            'HorizontalAlignment', 'center', 'BackgroundColor', [1 1 1 0.7], 'Margin', 1, 'Interpreter', 'none');
    end

    quiver(ax_iso, 0, 0, vz_2d(1), vz_2d(2), 'Color', 'k', 'LineWidth', 2, 'MaxHeadSize', 0.5, 'AutoScale', 'off');
    text(ax_iso, vz_2d(1)*1.1, vz_2d(2)*1.1, 'V_z', 'Color', 'k', 'FontWeight', 'bold');
    quiver(ax_iso, 0, 0, az_2d(1), az_2d(2), 'Color', 'r', 'LineWidth', 3, 'MaxHeadSize', 0.5, 'AutoScale', 'off');
    text(ax_iso, az_2d(1)*1.1, az_2d(2)*1.1, 'A_z', 'Color', 'r', 'FontWeight', 'bold');

    xlabel('UMAP 1'); ylabel('UMAP 2');
    title('Isolines of Source Activation', 'FontSize', 14);
    axis(ax_iso, 'tight');

    % [3] ФИЛЬТР (W)
    ax_w = nexttile(t, 3);
    topo.avg = wx;
    cfg_topo.figure = ax_w;
    ft_topoplotER(cfg_topo, topo);
    title(ax_w, 'Spatial Filter', 'FontSize', 14);

    % [4] ПАТТЕРН (A)
    ax_p = nexttile(t, 4);
    topo.avg = ax;
    cfg_topo.figure = ax_p;
    ft_topoplotER(cfg_topo, topo);
    title(ax_p, 'Spatial Pattern', 'FontSize', 14);

    % [5] ДИНАМИКА
    ax_dyn = nexttile(t, 5, [1 4]);
    hold(ax_dyn, 'on'); grid(ax_dyn, 'on');
    plot(ax_dyn, S_pow, 'LineWidth', 1.5, 'Color', [0 0.447 0.741]);
    plot(ax_dyn, zz * sign(r_val), 'LineWidth', 1.5, 'Color', [0.850 0.325 0.098]);

    % ---- Подписи по блокам (показываем все метки, даже если повторяются) ----
    xticks(ax_dyn, ticks(1:end-1));               % позиции начал блоков
    if length(block_labels) == length(ticks)-1
        xticklabels(ax_dyn, block_labels);        % метки блоков
    else
        xticklabels(ax_dyn, ...
            arrayfun(@num2str, 1:(length(ticks)-1), 'UniformOutput', false));
    end
    xtickangle(ax_dyn, 45);
    xlim(ax_dyn, [0 ticks(end)]);
    for k = 2:length(ticks)-1
        xline(ax_dyn, ticks(k), '--', 'Color', [0.3 0.3 0.3]);
    end
    legend(ax_dyn, {'Source Power Envelope', 'UMAP Canonical Target'}, 'Location', 'best');
    title(ax_dyn, sprintf('Component Dynamics (Correlation = %.3f)', r_val), 'FontSize', 14);
    ylabel(ax_dyn, 'Amplitude (Z-score)');

    % ============ 10. TIME-FREQUENCY КАРТА ============
    f_target = 1:0.5:30;
    window_length = floor(Fs * 1);
    TF_map = zeros(length(f_target), N_epoch_trial);

    for ep = 1:N_epoch_trial
        s_ep = wx' * squeeze(X_raw_epo(:,:,ep))';
        [pxx, ~] = pwelch(s_ep, hamming(window_length), [], f_target, Fs);
        TF_map(:, ep) = pxx;
    end

    baseline_idx = contains(labels, 'Rest');
    if sum(baseline_idx) == 0
        epochs_per_block = ticks(2) - ticks(1);
        baseline_idx = 1 : 2*epochs_per_block;
    end
    baseline_power = mean(TF_map(:, baseline_idx), 2);
    TF_map_db = 10 * log10(TF_map ./ repmat(baseline_power, 1, size(TF_map, 2)));

    fig_tf = figure('Color', 'w', 'Position', [100+i*30, 150+i*30, 1200, 300], ...
        'Name', sprintf('TF Dynamics | Component %d', i));
    clim_max = max(abs(prctile(TF_map_db(:), [5 95])));
    imagesc(1:N_epoch_trial, f_target, TF_map_db);
    axis('xy');
    colormap('jet');
    caxis([-clim_max clim_max]);
    cbar_tf = colorbar;
    cbar_tf.Label.String = 'Power Change (dB vs Rest)';
    title(sprintf('Broadband Time-Frequency Dynamics (C%d, r=%.2f)', i, r_val), 'FontSize', 12);
    ylabel('Frequency (Hz)', 'FontWeight', 'bold');

    hold on;
    for k = 2:length(ticks)-1
        xline(ticks(k), '--', 'Color', [1 1 1 0.8], 'LineWidth', 1.5);
    end
    hold off;

    xticks(ticks(1:end-1));
    if length(block_labels) == length(ticks)-1
        xticklabels(block_labels);
    else
        xticklabels(arrayfun(@num2str, 1:(length(ticks)-1), 'UniformOutput', false));
    end
    xtickangle(45);
    xlim([0 ticks(end)]);
end

%%