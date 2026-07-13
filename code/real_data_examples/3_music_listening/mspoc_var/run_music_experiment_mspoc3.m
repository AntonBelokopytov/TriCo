close all
clear
clc
ft_path = 'C:\Users\ansbel\Documents\GitHub\site-packages\fieldtrip';
if ~exist('ft_defaults','file')
    addpath(ft_path);
end
ft_defaults;

%%
sub_path = 'Tumyalis_music_epochs.fif';
cfg = [];
cfg.dataset = sub_path;
Xinf = ft_preprocessing(cfg);
Fs = Xinf.fsample;
topo = [];
topo.dimord = 'chan_time';
topo.label  = Xinf.elec.label;  
topo.time   = 0;
topo.elec   = Xinf.elec;
laycfg = [];
laycfg.elec = Xinf.elec;
lay = ft_prepare_layout(laycfg);     
cfg.marker       = 'labels';
cfg.layout       = lay;
cfg.comment      = 'no';
cfg.style        = 'fill';
cfg.markersymbol = 'o';
cfg.colorbar     = 'no'; 

%% ========================================================================
% 2. ЗАГРУЗКА ДАННЫХ ИЗ .mat (результаты Python с SSD)
% ========================================================================
load('results_for_matlab.mat');   % X_ssd, X_raw, Rmean2d, Rmean3d, Rmean10d,
                                  % labels, trials, sfreq, ch_names, ch_pos,
                                  % Cmean_ssd, covmats_ssd, W_ssd, A_ssd

Fs = sfreq;
n_channels_original = size(W_ssd, 1);   % исходное число каналов (38)
n_comp_ssd = size(X_ssd, 2);            % число SSD-компонент

%% ========================================================================
% 3. ПОДГОТОВКА ЭПОХ (уже нарезаны и спроецированы в SSD)
% ========================================================================
X_epo = X_ssd;          % (time, n_comp, windows)
X_raw_epo = X_raw;      % (time, channels, windows) – сырые для TF
N_epoch_trial = size(X_epo, 3);

%% ========================================================================
% 4. СПИСОК УСЛОВИЙ
% ========================================================================
conditions = unique(labels, 'stable');
num_clusters = length(conditions);

conditions

%% ========================================================================
% 5. КОВАРИАЦИОННЫЕ МАТРИЦЫ (загружены из Python)
% ========================================================================
Covs = covmats_ssd;   % (n_comp, n_comp, windows)

%% ========================================================================
% 6. ЦЕЛЕВАЯ ПЕРЕМЕННАЯ (UMAP)
% ========================================================================
Rmean2d = double(Rmean2d);
Rmean3d = double(Rmean3d);
Rmean10d = double(Rmean10d);
Z_target = Rmean10d;

%% ========================================================================
% 7. mSPoC (в SSD-пространстве)
% ========================================================================
% [W, Vz, ~, A, Az, out] = my_mspoc(X_epo, Z_target', 'Cxx', Cmean_ssd);
[W, Vz, ~, A, Az, out] = my_mspoc(X_epo, Z_target');

% Вычисление корреляций
corrs = zeros(size(W,2), 1);
for k = 1:size(W,2)
    Zpr = Z_target * Vz(:,k);
    Env = zeros(1, N_epoch_trial);
    for ep_idx = 1:N_epoch_trial
        Env(ep_idx) = log(W(:, k)' * W_ssd' * covmats_band(:,:,ep_idx) * W_ssd * W(:, k));
    end
    corrs(k) = corr(Env', Zpr);
end

[corrs_sorted_abs, sort_idx] = sort(abs(corrs), 'descend');
corrs_sorted = corrs(sort_idx);
W_sorted  = W(:, sort_idx);
A_sorted  = A(:, sort_idx);
Vz_sorted = Vz(:, sort_idx);
Az_sorted = Az(:, sort_idx);

disp('Correlations (sorted):');
disp(corrs_sorted');

%% ========================================================================
% 8. ПОДГОТОВКА ДЛЯ ВИЗУАЛИЗАЦИИ
% ========================================================================
ccx = zeros(1, num_clusters);
ccy = zeros(1, num_clusters);
for c = 1:num_clusters
    idx = strcmp(labels, conditions{c});
    ccx(c) = median(Rmean2d(idx,1));
    ccy(c) = median(Rmean2d(idx,2));
end
scale_f = max(sqrt(Rmean2d(:,1).^2 + Rmean2d(:,2).^2));

% Границы блоков
change_idx = find(diff(trials) ~= 0) + 1;
ticks = [0; change_idx(:); N_epoch_trial]';  

%% ========================================================================
% 9. ГЛОБАЛЬНЫЙ 3D UMAP
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
% 10. КОМПЛЕКСНАЯ ВИЗУАЛИЗАЦИЯ ТОП-КОМПОНЕНТ
% ========================================================================
n_to_plot = min(10, length(sort_idx));

for i = 1:n_to_plot
    r_val    = corrs_sorted(i);
    
    % Переводим фильтр и паттерн из SSD в сенсорное пространство
    wx = W_ssd * W_sorted(:, i);      
    ax = A_ssd * W_sorted(:, i);       
    
    % Нормировка знака
    [~, max_idx] = max(abs(wx)); wx = wx .* sign(wx(max_idx));
    [~, max_idx] = max(abs(ax)); ax = ax .* sign(ax(max_idx));
    
    % Мощность источника (в SSD-пространстве)
    S_pow = zeros(1, N_epoch_trial);
    for j = 1:N_epoch_trial
        S_pow(j) = log(wx' * covmats_band(:,:,j) * wx);
    end
    S_raw = S_pow;
    S_pow = (S_pow - mean(S_pow)) / std(S_pow);
    
    % Проекция целевой переменной
    zz = Z_target * Vz_sorted(:, i);
    zz = (zz - mean(zz)) / std(zz) * sign(r_val);
    
    corr(S_pow', zz)

    % --- РАСЧЕТ 2D ВЕКТОРОВ ДЛЯ СТРЕЛОК ---
    zz_c = zz;
    R2_c = Rmean2d;
    vz_2d_eff = pinv(R2_c) * zz_c;
    az_2d_eff = (R2_c' * zz_c) / (zz_c' * zz_c);
    vz_2d = vz_2d_eff / norm(vz_2d_eff) * scale_f * 0.4;
    az_2d = az_2d_eff / norm(az_2d_eff) * scale_f * 0.4;
    
    % ---- ФИГУРА ----
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
    xlabel(ax_umap, 'UMAP 1'); ylabel(ax_umap, 'UMAP 2'); zlabel(ax_umap, 'UMAP 3');
    title(ax_umap, '3D UMAP Space', 'FontSize', 14);
    
    % [2] 2D ИЗОЛИНИИ
    ax_iso = nexttile(t, 2);
    hold(ax_iso, 'on');
    
    MWS_raw = movmean(S_raw', 10);
    x_min = min(Rmean2d(:,1));  x_max = max(Rmean2d(:,1));
    y_min = min(Rmean2d(:,2));  y_max = max(Rmean2d(:,2));
    dx = x_max - x_min;  dy = y_max - y_min;
    pad_factor = 0.1;
    x_min_pad = x_min - pad_factor*dx;    x_max_pad = x_max + pad_factor*dx;
    y_min_pad = y_min - pad_factor*dy;    y_max_pad = y_max + pad_factor*dy;
    
    n_edge = 10;
    x_edge = [linspace(x_min_pad, x_max_pad, n_edge)'; linspace(x_min_pad, x_max_pad, n_edge)'; repmat(x_min_pad, n_edge, 1); repmat(x_max_pad, n_edge, 1)];
    y_edge = [repmat(y_min_pad, n_edge, 1); repmat(y_max_pad, n_edge, 1); linspace(y_min_pad, y_max_pad, n_edge)'; linspace(y_min_pad, y_max_pad, n_edge)'];
    x_corn = [x_min_pad; x_min_pad; x_max_pad; x_max_pad];
    y_corn = [y_min_pad; y_max_pad; y_min_pad; y_max_pad];
    x_edge = [x_edge; x_corn];  y_edge = [y_edge; y_corn];
    
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
        text(ax_iso, ccx(c), ccy(c), num2str(c), 'FontWeight','bold', 'HorizontalAlignment', 'center', ...
            'BackgroundColor', [1 1 1 0.7], 'Margin', 1);
    end
    
    quiver(ax_iso, 0, 0, vz_2d(1), vz_2d(2), 'Color', 'k', 'LineWidth', 2, 'MaxHeadSize', 0.5, 'AutoScale', 'off');
    text(ax_iso, vz_2d(1)*1.1, vz_2d(2)*1.1, 'V_z', 'Color', 'k', 'FontWeight', 'bold');
    quiver(ax_iso, 0, 0, az_2d(1), az_2d(2), 'Color', 'r', 'LineWidth', 3, 'MaxHeadSize', 0.5, 'AutoScale', 'off');
    text(ax_iso, az_2d(1)*1.1, az_2d(2)*1.1, 'A_z', 'Color', 'r', 'FontWeight', 'bold');
    
    xlabel(ax_iso, 'UMAP 1'); ylabel(ax_iso, 'UMAP 2');
    title(ax_iso, 'Isolines of Source Activation', 'FontSize', 14);
    axis(ax_iso, 'tight');
    
    % [3] ФИЛЬТР (W)
    ax_w = nexttile(t, 3);
    topo.avg = wx; cfg.figure = ax_w; cfg.colorbar = 'EastOutside';
    ft_topoplotER(cfg, topo); title(ax_w, 'Spatial Filter', 'FontSize', 14);
    
    % [4] ПАТТЕРН (A)
    ax_p = nexttile(t, 4);
    topo.avg = ax; cfg.figure = ax_p; cfg.colorbar = 'EastOutside';
    ft_topoplotER(cfg, topo); title(ax_p, 'Spatial Pattern', 'FontSize', 14);

    % [5] ДИНАМИКА
    ax_dyn = nexttile(t, 5, [1 4]);
    hold(ax_dyn, 'on'); grid(ax_dyn, 'on');
    plot(ax_dyn, S_pow, 'LineWidth', 1.5, 'Color', [0 0.447 0.741]);
    plot(ax_dyn, zz * sign(r_val), 'LineWidth', 1.5, 'Color', [0.850 0.325 0.098]);
    
    xticks(ax_dyn, ticks(1:end-1));
    if length(ticks)-1 == num_clusters
        xticklabels(ax_dyn, conditions);
    else
        xticklabels(ax_dyn, arrayfun(@num2str, 1:(length(ticks)-1), 'UniformOutput', false));
    end
    xtickangle(ax_dyn, 45);
    xlim(ax_dyn, [0 ticks(end)]);
    for k = 2:length(ticks)-1
        xline(ax_dyn, ticks(k), '--', 'Color', [0.3 0.3 0.3]);
    end
    legend(ax_dyn, {'Source Power Envelope', 'UMAP Canonical Target'}, 'Location', 'best');
    title(ax_dyn, sprintf('Component Dynamics (Correlation = %.3f)', r_val), 'FontSize', 14);
    ylabel(ax_dyn, 'Amplitude (Z-score)');
    
    % ax_dyn.XTickLabelInterpreter = 'none';  
    % =================================================================
    % 11. TIME-FREQUENCY КАРТА (отдельное окно)
    % =================================================================
    f_target = 1:0.5:30;
    window_length = floor(Fs * 1);
    TF_map = zeros(length(f_target), N_epoch_trial);
    
    for ep = 1:N_epoch_trial
        % Проецируем сырые данные через сенсорный фильтр wx
        s_ep_unfilt = wx' * squeeze(X_raw_epo(:,:,ep))';
        [pxx, ~] = pwelch(s_ep_unfilt, hamming(window_length), [], f_target, Fs);
        TF_map(:, ep) = pxx;
    end
    
    % Baseline по всем эпохам, помеченным как RS_EO (и 1, и 2)
    rs_eo_pattern = 'RS_EO';  % общая часть названия
    baseline_idx = contains(labels, rs_eo_pattern);
    if ~any(baseline_idx)
        % запасной вариант – взять первые два блока (если известно количество эпох на блок)
        epochs_per_block = ticks(2) - ticks(1);
        baseline_idx = 1 : 2*epochs_per_block;  % первые два блока
    end
    baseline_power = mean(TF_map(:, baseline_idx), 2);
    TF_map_db = 10 * log10(TF_map ./ repmat(baseline_power, 1, size(TF_map, 2)));    

    fig_tf_name = sprintf('TF Dynamics | Component %d (Rank %d)', i, i);
    figure('Color', 'w', 'Position', [100+i*30, 150+i*30, 1200, 300], 'Name', fig_tf_name);
    
    clim_max = max(abs(prctile(TF_map_db(:), [5 95])));
    imagesc(1:N_epoch_trial, f_target, TF_map_db);
    axis('xy');
    colormap('jet');
    caxis([-clim_max clim_max]);
    cbar_tf = colorbar;
    cbar_tf.Label.String = 'Power Change (dB vs Baseline)';
    title(sprintf('Broadband Time-Frequency Dynamics (C%d, r=%.2f)', i, r_val), 'FontSize', 12);
    ylabel('Frequency (Hz)', 'FontWeight', 'bold');
    
    hold on;
    for k = 2:length(ticks)-1
        xline(ticks(k), '--', 'Color', [1 1 1 0.8], 'LineWidth', 1.5);
    end
    hold off;
    
    xticks(ticks(1:end-1));
    if length(ticks)-1 == num_clusters
        xticklabels(conditions);
    else
        xticklabels(arrayfun(@num2str, 1:(length(ticks)-1), 'UniformOutput', false));
    end
    % set(gca, 'XTickLabelInterpreter', 'none')
    xtickangle(45);
    xlim([0 ticks(end)]);
end


%%