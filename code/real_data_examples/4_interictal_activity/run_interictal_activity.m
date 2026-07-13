close all
clear
clc

ft_path = 'C:\Users\ansbel\Documents\2Git\fieldtrip';
if ~exist('ft_defaults','file')
    addpath(ft_path);
end
ft_defaults;

%% =====================================================================
% 1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ
% =====================================================================
sub_path = 'C:\Users\ansbel\Documents\data\epi\Kostina\Maxfiltered MEG\sleep_1_sss.fif';
cfg = [];
cfg.dataset = sub_path;
Xinf = ft_preprocessing(cfg);

target_Fs = 250; 
cfg_resample = [];
cfg_resample.resamplefs = target_Fs;
cfg_resample.detrend    = 'no';
Xinf = ft_resampledata(cfg_resample, Xinf);
Fs = Xinf.fsample;

% Оставляем только ЭЭГ каналы
good_chans = contains(Xinf.label, 'EEG');
bad_chans  = contains(Xinf.label, 'STI') | contains(Xinf.label, 'EOG') | contains(Xinf.label, 'ECG');
data_idx   = good_chans & ~bad_chans;

if sum(data_idx) == 0
    warning('ЭЭГ каналы не найдены! Будут использованы все доступные каналы.');
    data_idx = 1:length(Xinf.label);
end

laycfg = [];
if isfield(Xinf, 'elec'), laycfg.elec = Xinf.elec; end
lay = ft_prepare_layout(laycfg);     

topo = [];
topo.dimord = 'chan_time';
topo.label  = Xinf.label(data_idx);  
topo.time   = 0;

%%
% Фильтрация
high_pass = 1;
low_pass = min(70, (Fs/2) - 5); 
[b,a] = butter(3, [high_pass, low_pass]/(Fs/2));   
Xfilt  = filtfilt(b, a, Xinf.trial{1}(data_idx, :)')';

%% =====================================================================
% 2. SVD И PCA
% =====================================================================
[U,S,~] = svd(Xfilt,'econ');           
S = diag(S);

tol = max(size(Xfilt)) * eps(S(1));
r = sum(S > tol);

ve = S.^2;
var_explained = cumsum(ve) / sum(ve);
var_explained(end) = 1;

n_components = find(var_explained >= 1, 1);
if isempty(n_components), n_components = r; end
n_components = max(min(n_components, r), 1);
U = U(:, 1:n_components);               
Xfiltpca = U' * Xfilt;

%% =====================================================================
% 3. НАРЕЗКА НА ЭПОХИ
% =====================================================================
Wsize = 0.3;    % Window size in seconds
Ssize = 0.15;  % Step size in seconds
X_epo = epoch_data(Xfiltpca', Fs, Wsize, Ssize);
N_epochs = size(X_epo, 3);

Covs = zeros(n_components, n_components, N_epochs);
for i = 1:N_epochs
    Covs(:,:,i) = cov(X_epo(:,:,i)); 
end
Tcovs = Tangent_space(Covs);           

%% =====================================================================
% 4. ЧТЕНИЕ АННОТАЦИЙ СПАЙКОВ И РАЗМЕТКА ЭПОХ
% =====================================================================
annot_file = 'C:\Users\ansbel\Documents\data\epi\Kostina\Maxfiltered MEG\sleep_1.fif.txt';
spike_times = [];

if exist(annot_file, 'file')
    fid = fopen(annot_file, 'r');
    lines = textscan(fid, '%s', 'Delimiter', '\n');
    fclose(fid);
    lines = lines{1};

    for i = 1:length(lines)
        str = lines{i};
        % Ищем строки с разметкой
        if contains(str, '|ED|') || contains(str, '=2|')
            parts = strsplit(str, '|');
            if length(parts) >= 5
                % Заменяем запятые на точки для правильной конвертации в числа
                t_start = str2double(strrep(parts{4}, ',', '.'));
                t_end   = str2double(strrep(parts{5}, ',', '.'));
                
                if ~isnan(t_start) && ~isnan(t_end)
                    spike_times = [spike_times; t_start, t_end];
                end
            end
        end
    end
    fprintf('>>> Найдено спайков в разметке: %d\n', size(spike_times, 1));
else
    warning('Файл %s не найден! Построение без подсветки спайков.', annot_file);
end

% Создаем логический массив: какие из N_epochs содержат спайк
is_spike_epoch = false(1, N_epochs);
for ep_idx = 1:N_epochs
    epoch_start = (ep_idx - 1) * Ssize;
    epoch_end   = epoch_start + Wsize;
    
    for sp = 1:size(spike_times, 1)
        % Проверка на пересечение отрезков времени: (Start1 <= End2) && (End1 >= Start2)
        if epoch_start <= spike_times(sp, 2) && epoch_end >= spike_times(sp, 1)
            is_spike_epoch(ep_idx) = true;
            break;
        end
    end
end
fprintf('>>> Эпох со спайками: %d из %d\n', sum(is_spike_epoch), N_epochs);

%% =====================================================================
% 5. UMAP EMBEDDING
% =====================================================================
clear u
u = UMAP("n_neighbors", 30, "n_components", 3, "min_dist", 0.1); 
u.metric = 'euclidean';
R = u.fit_transform(Tcovs');
Rmean = R - mean(R,1);

%%
figure('Color', 'w');
hold on; grid on;

% 1. Рисуем непрерывную траекторию сна тонкой нитью (plot3)
plot3(R(:,1), R(:,2), R(:,3), 'Color', [0.6 0.6 0.6 0.3], 'LineWidth', 0.5);

% 2. Рисуем фоновое ЭЭГ (мелкие серые точки)
scatter3(R(~is_spike_epoch, 1), R(~is_spike_epoch, 2), R(~is_spike_epoch, 3), ...
    15, [0.7 0.7 0.7], 'filled', 'MarkerFaceAlpha', 0.4);

% 3. Рисуем спайки поверх (крупные красные точки с черной обводкой)
if any(is_spike_epoch)
    scatter3(R(is_spike_epoch, 1), R(is_spike_epoch, 2), R(is_spike_epoch, 3), ...
        45, [1 0 0], 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
    legend({'Trajectory', 'Background EEG', 'Spikes'}, 'Location', 'best');
else
    legend({'Trajectory', 'Background EEG'}, 'Location', 'best');
end

xlabel('UMAP 1'); ylabel('UMAP 2'); zlabel('UMAP 3');
view(-45, 30); % Удобный угол обзора для 3D

%% =====================================================================
% 6. mSPoC COMPUTATION
% =====================================================================
[W, Vz, ~, A, Az, ~] = my_mspoc(X_epo, Rmean');
corrs = zeros(1, size(W,2));

for k = 1:size(W,2)
    Zpr = Vz(:,k)' * Rmean';
    Env = zeros(1, N_epochs);
    for ep_idx = 1:N_epochs
        Env(ep_idx) = log(W(:, k)' * Covs(:,:,ep_idx) * W(:, k));
    end
    corrs(k) = corr(Env', Zpr');
end

[corrs_sorted_abs, sort_idx] = sort(abs(corrs), 'descend');
corrs_sorted = corrs(sort_idx); 
W_sorted  = W(:, sort_idx);
A_sorted  = A(:, sort_idx);
Vz_sorted = Vz(:, sort_idx) .* sign(corrs_sorted);

%% =====================================================================
% 7. КОМПЛЕКСНАЯ ВИЗУАЛИЗАЦИЯ
% =====================================================================
n_to_plot = min(10, length(sort_idx));
% Центр каждой эпохи в секундах
time_axis_sec = (0:N_epochs-1) * Ssize + Wsize/2; 

for i = 1:n_to_plot
    comp_idx = sort_idx(i);
    r_val    = corrs_sorted(i);
    
    wx = U * W_sorted(:, i);
    ax = U * A_sorted(:, i);
    
    [~, max_idx] = max(abs(wx)); wx = wx .* sign(wx(max_idx));
    [~, max_idx] = max(abs(ax)); ax = ax .* sign(ax(max_idx));
    
    v = Vz_sorted(:, i);
    v = v / norm(v); 
    
    S_pow = zeros(1, N_epochs);
    for j = 1:N_epochs
        S_pow(j) = log(wx' * U * Covs(:,:,j) * U' * wx);
    end
    S_raw = S_pow; 
    S_pow = (S_pow - mean(S_pow)) / std(S_pow); 
    
    zz = Rmean * Vz_sorted(:, i);
    zz = (zz - mean(zz)) / std(zz) * sign(r_val);
    
    % -----------------------------------------------------------------
    fig_name = sprintf('Component %d | Corr = %.3f', comp_idx, r_val);
    figure('Color', 'w', 'Position', [50, 50, 1600, 700], 'Name', fig_name);
    t = tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    % [1] 3D UMAP (Разделение на Фон и Спайки)
    ax_umap = nexttile(t, 1); hold(ax_umap, 'on'); grid(ax_umap, 'on');
    
    % Рисуем фоновый сон (серым, полупрозрачным)
    h_bg = scatter3(ax_umap, Rmean(~is_spike_epoch, 1), Rmean(~is_spike_epoch, 2), Rmean(~is_spike_epoch, 3), ...
        15, [0.7 0.7 0.7], 'filled', 'MarkerFaceAlpha', 0.4);
        
    % Рисуем спайки (красным, более крупными точками)
    if any(is_spike_epoch)
        h_sp = scatter3(ax_umap, Rmean(is_spike_epoch, 1), Rmean(is_spike_epoch, 2), Rmean(is_spike_epoch, 3), ...
            40, [1 0 0], 'filled', 'MarkerFaceAlpha', 0.9, 'MarkerEdgeColor', 'k');
        legend(ax_umap, [h_bg, h_sp], {'Background EEG', 'Spikes'}, 'Location', 'best');
    else
        legend(ax_umap, h_bg, {'Background EEG'}, 'Location', 'best');
    end
    
    scale_f = max(abs(Rmean(:))) * 0.9;
    quiver3(ax_umap, 0, 0, 0, v(1)*scale_f, v(2)*scale_f, v(3)*scale_f, 'Color', 'k', 'LineWidth', 4, 'MaxHeadSize', 0.5, 'AutoScale', 'off');
    view(ax_umap, -45, 30);
    xlabel('UMAP 1'); ylabel('UMAP 2'); zlabel('UMAP 3');
    title(ax_umap, 'Spike Distribution in UMAP', 'FontSize', 14);
    
    % [2] 2D ИЗОЛИНИИ
    ax_iso = nexttile(t, 2); hold(ax_iso, 'on');
    MWS_raw = movmean(S_raw', 5); 
    F = scatteredInterpolant(Rmean(:,1), Rmean(:,2), MWS_raw, 'natural', 'linear');
    
    xrange = linspace(min(Rmean(:,1))-1, max(Rmean(:,1))+1, 100);
    yrange = linspace(min(Rmean(:,2))-1, max(Rmean(:,2))+1, 100);
    [Xg, Yg] = meshgrid(xrange, yrange);
    ProjGrid = F(Xg, Yg);
    pcolor(ax_iso, Xg, Yg, ProjGrid); shading(ax_iso, 'interp'); colormap(ax_iso);
    contour(ax_iso, Xg, Yg, ProjGrid, 10, 'k', 'LineWidth', 0.05);
    
    vz_2d = Vz_sorted(1:2, i) / norm(Vz_sorted(1:2, i)) * scale_f * 0.4;
    quiver(ax_iso, 0, 0, vz_2d(1), vz_2d(2), 'Color', 'k', 'LineWidth', 2, 'MaxHeadSize', 0.5, 'AutoScale', 'off');
    xlabel('UMAP 1'); ylabel('UMAP 2');
    title('Source Power Isolines', 'FontSize', 14); axis tight;
    
    % [3] FILTER
    ax_w = nexttile(t, 3);
    topo.avg = wx;
    cfg.figure = ax_w;
    ft_topoplotER(cfg, topo); title('Spatial Filter', 'FontSize', 14);
    
    % [4] PATTERN
    ax_p = nexttile(t, 4);
    topo.avg = ax;
    cfg.figure = ax_p;
    ft_topoplotER(cfg, topo); title('Spatial Pattern', 'FontSize', 14);
    
    % [5] DYNAMICS
    ax_dyn = nexttile(t, 5, [1 4]); hold(ax_dyn, 'on'); grid(ax_dyn, 'on');
    
    % Подсветка временных отрезков спайков красными прозрачными полосами
    yl = [-3, max(max(S_pow), max(zz))*1.1]; 
    ylim(ax_dyn, yl);
    for sp = 1:size(spike_times, 1)
        patch(ax_dyn, [spike_times(sp,1) spike_times(sp,2) spike_times(sp,2) spike_times(sp,1)], ...
                      [yl(1) yl(1) yl(2) yl(2)], [1 0 0], 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
    
    plot(ax_dyn, time_axis_sec, S_pow, 'LineWidth', 1.5, 'Color', [0 0.447 0.741], 'DisplayName', 'Source Power Envelope');
    plot(ax_dyn, time_axis_sec, zz * sign(r_val), 'LineWidth', 1.5, 'Color', [0.850 0.325 0.098], 'DisplayName', 'UMAP Canonical Target');
    
    legend(ax_dyn, 'Location', 'best');
    title(sprintf('Dynamics over time (Correlation = %.3f)', r_val), 'FontSize', 14);
    xlabel('Time (seconds)'); ylabel('Z-score');
    axis tight;
end