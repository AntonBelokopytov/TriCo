close all
clear
clc
ft_path = 'C:\Users\ansbel\Documents\GitHub\site-packages\fieldtrip';
if ~exist('ft_defaults','file')
    addpath(ft_path);
end
ft_defaults;
np = py.importlib.import_module('numpy');
umap_module = py.importlib.import_module('umap');

%%
sub_path = 'DmiAna_music_epochs.fif';
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

%%
conditions = {'(1) RS EC 1', '(2) RS EO 1', '(3) 2Hz', '(4) 05Hz', '(5) 4Hz', '(6) 1Hz', '(7) 3Hz', ...
              '(8) NoRy 1','(9) Waltz 1','(10) Waltz 2','(11) NoRy 2','(12) NoRy 3','(13) Waltz 3', ...
              '(14) NoRy 4','(15) Waltz 4','(16) NoRy 5','(17) Waltz 5','(18) RS EC 2','(19) RS EO 2', ...
              '(20) Waltz 6','(21) Waltz 7','(22) Waltz 8'};

%% =====================================================================
% BANDPASS FILTERING
% =====================================================================
[b,a] = butter(3,[15,25]/(Fs/2));   
n_channels = 38;                    
Xfilt = [];
Epfilt = [];
Epochs_filt = []; Epochs = [];
Xraw = [];
for i = 1:numel(Xinf.trial)
    Ep_raw = Xinf.trial{i}(1:n_channels,:);            
    Epfilt  = filtfilt(b,a,Ep_raw')';    
    Epfilt = Epfilt(:,Fs/2:end-Fs/2);    
    
    Xraw = cat(2, Xraw, Ep_raw);
    Xfilt = cat(2,Xfilt,Epfilt);         
    Epochs_filt(:,:,i) = Epfilt;         
    Epochs(:,:,i) = Ep_raw;         
end

%% =====================================================================
% SPATIO-SPECTRAL DECOMPOSITION (SSD)
% =====================================================================
freq_noise_broad = [13 27];       
freq_noise_stop  = [14.5 25.5];   

X_signal = Xfilt; 
X_noise_broad = bandpass(Xraw', freq_noise_broad, Fs)';
[b_stop, a_stop] = butter(3, freq_noise_stop/(Fs/2), 'stop');
X_noise = filtfilt(b_stop, a_stop, X_noise_broad')';

C_signal = cov(X_signal'); 
C_noise  = cov(X_noise');

reg_coeff = 1e-5; 
C_noise_reg = C_noise + reg_coeff * trace(C_noise) * eye(size(C_noise));

[W_all, D] = eig(C_signal, C_noise_reg);
eigenvalues = diag(D); 
[~, sort_idx_ssd] = sort(eigenvalues, 'descend');
W_all = W_all(:, sort_idx_ssd);

n_components = 25; 
W_ssd = W_all(:, 1:n_components);
A_ssd = C_signal * W_ssd / (W_ssd' * C_signal * W_ssd);

Epfilt_pca = zeros(n_components, size(Epochs_filt,2), size(Epochs_filt,3));
for i = 1:size(Epochs_filt,3)
    Epfilt_pca(:,:,i) = W_ssd' * Epochs_filt(:,:,i);
end

%% =====================================================================
% EPOCH SEGMENTATION
% =====================================================================
Wsize = 2;  
Ssize = 0.5;  
X_epo = []; X_raw_epo = []; time = []; % Добавили X_raw_epo

for i=1:size(Epfilt_pca,3)
    % Окна для mSPoC (отфильтрованные и в SSD-пространстве)
    ep_wins = epoch_data(Epfilt_pca(:,:,i)', Fs, Wsize, Ssize);
    X_epo = cat(3, X_epo, ep_wins); 
    
    % Окна для спектрограммы (сырые данные в исходном пространстве сенсоров)
    ep_wins_raw = epoch_data(Epochs(:,:,i)', Fs, Wsize, Ssize);
    X_raw_epo = cat(3, X_raw_epo, ep_wins_raw);
    
    timeline = 0.5 + ( Wsize/2:Ssize:(size(ep_wins,3)*Ssize+Ssize) );
    if i>1
        timeline = timeline + time(end) + Wsize-Ssize;
    end
    time = [time,timeline];
end

Covs = []; Covs_vec = [];
for i=1:size(X_epo,3)
    C = cov(X_epo(:,:,i));
    Covs(:,:,i) = C;
    Covs_vec(i,:) = C(triu(true(size(C))));
end

Cmean = riemann_mean(Covs);
Tcovs = Tangent_space(Covs,Cmean);          
N_epoch_trial = size(ep_wins,3);

%% =====================================================================
% UMAP EMBEDDING (Hyperboloid)
% =====================================================================
reducer10d = umap_module.UMAP(pyargs('n_neighbors', 20, 'n_components', 10, 'min_dist', 0.1, 'metric', 'euclidean', 'output_metric', 'euclidean'));
reducer3d = umap_module.UMAP(pyargs('n_neighbors', 20, 'n_components', 3, 'min_dist', 0.1, 'metric', 'euclidean', 'output_metric', 'euclidean'));
reducer2d = umap_module.UMAP(pyargs('n_neighbors', 20, 'n_components', 2, 'min_dist', 0.1, 'metric', 'euclidean', 'output_metric', 'euclidean'));

Tcovs_aug = [Tcovs'; zeros(1, size(Tcovs,1))]'; 
Tcovs_aug_np = np.array(Tcovs_aug');
reducer10d.fit(Tcovs_aug_np);
reducer3d.fit(Tcovs_aug_np);
reducer2d.fit(Tcovs_aug_np);

Rmean10d = double(reducer10d.transform(Tcovs_aug_np));
Rmean3d = double(reducer3d.transform(Tcovs_aug_np));
Rmean2d = double(reducer2d.transform(Tcovs_aug_np));

%% =====================================================================
% VISUALIZE 3D UMAP (С ЛЕГЕНДОЙ)
% =====================================================================
num_clusters = 22;
cmap = jet(num_clusters);

figure('Name', 'Global 3D UMAP', 'Color', 'w', 'Position', [100, 100, 900, 700]);
hold on; grid on;
h_scat = gobjects(num_clusters, 1);
mask = 1:N_epoch_trial;

for c = 1:num_clusters
    if mask(end) <= size(Rmean3d, 1)
        idx = mask;
    else
        idx = mask(1):size(Rmean3d, 1);
    end
    mask = mask + N_epoch_trial;
    
    h_scat(c) = scatter3(Rmean3d(idx,1), Rmean3d(idx,2), Rmean3d(idx,3), ...
        25, cmap(c,:), 'filled', 'MarkerFaceAlpha', 0.7);
end

xlabel('UMAP component 1'); ylabel('UMAP component 2'); zlabel('UMAP component 3');
title('3D Hyperboloid UMAP Embedding', 'FontSize', 14);
view(-45, 30);
% Создаем чистую легенду
% legend(h_scat(isgraphics(h_scat)), conditions, 'Location', 'bestoutside', 'Interpreter', 'none');
hold off;

%% =====================================================================
% ИНИЦИАЛИЗАЦИЯ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ ДЛЯ mSPoC
% =====================================================================
% Вы можете легко менять Z_target на Rmean3d или Rmean2d. 
% Визуализация автоматически подстроится и спроецирует результат на 2D/3D.
Z_target = Rmean10d_centered;  

%% =====================================================================
% VISUALIZE 3D UMAP (С ЛЕГЕНДОЙ)
% =====================================================================
% embedding_centered_3 = hyperbolic_recenter(Rmean3d);

num_clusters = 22;
cmap = jet(num_clusters);

figure('Name', 'Global 3D UMAP', 'Color', 'w', 'Position', [100, 100, 900, 700]);
hold on; grid on;
h_scat = gobjects(num_clusters, 1);
mask = 1:N_epoch_trial;

for c = 1:num_clusters
    if mask(end) <= size(Rmean3d_centered, 1)
        idx = mask;
    else
        idx = mask(1):size(Rmean3d_centered, 1);
    end
    mask = mask + N_epoch_trial;
    
    h_scat(c) = scatter3(Rmean3d_centered(idx,1), Rmean3d_centered(idx,2), Rmean3d_centered(idx,3), ...
        25, cmap(c,:), 'filled', 'MarkerFaceAlpha', 0.7);
end

xlabel('UMAP component 1'); ylabel('UMAP component 2'); zlabel('UMAP component 3');
title('3D Hyperboloid UMAP Embedding', 'FontSize', 14);
view(-45, 30);
% Создаем чистую легенду
% legend(h_scat(isgraphics(h_scat)), conditions, 'Location', 'bestoutside', 'Interpreter', 'none');
hold off;

%% =====================================================================
% mSPoC COMPUTATION
% =====================================================================
[W, Vz, ~, A, Az, out] = my_mspoc(X_epo, Z_target', 'Cxx', Cmean);

Epochs_cov = [];
for i=1:size(X_epo,3)
    Epochs_cov(:,:,i) = cov(X_epo(:,:,i));
end

corrs = [];
for k = 1:size(W,2)
    Zpr = Vz(:,k)' * Z_target';
    Env = zeros(1, size(X_epo,3));
    for ep_idx = 1:size(X_epo,3)
        Env(ep_idx) = log(W(:, k)' * Epochs_cov(:,:,ep_idx) * W(:, k));
    end
    corrs(k) = corr(Env', Zpr');
end

[corrs_sorted_abs, sort_idx] = sort(abs(corrs), 'descend');
corrs_sorted = corrs(sort_idx); 
W_sorted  = W(:, sort_idx);
A_sorted  = A(:, sort_idx);
Vz_sorted = Vz(:, sort_idx) .* sign(corrs_sorted);
Az_sorted = Az(:, sort_idx) .* sign(corrs_sorted);
corrs_sorted

%% =====================================================================
% ПОДГОТОВКА ДАННЫХ ДЛЯ ВИЗУАЛИЗАЦИИ
% =====================================================================
ccx = zeros(1, num_clusters);
ccy = zeros(1, num_clusters);
mask = 1:N_epoch_trial;

for i = 1:num_clusters
    if mask(end) <= size(Rmean2d_centered, 1)
        idx = mask;
    else
        idx = mask(1);
    end
    mask = mask + N_epoch_trial;
    ccx(i) = median(Rmean2d_centered(idx,1)); 
    ccy(i) = median(Rmean2d_centered(idx,2)); 
end

scale_f = max(sqrt(Rmean2d_centered(:,1).^2 + Rmean2d_centered(:,2).^2));

%% =====================================================================
% КОМПЛЕКСНАЯ ВИЗУАЛИЗАЦИЯ (ТОП КОМПОНЕНТЫ)
% =====================================================================
n_to_plot = min(10, length(sort_idx));
tstep = 235; 
ticks = 0:tstep:size(Covs,3);
U = pinv(W_ssd');

for i = 1:n_to_plot
    comp_idx = sort_idx(i);
    r_val    = corrs_sorted(i);
    
    wx = U * W(:, comp_idx); 
    ax = A_ssd * A(:, comp_idx);    

    [~, max_idx] = max(abs(wx)); wx = wx .* sign(wx(max_idx));
    [~, max_idx] = max(abs(ax)); ax = ax .* sign(ax(max_idx));
    
    S_pow = zeros(1, size(Covs,3));
    for j = 1:size(Covs,3)
        S_pow(j) = log(W(:, comp_idx)' * Covs(:,:,j) * W(:, comp_idx));
    end
    S_raw = S_pow; 
    S_pow = (S_pow - mean(S_pow)) / std(S_pow); 
    % S_pow = S_pow / std(S_pow); 
    
    zz = Z_target * Vz_sorted(:, i);
    zz = (zz - mean(zz)) / std(zz) * sign(r_val);
    
    % --- РАСЧЕТ ЭКВИВАЛЕНТНЫХ 2D ВЕКТОРОВ ---
    % Проецируем 1D таргет (zz) обратно на 2D пространство для отрисовки стрелок
    zz_c = zz;
    R2_c = Rmean2d_centered;
    vz_2d_eff = pinv(R2_c) * zz_c;               % Фильтр в 2D (Linear Regression)
    az_2d_eff = (R2_c' * zz_c) / (zz_c' * zz_c); % Паттерн в 2D (Covariance)
    
    vz_2d = vz_2d_eff / norm(vz_2d_eff) * scale_f * 0.4;
    az_2d = az_2d_eff / norm(az_2d_eff) * scale_f * 0.4;
    
    % -----------------------------------------------------------------
    fig_name = sprintf('Component %d (Rank %d) | Corr = %.3f', comp_idx, i, r_val);
    figure('Color', 'w', 'Position', [50+i*30, 50+i*30, 1600, 700], 'Name', fig_name);
    
    t = tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    % [1] 3D UMAP
    ax_umap = nexttile(t, 1);
    hold(ax_umap, 'on'); grid(ax_umap, 'on');
    mask = 1:N_epoch_trial;
    for c = 1:num_clusters
        if mask(end) <= size(Rmean3d, 1)
            idx = mask;
        else
            idx = mask(1):size(Rmean3d, 1);
        end
        mask = mask + N_epoch_trial;
        scatter3(ax_umap, Rmean3d(idx,1), Rmean3d(idx,2), Rmean3d(idx,3), 15, repmat(cmap(c,:), length(idx), 1), ...
            'filled', 'MarkerFaceAlpha', 0.4);
    end
    view(ax_umap, -45, 30);
    xlabel(ax_umap, 'UMAP 1'); ylabel(ax_umap, 'UMAP 2'); zlabel(ax_umap, 'UMAP 3');
    title(ax_umap, '3D UMAP Space', 'FontSize', 14);
    
    % [2] 2D ИЗОЛИНИИ
    ax_iso = nexttile(t, 2);
    hold(ax_iso, 'on');
    
    MWS_raw = movmean(S_raw', 20);
    x_min = min(Rmean2d_centered(:,1));  x_max = max(Rmean2d_centered(:,1));
    y_min = min(Rmean2d_centered(:,2));  y_max = max(Rmean2d_centered(:,2));
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
    X_all = [Rmean2d_centered(:,1); x_edge]; Y_all = [Rmean2d_centered(:,2); y_edge]; MWS_all = [MWS_raw; MWS_edge];
    
    F = scatteredInterpolant(X_all, Y_all, MWS_all, 'natural', 'linear');
    [Xg, Yg] = meshgrid(linspace(x_min_pad, x_max_pad, 100), linspace(y_min_pad, y_max_pad, 100));
    ProjGrid = F(Xg, Yg);
    ProjGrid(ProjGrid > max(MWS_raw)) = max(MWS_raw);

    pcolor(ax_iso, Xg, Yg, ProjGrid); shading(ax_iso, 'interp'); colormap(ax_iso);
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
    
    % [5] ДИНАМИКА ИСТОЧНИКА VS ПРОЕКЦИЯ UMAP
    ax_dyn = nexttile(t, 5, [1 4]);
    hold(ax_dyn, 'on'); grid(ax_dyn, 'on');
    
    plot(ax_dyn, S_pow, 'LineWidth', 1.5, 'Color', [0 0.447 0.741]);
    plot(ax_dyn, zz * sign(r_val), 'LineWidth', 1.5, 'Color', [0.850 0.325 0.098]);
    
    xticks(ax_dyn, ticks(1:end-1)); xticklabels(ax_dyn, conditions); xtickangle(ax_dyn, 45); xlim(ax_dyn, [0 ticks(end)]);
    for k = 1:length(ticks(2:end-1)) xline(ax_dyn, ticks(k), '--', 'Color', [0.3 0.3 0.3]); end
    
    legend(ax_dyn, {'Source Power Envelope', 'UMAP Canonical Target'}, 'Location', 'best');
    title(ax_dyn, sprintf('Component Dynamics (Correlation = %.3f)', r_val), 'FontSize', 14); ylabel(ax_dyn, 'Amplitude (Z-score)');


    % =================================================================
    % ОТДЕЛЬНЫЙ ГРАФИК TIME-FREQUENCY
    % =================================================================
    f_target = 1:0.5:25; % Чуть шире диапазон, чтобы видеть альфу и бету
    window_length = floor(Fs * 1); % Окно 1 секунда для pwelch
    TF_map = zeros(length(f_target), size(X_raw_epo, 3));
    
    for ep = 1:size(X_raw_epo, 3)
        % Проецируем 2-секундное окно сырых данных через найденный пространственный фильтр
        s_ep_unfilt = wx' * squeeze(X_raw_epo(:,:,ep))'; 
        [pxx, ~] = pwelch(s_ep_unfilt, hamming(window_length), [], f_target, Fs);
        % Сохраняем абсолютную мощность (пока не логарифмируем)
        TF_map(:, ep) = pxx; 
    end
    
    % --- ВАРИАНТ 1: Baseline Correction (Рекомендуемый) ---
    % Находим индексы эпох, относящихся к Resting State (например, EO)
    % Предполагаем, что condition_idx хранит метки условий для каждой эпохи X_epo
    % (тебе нужно убедиться, что condition_idx у тебя создан при нарезке)
    
    % Ищем эпохи, попадающие в 1-е (RS EC) или 2-е (RS EO) условия. 
    % Давай возьмем RS EO (открытые глаза) как более релевантный бейзлайн для задач
    % В твоем списке conditions: '(2) RS EO 1' и '(19) RS EO 2'
    
    % (Упрощенный поиск бейзлайна - берем просто первые N эпох, 
    % если у тебя нет вектора меток. Замени на реальные индексы!)
    baseline_idx = (ticks(2)+1):ticks(3); % Берем первую сессию (RS EC 1) как бейзлайн
    
    % Средняя абсолютная мощность в бейзлайне
    baseline_power = mean(TF_map(:, baseline_idx), 2);
    
    % Переводим в децибелы относительно бейзлайна (ERD/ERS)
    % Формула: 10 * log10(Power / Baseline)
    TF_map_db = 10 * log10(TF_map ./ repmat(baseline_power, 1, size(TF_map, 2)));
    
    % --- ВАРИАНТ 2: Z-score по частотам (Закомментировано, если захочешь) ---
    % TF_map_log = 10 * log10(TF_map);
    % TF_map_db = (TF_map_log - mean(TF_map_log, 2)) ./ std(TF_map_log, 0, 2);
    % --------------------------------------------------------

    fig_tf_name = sprintf('TF Dynamics | Component %d (Rank %d)', comp_idx, i);
    figure('Color', 'w', 'Position', [100+i*30, 150+i*30, 1200, 300], 'Name', fig_tf_name);
    
    % Отрисовка
    % Важно ограничить цветовую шкалу, чтобы выбросы не ломали контраст
    clim_max = max(abs(prctile(TF_map_db(:), [5 95]))); 
    
    imagesc(1:size(X_raw_epo,3), f_target, TF_map_db);
    axis('xy');
    colormap('jet');
    caxis([-clim_max clim_max]); % Симметричная шкала для десинхронизации/синхронизации
    
    cbar_tf = colorbar; 
    cbar_tf.Label.String = 'Power Change (dB vs Baseline)';
    title(sprintf('Broadband Time-Frequency Dynamics (C%d, r=%.2f)', comp_idx, r_val), 'FontSize', 12);
    ylabel('Frequency (Hz)', 'FontWeight', 'bold');
    
    % Добавляем линии условий и подписи оси X
    hold on;
    for k = 1:length(ticks(2:end-1))
        xline(ticks(k), '--', 'Color', [1 1 1 0.8], 'LineWidth', 1.5);
    end
    hold off;
    
    xticks(ticks(1:end-1)); 
    xticklabels(conditions); 
    xtickangle(45); 
    xlim([0 ticks(end)]);
end 
