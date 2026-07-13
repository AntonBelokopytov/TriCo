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

%%
conditions = {'(1) RS EC 1', '(2) RS EO 1', '(3) 2Hz', '(4) 05Hz', '(5) 4Hz', '(6) 1Hz', '(7) 3Hz', ...
              '(8) NoRy 1','(9) Waltz 1','(10) Waltz 2','(11) NoRy 2','(12) NoRy 3','(13) Waltz 3', ...
              '(14) NoRy 4','(15) Waltz 4','(16) NoRy 5','(17) Waltz 5','(18) RS EC 2','(19) RS EO 2', ...
              '(20) Waltz 6','(21) Waltz 7','(22) Waltz 8'};

%% =====================================================================
% BANDPASS FILTERING
% =====================================================================
[b,a] = butter(3,[15,25]/(Fs/2));   % 3rd-order Butterworth filter
n_channels = 38;                    % Only EEG channels
Xfilt = [];
Epfilt = [];
Epochs_filt = [];

Xraw = [];
for i = 1:numel(Xinf.trial)
    Ep_raw = Xinf.trial{i}(1:n_channels,:);            
    Epfilt  = filtfilt(b,a,Ep_raw')';    % Zero-phase filtering
    Epfilt = Epfilt(:,Fs/2:end-Fs/2);    % Trim edges
    
    Xraw = cat(2, Xraw, Ep_raw);
    Xfilt = cat(2,Xfilt,Epfilt);         % Concatenate for SVD
    Epochs_filt(:,:,i) = Epfilt;         % Store filtered epoch
end

%% =====================================================================
% SPATIO-SPECTRAL DECOMPOSITION (SSD) - ORIGINAL APPROACH
% =====================================================================
% 1. Настройка частотных диапазонов (широкий шум минус целевой сигнал)
freq_noise_broad = [13 27];       % Широкая полоса (захватывает и фланкеры, и сигнал)
freq_noise_stop  = [14.5 25.5];   % То, что мы ВЫРЕЗАЕМ из шума (целевая полоса)
% freq_noise_broad = [13 40];       % Широкая полоса (захватывает и фланкеры, и сигнал)
% freq_noise_stop  = [14.5 25.5];   % То, что мы ВЫРЕЗАЕМ из шума (целевая полоса)

% 2. Подготовка данных
X_signal = Xfilt; 

% Получаем шум классическим методом (Band-pass -> Band-stop)
X_noise_broad = bandpass(Xraw', freq_noise_broad, Fs)';
[b_stop, a_stop] = butter(3, freq_noise_stop/(Fs/2), 'stop');
X_noise = filtfilt(b_stop, a_stop, X_noise_broad')';

% 3. Вычисление матриц ковариации
C_signal = cov(X_signal'); 
C_noise  = cov(X_noise');

% 4. РЕГУЛЯРИЗАЦИЯ (ДИАГОНАЛЬНАЯ ЗАГРУЗКА) - Защита от вырожденности матрицы!
% Добавляем крошечную константу на главную диагональ матрицы шума.
% Это делает её строго положительно определенной (full rank) без искажения физики.
reg_coeff = 1e-5; % 0.001% от дисперсии
C_noise_reg = C_noise + reg_coeff * trace(C_noise) * eye(size(C_noise));

% 5. Обобщенное разложение на собственные значения (GEVD)
[W_all, D] = eig(C_signal, C_noise_reg);
eigenvalues = diag(D); % Отношение С/Ш

% 6. Сортировка компонент по убыванию SNR
[~, sort_idx_ssd] = sort(eigenvalues, 'descend');
W_all = W_all(:, sort_idx_ssd);

%%
% 7. Выбор количества компонент
n_components = 30; 
W_ssd = W_all(:, 1:n_components);
% W_ssd = eye(size(W_ssd));

% Формула паттерна: A = C_s * W * (W' * C_s * W)^-1
A_ssd = C_signal * W_ssd / (W_ssd' * C_signal * W_ssd);
% A_ssd = eye(size(A_ssd));

% 8. Проекция непрерывных данных и эпох в пространство SSD
Xfiltpca = W_ssd' * Xfilt; 

Epfilt_pca = zeros(n_components, size(Epochs_filt,2), size(Epochs_filt,3));
for i = 1:size(Epochs_filt,3)
    Epfilt_pca(:,:,i) = W_ssd' * Epochs_filt(:,:,i);
end

% =====================================================================
% EPOCH SEGMENTATION
% =====================================================================
Wsize = 2;  % Window size in seconds
Ssize = 0.5;  % Step size in seconds

X_epo = []; time = [];
for i=1:size(Epfilt_pca,3)
    ep_wins = epoch_data(Epfilt_pca(:,:,i)', Fs, Wsize, Ssize);
    X_epo = cat(3,X_epo,ep_wins); 
    timeline = 0.5 + ( Wsize/2:Ssize:(size(ep_wins,3)*Ssize+Ssize) );
    if i>1
        timeline = timeline + time(end) + Wsize-Ssize;
    end
    time = [time,timeline];
end

% Covariance matrices of epochs
Covs = []; Covs_vec = [];
for i=1:size(X_epo,3)
    C = cov(X_epo(:,:,i));
    Covs(:,:,i) = C;
    Covs_vec(i,:) = C(triu(true(size(C))));
end

% Covs = [];
% for i=1:size(X_epo,3)
%     C = cov(X_epo(:,:,i));
%     C = C + 1e-6 * trace(C) * eye(size(C)); 
%     Covs(:,:,i) = C;
% end

Cmean = riemann_mean(Covs);
Tcovs = Tangent_space(Covs,Cmean);          
N_epoch_trial = size(ep_wins,3);

figure;
plot(Tcovs')

%%
reducer10d = umap_module.UMAP(pyargs(...
    'n_neighbors', 20, ...
    'n_components', 10, ...
    'min_dist', 0.1, ...
    'metric', 'euclidean', ...
    'output_metric', 'hyperboloid' ...
));

reducer3d = umap_module.UMAP(pyargs(...
    'n_neighbors', 20, ...
    'n_components', 3, ...
    'min_dist', 0.1, ...
    'metric', 'euclidean', ...
    'output_metric', 'hyperboloid' ...
));

reducer2d = umap_module.UMAP(pyargs(...
    'n_neighbors', 20, ...
    'n_components', 2, ...
    'min_dist', 0.1, ...
    'metric', 'euclidean', ...
    'output_metric', 'hyperboloid' ...
));

% Данные: Tcovs размера (число_окон x D)
Tcovs_np = np.array(Tcovs');   % Python-массив, строки – окна

% --- Обучение UMAP ---
reducer10d.fit(Tcovs_np);
reducer3d.fit(Tcovs_np);
reducer2d.fit(Tcovs_np);

% --- Трансформация всех окон ---
R10d = double(reducer10d.transform(Tcovs_np));
R3d = double(reducer3d.transform(Tcovs_np));
R2d = double(reducer2d.transform(Tcovs_np));

% --- Образ Cmean (нулевой вектор) ---
origin_vec = np.array(zeros(2,size(Tcovs,1)));  % 1xD нулей
origin_3d = double(reducer3d.transform(origin_vec));    % 1x3
origin_2d = double(reducer2d.transform(origin_vec));    % 1x2

% --- Сдвиг начала координат в Cmean ---
% Rmean3d = R3d - origin_3d(1,:);   % теперь (0,0,0) = образ Cmean
% Rmean2d = R2d - origin_2d(1,:);   % теперь (0,0)   = образ Cmean
Rmean10d = R10d;   % теперь (0,0,0) = образ Cmean
Rmean3d = R3d;   % теперь (0,0,0) = образ Cmean
Rmean2d = R2d;   % теперь (0,0)   = образ Cmean

% =====================================================================
% UMAP EMBEDDING
% =====================================================================
% clear u
% u = UMAP("n_neighbors",20,"n_components",3,"min_dist",0);
% u.metric = 'euclidean';
% u.target_metric = 'euclidean';
% 
% % Low-dimensional embedding of epochs
% R = u.fit_transform(Tcovs');
% Rmean = R - mean(R,1);

%% =====================================================================
% VISUALIZE UMAP
% =====================================================================
figure
set(gcf, 'Color', 'w');
% scatter3(Rmean3d(:,1),Rmean3d(:,2),Rmean3d(:,3));
% scatter3(Rmean10d(:,4),Rmean10d(:,5),Rmean10d(:,6));
xlabel('UMAP component 1')
ylabel('UMAP component 2')
zlabel('UMAP component 3')

%%
% =====================================================================
% mSPoC COMPUTATION
% =====================================================================
% Получаем фильтры X (W), направления Y/UMAP (Vz), паттерны (A) и корреляции
[W, Vz, ~, A, Az, out] = my_mspoc(X_epo, Rmean10d');

Epochs_cov = [];
for i=1:size(X_epo,3)
    Epochs_cov(:,:,i) = cov(X_epo(:,:,i));
end

corrs = [];
for k = 1:size(W,2)
    Zpr = Vz(:,k)' * Rmean10d';
    Env = zeros(1, size(X_epo,3));
    for ep_idx = 1:size(X_epo,3)
        Env(ep_idx) = (W(:, k)' * Epochs_cov(:,:,ep_idx) * W(:, k));
    end
    corrs(k) = corr(Env', Zpr');
end
corrs

[corrs_sorted_abs, sort_idx] = sort(abs(corrs), 'descend');
corrs_sorted = corrs(sort_idx); 
W_sorted  = W(:, sort_idx);
A_sorted  = A(:, sort_idx);
Vz_sorted = Vz(:, sort_idx) .* sign(corrs_sorted);
Az_sorted = Az(:, sort_idx) .* sign(corrs_sorted);

% % === РАСЧЕТ P-VALUES ===
% p_values = zeros(size(corrs_sorted));
% for i = 1:length(corrs_sorted)
%     p_values(i) = sum(max_corr_null >= abs(corrs_sorted(i))) / n_perms;
% end
% 
% % === ВИЗУАЛИЗАЦИЯ НУЛЬ-РАСПРЕДЕЛЕНИЯ ===
% figure('Color', 'w', 'Name', 'Permutation Test Null Distribution');
% histogram(max_corr_null, 30, 'Normalization', 'probability', 'FaceColor', [0.7 0.7 0.7]);
% hold on; grid on;
% colors = lines(10); % Выделим топ-3 истинных компоненты
% for i = 1:min(10, length(corrs_sorted))
%     xline(abs(corrs_sorted(i)), 'Color', colors(i,:), 'LineWidth', 2.5, ...
%         'Label', sprintf('C%d (p=%.3f)', i, p_values(i)), 'LabelOrientation', 'horizontal');
% end
% xlabel('Maximum Absolute Correlation');
% ylabel('Probability');
% title('Max-Statistic Null Distribution (Raw Time Series Shift)');

%% =====================================================================
% ПОДГОТОВКА ДАННЫХ ДЛЯ ВИЗУАЛИЗАЦИИ
% =====================================================================
x = Rmean3d(:,1);
y = Rmean3d(:,2);
z = Rmean3d(:,3);
num_clusters = 22;
cmap = jet(num_clusters);
% Центры кластеров всё равно нужны для 2D‑панели (номера условий)
ccx = zeros(1, num_clusters);
ccy = zeros(1, num_clusters);
mask = 1:N_epoch_trial;

for i = 1:num_clusters
    if mask(end) <= numel(x)
        sc_x = Rmean2d(mask,1); sc_y = Rmean2d(mask,2);
    else
        sc_x = Rmean2d(mask(1):end,1); sc_y = Rmean2d(mask(1):end,2);
    end
    mask = mask + N_epoch_trial;
    ccx(i) = median(sc_x); ccy(i) = median(sc_y); 
end

% Масштабный коэффициент для стрелок Vz/Az в 2D
scale_f = max(sqrt(Rmean2d(:,1).^2 + Rmean2d(:,2).^2));   % максимальное расстояние от центра

% =====================================================================
% КОМПЛЕКСНАЯ ВИЗУАЛИЗАЦИЯ (ТОП КОМПОНЕНТЫ)
% =====================================================================
n_to_plot = min(10, length(sort_idx));
tstep = 235; 
ticks = 0:tstep:size(Covs,3);

U = pinv(W_ssd');
for i = 1:n_to_plot
    comp_idx = sort_idx(i);
    r_val    = corrs_sorted(i);
    
    % Переводим фильтры и паттерны из PCA обратно в сенсорное пространство
    wx = U * W(:, comp_idx); % Или используй pinv, если матрица плохо обусловлена
    ax = A_ssd * A(:, comp_idx);    

    % Коррекция знака (максимальный по модулю вес делаем положительным)
    [~, max_idx] = max(abs(wx));
    wx = wx .* sign(wx(max_idx));
    [~, max_idx] = max(abs(ax));
    ax = ax .* sign(ax(max_idx));
    
    % Вычисляем динамику мощности источника (S_pow)
    S_pow = zeros(1, size(Covs,3));
    for j = 1:size(Covs,3)
        S_pow(j) = (W(:, comp_idx)' * Covs(:,:,j) * W(:, comp_idx));
    end
    S_raw = S_pow; 
    S_pow = (S_pow - mean(S_pow)) / std(S_pow); 
    
    % Каноническая проекция UMAP на найденное направление
    zz = Rmean10d * Vz_sorted(:, i);
    zz = (zz - mean(zz)) / std(zz) * sign(r_val);
    
    % -----------------------------------------------------------------
    % СОЗДАНИЕ ФИГУРЫ И СЕТКИ 2x4
    % -----------------------------------------------------------------
    fig_name = sprintf('Component %d (Rank %d) | Corr = %.3f', comp_idx, i, r_val);
    figure('Color', 'w', 'Position', [50+i*30, 50+i*30, 1600, 700], 'Name', fig_name);
    
    t = tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    % [1] 3D UMAP — ТОЛЬКО ТОЧКИ
    ax_umap = nexttile(t, 1);
    hold(ax_umap, 'on'); grid(ax_umap, 'on');
    
    mask = 1:N_epoch_trial;
    for c = 1:num_clusters
        if mask(end) <= numel(x)
            sc_x = x(mask); sc_y = y(mask); sc_z = z(mask);
        else
            sc_x = x(mask(1):end); sc_y = y(mask(1):end); sc_z = z(mask(1):end);
        end
        mask = mask + N_epoch_trial;
        scatter3(ax_umap, sc_x, sc_y, sc_z, 15, repmat(cmap(c,:), length(sc_x), 1), ...
            'filled', 'MarkerFaceAlpha', 0.4);
    end
    
    view(ax_umap, -45, 30);
    xlabel(ax_umap, 'UMAP 1'); ylabel(ax_umap, 'UMAP 2'); zlabel(ax_umap, 'UMAP 3');
    title(ax_umap, '3D UMAP Embedding', 'FontSize', 14);
    
    % [2] 2D ИЗОЛИНИИ ИСТИННОЙ МОЩНОСТИ ИСТОЧНИКА
    ax_iso = nexttile(t, 2);
    hold(ax_iso, 'on');
    
    MWS_raw = movmean(S_raw', 20);
    
    % Определяем границы данных с небольшим отступом
    x_min = min(Rmean2d(:,1));  x_max = max(Rmean2d(:,1));
    y_min = min(Rmean2d(:,2));  y_max = max(Rmean2d(:,2));
    dx = x_max - x_min;  dy = y_max - y_min;
    pad_factor = 0.1;
    x_min_pad = x_min - pad_factor*dx;    x_max_pad = x_max + pad_factor*dx;
    y_min_pad = y_min - pad_factor*dy;    y_max_pad = y_max + pad_factor*dy;
    
    n_edge = 10;
    x_edge = [linspace(x_min_pad, x_max_pad, n_edge)';     
              linspace(x_min_pad, x_max_pad, n_edge)';     
              repmat(x_min_pad, n_edge, 1);                
              repmat(x_max_pad, n_edge, 1)];               
    y_edge = [repmat(y_min_pad, n_edge, 1);
              repmat(y_max_pad, n_edge, 1);
              linspace(y_min_pad, y_max_pad, n_edge)';
              linspace(y_min_pad, y_max_pad, n_edge)'];
    x_corn = [x_min_pad; x_min_pad; x_max_pad; x_max_pad];
    y_corn = [y_min_pad; y_max_pad; y_min_pad; y_max_pad];
    x_edge = [x_edge; x_corn];  
    y_edge = [y_edge; y_corn];
    
    MWS_edge = ones(size(x_edge)) * min(MWS_raw);   
    X_all = [Rmean2d(:,1); x_edge];
    Y_all = [Rmean2d(:,2); y_edge];
    MWS_all = [MWS_raw; MWS_edge];
    
    F = scatteredInterpolant(X_all, Y_all, MWS_all, 'natural', 'linear');

    xrange = linspace(x_min_pad, x_max_pad, 100);
    yrange = linspace(y_min_pad, y_max_pad, 100);
    [Xg, Yg] = meshgrid(xrange, yrange);
    ProjGrid = F(Xg, Yg);
    % ProjGrid(ProjGrid < 0) = 0;
    ProjGrid(ProjGrid > max(MWS_raw)) = max(MWS_raw);

    pcolor(ax_iso, Xg, Yg, ProjGrid);
    shading(ax_iso, 'interp');
    colormap(ax_iso);
    cbar = colorbar(ax_iso);
    cbar.Label.String = 'Source power';
    contour(ax_iso, Xg, Yg, ProjGrid, 10, 'k', 'LineWidth', 0.05);
    
    % Центры кластеров на 2D‑карте
    for c = 1:num_clusters
        text(ax_iso, ccx(c), ccy(c), num2str(c), 'FontWeight','bold', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'BackgroundColor', [1 1 1 0.7], 'Margin', 1);
    end
    
    % Векторы Vz и Az в 2D
    vz_2d = Vz_sorted(1:2, i);
    az_2d = Az_sorted(1:2, i);
    vz_2d = vz_2d / norm(vz_2d) * scale_f * 0.4;
    az_2d = az_2d / norm(az_2d) * scale_f * 0.4;
    
    quiver(ax_iso, 0, 0, vz_2d(1), vz_2d(2), ...
        'Color', 'k', 'LineWidth', 2, 'MaxHeadSize', 0.5, 'AutoScale', 'off');
    text(ax_iso, vz_2d(1)*1.1, vz_2d(2)*1.1, 'V_z', 'Color', 'k', 'FontWeight', 'bold');
    quiver(ax_iso, 0, 0, az_2d(1), az_2d(2), ...
        'Color', 'r', 'LineWidth', 3, 'MaxHeadSize', 0.5, 'AutoScale', 'off');
    text(ax_iso, az_2d(1)*1.1, az_2d(2)*1.1, 'A_z', 'Color', 'r', 'FontWeight', 'bold');

    xlabel(ax_iso, 'UMAP 1'); ylabel(ax_iso, 'UMAP 2');
    title(ax_iso, 'Isolines of Source Activation', 'FontSize', 14);
    axis(ax_iso, 'tight');
    
    % [3] ФИЛЬТР (W)
    ax_w = nexttile(t, 3);
    topo.avg = wx;
    cfg.figure = ax_w;
    cfg.colorbar = 'EastOutside';
    ft_topoplotER(cfg, topo);
    title(ax_w, 'Spatial Filter', 'FontSize', 14);
    
    % [4] ПАТТЕРН (A)
    ax_p = nexttile(t, 4);
    topo.avg = ax;
    cfg.figure = ax_p;
    cfg.colorbar = 'EastOutside';
    ft_topoplotER(cfg, topo);
    title(ax_p, 'Spatial Pattern', 'FontSize', 14);
    
    % [5] ДИНАМИКА ИСТОЧНИКА VS ПРОЕКЦИЯ UMAP
    ax_dyn = nexttile(t, 5, [1 4]);
    hold(ax_dyn, 'on'); grid(ax_dyn, 'on');
    
    plot(ax_dyn, S_pow, 'LineWidth', 1.5, 'Color', [0 0.447 0.741]);
    plot(ax_dyn, zz * sign(r_val), 'LineWidth', 1.5, 'Color', [0.850 0.325 0.098]);
    
    xticks(ax_dyn, ticks(1:end-1));
    xticklabels(ax_dyn, conditions);
    xtickangle(ax_dyn, 45);
    xlim(ax_dyn, [0 ticks(end)]);
    
    for k = 1:length(ticks(2:end-1))
        xline(ax_dyn, ticks(k), '--', 'Color', [0.3 0.3 0.3]);
    end
    
    legend(ax_dyn, {'Source Power Envelope', 'UMAP Canonical Target'}, 'Location', 'best');
    title(ax_dyn, sprintf('Component Dynamics (Correlation = %.3f)', r_val), 'FontSize', 14);
    ylabel(ax_dyn, 'Amplitude (Z-score)');
end

%% =====================================================================
% АГРЕГИРОВАННАЯ КАРТА ДОМИНИРУЮЩИХ ИСТОЧНИКОВ (ТОП-K)
% =====================================================================
n_agg = min(5, length(sort_idx));   % число отображаемых источников (можно менять)
comp_indices = sort_idx(1:n_agg);   % индексы выбранных компонент

% 1. Собираем сглаженные мощности для выбранных компонент
fprintf('Сбор мощностей для агрегированной карты...\n');
S_all = zeros(size(Covs,3), n_agg);   % окна × компоненты
for k = 1:n_agg
    idx = comp_indices(k);
    w = U * W_sorted(:, idx);
    pow = zeros(size(Covs,3), 1);
    for j = 1:size(Covs,3)
        pow(j) = log(w' * U * Covs(:,:,j) * U' * w);
    end
    S_all(:,k) = movmean(pow, 20);   % сглаживание, как раньше
end

% 2. Нормализация (z-score) – каждая компонента отдельно
S_z = (S_all - mean(S_all,1)) ./ std(S_all,0,1);

% 3. Границы сетки (используем те же, что и раньше)
x_min = min(Rmean2d(:,1));  x_max = max(Rmean2d(:,1));
y_min = min(Rmean2d(:,2));  y_max = max(Rmean2d(:,2));
dx = x_max - x_min;  dy = y_max - y_min;
pad_factor = 0.1;
x_min_pad = x_min - pad_factor*dx;    x_max_pad = x_max + pad_factor*dx;
y_min_pad = y_min - pad_factor*dy;    y_max_pad = y_max + pad_factor*dy;
xrange = linspace(x_min_pad, x_max_pad, 100);
yrange = linspace(y_min_pad, y_max_pad, 100);
[Xg, Yg] = meshgrid(xrange, yrange);

% 4. Интерполяция z-оценок для каждой компоненты и определение победителя
winner_grid = zeros(size(Xg));   % индекс доминирующей компоненты
max_z_grid  = zeros(size(Xg));   % её z-оценка (для прозрачности/яркости)

% Добавляем паддинг нулями по краям (чтобы интерполяция была стабильной)
x_edge = [linspace(x_min_pad, x_max_pad, 10)';     
          linspace(x_min_pad, x_max_pad, 10)';     
          repmat(x_min_pad, 10, 1);                
          repmat(x_max_pad, 10, 1)];               
y_edge = [repmat(y_min_pad, 10, 1);
          repmat(y_max_pad, 10, 1);
          linspace(y_min_pad, y_max_pad, 10)';
          linspace(y_min_pad, y_max_pad, 10)'];
x_corn = [x_min_pad; x_min_pad; x_max_pad; x_max_pad];
y_corn = [y_min_pad; y_max_pad; y_min_pad; y_max_pad];
x_edge = [x_edge; x_corn];  y_edge = [y_edge; y_corn];
X_all = [Rmean2d(:,1); x_edge];
Y_all = [Rmean2d(:,2); y_edge];

Z_grid = zeros([size(Xg), n_agg]);
for k = 1:n_agg
    % значения на краях делаем минимальными, чтобы победитель был только внутри облака
    edge_vals = min(S_z(:,k)) * ones(size(x_edge));
    V_all = [S_z(:,k); edge_vals];
    F = scatteredInterpolant(X_all, Y_all, V_all, 'natural', 'linear');
    Z_grid(:,:,k) = F(Xg, Yg);
end

[~, winner_grid] = max(Z_grid, [], 3);   % индекс победителя (1..n_agg)

% 5. Визуализация
figure('Color', 'w', 'Position', [100, 100, 1200, 900]);
ax_agg = axes;
hold(ax_agg, 'on');

% Цветовая палитра для источников (качественная)
cmap_sources = lines(n_agg);
colormap(ax_agg, cmap_sources);

% Рисуем pcolor с дискретными значениями
pcolor(ax_agg, Xg, Yg, winner_grid);
shading(ax_agg, 'flat');
caxis(ax_agg, [0.5, n_agg+0.5]);   % чтобы каждый цвет занимал ровно один индекс

% Цветовая шкала с подписями источников
cbar = colorbar(ax_agg);
cbar.Ticks = 1:n_agg;
cbar.TickLabels = compose("Source %d", 1:n_agg);
cbar.Label.String = 'Dominant source';

% Добавляем контуры для каждой компоненты (тонкие линии своего цвета)
for k = 1:n_agg
    contour(ax_agg, Xg, Yg, Z_grid(:,:,k), 5, 'Color', cmap_sources(k,:), 'LineWidth', 1);
end

% Центры кластеров условий
for c = 1:num_clusters
    text(ax_agg, ccx(c), ccy(c), num2str(c), 'FontWeight','bold', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'BackgroundColor', [1 1 1 0.7], 'Margin', 1);
end

% Векторы Vz и Az для всех отображаемых источников
for k = 1:n_agg
    idx = comp_indices(k);
    vz_2d = Vz_sorted(1:2, idx);
    az_2d = Az_sorted(1:2, idx);
    % масштабирование
    vz_2d = vz_2d / norm(vz_2d) * scale_f * 0.3;
    az_2d = az_2d / norm(az_2d) * scale_f * 0.3;
    quiver(ax_agg, 0, 0, vz_2d(1), vz_2d(2), ...
        'Color', cmap_sources(k,:), 'LineWidth', 2, 'MaxHeadSize', 0.5, 'AutoScale', 'off');
    quiver(ax_agg, 0, 0, az_2d(1), az_2d(2), ...
        'Color', cmap_sources(k,:), 'LineWidth', 1.5, 'LineStyle', '--', 'MaxHeadSize', 0.5, 'AutoScale', 'off');
    % подписи к векторам (только для первого, чтобы не загромождать)
    if k == 1
        text(ax_agg, vz_2d(1)*1.1, vz_2d(2)*1.1, 'V_z', 'Color', cmap_sources(k,:), 'FontWeight', 'bold');
        text(ax_agg, az_2d(1)*1.1, az_2d(2)*1.1, 'A_z', 'Color', cmap_sources(k,:), 'FontWeight', 'bold');
    end
end

xlabel(ax_agg, 'UMAP 1'); ylabel(ax_agg, 'UMAP 2');
title(ax_agg, sprintf('Dominant mSPoC sources (top %d)', n_agg), 'FontSize', 14);
axis(ax_agg, 'tight');
grid(ax_agg, 'on');
hold off;

%%
%%
%%
%%
%% =====================================================================
% PERMUTATION TEST (RAW TIME SERIES CIRCULAR SHIFT)
% =====================================================================
n_perms = 100; % Начни со 100 для оценки времени, для статьи лучше 500-1000

total_samples = size(Xfiltpca, 2);
min_shift = Fs * 10; % Минимальный сдвиг - 10 секунд, чтобы точно разорвать автокорреляцию
shifts = randi([min_shift, total_samples - min_shift], n_perms, 1);

max_corr_null = zeros(n_perms, 1);

% Получаем размерности из оригинальных данных
n_trials = size(Epochs_filt, 3);
samples_per_trial = size(Epochs_filt, 2);
n_comps_pca = size(Xfiltpca, 1);

fprintf('Running %d permutations on RAW TIME SERIES (Full Pipeline)...\n', n_perms);

parfor_progress(n_perms);
parfor (p = 1:n_perms, 3)
% for p = 1:n_perms
    % fprintf('Permutation %d/%d...\n', p, n_perms);
    
    % 1. Циркулярный сдвиг непрерывного сигнала в PCA-пространстве
    Xfiltpca_shifted = circshift(Xfiltpca, shifts(p), 2);
    
    % Возвращаем структуру 3D-массива (компоненты х время х условия)
    Epfilt_pca_shifted = reshape(Xfiltpca_shifted, [n_comps_pca, samples_per_trial, n_trials]);
    
    % 2. Повторная нарезка на эпохи
    X_epo_perm = [];
    for i = 1:n_trials
        ep_wins = epoch_data(Epfilt_pca_shifted(:,:,i)', Fs, Wsize, Ssize);
        X_epo_perm = cat(3, X_epo_perm, ep_wins); 
    end
    
    % 3. Повторный расчет ковариационных матриц
    Epochs_cov_perm = zeros(n_comps_pca, n_comps_pca, size(X_epo_perm,3));
    for i = 1:size(X_epo_perm,3)
        Epochs_cov_perm(:,:,i) = cov(X_epo_perm(:,:,i));
    end
       
    % 6. mSPoC на полностью сдвинутых данных
    [W_perm, Vz_perm] = my_mspoc(X_epo_perm, Rmean');
    
    % 7. Извлечение корреляций
    corrs_perm = zeros(1, size(W_perm, 2));
    for k = 1:size(W_perm,2)
        Zpr_perm = Vz_perm(:,k)' * Rmean';
        Env_perm = zeros(1, size(X_epo_perm,3));
        for ep_idx = 1:size(X_epo_perm,3)
            Env_perm(ep_idx) = log(W_perm(:, k)' * Epochs_cov_perm(:,:,ep_idx) * W_perm(:, k));
        end
        corrs_perm(k) = corr(Env_perm', Zpr_perm');
    end
    
    % 8. Сохраняем МАКСИМАЛЬНУЮ корреляцию по всем найденным компонентам (контроль FWER)
    max(abs(corrs_perm))
    max_corr_null(p) = max(abs(corrs_perm));
    parfor_progress;
end
parfor_progress(0);

%%
