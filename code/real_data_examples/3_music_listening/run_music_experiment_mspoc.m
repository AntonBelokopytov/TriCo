close all
clear
clc

ft_path = 'C:\Users\ansbel\Documents\2Git\fieldtrip';
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

for i = 1:numel(Xinf.trial)
    Ep_raw = Xinf.trial{i}(1:n_channels,:);            
    Epfilt  = filtfilt(b,a,Ep_raw')';    % Zero-phase filtering
    Epfilt = Epfilt(:,Fs/2:end-Fs/2);    % Trim edges
    
    Xfilt = cat(2,Xfilt,Epfilt);         % Concatenate for SVD
    Epochs_filt(:,:,i) = Epfilt;         % Store filtered epoch
end

%% =====================================================================
% SVD AND PCA
% =====================================================================
[U,S,~] = svd(Xfilt,'econ');           % Singular Value Decomposition
S = diag(S);

% Estimate effective rank
tol = max(size(Xfilt)) * eps(S(1));
r = sum(S > tol);

% Cumulative variance explained
ve = S.^2;
var_explained = cumsum(ve) / sum(ve);
var_explained(end) = 1;

% Number of components explaining at least 99% variance
n_components = find(var_explained>=0.99, 1);
n_components = max(min(n_components, r), 1);
U = U(:,1:n_components);               % Keep relevant PCA components

% Project epochs onto PCA components
Epfilt_pca = [];
for i = 1:size(Epochs_filt,3)
    Epfilt_pca(:,:,i) = U'*Epochs_filt(:,:,i);
end
Xfiltpca = U'*Xfilt;

%% =====================================================================
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

Tcovs = Tangent_space(Covs);           % Tangent space projection
N_epoch_trial = size(ep_wins,3);

%% =====================================================================
% UMAP EMBEDDING
% =====================================================================
clear u
u = UMAP("n_neighbors",20,"n_components",3,"min_dist",0);
u.metric = 'euclidean';
u.target_metric = 'euclidean';

% Low-dimensional embedding of epochs
R = u.fit_transform(Tcovs');
Rmean = R - mean(R,1);

%% =====================================================================
% VISUALIZE UMAP
% =====================================================================
figure
set(gcf, 'Color', 'w');
scatter3(R(:,1),R(:,2),R(:,3));
xlabel('UMAP component 1')
ylabel('UMAP component 2')
zlabel('UMAP component 3')

%% =====================================================================
% Temporal evolution of UMAP components
% =====================================================================
figure;
plot(R)                                  % Plot each UMAP component over epochs
set(gcf, 'Color', 'w');

% Define tick positions separating experimental conditions
tstep = 235;                             % number of windows per condition
ticks = 0:tstep:size(R,1);

% Corresponding time labels (in seconds)
times = 0:120:22*120;                    % cumulative duration across conditions
xticks(ticks)
xticklabels(times)
xlim([0,size(R,1)])
ylabel('UMAP component coordinate')
xlabel('time, sec')
legend({'component 1', 'component 2', 'component 3'})

%% =====================================================================
% mSPoC COMPUTATION
% =====================================================================
% Получаем фильтры X (W), направления Y/UMAP (Vz), паттерны (A) и корреляции
[W, Vz, ~, A, ~, out] = my_mspoc(X_epo, Rmean');
corrs = out.corr_values;

[corrs_sorted_abs, sort_idx] = sort(abs(corrs), 'descend');
corrs_sorted = corrs(sort_idx); 
W_sorted  = W(:, sort_idx);
A_sorted  = A(:, sort_idx);
Vz_sorted = Vz(:, sort_idx);

%% =====================================================================
% ПОДГОТОВКА ДАННЫХ ДЛЯ ВИЗУАЛИЗАЦИИ
% =====================================================================
x = Rmean(:,1);
y = Rmean(:,2);
z = Rmean(:,3);
num_clusters = 22;
cmap = jet(num_clusters);
ccx = zeros(1, num_clusters);
ccy = zeros(1, num_clusters);
ccz = zeros(1, num_clusters);
mask = 1:N_epoch_trial;

% --- Вычисляем центры кластеров для 3D графика ---
for i = 1:num_clusters
    if mask(end) <= numel(x)
        sc_x = x(mask); sc_y = y(mask); sc_z = z(mask);
    else
        sc_x = x(mask(1):end); sc_y = y(mask(1):end); sc_z = z(mask(1):end);
    end
    mask = mask + N_epoch_trial;
    ccx(i) = mean(sc_x); ccy(i) = mean(sc_y); ccz(i) = mean(sc_z);
end

% =====================================================================
% КОМПЛЕКСНАЯ ВИЗУАЛИЗАЦИЯ (ТОП КОМПОНЕНТЫ)
% =====================================================================
n_to_plot = min(10, length(sort_idx));
tstep = 235; 
ticks = 0:tstep:size(Covs,3);

for i = 1:n_to_plot
    comp_idx = sort_idx(i);
    r_val    = corrs_sorted(i);
    
    % Переводим фильтры и паттерны из PCA обратно в сенсорное пространство
    wx = U * W_sorted(:, i);
    ax = U * A_sorted(:, i);
    
    % Коррекция знака (максимальный по модулю вес делаем положительным)
    [~, max_idx] = max(abs(wx));
    wx = wx .* sign(wx(max_idx));
    [~, max_idx] = max(abs(ax));
    ax = ax .* sign(ax(max_idx));
    
    % Вектор направления UMAP
    v = Vz_sorted(:, i);
    v = v / norm(v); % Нормализуем для отрисовки стрелки
    
    % Вычисляем динамику мощности источника (S_pow)
    S_pow = zeros(1, size(Covs,3));
    for j = 1:size(Covs,3)
        S_pow(j) = wx' * U * Covs(:,:,j) * U' * wx;
    end
    S_pow = (S_pow - mean(S_pow)) / std(S_pow);
    
    % Каноническая проекция UMAP на найденное направление
    zz = Vz_sorted(:, i)' * Rmean';
    zz = (zz - mean(zz)) / std(zz) * sign(r_val);
    
    % -----------------------------------------------------------------
    % СОЗДАНИЕ ФИГУРЫ И СЕТКИ 2x4
    % -----------------------------------------------------------------
    fig_name = sprintf('Component %d (Rank %d) | Corr = %.3f', comp_idx, i, r_val);
    figure('Color', 'w', 'Position', [50+i*30, 50+i*30, 1600, 700], 'Name', fig_name);
    
    t = tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    % [1] БЛОК СЛЕВА (занимает 2х2): 3D UMAP И ВЕКТОР НАПРАВЛЕНИЯ
    ax_umap = nexttile(t, 1, [2 2]);
    hold(ax_umap, 'on'); grid(ax_umap, 'on');
    
    % Линия, соединяющая центры кластеров
    plot3(ax_umap, ccx, ccy, ccz, 'k', 'LineWidth', 1);
    
    % Раскрашиваем точки облака и добавляем центры с легендой
    legend_handles = gobjects(num_clusters, 1);
    mask = 1:N_epoch_trial;
    for c = 1:num_clusters
        if mask(end) <= numel(x)
            sc_x = x(mask); sc_y = y(mask); sc_z = z(mask);
        else
            sc_x = x(mask(1):end); sc_y = y(mask(1):end); sc_z = z(mask(1):end);
        end
        mask = mask + N_epoch_trial;
        
        % Раскрашенные точки эпох
        scatter3(ax_umap, sc_x, sc_y, sc_z, 15, repmat(cmap(c,:), length(sc_x), 1), 'filled', 'MarkerFaceAlpha', 0.4);
        
        % Крупные центры кластеров (сохраняем handle для легенды)
        legend_handles(c) = scatter3(ax_umap, ccx(c), ccy(c), ccz(c), 150, cmap(c,:), 'filled', 'MarkerEdgeColor', 'k');
        
        % Подпись номера прямо на центре
        text(ax_umap, ccx(c), ccy(c), ccz(c), num2str(c), 'FontSize', 12, 'FontWeight', 'bold', ...
            'BackgroundColor', [0.95 0.95 0.95], 'HorizontalAlignment', 'center', 'Margin', 1);
    end
    
    % Добавляем легенду с текстовыми подписями условий к 3D графику
    legend(ax_umap, legend_handles, conditions, 'Location', 'westoutside', 'FontSize', 9);
    
    % Вектор направления (Красная стрелка из центра)
    scale_f = max(abs(Rmean(:))) * 0.9;
    quiver3(ax_umap, 0, 0, 0, v(1)*scale_f, v(2)*scale_f, v(3)*scale_f, ...
        'Color', 'r', 'LineWidth', 4, 'MaxHeadSize', 0.5, 'AutoScale', 'off');
    text(ax_umap, v(1)*scale_f*1.1, v(2)*scale_f*1.1, v(3)*scale_f*1.1, ...
        sprintf('Vz (r=%.2f)', r_val), 'Color', 'r', 'FontSize', 14, 'FontWeight', 'bold');
        
    view(ax_umap, -45, 30);
    xlabel(ax_umap, 'UMAP 1'); ylabel(ax_umap, 'UMAP 2'); zlabel(ax_umap, 'UMAP 3');
    title(ax_umap, 'Canonical Direction in UMAP Space', 'FontSize', 14);
    
    % [2] ВЕРХ СПРАВА: ФИЛЬТР (W)
    ax_w = nexttile(t, 3);
    topo.avg = wx;
    cfg.figure = ax_w;
    cfg.colorbar = 'EastOutside';
    ft_topoplotER(cfg, topo); 
    title(ax_w, 'Spatial Filter', 'FontSize', 14);
    
    % [3] ВЕРХ СПРАВА: ПАТТЕРН (A)
    ax_p = nexttile(t, 4);
    topo.avg = ax;
    cfg.figure = ax_p;
    cfg.colorbar = 'EastOutside';
    ft_topoplotER(cfg, topo);
    title(ax_p, 'Spatial Pattern', 'FontSize', 14);
    
    % [4] НИЗ СПРАВА: ДИНАМИКА ИСТОЧНИКА VS ПРОЕКЦИЯ UMAP
    ax_dyn = nexttile(t, 7, [1 2]);
    hold(ax_dyn, 'on'); grid(ax_dyn, 'on');
    
    plot(ax_dyn, S_pow, 'LineWidth', 1.5, 'Color', [0 0.447 0.741]); 
    plot(ax_dyn, zz, 'LineWidth', 1.5, 'Color', [0.850 0.325 0.098]);
    
    xticks(ax_dyn, ticks(1:end-1));
    xticklabels(ax_dyn, conditions);
    xtickangle(ax_dyn, 45);
    xlim(ax_dyn, [0 ticks(end)]);
    
    % Вертикальные разделители условий
    for k = 1:length(ticks(2:end-1))
        xline(ax_dyn, ticks(k), '--', 'Color', [0.3 0.3 0.3]);
    end
    
    legend(ax_dyn, {'Source Power Envelope', 'UMAP Canonical Target'}, 'Location', 'best');
    title(ax_dyn, sprintf('Component Dynamics (Correlation = %.3f)', r_val), 'FontSize', 14);
    ylabel(ax_dyn, 'Amplitude (Z-score)');
end

%% =====================================================================
