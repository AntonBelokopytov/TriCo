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
% 1. ЛОКАЛЬНЫЙ АНАЛИЗ ПРОСТРАНСТВА (АТЛАС КАРТ)
% =====================================================================
n_regions = 10;
[region_idx, region_centers] = kmeans(Rmean, n_regions, 'Replicates', 10);

% Визуализация областей
figure('Color','w','Name','UMAP Local Regions', 'Position', [100 100 800 600]);
cmap_reg = lines(n_regions);
for r = 1:n_regions
    idx = (region_idx == r);
    scatter3(Rmean(idx,1), Rmean(idx,2), Rmean(idx,3), 20, cmap_reg(r,:), 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
end
scatter3(region_centers(:,1), region_centers(:,2), region_centers(:,3), 200, 'kx', 'LineWidth', 3);
title('UMAP Partitioned into Local Coordinate Charts');
xlabel('UMAP 1'); ylabel('UMAP 2'); zlabel('UMAP 3');
legend(arrayfun(@(x) sprintf('Region %d', x), 1:n_regions, 'UniformOutput', false));
view(-45, 30); grid on;

%% 2. ЗАПУСК ЛОКАЛЬНОГО mSPoC ДЛЯ КАЖДОГО РЕГИОНА
W_loc = cell(n_regions, 1);
A_loc = cell(n_regions, 1);
Vz_loc = cell(n_regions, 1);
corrs_loc = cell(n_regions, 1);

for r = 1:n_regions
    mask = (region_idx == r);
    if sum(mask) < 20
        warning(['Регион ', num2str(r), ' содержит слишком мало точек (', num2str(sum(mask)), ').']);
        continue;
    end
    
    X_epo_loc = X_epo(:, :, mask);
    Rmean_loc = Rmean(mask, :)';
    
    [W_c, Vz_c, ~, A_c, ~, out_c] = my_mspoc(X_epo_loc, Rmean_loc);
    
    % Сортировка по силе корреляции внутри региона
    [~, sort_idx] = sort(abs(out_c.corr_values), 'descend');
    
    W_loc{r} = U * W_c(:, sort_idx);
    A_loc{r} = U * A_c(:, sort_idx);
    Vz_loc{r} = Vz_c(:, sort_idx);
    corrs_loc{r} = out_c.corr_values(sort_idx);
    
    % Коррекция знака
    for c = 1:size(W_loc{r}, 2)
        [~, max_idx] = max(abs(W_loc{r}(:, c)));
        sgn = sign(W_loc{r}(max_idx, c));
        W_loc{r}(:, c) = W_loc{r}(:, c) * sgn;
        A_loc{r}(:, c) = A_loc{r}(:, c) * sgn;
        % Направление Vz не инвертируем здесь, оно привязано к корреляции
    end
end

%% 3. ВИЗУАЛИЗАЦИЯ ТОП-3 ИСТОЧНИКОВ ДЛЯ КАЖДОГО РЕГИОНА
ticks = 0:235:size(Covs,3); 

for r = 1:n_regions
    if isempty(A_loc{r}), continue; end
    n_to_plot = min(3, size(A_loc{r}, 2));
    
    fig_name = sprintf('Local System: Region %d', r);
    figure('Color', 'w', 'Position', [50+r*20, 50+r*20, 1400, 800], 'Name', fig_name);
    t = tiledlayout(n_to_plot, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    for c = 1:n_to_plot
        wx = W_loc{r}(:, c);
        ax = A_loc{r}(:, c);
        r_val = corrs_loc{r}(c);
        
        % [1] Топография Фильтра
        ax_w = nexttile(t, (c-1)*4 + 1);
        topo.avg = wx; cfg.figure = ax_w; 
        if isfield(cfg, 'zlim'), cfg = rmfield(cfg, 'zlim'); end
        ft_topoplotER(cfg, topo); title(ax_w, sprintf('Filter #%d', c));
        
        % [2] Топография Паттерна
        ax_p = nexttile(t, (c-1)*4 + 2);
        topo.avg = ax; cfg.figure = ax_p; 
        if isfield(cfg, 'zlim'), cfg = rmfield(cfg, 'zlim'); end
        ft_topoplotER(cfg, topo); title(ax_p, sprintf('Pattern #%d', c));
        
        % [3] Вычисление глобальной и локальной мощности
        S_pow_global = zeros(1, size(Covs,3));
        for j = 1:size(Covs,3), S_pow_global(j) = wx' * U * Covs(:,:,j) * U' *  wx; end
        S_pow_global = (S_pow_global - mean(S_pow_global)) / std(S_pow_global);
        
        S_pow_local = NaN(1, size(Covs,3));
        S_pow_local(region_idx == r) = S_pow_global(region_idx == r);
        
        % Вектор Vz на локальных точках
        zz_global = NaN(1, size(Covs,3));
        zz_local = Vz_loc{r}(:, c)' * Rmean(region_idx == r, :)';
        zz_local = (zz_local - mean(zz_local)) / std(zz_local) * sign(r_val);
        zz_global(region_idx == r) = zz_local;
        
        % [4] Динамика (Серое - глобально, Цветное - локально)
        ax_dyn = nexttile(t, (c-1)*4 + 3, [1 2]); hold on; grid on;
        plot(ax_dyn, S_pow_global, 'Color', [0.8 0.8 0.8], 'LineWidth', 1); % Фон
        plot(ax_dyn, S_pow_local, 'Color', [0 0.447 0.741], 'LineWidth', 1.5);
        plot(ax_dyn, zz_global, 'Color', [0.850 0.325 0.098], 'LineWidth', 1.5);
        
        xlim(ax_dyn, [0 ticks(end)]); xticks(ax_dyn, ticks(1:end-1)); 
        xticklabels(ax_dyn, []); % Уберем подписи для компактности
        for tk = 1:length(ticks), xline(ax_dyn, ticks(tk), '--', 'Color', [0.5 0.5 0.5]); end
        title(ax_dyn, sprintf('Dynamics in Region %d (Corr = %.2f)', r, r_val));
    end
    sgtitle(sprintf('ATLAS CHART: Local Coordinates for Region %d', r), 'FontSize', 16, 'FontWeight', 'bold');
end

%% 4. СШИВКА (ALIGNMENT) ИСТОЧНИКОВ МЕЖДУ РЕГИОНАМИ
base_region = 1;
A_base = A_loc{base_region};
n_comps = size(A_base, 2);

matched_A  = zeros(size(A_base, 1), n_comps, n_regions);
matched_W  = zeros(size(A_base, 1), n_comps, n_regions);
matched_Vz = zeros(size(Vz_loc{1}, 1), n_comps, n_regions);

matched_A(:,:,base_region)  = A_base;
matched_W(:,:,base_region)  = W_loc{base_region};
matched_Vz(:,:,base_region) = Vz_loc{base_region};

% Для аналитики: сохраним максимальные корреляции сшивки
match_quality = zeros(n_comps, n_regions);
match_quality(:, base_region) = 1;

for r = 1:n_regions
    if r == base_region || isempty(A_loc{r}), continue; end
    
    A_target = A_loc{r};
    corr_mat = corr(A_base, A_target);
    
    for c = 1:n_comps
        [max_corr, best_target_idx] = max(abs(corr_mat(c, :)));
        sign_flip = sign(corr_mat(c, best_target_idx));
        
        match_quality(c, r) = max_corr;
        
        matched_A(:, c, r)  = A_loc{r}(:, best_target_idx) * sign_flip;
        matched_W(:, c, r)  = W_loc{r}(:, best_target_idx) * sign_flip;
        matched_Vz(:, c, r) = Vz_loc{r}(:, best_target_idx) * sign_flip;
        
        corr_mat(:, best_target_idx) = 0; % Исключаем повторное использование
    end
end

% Вывод качества сшивки
figure('Color', 'w', 'Name', 'Alignment Quality', 'Position', [100 100 600 400]);
imagesc(match_quality); colormap(hot); colorbar; caxis([0 1]);
title('Cross-Region Source Alignment Quality (Absolute Correlation)');
xlabel('Target Region'); ylabel('Base Source Component ID');
xticks(1:n_regions); yticks(1:n_comps);

%% 5. ВЕКТОРНЫЕ ПОЛЯ (ИЗОЛИНИИ) И ЭВОЛЮЦИЯ ПАТТЕРНА
n_matched_to_plot = min(1, n_comps);

% Автоматически рассчитываем ширину сетки 
% (2 колонки под UMAP + нужное количество колонок под топографии)
n_topo_cols = ceil(n_regions / 2);
total_cols = n_topo_cols + 2; 

for c = 1:n_matched_to_plot
    % Динамически расширяем окно фигуры в зависимости от кол-ва колонок
    figure('Color', 'w', 'Name', sprintf('Manifold Vector Field - Source %d', c), ...
           'Position', [100, 100, 400 + total_cols*200, 600]);
    
    t = tiledlayout(2, total_cols, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    % [ЛЕВАЯ ЧАСТЬ] Векторное поле на 3D UMAP (занимает 2х2)
    ax_vf = nexttile(t, 1, [2 2]);
    hold(ax_vf, 'on'); grid(ax_vf, 'on');
    
    % Рисуем бледный UMAP
    scatter3(ax_vf, Rmean(:,1), Rmean(:,2), Rmean(:,3), 10, [0.8 0.8 0.8], 'filled', 'MarkerFaceAlpha', 0.1);
    
    % Генерируем "изолинии" (плотное поле градиентов)
    scale_f = max(abs(Rmean(:))) * 0.15; % Длина стрелочек
    step = 5; % Отрисовываем каждую 5-ю точку для красоты
    
    for p = 1:step:size(Rmean,1)
        r = region_idx(p);
        if isempty(A_loc{r}) || match_quality(c, r) < 0.4, continue; end % Игнорируем плохие сшивки
        
        vz = matched_Vz(:, c, r);
        vz = vz / norm(vz); % Направление градиента
        
        % Отрисовываем мини-стрелочку градиента в этой точке пространства
        quiver3(ax_vf, Rmean(p,1), Rmean(p,2), Rmean(p,3), ...
                vz(1)*scale_f, vz(2)*scale_f, vz(3)*scale_f, ...
                'Color', cmap_reg(r,:), 'LineWidth', 1.5, 'MaxHeadSize', 2, 'AutoScale', 'off');
    end
    
    view(ax_vf, -45, 30);
    title(ax_vf, sprintf('Manifold Gradient Field (Isolines) for Source #%d', c));
    xlabel('UMAP 1'); ylabel('UMAP 2'); zlabel('UMAP 3');
    
    % [ПРАВАЯ ЧАСТЬ] Эволюция топографии паттерна по регионам
    for r = 1:n_regions
        % Динамический расчет индекса ячейки (распределение в 2 ряда)
        if r <= n_topo_cols
            tile_idx = r + 2; % Первая строка (смещение на 2 колонки из-за UMAP)
        else
            tile_idx = total_cols + (r - n_topo_cols) + 2; % Вторая строка
        end
        
        ax_topo = nexttile(t, tile_idx); 
        
        % Если для данного региона нет уверенной сшивки — пропускаем отрисовку
        if match_quality(c, r) < 0.4
            title(ax_topo, sprintf('Reg %d\n(No Match)', r));
            axis(ax_topo, 'off');
            continue;
        end
        
        topo.avg = matched_A(:, c, r);
        cfg.figure = ax_topo;
        if isfield(cfg, 'zlim'), cfg = rmfield(cfg, 'zlim'); end
        ft_topoplotER(cfg, topo);
        
        % Цветной заголовок, соответствующий цвету градиентов на 3D графике
        title(ax_topo, sprintf('Reg %d\n(Match: %.2f)', r, match_quality(c,r)), ...
              'Color', cmap_reg(r,:), 'FontWeight', 'bold');
    end
    sgtitle(sprintf('Global Matched Source #%d: Evolution Across %d Regions', c, n_regions), ...
            'FontSize', 16, 'FontWeight', 'bold');
end