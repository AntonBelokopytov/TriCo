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
    i
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
plot(R)                          % Plot each UMAP component over epochs
set(gcf, 'Color', 'w');

% Define tick positions separating experimental conditions
tstep = 235;                     % number of windows per condition
ticks = 0:tstep:size(R,1);

% Corresponding time labels (in seconds)
times = 0:120:22*120;            % cumulative duration across conditions

xticks(ticks)
xticklabels(times)
xlim([0,size(R,1)])

ylabel('UMAP component coordinate')
xlabel('time, sec')

legend({'component 1', 'component 2', 'component 3'})

%% =====================================================================
% eSPoC COMPUTATION
% =====================================================================

% Run eSPoC
[W, A, Vf, Vz, corrs, ~, ~, Epochs_cov, VecCov] = espoc(X_epo, R');
% [W, Vz, ~, A] = mspoc(X_epo, R');

% Plot correlation values
figure;
set(gcf,'Color','w');

stem(corrs','LineWidth',1.5);
grid on

xlabel('Local component index')
ylabel('Correlation')
title('eSPoC correlation values')

legend({'Global 1','Global 2','Global 3'}, 'Location','best')

xlim([1 size(corrs,2)])

%% =====================================================================
% Canonical projections and cluster structure in canonical space
% =====================================================================
% Global covariance activation (canonical components in feature space)
gl_src = Vf' * VecCov;

% Canonical projection of embedding
emb_can_pr = Vz' * R';

% Correlation between canonical covariance activations
% and canonical embedding projections
corr(gl_src', emb_can_pr')

figure; 
set(gcf,'Color','w');
tiledlayout(3,2)

% =====================================================================
% 3D visualization of canonical space (cluster means)
% =====================================================================

nexttile(1,[3,1])

% Canonical coordinates
x = Vz(:,1)' * Rmean';
y = Vz(:,2)' * Rmean';
z = Vz(:,3)' * Rmean';

num_clusters = 22;
cmap = jet(num_clusters);

ccx=[]; ccy=[]; ccz=[];
mask = 1:N_epoch_trial;

% --- Compute cluster centers ---
for i = 1:num_clusters
    if mask(end) <= numel(x)
        sc_x = x(mask);
        sc_y = y(mask);
        sc_z = z(mask);
    else
        sc_x = x(mask(1):end);
        sc_y = y(mask(1):end);
        sc_z = z(mask(1):end);
    end
    mask = mask + N_epoch_trial;

    ccx = [ccx, mean(sc_x)];
    ccy = [ccy, mean(sc_y)];
    ccz = [ccz, mean(sc_z)];
end

% Connect cluster centers
plot3(ccx, ccy, ccz, 'k', 'LineWidth', 1);
hold on; grid on

legend_handles = gobjects(num_clusters,1);

% --- Plot clusters and their centers ---
mask = 1:N_epoch_trial;
for i = 1:num_clusters

    if mask(end) <= numel(x)
        sc_x = x(mask);
        sc_y = y(mask);
        sc_z = z(mask);
    else
        sc_x = x(mask(1):end);
        sc_y = y(mask(1):end);
        sc_z = z(mask(1):end);
    end
    mask = mask + N_epoch_trial;

    cx = mean(sc_x);
    cy = mean(sc_y);
    cz = mean(sc_z);

    % Cluster points
    scatter3(sc_x, sc_y, sc_z, 10, ...
        repmat(cmap(i,:), length(sc_x), 1), ...
        'filled', ...
        'MarkerFaceAlpha', 0.3);

    % Cluster center (for legend)
    legend_handles(i) = scatter3(cx, cy, cz, 120, cmap(i,:), 'filled');

    % Label cluster index
    text(cx, cy, cz, num2str(i), ...
        'FontSize', 16, ...
        'FontWeight', 'bold', ...
        'Color', 'k', ...
        'BackgroundColor', [0.95 0.95 0.95], ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle');
end

legend(legend_handles, conditions, 'Location', 'northeastoutside');

view(-45, 30);
xlabel('Canonical axis 1')
ylabel('Canonical axis 2')
zlabel('Canonical axis 3')

% =====================================================================
% Temporal dynamics of canonical components
% =====================================================================

tstep = 235;
ticks = 0:tstep:size(R,1);

for i = 1:3

    nexttile(i*2)

    % Normalize signals for visualization
    gl_src_n = (gl_src(i,:) - mean(gl_src(i,:))) / std(gl_src(i,:));
    emb_can_pr_n = (emb_can_pr(i,:) - mean(emb_can_pr(i,:))) / std(emb_can_pr(i,:));

    plot(gl_src_n,'blue')
    hold on
    plot(emb_can_pr_n,'red')

    title(['component ', num2str(i), ...
           ' | corr = ', ...
           num2str(corr(gl_src_n',emb_can_pr_n'),'%.2f')])

    grid()

    xticks(ticks(1:end-1));
    xlim([0, ticks(end)])

    % Condition indices on x-axis
    conditions_num = arrayfun(@(x) ['(' num2str(x) ')'], ...
                              1:22, ...
                              'UniformOutput', false);
    xticklabels(conditions_num);

    if i == 1
        legend('Global source signal', 'UMAP canonical projection')
    end

    if i == 3
        xlabel('Experimental conditions')
    end
end

% Replace numeric labels with condition names
xticklabels(conditions);

%% =====================================================================
% PERMUTATION TEST (CIRCULAR TIME SHIFTS)
% =====================================================================
clear corrmax corrmin

chs = size(Xfiltpca,1);
samples_per_epoch = 59501;
nEpochs = 22;

parfor i=1:1000
    i

    % Circular time shift to destroy temporal alignment
    r_idx = fix(rand*size(Xfiltpca,2));
    XCirc = circshift(Xfiltpca,[0,r_idx]);
    
    Eps_circ = reshape( ...
    XCirc(:, 1:samples_per_epoch*nEpochs), ...
    chs, samples_per_epoch, nEpochs);

    X_test = zeros(size(X_epo));
    en = 0;
    for j=1:nEpochs
        ep_wins = epoch_data(Eps_circ(:,:,j)', Fs, Wsize, Ssize);
        
        st = en + 1;
        en = en + size(ep_wins,3);
        
        X_test(:,:,st:en) = ep_wins; 
    end
    
    [~,~,~,~,corrs] = espoc(X_test, R');
    corrmax(:,i) = max(corrs,[],2);
    corrmin(:,i) = min(corrs,[],2);
end

%% =====================================================================
% SIGNIFICANCE THRESHOLDS
% =====================================================================
corrmax1 = sort(max(corrmax,[],1),'descend');
corrmin1 = sort(min(corrmin,[],1));

alpha = 0.05;
% Determine one-sided alpha = 0.05 thresholds
i=1; while 1 - sum(corrmax1(i) > corrmax1)/numel(corrmax1) <= alpha, i=i+1; end
max_val = corrmax1(i)

% Determine one-sided alpha = 0.05 thresholds
i=1; while 1 - sum(corrmin1(i) < corrmin1)/numel(corrmin1) <= alpha, i=i+1; end
min_val = corrmin1(i)

figure; stem(corrs'); hold on
yline(max_val)
yline(min_val)

%% =====================================================================
% Canonical space visualization (cluster means)
% =====================================================================
x = Vz(:,1)'*Rmean';
y = Vz(:,2)'*Rmean';
z = Vz(:,3)'*Rmean';

% x = Rmean(:,1);
% y = Rmean(:,2);
% z = Rmean(:,3);

figure; set(gcf,'Color','w');

num_clusters = 22;
cmap = jet(num_clusters);

ccx=[]; ccy=[]; ccz=[];
mask = 1:N_epoch_trial;

% --- считаем центры ---
for i = 1:num_clusters
    if mask(end) <= numel(x)
        sc_x = x(mask);
        sc_y = y(mask);
        sc_z = z(mask);
    else
        sc_x = x(mask(1):end);
        sc_y = y(mask(1):end);
        sc_z = z(mask(1):end);
    end
    mask = mask + N_epoch_trial;

    ccx = [ccx, mean(sc_x)];
    ccy = [ccy, mean(sc_y)];
    ccz = [ccz, mean(sc_z)];
end

plot3(ccx, ccy, ccz, 'k', 'LineWidth', 1);
hold on; grid on

% --- легенда будет по центрам ---
legend_handles = gobjects(num_clusters,1);

% --- рисуем кластеры и центры ---
mask = 1:N_epoch_trial;
for i = 1:num_clusters
    if mask(end) <= numel(x)
        sc_x = x(mask);
        sc_y = y(mask);
        sc_z = z(mask);
    else
        sc_x = x(mask(1):end);
        sc_y = y(mask(1):end);
        sc_z = z(mask(1):end);
    end
    mask = mask + N_epoch_trial;

    cx = mean(sc_x);
    cy = mean(sc_y);
    cz = mean(sc_z);

    % точки кластера (без легенды)
    scatter3(sc_x, sc_y, sc_z, 10, ...
        repmat(cmap(i,:), length(sc_x), 1), ...
        'filled', ...
        'MarkerFaceAlpha', 0.3);
    hold on

    % центр (для легенды)
    legend_handles(i) = scatter3(cx, cy, cz, 120, cmap(i,:), 'filled');

    % номер с фоном
    text(cx, cy, cz, num2str(i), ...
        'FontSize', 16, ...
        'FontWeight', 'bold', ...
        'Color', 'k', ... % цвет букв
        'BackgroundColor', [0.95 0.95 0.95], ... % цвет фона
        'Margin', 0.00001, ... % отступ вокруг текста
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle');
end

conditions = {'(1) RS EC 1', '(2) RS EO 1', '(3) 2Hz', '(4) 05Hz', '(5) 4Hz', '(6) 1Hz', '(7) 3Hz', ...
              '(8) NoRy 1','(9) Waltz 1','(10) Waltz 2','(11) NoRy 2','(12) NoRy 3','(13) Waltz 3', ...
              '(14) NoRy 4','(15) Waltz 4','(16) NoRy 5','(17) Waltz 5','(18) RS EC 2','(19) RS EO 2', ...
              '(20) Waltz 6','(21) Waltz 7','(22) Waltz 8'};

legend(legend_handles, conditions, 'Location', 'northeastoutside');

view(-45, 30);

% xlabel('UMAP component 1')
% ylabel('UMAP component 2')
% zlabel('UMAP component 3')
xlabel('Canonical axis 1')
ylabel('Canonical axis 2')
zlabel('Canonical axis 3')

%%
% Select global source and local component
gl_src_idx  = 1;
lcl_src_idx = 1;

% Back-project spatial pattern and spatial filter to sensor space
ax = U*A(gl_src_idx,:,lcl_src_idx)';
wx = U*W(gl_src_idx,:,lcl_src_idx)';
% ax = U*A(:,lcl_src_idx);
% wx = U*W(:,lcl_src_idx);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Fix sign ambiguity (make dominant coefficient positive)
[~, idx] = max(abs(wx));
wx = wx.*sign(wx(idx));

[~, idx] = max(abs(ax));
ax = ax.*sign(ax(idx));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
set(gcf,'Color','w');

% Layout: filter | pattern | dynamics
t = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

% Display correlation value in title
sgtitle(['Source Envelope - UMAP correlation: ', ...
         num2str(corrs(gl_src_idx,lcl_src_idx))])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ---- Filter topography ----
ax1 = nexttile(t,1); 
title(ax1,'Filter');

topo.avg   = wx;
cfg.figure = ax1;
ft_topoplotER(cfg, topo); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ---- Pattern topography ----
ax2 = nexttile(t,2);       
title(ax2,'Pattern');

topo.avg   = ax;
cfg.figure = ax2;
ft_topoplotER(cfg, topo); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ---- Source dynamics vs canonical projection ----
ax3 = nexttile(t,3,[1,2]);
hold on; grid on        
title(ax3,'Latent Source Signal & UMAP canonical projection');

% Compute source power from covariance matrices
S = [];
for i = 1:size(Epochs_cov,3)
    S(i) = wx' * U * Covs(:,:,i) * U' * wx;
end

% Z-score normalization
S = (S-mean(S))/std(S);
plot(S,'LineWidth',1)

% Canonical projection of embedding
zz = Vz(:,gl_src_idx)'*Rmean';
zz = (zz-mean(zz))/std(zz) * sign(corrs(gl_src_idx,lcl_src_idx));
plot(zz,'LineWidth',1,'Color','red')

% Condition boundaries on x-axis
tstep = 235;
ticks = 0:tstep:size(S,2);

xticks(ticks(1:end-1));
xlim([0 ticks(end)])

% Condition labels
conditions = {'(1) RS EC 1','(2) RS EO 1','(3) 2Hz','(4) 0.5Hz',...
              '(5) 4Hz','(6) 1Hz','(7) 3Hz','(8) NoRy 1','(9) Waltz 1',...
              '(10) Waltz 2','(11) NoRy 2','(12) NoRy 3','(13) Waltz 3',...
              '(14) NoRy 4','(15) Waltz 4','(16) NoRy 5','(17) Waltz 5',...
              '(18) RS EC 2','(19) RS EO 2','(20) Waltz 6',...
              '(21) Waltz 7','(22) Waltz 8'};

xticklabels(conditions);
xtickangle(45);

xlabel('Experimental conditions')
ylabel('Source signal')

legend('Source Signal Envelope','UMAP canonical projection')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Настройка фигуры и layout
figure;
set(gcf, 'Color', 'w');

% Layout: 4 строки (по компонентам) × 5 колонок
% [filter | pattern | dynamics(3 columns)]
t = tiledlayout(4,5,'TileSpacing','compact','Padding','compact');

gl_src_idx = 1;   % индекс глобального источника

% Каноническая проекция выбранного глобального источника
zz = Vz(:,gl_src_idx)' * Rmean';
zz = (zz - mean(zz)) / std(zz) * sign(corrs(gl_src_idx,1));

comp_indices = [2 3 4 5];  % локальные компоненты для отображения
all_right_axes = [];       % оси с динамикой (правая часть)

ticks = 0:235:size(S,2);   % границы экспериментальных условий

for i = 1:length(comp_indices)

    lcl_src_idx = comp_indices(i);   % текущий локальный источник

    % --- Spatial pattern & filter (в сенсорном пространстве)
    ax = U*A(gl_src_idx,:,lcl_src_idx)';
    wx = U*W(gl_src_idx,:,lcl_src_idx)';

    % Выравнивание знака
    [~, idx] = max(abs(wx)); wx = wx .* sign(wx(idx));
    [~, idx] = max(abs(ax)); ax = ax .* sign(ax(idx));

    row = (i-1)*5;

    % --- Column 1: spatial filter
    ax_wx = nexttile(t, row + 1);
    topo.avg = wx;
    cfg.figure = ax_wx;
    ft_topoplotER(cfg, topo);
    if i==1
        title('Filter')
    end

    % --- Column 2: spatial pattern
    ax_ax = nexttile(t, row + 2);
    topo.avg = ax;
    cfg.figure = ax_ax;
    ft_topoplotER(cfg, topo);
    if i==1
        title('Pattern')
    end

    % --- Columns 3–5: динамика источника
    ax_plot = nexttile(t, row + 3, [1 3]);
    all_right_axes = [all_right_axes ax_plot];

    % Мощность источника из ковариационных матриц
    S = zeros(1,size(Epochs_cov,3));
    for j = 1:size(Epochs_cov,3)
        S(j) = wx' * U * Covs(:,:,j) * U' * wx;
    end
    S = (S - mean(S)) / std(S);

    plot(ax_plot, S, 'LineWidth', 1); hold on;
    plot(ax_plot, zz, 'LineWidth', 1, 'Color', 'red');
    grid on

    xticks(ax_plot, ticks(1:end-1));
    xticklabels(ax_plot, []);
    xlim([0, ticks(end)])

    title(['corr = ', num2str(corrs(gl_src_idx,lcl_src_idx),'%.2f')])
end

% --- Подписи условий только на последней правой оси
conditions = {'(1) RS EC 1','(2) RS EO 1','(3) 2Hz','(4) 0.5Hz',...
              '(5) 4Hz','(6) 1Hz','(7) 3Hz','(8) NoRy 1','(9) Waltz 1',...
              '(10) Waltz 2','(11) NoRy 2','(12) NoRy 3','(13) Waltz 3',...
              '(14) NoRy 4','(15) Waltz 4','(16) NoRy 5','(17) Waltz 5',...
              '(18) RS EC 2','(19) RS EO 2','(20) Waltz 6',...
              '(21) Waltz 7','(22) Waltz 8'};

conditions_num = arrayfun(@(x) ['(' num2str(x) ')'], 1:22, 'UniformOutput', false);

% Короткие подписи на первых трёх строках
for i = 1:3
    ax = all_right_axes(i);
    xticklabels(ax, conditions_num);
end

% Полные подписи на последней строке
ax_last = all_right_axes(end);
xticklabels(ax_last, conditions);
xtickangle(ax_last, 45);
xlabel('Experimental conditions')

% Вертикальные линии, разделяющие условия
for i = 1:4
    ax = all_right_axes(i);
    for k = 1:length(ticks(2:end-1))
        xline(ax, ticks(k), '--', 'Color', [0.1 0.1 0.1]);
    end
end

legend(all_right_axes(1), ...
       'Source Signal Envelope', ...
       'UMAP canonical projection');

%%

%%

%%
