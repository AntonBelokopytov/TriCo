close all
clear
clc

ft_path = 'D:\OS(CURRENT)\scripts\eSPoC_UMAP\0_2AOS\fieldtrip';

if ~exist('ft_defaults','file')
    addpath(ft_path);
end

ft_defaults;

%% =====================================================================
% LOAD DATA
% =====================================================================
sub_path = 'data/Patient1_OFF_2-35Hz_pp2s_epochs.fif';

cfg = [];
cfg.dataset = sub_path;
Xinf = ft_preprocessing(cfg);    % Load EEG/MEG data
Fs = Xinf.fsample;               % Sampling frequency

% Initialize topography structure
topo = [];
topo.dimord = 'chan_time';
topo.label  = Xinf.elec.label;  
topo.time   = 0;
topo.elec   = Xinf.elec;
topo.time    = 0;

% Prepare FieldTrip layout for topography plotting
laycfg = [];
laycfg.elec = Xinf.elec;
lay = ft_prepare_layout(laycfg);     

cfg.marker       = 'labels';
cfg.layout       = lay;
cfg.comment      = 'no';
cfg.style        = 'fill';
cfg.markersymbol = 'o';
cfg.colorbar     = 'yes'; 
cfg.layout.pos(:, 1:2) = cfg.layout.pos(:, 1:2) * 1.1; 
cfg.layout.pos(:, 2) = cfg.layout.pos(:, 2) - 0.05;

%% =====================================================================
% BANDPASS FILTERING (ALPHA BAND)
% =====================================================================
[b,a] = butter(3,[8,12]/(Fs/2));   % 3rd-order Butterworth filter
n_channels = 38;                   % Only EEG channels

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

%% =====================================================================
% EPOCH SEGMENTATION
% =====================================================================

Wsize = 0.5;  % Window size (s)
Ssize = 0.5;  % Step size (s)

X_epo = []; 
time = [0];

for i = 1:size(Epfilt_pca,3)

    % Segment each trial into short overlapping windows
    ep_wins = epoch_data(Epfilt_pca(:,:,i)', Fs, Wsize, Ssize);

    % Concatenate windows across trials
    X_epo = cat(3, X_epo, ep_wins); 

    % Build time axis (center of each window)
    timeline = time(end) + (Wsize - Ssize) + (1:size(ep_wins,3))/2;
    time = [time, timeline];
end

time = time(2:end);   % remove initial zero

% =====================================================================
% Covariance matrices of segmented epochs
% =====================================================================

Covs = [];
for i = 1:size(X_epo,3)
    Covs(:,:,i) = cov(X_epo(:,:,i));   % covariance of each window
end

% Project covariance matrices to tangent space (Euclidean representation)
Tcovs = Tangent_space(Covs);

% Number of windows per trial (used later for coloring)
N_epoch_trial = size(ep_wins,3);

%% =====================================================================
% UMAP EMBEDDING
% =====================================================================
clear u
u = UMAP("n_neighbors",30,"n_components",3);
u.metric = 'euclidean';
u.target_metric = 'euclidean';

% Low-dimensional embedding of epochs
R = u.fit_transform(Tcovs');

%% =====================================================================
% VISUALIZE UMAP
% =====================================================================
figure
scatter3(R(:,1),R(:,2),R(:,3));
xlabel('UMAP component 1')
ylabel('UMAP component 2')
zlabel('UMAP component 3')

%% =====================================================================
% eSPoC COMPUTATION
% =====================================================================
[W, A, Vf, Vz, corrs, VecCov, Epochs_cov, eigenvalues] = espoc(X_epo, R');
figure; stem(corrs'); legend('1','2','3')

% Zero-mean UMAP coordinates across epochs
Rmean = R - mean(R,1);

%%
% Global source activation in covariance space
gl_src = Vf'*VecCov;
emb_can_pr = Vz'*Rmean';

corr(gl_src',emb_can_pr')

figure; set(gcf,'Color','w');
tiledlayout(3,2)

nexttile(1,[3,1])
x = Vz(:,1)'*Rmean'; y = Vz(:,2)'*Rmean'; z = Vz(:,3)'*Rmean';
start_index = 0; 

cmap = parula(N_epoch_trial); 
data_points_per_iteration = N_epoch_trial;
for i=1:80
    current_indices = start_index + (1:data_points_per_iteration);
    if max(current_indices) > length(x), warning('Index out of bounds.'); break; end
    color_data = 1:data_points_per_iteration;
    scatter3(x(current_indices), y(current_indices), z(current_indices), 30, color_data, 'filled'); 
    hold on
    start_index = start_index + data_points_per_iteration;
end

colormap(cmap);
c = colorbar;
time_step = 0.5;
nPoints = data_points_per_iteration;
time_labels = (1:nPoints) * time_step;
c.Ticks = linspace(1,nPoints,nPoints);
c.TickLabels = arrayfun(@(t) sprintf('%.1f',t), time_labels, 'UniformOutput', false);
c.Label.String = 'Time within trial (s)';

xlabel('UMAP canonical axis 1')
ylabel('UMAP canonical axis 2')
zlabel('UMAP canonical axis 3')

for i=1:3
    nexttile(i*2)

    gl_src_n = gl_src(i,:);
    gl_src_n = (gl_src_n - mean(gl_src_n)) / std(gl_src_n);

    emb_can_pr_n = emb_can_pr(i,:);
    emb_can_pr_n = (emb_can_pr_n - mean(emb_can_pr_n)) / std(emb_can_pr_n);

    plot(gl_src_n,'blue')
    hold on
    plot(emb_can_pr_n,'red')

    title(['component ', num2str(i), ' | ', 'corr: ', round(num2str(corr(gl_src_n',emb_can_pr_n'),2))])
    grid()

    xticks(40:40:size(Rmean,1))
    xticklabels(time(40:40:end))
    
    if i==1
        legend('Global source signal', 'UMAP canonical projection')
    end
    if i==3
        xlabel('t, sec')
    end
    
end

%% =====================================================================
% UMAP SCATTER PLOTS WITH COLOR GRADIENT
% =====================================================================

figure; set(gcf,'Color','w');
t = tiledlayout(1,2);

% ---------------------------------------------------------------------
% Original embedding (UMAP space)
% ---------------------------------------------------------------------

nexttile
x = Rmean(:,1); 
y = Rmean(:,2); 
z = Rmean(:,3);

% Colormap for time progression within each trial
cmap = parula(N_epoch_trial); 

start_index = 0; 
data_points_per_iteration = N_epoch_trial;

% Plot each trial separately, coloring points by time within trial
for i = 1:80
    current_indices = start_index + (1:data_points_per_iteration);

    if max(current_indices) > length(x)
        warning('Index out of bounds.'); 
        break; 
    end

    color_data = 1:data_points_per_iteration;

    scatter3(x(current_indices), ...
             y(current_indices), ...
             z(current_indices), ...
             30, color_data, 'filled'); 
    hold on

    start_index = start_index + data_points_per_iteration;
end

title('Initial UMAP graph embedding')
xlabel('UMAP component 1')
ylabel('UMAP component 2')
zlabel('UMAP component 3')
view(-45, 30);

% ---------------------------------------------------------------------
% Projection onto canonical eSPoC axes
% ---------------------------------------------------------------------

nexttile

% Rotate embedding into canonical space
x = Vz(:,1)' * Rmean';
y = Vz(:,2)' * Rmean';
z = Vz(:,3)' * Rmean';

start_index = 0;

% Same visualization but in canonical coordinates
for i = 1:80
    current_indices = start_index + (1:data_points_per_iteration);

    if max(current_indices) > length(x)
        warning('Index out of bounds.'); 
        break; 
    end

    color_data = 1:data_points_per_iteration;

    scatter3(x(current_indices), ...
             y(current_indices), ...
             z(current_indices), ...
             30, color_data, 'filled'); 
    hold on

    start_index = start_index + data_points_per_iteration;
end

% Colorbar shows time within trial
colormap(cmap);
c = colorbar;

time_step = 0.5;  % window step (s)
nPoints = data_points_per_iteration;
time_labels = (1:nPoints) * time_step;

c.Ticks = linspace(1, nPoints, nPoints);
c.TickLabels = arrayfun(@(t) sprintf('%.1f',t), ...
                        time_labels, ...
                        'UniformOutput', false);
c.Label.String = 'Time within trial (s)';

title('Projection on canonical eSPoC axes')
xlabel('Canonical axis 1')
ylabel('Canonical axis 2')
zlabel('Canonical axis 3')
view(-45, 30);

%% =====================================================================
% SOURCE AND COMPONENT SELECTION
% =====================================================================

% Select global source index and local eigen-component
gl_src_idx  = 1; 

[~, idx] = max(abs(corrs(1,:)));
lcl_src_idx = idx;

% Back-project spatial filter and pattern from PCA space to sensor space
wx = U * W(gl_src_idx,:,lcl_src_idx)';    % spatial filter
ax = U * A(gl_src_idx,:,lcl_src_idx)';    % spatial pattern

% Fix sign ambiguity (filters/patterns are defined up to sign)
[~, idx] = max(abs(wx)); wx = wx .* sign(wx(idx));
[~, idx] = max(abs(ax)); ax = ax .* sign(ax(idx));

% =====================================================================
% TOPOGRAPHY AND TIME SERIES PLOTS
% =====================================================================

figure
set(gcf,'Color','w');
t = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

% Show correlation value in title
sgtitle(['Source Envelope - UMAP correlation: ', ...
         num2str(corrs(gl_src_idx,lcl_src_idx))])

% --- Spatial filter topography
ax1 = nexttile(t,1); 
title(ax1,'Filter');
topo.avg = wx; 
cfg.figure = ax1;
ft_topoplotER(cfg, topo); 

% --- Spatial pattern (forward model)
ax2 = nexttile(t,2); 
title(ax2,'Pattern');
topo.avg = ax; 
cfg.figure = ax2;
ft_topoplotER(cfg, topo); 

% --- Source time series and embedding projection
ax3 = nexttile(t,3,[1,2]); 
hold on; grid on
title(ax3,'Latent Source Signal & UMAP canonical projection');

% Project filtered data onto spatial filter
S = wx' * Xfilt;

% Normalize for visualization
S = (S - mean(S)) / std(S);

% Plot source signal and its envelope
plot(S)
plot(abs(hilbert(S)))   % analytic envelope

% Canonical projection of embedding
zz = Vz(:,gl_src_idx)' * Rmean';
zz = sign(corrs(gl_src_idx,lcl_src_idx)) * (zz - mean(zz)) / std(zz);
plot(time*Fs, zz, 'LineWidth', 1.5, 'Color','green')

% Format time axis
xticks(0:10*Fs:size(Xfilt,2))
xticklabels(0:10:(size(Xfilt,2)/Fs))
xlim([0,size(Xfilt,2)])

legend('Source signal', ...
       'Source Signal Envelope', ...
       'UMAP canonical projection')

ylabel('Source signal')
xlabel('t, sec')

%% =====================================================================
% Source and component selection
% =====================================================================

% Select global source and local eigenmode
gl_src_idx  = 1;
[~, idx] = max(abs(corrs(1,:)));
lcl_src_idx = idx;

% Back-project spatial filter and pattern to sensor space (undo PCA)
wx = U * W(gl_src_idx,:,lcl_src_idx)';   % spatial filter
ax = U * A(gl_src_idx,:,lcl_src_idx)';   % spatial pattern

% Fix sign ambiguity (filters defined up to sign)
[~, idx] = max(abs(wx));
wx = wx .* sign(wx(idx));

[~, idx] = max(abs(ax));
ax = ax .* sign(ax(idx));

% =====================================================================
% Extract source signal and envelope for all epochs
% =====================================================================

clear Yseg Yenv
nEpochs = size(Epochs_filt, 3);

for ep_idx = 1:nEpochs
    Ep = Epochs_filt(:,:,ep_idx);   % current filtered epoch
    
    s  = wx' * Ep;                  % source time series
    en = abs(hilbert(s));           % analytic envelope
    
    Yseg(:,ep_idx) = s;             % store signal
    Yenv(:,ep_idx) = en;            % store envelope
end

% =====================================================================
% Create visualization figure
% =====================================================================

figure('Color','w');

% Time axis for visualization
tsec_env = linspace(-2.5, 2.5, size(Yenv,1));
tsec_erp = tsec_env;
E = size(Yenv,2);

% Envelope statistics across epochs
env_mean = mean(Yenv,2,'omitnan');
sd       = std(Yenv,0,2,'omitnan');

t = tiledlayout(3,2,'TileSpacing','compact','Padding','compact');

% ---- Envelope heatmap ----
axH = nexttile(t,1,[3 1]);
imagesc(axH, tsec_env, 1:E, Yenv');
set(axH,'YDir','normal','Color','w');
grid(axH,'on');
xline(axH,0,'k--','LineWidth',1);
xlabel(axH,'time, s');
ylabel(axH,'epoch');
title(axH,'Envelope per epoch');
colorbar(axH);
caxis(axH,[0 4]);

% ---- Source activity (all epochs) ----
axERP = nexttile(t,2);
hold(axERP,'on');
grid(axERP,'on');
set(axERP,'Color','w');
plot(axERP, tsec_erp, Yseg);
xline(axERP,0,'k--','LineWidth',1);
xlabel(axERP,'time, s');
ylabel(axERP,'amplitude');
title(axERP,'Source activity (all epochs)');

% ---- Mean envelope ± SD ----
axENV = nexttile(t,4);
hold(axENV,'on');
grid(axENV,'on');
set(axENV,'Color','w');

xfill = [tsec_env, fliplr(tsec_env)];
yfill = [(env_mean + sd).', fliplr((env_mean - sd).')];

fill(axENV, xfill, yfill, [0.3 0.5 1.0], ...
     'FaceAlpha',0.2, 'EdgeColor','none');

plot(axENV, tsec_env, env_mean, ...
     'Color',[0.1 0.3 0.9], 'LineWidth',2);

xline(axENV,0,'k--','LineWidth',1);
xlabel(axENV,'time, s');
ylabel(axENV,'envelope (a.u.)');
title(axENV,'Mean envelope \pm SD');

% =====================================================================
% Filter and pattern topographies
% =====================================================================

% Create nested panel in bottom-right tile
axTmp = nexttile(t, 6);
pos6  = axTmp.OuterPosition;
delete(axTmp);

figCol = get(gcf,'Color');

p6 = uipanel('Parent', gcf, ...
             'Units','normalized', ...
             'Position', pos6, ...
             'BorderType','none', ...
             'BackgroundColor', figCol);

t6 = tiledlayout(p6, 1, 2, ...
                 'TileSpacing','compact', ...
                 'Padding','compact');

% ---- Filter topography ----
axFiltTopo = nexttile(t6, 1);
set(axFiltTopo, 'Color','w');

cfg.marker = 'no';
topo.avg  = wx;
cfg.figure = axFiltTopo;

ft_topoplotER(cfg, topo);
title(axFiltTopo,'Filter');

% ---- Pattern topography ----
axPatTopo = nexttile(t6, 2);
set(axPatTopo, 'Color','w');

topo.avg = ax;
cfg.figure = axPatTopo;

ft_topoplotER(cfg, topo);
title(axPatTopo,'Pattern');

% Ensure white background
set(findall(p6, 'type','axes'), 'Color', figCol);

% ---- Link time axes ----
linkaxes([axH axERP axENV], 'x');

%%
