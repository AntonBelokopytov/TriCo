close all
clear
clc

ft_path = 'C:\Users\ansbel\Documents\2Git\fieldtrip';

if ~exist('ft_defaults','file')
    addpath(ft_path);
end

ft_defaults;

%% =====================================================================
%  Load continuous MEG/EEG data
% =====================================================================
sub_path = 'sub1_rest_ec_eo_raw.fif';

cfg = [];
cfg.dataset = sub_path;
Xinf = ft_preprocessing(cfg);

% X: time x channels
X = Xinf.trial{1}';
Fs = Xinf.fsample;

% Prepare FieldTrip structures for topographic plotting
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
cfg.colorbar     = 'yes'; 

%% =====================================================================
%  Band-pass filtering and PCA (dimensionality reduction)
% =====================================================================

% Alpha band filtering
[b,a] = butter(3,[8,12]/(Fs/2)); 
Xfilt = filtfilt(b,a,X)';

% PCA via SVD
[U,S,~] = svd(Xfilt,'econ');
S = diag(S);

% Estimate effective rank of the data
tol = max(size(X)) * eps(S(1));
r = sum(S > tol);

% Cumulative variance explained
ve = S.^2;
var_explained = cumsum(ve) / sum(ve);
var_explained(end) = 1;

% Number of components explaining at least 99% variance
n_components = find(var_explained>=0.99, 1);
n_components = max(min(n_components, r), 1);

% PCA projection
U = U(:,1:n_components);
Xpca = U'*Xfilt;

%% =====================================================================
%  Epoching
% =====================================================================

Ws = 2;   % window size (seconds)
Ss = 1;   % step size (seconds)

epochs = epoch_data(Xpca',Fs,Ws,Ss);

%% =====================================================================
%  Covariance matrices and vectorization
% =====================================================================

covs = [];
Covs_vec = [];

for i = 1:size(epochs,3)
    C = cov(epochs(:,:,i));
    covs(:,:,i) = C;
    % Vectorize upper triangular part
    Covs_vec(i,:) = C(triu(true(size(C))));
end

%% =====================================================================
%  Tangent space projection of covariance matrices
% =====================================================================
Tcovs = Tangent_space(covs);

%% =====================================================================
%  UMAP embedding of covariance features
% =====================================================================
clear u
u = UMAP("n_neighbors",30,"n_components",3);
u.metric = 'euclidean';
u.target_metric = 'euclidean';

% Low-dimensional embedding of epochs
R = u.fit_transform(Tcovs');

%% =====================================================================
%  Visualization of UMAP space (EC vs EO)
% =====================================================================
figure

N = fix(size(epochs,3)/2);

scatter3(R(1:N,1),R(1:N,2),R(1:N,3));
hold on
scatter3(R(N+1:end,1),R(N+1:end,2),R(N+1:end,3))

legend('Eyes closed','Eyes opened')
xlabel('UMAP component 1')
ylabel('UMAP component 2')
zlabel('UMAP component 3')

%% =====================================================================
%  eSPoC: correlate source envelopes with UMAP coordinates
% =====================================================================
[W, A, Vf, Vz, corrs, VecCov, Epochs_cov, eigenvalues] = espoc(epochs, R');

figure; stem(corrs');
legend('1','2','3')

Rmean = R - mean(R,1);

%%
gl_src = Vf'*VecCov;
emb_can_pr = Vz'*Rmean';

corr(gl_src',emb_can_pr')

figure; set(gcf,'Color','w');
tiledlayout(3,2)

nexttile(1,[3,1])
xx = Vz(:,1)'*R';
yy = Vz(:,2)'*R';
zz = Vz(:,3)'*R';

scatter3(xx(1:N),yy(1:N),zz(1:N)); hold on
scatter3(xx(N+1:end),yy(N+1:end),zz(N+1:end));
legend('Eyes closed', 'Eyes opened')

xlabel('UMAP canonical axis 1')
ylabel('UMAP canonical axis 2')
zlabel('UMAP canonical axis 3')

Step = 10;
t = 0:Step*Fs:size(X,1);
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

    xticks(t/Fs)
    xticklabels(t/Fs)
    
    xline(120, 'LineWidth', 3, 'color', 'red','LineStyle','--')

    if i==1
        legend('Global source signal', 'UMAP canonical projection')
    end
    if i==3
        xlabel('t, sec')
    end
    
end

%% =====================================================================
%  Permutation test using circular time shifts
% =====================================================================

clear corrmax corrmin

parfor i=1:1000
    i
    
    % Circular time shift destroys temporal alignment
    r_idx = fix(rand*size(Xpca,2));
    XCirc = circshift(Xpca,[0,r_idx]);
    
    X_test = epoch_data(XCirc', Fs, Ws, Ss);
    
    [~, ~, ~, ~, corrs] = espoc(X_test, R');

    % Store extreme correlations across components
    corrmax(:,i) = max(corrs,[],2);
    corrmin(:,i) = min(corrs,[],2);
end

%% =====================================================================
%  Significance thresholds (two-sided, alpha = 0.05)
% =====================================================================

% Build null distributions of extreme statistics
corrmax1 = sort(max(corrmax, [], 1), 'descend');
corrmin1 = sort(min(corrmin, [], 1));

% Upper threshold (positive correlations)
i = 1;
while 1 - sum(corrmax1(i) > corrmax1) / numel(corrmax1) <= 0.05
    i = i + 1;
end
max_val = corrmax1(i);

% Lower threshold (negative correlations)
i = 1;
while 1 - sum(corrmin1(i) < corrmin1) / numel(corrmin1) <= 0.05
    i = i + 1;
end
min_val = corrmin1(i);

% Plot observed correlations with significance thresholds
figure; stem(corrs'); hold on
yline(max_val)
yline(min_val)
legend('1','2','3')

%% =====================================================================
%  UMAP space visualization
% =====================================================================

% Coordinates of epochs in original UMAP space
x = Rmean(:,1);
y = Rmean(:,2);
z = Rmean(:,3);

figure; set(gcf,'Color','w');
t = tiledlayout(2,2, 'TileSpacing','compact', 'Padding','compact');

% 3D scatter of UMAP embedding (EC vs EO)
nexttile(t, 2);
scatter3(x(1:N), y(1:N), z(1:N)); hold on
scatter3(x(N+1:end), y(N+1:end), z(N+1:end));
legend('Eyes closed', 'Eyes opened')

xlabel('UMAP component 1')
ylabel('UMAP component 2')
zlabel('UMAP component 3')

% =====================================================================
%  Marginal distributions along UMAP axes
% =====================================================================

% Distribution along axis 3
nexttile(t, 1);
EDGES = linspace(min(z), max(z), 20);
histogram(z(1:N), EDGES); hold on
histogram(z(N+1:end), EDGES);
legend('Eyes closed', 'Eyes opened')
title('Axis 3');

% Distribution along axis 2
nexttile(t, 3);
EDGES = linspace(min(y), max(y), 20);
histogram(y(1:N), EDGES); hold on
histogram(y(N+1:end), EDGES);
title('Axis 2');

% Distribution along axis 1
nexttile(t, 4);
EDGES = linspace(min(x), max(x), 20);
histogram(x(1:N), EDGES); hold on
histogram(x(N+1:end), EDGES);
title('Axis 1');

%% =====================================================================
%  Canonical projection of UMAP coordinates
% =====================================================================

% Rotate UMAP embedding into canonical space (via CCA)
x = Vz(:,1)' * Rmean';
y = Vz(:,2)' * Rmean';
z = Vz(:,3)' * Rmean';

figure; set(gcf,'Color','w');
t = tiledlayout(2,2, 'TileSpacing','compact', 'Padding','compact');

% =====================================================================
%  Canonical space visualization
% =====================================================================

% 3D scatter in canonical coordinate system
nexttile(t, 2);
scatter3(x(1:N), y(1:N), z(1:N)); hold on
scatter3(x(N+1:end), y(N+1:end), z(N+1:end));
legend('Eyes closed', 'Eyes opened')

xlabel('Canonical axis 1')
ylabel('Canonical axis 2')
zlabel('Canonical axis 3')

% =====================================================================
%  Marginal distributions along canonical axes
% =====================================================================

% Axis 3
nexttile(t, 1);
EDGES = linspace(min(z), max(z), 20);
histogram(z(1:N), EDGES); hold on
histogram(z(N+1:end), EDGES);
legend('Eyes closed', 'Eyes opened')
title('Axis 3');

% Axis 2
nexttile(t, 3);
EDGES = linspace(min(y), max(y), 20);
histogram(y(1:N), EDGES); hold on
histogram(y(N+1:end), EDGES);
title('Axis 2');

% Axis 1
nexttile(t, 4);
EDGES = linspace(min(x), max(x), 20);
histogram(x(1:N), EDGES); hold on
histogram(x(N+1:end), EDGES);
title('Axis 1');

%% =====================================================================
%  Spatial filters, patterns, and source visualization
% =====================================================================
% Select global source index and local component (eigenmode)
gl_src_idx  = 1;
lcl_src_idx = 10;

% Project filter and pattern back to sensor space (undo PCA)
ax = U * A(gl_src_idx,:,lcl_src_idx)';
wx = U * W(gl_src_idx,:,lcl_src_idx)';

% Fix sign ambiguity (filters/patterns are defined up to sign)
[~, idx] = max(abs(wx));
wx = wx .* sign(wx(idx));

[~, idx] = max(abs(ax));
ax = ax .* sign(ax(idx));

% =====================================================================
%  Topography and temporal dynamics of the extracted source
% =====================================================================

figure
set(gcf,'Color','w');
t = tiledlayout(2,2, 'TileSpacing','compact', 'Padding','compact');
sgtitle(['Source Envelope - UMAP correlation: ', ...
         num2str(corrs(gl_src_idx,lcl_src_idx))])

% --- Spatial filter topography
ax1 = nexttile(t, 1); 
title(ax1,'Filter');
topo.avg   = wx;
cfg.figure = ax1;
ft_topoplotER(cfg, topo); 

% --- Spatial pattern (forward model)
ax2 = nexttile(t, 2);       
title(ax2,'Pattern');
topo.avg   = ax;
cfg.figure = ax2;
ft_topoplotER(cfg, topo); 

% --- Temporal dynamics
ax3 = nexttile(t, 3, [1,2]); hold on; grid();        
title(ax3,'Latent Source Signal & UMAP canonical projection');

% Extract source time series
S = X * wx;

% Normalize
S = (S - mean(S)) / std(S);

% Plot source signal and its envelope
plot(S)
plot(abs(hilbert(S)))

% Canonical UMAP projection (normalized)
zz = Vz(:,gl_src_idx)' * Rmean';
zz = - (zz - mean(zz)) / std(zz);

plot((1:size(zz,2)) * Ss * Fs, zz, ...
     'LineWidth',2, 'Color','green')

% Time axis formatting
Step = 10;
t = 0:Step*Fs:size(X,1);
xticks(t)
xticklabels(t/Fs)
xlim([0,size(X,1)])

legend('Source signal', ...
       'Source Signal Envelope', ...
       'UMAP canonical projection')

ylabel('Source signal')
xlabel('t, sec')

%%
