function [W, A, Vf, Vz, corrs, eigenvalues, cca_corrs, Epochs_cov, Feat] = espoc(X_epochs, Z, varargin)
% Extended Source Power Co-modulation (eSPoC)
%
% This function implements the eSPoC framework for explaining variability
% of EEG/MEG covariance features with respect to a multidimensional external
% regressor (e.g., UMAP embedding coordinates).
%
% INPUT:
%   X_epochs  - Band-pass filtered epoched data
%               size: [n_samples_per_epoch, n_channels, n_epochs]
%
%   z         - Multidimensional external regressor
%               size: [n_regressors, n_epochs]
%               (e.g., embedding coordinates)
%
% OPTIONAL PARAMETERS:
%   'X_min_var_explained' - Fraction of variance (0–1) retained during PCA
%                           in covariance feature space (default: 1)
%
%   'whitening_reg'       - Regularization parameter for covariance whitening
%                           (default: 1e-4)
%
%   'cca_mode'            - 'regularized' (default) or 'standard'
%
%   'cca_reg'             - Regularization parameter for CCA (0–1)
%
%   'ww_reg'              - Regularization during projection from
%                           covariance feature space to rank-1 matrices
%
% OUTPUT:
%   W           - Spatial filters in sensor space
%   A           - Corresponding spatial patterns (forward-model patterns)
%   Vf          - Canonical vectors in covariance feature space
%   Vz          - Canonical vectors in regressor space
%   corrs       - Correlation between reconstructed source power and projected regressor
%   eigenvalues - Eigenvalues of reconstructed covariance matrices
%   cca_corrs   - Canonical correlations (correlation of feature subspace 
%                 projections with external variable projections)

opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, ...
                  'X_min_var_explained', 1, ...
                  'whitening_reg', 0.00001, ...
                  'cca_mode', 'regularized', ...
                  'cca_reg', 0.00001);

% 1. Извлечение признаков с использованием быстрых тензорных вычислений
[Feat, Wm, Cxx, Epochs_cov] = get_white_covariance_series(X_epochs, opt);

% 2. Снижение размерности
[Featdr, Uf] = project_to_pc(Feat, opt.X_min_var_explained);
Cff = cov(Featdr');

% 3. Канонический корреляционный анализ (CCA)
if strcmp(opt.cca_mode, 'regularized')
    [Vfdr, Vz] = cca(Featdr', Z', opt);
elseif strcmp(opt.cca_mode, 'standard') 
    [Vfdr, Vz] = canoncorr(Featdr', Z');
end

% =========================================================================
% Вычисление корреляций признакового подпространства (Canonical Corrs)
% =========================================================================
Feat_proj = Featdr' * Vfdr;
Z_proj = Z' * Vz;

n_cca_comps = size(Vfdr, 2);
cca_corrs = zeros(n_cca_comps, 1);
for c = 1:n_cca_comps
    % Вычисляем корреляцию между спроецированными признаками и спроецированным Z
    cca_corrs(c) = corr(Feat_proj(:, c), Z_proj(:, c));
end
% =========================================================================

% Возврат фильтров из пространства сниженной размерности
Vf = Uf * Vfdr;
Af = Uf * (Cff * Vfdr);

n_global = size(Af, 2);
n_epochs = size(Epochs_cov, 3);

% Инициализация массивов для предотвращения динамического изменения размера 
W = []; A = []; corrs = []; eigenvalues = []; 

% Проекция и нормализация фильтров
for global_src_idx = 1:n_global
    [w, a, s] = project_to_manifold(Af(:,global_src_idx), Wm, Cxx);
    n_local = size(w, 2);
    
    Zpr = Vz(:,global_src_idx)' * Z;
    
    Env = zeros(n_epochs, n_local);    
    for local_src_idx = 1:n_local
        w_loc = w(:, local_src_idx);
        for ep_idx = 1:n_epochs
            Env(ep_idx, local_src_idx) = w_loc' * Epochs_cov(:,:,ep_idx) * w_loc;
        end
    end
    
    cr = corr(Env, Zpr');
    
    eigenvalues(global_src_idx, 1:n_local) = s';
    corrs(global_src_idx, 1:n_local) = cr';
    W(global_src_idx, :, 1:n_local) = w;
    A(global_src_idx, :, 1:n_local) = a;
end

if size(W,1) == 1
    corrs = squeeze(corrs);
    W = squeeze(W);
    A = squeeze(A);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [F, Wm, Cxx, Epochs_cov] = get_white_covariance_series(X_epochs, opt)
[n_samples, n_channels, n_epochs] = size(X_epochs);
n_features = (n_channels^2 - n_channels)/2 + n_channels;
X_mean = mean(X_epochs, 1);
X_cen = X_epochs - X_mean;
try
    Epochs_cov = pagemtimes(X_cen, 'transpose', X_cen, 'none') / (n_samples - 1);
catch
    Epochs_cov = zeros(n_channels, n_channels, n_epochs);
    for ep_idx = 1:n_epochs
        X_ep = X_cen(:, :, ep_idx);
        Epochs_cov(:,:,ep_idx) = (X_ep' * X_ep) / (n_samples - 1);
    end
end
Cxx = mean(Epochs_cov, 3);
% Отбеливающая матрица
Cxx_r = Cxx + opt.whitening_reg * eye(n_channels) * trace(Cxx) / n_channels;
iWm = sqrtm(Cxx_r);    
Wm = eye(n_channels) / iWm;
% Вычисление отбеленных признаков
F = zeros(n_features, n_epochs);
try
    % Тензорное отбеливание
    Epochs_covW = pagemtimes(Wm, pagemtimes(Epochs_cov, Wm'));
    for ep_idx = 1:n_epochs
        F(:, ep_idx) = cov2upper(Epochs_covW(:,:,ep_idx));
    end
catch
    % Fallback
    for ep_idx = 1:n_epochs
        XcovW = Wm * Epochs_cov(:,:,ep_idx) * Wm';
        F(:, ep_idx) = cov2upper(XcovW);
    end
end
F = F - mean(F, 2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [W, A, s] = project_to_manifold(V, Wm, Cxx)
% Cxx_r = Cxx + 0.00001 * eye(size(Cxx,1)) * trace(Cxx) / size(Cxx,1);

WW = upper2cov(V);
[Uw, S, ~] = eig(WW); [s,idx] = sort(diag(S),'descend'); Uw = Uw(:,idx);
n_local = size(Uw, 2);
n_channels = size(Wm, 1);
W = zeros(n_channels, n_local);
A = zeros(n_channels, n_local);
% Нормализация и восстановление паттернов
for local_src_idx = 1:n_local
    wi = Wm * Uw(:,local_src_idx);
    Wprn = wi / sqrt(wi' * Cxx * wi);
    W(:,local_src_idx) = Wprn;
    A(:,local_src_idx) = Cxx * Wprn / (Wprn' * Cxx * Wprn);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [v] = cov2upper(C)
upper_mask = triu(true(size(C)));
upper_triu_mask = triu(true(size(C)), 1);
C(upper_triu_mask) = C(upper_triu_mask) * sqrt(2);
v = C(upper_mask);
v = v(:);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function C = upper2cov(v)
n = (-1 + sqrt(1 + 8 * numel(v))) / 2;
C = zeros(n);
upper_mask = triu(true(n));
C(upper_mask) = v;
upper_triu_mask = triu(true(n), 1);
C(upper_triu_mask) = C(upper_triu_mask) / sqrt(2);
C = C + triu(C, 1)';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X_proj, U] = project_to_pc(X, min_var_explained)
X = X - mean(X,2);
[U, S, ~] = svd(X, "econ");
S = diag(S);
tol_rank = max(size(X)) * eps(S(1));
r = sum(S > tol_rank);
ve = S.^2;
var_explained = cumsum(ve) / sum(ve);
var_explained(end) = 1;
tol = 1e-12;
n_components = find(var_explained >= min_var_explained - tol, 1);
if isempty(n_components)
    n_components = r;
end
n_components = max(min(n_components, r), 1);
U = U(:, 1:n_components);
X_proj = U' * X;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Vx, Vy, Cxx, Cyy] = cca(X, Y, opt)
gamma = opt.cca_reg;
X = X - mean(X,1);  
Y = Y - mean(Y,1);
[n, ~] = size(X);
Cxx = (X' * X) / (n-1);
Cyy = (Y' * Y) / (n-1);
Cxy = (X' * Y) / (n-1);
scale_x = trace(Cxx) / size(Cxx,1);
% scale_y = trace(Cyy) / size(Cyy,1);
Sxx_r = (1-gamma)*Cxx + gamma*scale_x*eye(size(Cxx));
% Syy_r = (1-gamma)*Cyy + gamma*scale_y*eye(size(Cyy));
Rx = chol(Sxx_r, 'upper');
Ry = chol(Cyy, 'upper');
K = Rx' \ (Cxy / Ry);            
[Ux, ~, Uy] = svd(K, 'econ');
Vx = Rx \ Ux; 
Vy = Ry \ Uy; 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OTHER HELPERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function opt = propertylist2struct(varargin)
opt= [];
if nargin==0, return; end
if isstruct(varargin{1}) | isempty(varargin{1}),
  opt= varargin{1}; iListOffset= 1;
else
  iListOffset = 0;
end
nFields= (nargin-iListOffset)/2;
if nFields~=round(nFields), error('Invalid parameter/value list'); end
for ff= 1:nFields,
  fld = varargin{iListOffset+2*ff-1};
  if ~ischar(fld), error('Invalid parameter/value list'); end
  opt.(fld)= varargin{iListOffset+2*ff};
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [opt, isdefault]= set_defaults(opt, varargin)
isdefault= [];
if ~isempty(opt),
  for Fld=fieldnames(opt)', isdefault.(Fld{1})= 0; end
end
defopt = propertylist2struct(varargin{:});
for Fld= fieldnames(defopt)',
  fld= Fld{1};
  if ~isfield(opt, fld),
    [opt.(fld)]= deal(defopt.(fld));
    isdefault.(fld)= 1;
  end
end
end