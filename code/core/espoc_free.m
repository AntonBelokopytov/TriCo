function [W, A, Vf, Vz, corrs, eigenvalues, cca_corrs, Epochs_cov_init, Feat_init] = espoc_free(X_epochs, Z, varargin)
% Extended Source Power Co-modulation (eSPoC) - Iterative Deflation
%
% This function implements the eSPoC framework for explaining variability
% of EEG/MEG covariance features with respect to a multidimensional external
% regressor (e.g., UMAP embedding coordinates).
%
% В этой версии снято ограничение на ортогональность Z. Дефляция (отбеливание) 
% применяется только к пространственным данным X на каждой итерации.
%
% OPTIONAL PARAMETERS:
%   'n_components'        - Number of components to extract iteratively (default: 5)
%
% OUTPUT:
%   ...
%   Epochs_cov_init, Feat_init - Covariances and features from the FIRST 
%                                (pre-deflation) iteration.

opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, ...
                  'X_min_var_explained', 0.9, ...
                  'whitening_reg', 0.00001, ...
                  'cca_mode', 'regularized', ...
                  'cca_reg', 0.00001, ...
                  'n_components', 3);

[n_samples, n_channels, n_epochs] = size(X_epochs);
n_features = (n_channels^2 - n_channels)/2 + n_channels;
n_comps = opt.n_components;

% Инициализация массивов
W = zeros(n_channels, n_comps); 
A = zeros(n_channels, n_comps); 
Vf = zeros(n_features, n_comps);
Vz = zeros(size(Z, 1), n_comps);
corrs = zeros(n_comps, 1); 
eigenvalues = zeros(n_comps, 1);
cca_corrs = zeros(n_comps, 1);

% Копия данных для итеративной дефляции
X_current = X_epochs;

for c = 1:n_comps
    % 1. Извлечение признаков (на текущих дефлированных данных)
    [Feat, Wm, Cxx, Epochs_cov] = get_white_covariance_series(X_current, opt);
    
    % Сохраняем исходные матрицы для вывода (с первой итерации)
    if c == 1
        Epochs_cov_init = Epochs_cov;
        Feat_init = Feat;
    end
    
    % 2. Снижение размерности
    [Featdr, Uf] = project_to_pc(Feat, opt.X_min_var_explained);
    Cff = cov(Featdr');
    
    % 3. Канонический корреляционный анализ (CCA)
    if strcmp(opt.cca_mode, 'regularized')
        [Vfdr_all, Vz_all] = cca(Featdr', Z', opt);
    elseif strcmp(opt.cca_mode, 'standard') 
        [Vfdr_all, Vz_all] = canoncorr(Featdr', Z');
    end
    
    % Извлекаем ТОЛЬКО первую компоненту (с наибольшей канонической корреляцией)
    vfdr_1 = Vfdr_all(:, 1);
    vz_1 = Vz_all(:, 1);
    
    % Вычисление корреляции признакового подпространства (Canonical Corr)
    feat_proj = Featdr' * vfdr_1;
    z_proj = Z' * vz_1;
    cca_corrs(c) = corr(feat_proj, z_proj);
    
    % 4. Возврат фильтров из пространства сниженной размерности
    Vf_orig = Uf * vfdr_1;
    Af_orig = Uf * (Cff * vfdr_1);
    
    % Проекция на риманово многообразие ранга 1
    [w_all, a_all, s_all] = project_to_manifold(Af_orig, Wm, Cxx);
    
    % Выбираем фильтр с наибольшей дисперсией (project_to_manifold уже возвращает их отсортированными)
    w_best = w_all(:, 1);
    a_best = a_all(:, 1);
    s_best = s_all(1);
    
    % Вычисляем огибающую мощности для текущей компоненты
    env = zeros(n_epochs, 1);    
    for ep_idx = 1:n_epochs
        env(ep_idx) = w_best' * Epochs_cov(:,:,ep_idx) * w_best;
    end
    cr = corr(env, z_proj);
    
    % Сохраняем результаты текущей итерации
    W(:, c) = w_best;
    A(:, c) = a_best;
    Vf(:, c) = Vf_orig;
    Vz(:, c) = vz_1;
    corrs(c) = cr;
    eigenvalues(c) = s_best;
    
    % =========================================================================
    % 5. ДЕФЛЯЦИЯ (Снятие ортогональности внешней переменной)
    % =========================================================================
    % Отбеливаем данные ТОЛЬКО относительно найденного пространственного фильтра.
    % Матрица внешней переменной Z не изменяется!
    
    % Матрица проекции для дефляции сигнала X (размерности [samples x channels])
    % P = I - w * a' гарантирует, что X_new = X - (X * w) * a'
    P = eye(n_channels) - w_best * a_best';
    
    for ep_idx = 1:n_epochs
        X_current(:,:,ep_idx) = X_current(:,:,ep_idx) * P;
    end
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
% Wm = eye(n_channels);
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
[Uw, S, ~] = eig(WW); [s,idx] = sort(diag(abs(S)),'descend'); Uw = Uw(:,idx);
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