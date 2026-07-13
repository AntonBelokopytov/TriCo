function [W, A, Vz] = espoc_def(X_epochs, Z, varargin)
% Deflationary Extended Source Power Co-modulation (eSPoC)
%
% INPUT:
%   X_epochs  - Band-pass filtered epoched data [n_samples, n_channels, n_epochs]
%   Z         - Multidimensional external regressor [n_regressors, n_epochs]
%
% OPTIONAL PARAMETERS:
%   'n_components'        - Number of deflationary components to extract (default: 1)
%   'X_min_var_explained' - Fraction of variance retained during PCA (default: 1)
%   'whitening_reg'       - Regularization for covariance whitening (default: 1e-4)
%   'cca_mode'            - 'regularized' (default) or 'standard'
%   'cca_reg'             - Regularization parameter for CCA (0-1)
%
% OUTPUT:
%   W           - Spatial filters in sensor space [n_channels, n_components]
%   A           - Spatial patterns [n_channels, n_components]
%   Vf          - Cell array of canonical feature vectors (sizes decrease due to deflation)
%   Vz          - Cell array of canonical regressor vectors
%   corrs       - Correlation between source power and projected regressor
%   eigenvalues - Largest absolute eigenvalue of the covariance feature matrix 

opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, ...
                  'n_components', 5, ...
                  'X_min_var_explained', 1, ...
                  'whitening_reg', 0.00001, ...
                  'cca_mode', 'regularized', ...
                  'cca_reg', 0.0001);

[~, n_channels, n_epochs] = size(X_epochs);

Epochs_cov = zeros(n_channels, n_channels, n_epochs);
for ep_idx = 1:n_epochs
    Epochs_cov(:,:,ep_idx) = cov(X_epochs(:,:,ep_idx));
end
Cxx_orig = mean(Epochs_cov, 3);

Cxx_r = Cxx_orig + opt.whitening_reg * eye(n_channels) * trace(Cxx_orig) / n_channels;
M = eye(n_channels) / sqrtm(Cxx_r); 

Cxxe_white = zeros(n_channels, n_channels, n_epochs);
for ep_idx = 1:n_epochs
    Cxxe_white(:,:,ep_idx) = M * Epochs_cov(:,:,ep_idx) * M';
end

n_curr_channels = n_channels;
Cxxe_curr = Cxxe_white;
W_white = []; 

corrs = zeros(opt.n_components, 1);
Vz = zeros(size(Z,1), opt.n_components);

for comp_idx = 1:opt.n_components
    
    n_features = (n_curr_channels^2 - n_curr_channels)/2 + n_curr_channels;
    F = zeros(n_features, n_epochs);
    for ep_idx = 1:n_epochs
        F(:, ep_idx) = cov2upper(Cxxe_curr(:,:,ep_idx));
    end
    F = F - mean(F,2);
    
    [Featdr, Uf] = project_to_pc(F, opt.X_min_var_explained);
    Cff = cov(Featdr');

    % Canonical Correlation Analysis
    if strcmp(opt.cca_mode, 'regularized')
        [Vfdr, Vz_curr] = cca(Featdr', Z', opt);
    elseif strcmp(opt.cca_mode, 'standard') 
        [Vfdr, Vz_curr] = canoncorr(Featdr', Z');
    end
    Vz(:,comp_idx) = Vz_curr(:,1);

    Vf_full = Uf * Cff * Vfdr;
    % Vf_full = Uf * Vfdr;
    Af = Vf_full(:,1);
    
    WW = upper2cov(Af);
    
    [Uw, ~, ~] = svd(WW);
    
    w_corrs = [];
    for w_i=1:size(Uw,2)
        if comp_idx == 1
            w_curr = Uw(:,w_i);
        else
            w_curr = B_accum * Uw(:,w_i);
        end
        Zpr = Vz_curr(:,1)' * Z;
        Env = zeros(1, n_epochs);
        for ep_idx = 1:n_epochs
            Env(ep_idx) = w_curr' * Epochs_cov(:,:,ep_idx) * w_curr;
        end
        w_corrs(w_i) = corr(Env', Zpr');
    end
    [~, idxs] = sort(abs(w_corrs),'descend');

    w_sub = Uw(:,idxs(1));
    if comp_idx == 1
        w_white = w_sub;
        W_white = w_white;
    else
        w_white = B_accum * w_sub;
        W_white = [W_white, w_white];
    end
    
    % ДЕФЛЯЦИЯ
    if comp_idx < opt.n_components
        B_accum = null(W_white'); 
        
        n_sub_channels = size(B_accum, 2);
        Cxxe_next = zeros(n_sub_channels, n_sub_channels, n_epochs);
        for ep_idx = 1:n_epochs
            Cxxe_next(:,:,ep_idx) = B_accum' * Cxxe_white(:,:,ep_idx) * B_accum;
        end
        Cxxe_curr = Cxxe_next;
        n_curr_channels = n_sub_channels;
    end
end

% 3. Проекция фильтров из отбеленного пространства обратно в исходное сенсорное
W = M' * W_white;
A = zeros(n_channels, opt.n_components);

% Нормализация фильтров (к единичной дисперсии) и вычисление паттернов
for k = 1:opt.n_components
    W(:, k) = W(:, k) / sqrt(W(:, k)' * Cxx_orig * W(:, k));
    A(:, k) = Cxx_orig * W(:, k); 
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% HELPER FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [v] = cov2upper(C)
upper_mask = triu(true(size(C)));
upper_triu_mask = triu(true(size(C)), 1);
C(upper_triu_mask) = C(upper_triu_mask) * sqrt(2);
v = C(upper_mask);
v = v(:);
end

function C = upper2cov(v)
n = (-1 + sqrt(1 + 8 * numel(v))) / 2;
C = zeros(n);
upper_mask = triu(true(n));
C(upper_mask) = v;
upper_triu_mask = triu(true(n), 1);
C(upper_triu_mask) = C(upper_triu_mask) / sqrt(2);
C = C + triu(C, 1)';
end

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

function [Vx, Vy, Cxx, Cyy] = cca(X, Y, opt)
gamma = opt.cca_reg;
X = X - mean(X,1);  
Y = Y - mean(Y,1);
[n, ~] = size(X);
Cxx = (X' * X) / (n-1);
Cyy = (Y' * Y) / (n-1);
Cxy = (X' * Y) / (n-1);
scale_x = trace(Cxx) / size(Cxx,1);
scale_y = trace(Cyy) / size(Cyy,1);
Sxx_r = (1-gamma)*Cxx + gamma*scale_x*eye(size(Cxx));
Syy_r = (1-gamma)*Cyy + gamma*scale_y*eye(size(Cyy));
Rx = chol(Sxx_r, 'upper');
Ry = chol(Syy_r, 'upper');
K = Rx' \ (Cxy / Ry);            
[Ux, ~, Uy] = svd(K, 'econ');
Vx = Rx \ Ux; 
Vy = Ry \ Uy; 
end

function opt = propertylist2struct(varargin)
opt = [];
if nargin==0
  return;
end
if isstruct(varargin{1}) || isempty(varargin{1})
  opt = varargin{1};
  iListOffset = 1;
else
  iListOffset = 0;
end
nFields = (nargin-iListOffset)/2;
for ff = 1:nFields
  fld = varargin{iListOffset+2*ff-1};
  opt.(fld) = varargin{iListOffset+2*ff};
end
end

function [opt, isdefault] = set_defaults(opt, varargin)
isdefault = [];
if ~isempty(opt)
  for Fld = fieldnames(opt)'
    isdefault.(Fld{1}) = 0;
  end
end
defopt = propertylist2struct(varargin{:});
for Fld = fieldnames(defopt)'
  fld = Fld{1};
  if ~isfield(opt, fld)
    [opt.(fld)] = deal(defopt.(fld));
    isdefault.(fld) = 1;
  end
end
end