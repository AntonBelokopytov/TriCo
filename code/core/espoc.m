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

% Определяем размер n_local на первой итерации для выделения памяти
[w_tmp, ~, ~] = project_to_manifold(Af(:,1), Wm, Cxx);
n_local = size(w_tmp, 2);
n_channels = size(w_tmp, 1);

% Инициализация массивов для предотвращения динамического изменения размера 
W = zeros(n_global, n_channels, n_local);
A = zeros(n_global, n_channels, n_local);
corrs = zeros(n_global, n_local);
eigenvalues = zeros(n_global, n_local);

% Проекция и нормализация фильтров
for global_src_idx = 1:n_global
    [w, a, s] = project_to_manifold(Af(:,global_src_idx), Wm, Cxx);
    
    Zpr = Vz(:,global_src_idx)' * Z;
    
    % Векторизованное вычисление огибающей мощности
    % Env: [n_epochs, n_local]
    if exist('pagemtimes', 'builtin') || exist('pagemtimes', 'file')
        % Tensor product: W' * Cov * W
        Cov_W = pagemtimes(Epochs_cov, w);
        % w is [n_channels, n_local], Cov is [n_channels, n_channels, n_epochs]
        % We need diag(w' * Cov * w) for each epoch
        Env = zeros(n_epochs, n_local);
        for ep_idx = 1:n_epochs
            Env(ep_idx, :) = sum(w .* Cov_W(:,:,ep_idx), 1);
        end
    else
        Env = zeros(n_epochs, n_local);
        for local_src_idx = 1:n_local
            w_loc = w(:, local_src_idx);
            for ep_idx = 1:n_epochs
                Env(ep_idx, local_src_idx) = w_loc' * Epochs_cov(:,:,ep_idx) * w_loc;
            end
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

