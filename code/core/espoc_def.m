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

[n_samples, n_channels, n_epochs] = size(X_epochs);

if exist('pagemtimes', 'builtin') || exist('pagemtimes', 'file')
    X_mean = mean(X_epochs, 1);
    X_cen = X_epochs - X_mean;
    Epochs_cov = pagemtimes(X_cen, 'transpose', X_cen, 'none') / (n_samples - 1);
else
    Epochs_cov = zeros(n_channels, n_channels, n_epochs);
    for ep_idx = 1:n_epochs
        Epochs_cov(:,:,ep_idx) = cov(X_epochs(:,:,ep_idx));
    end
end
Cxx_orig = mean(Epochs_cov, 3);

Cxx_r = Cxx_orig + opt.whitening_reg * eye(n_channels) * trace(Cxx_orig) / n_channels;
M = eye(n_channels) / sqrtm(Cxx_r); 

if exist('pagemtimes', 'builtin') || exist('pagemtimes', 'file')
    Cxxe_white = pagemtimes(M, pagemtimes(Epochs_cov, M'));
else
    Cxxe_white = zeros(n_channels, n_channels, n_epochs);
    for ep_idx = 1:n_epochs
        Cxxe_white(:,:,ep_idx) = M * Epochs_cov(:,:,ep_idx) * M';
    end
end

n_curr_channels = n_channels;
Cxxe_curr = Cxxe_white;
W_white = []; 

corrs = zeros(opt.n_components, 1);
Vz = zeros(size(Z,1), opt.n_components);

for comp_idx = 1:opt.n_components
    
    n_features = (n_curr_channels^2 - n_curr_channels)/2 + n_curr_channels;

    % Векторизованное извлечение признаков (без цикла по эпохам)
    upper_mask = triu(true(n_curr_channels));
    upper_triu_mask = triu(true(n_curr_channels), 1);

    Cxxe_curr_mod = Cxxe_curr;
    % Умножаем внедиагональные элементы на sqrt(2) сразу для всех эпох
    % Поскольку upper_triu_mask двумерная, мы применяем ее к каждому срезу
    if exist('pagemtimes', 'builtin') || exist('pagemtimes', 'file')
         % Быстрый способ умножить нужные элементы
         diag_mask = eye(n_curr_channels);
         sqrt2_mask = diag_mask + (~diag_mask) * sqrt(2);
         Cxxe_curr_mod = Cxxe_curr_mod .* sqrt2_mask;
    else
         for ep_idx = 1:n_epochs
             tmp = Cxxe_curr_mod(:,:,ep_idx);
             tmp(upper_triu_mask) = tmp(upper_triu_mask) * sqrt(2);
             Cxxe_curr_mod(:,:,ep_idx) = tmp;
         end
    end

    % Извлекаем элементы для всех эпох
    % Преобразуем 3D тензор в 2D матрицу признаков
    Cxxe_curr_reshaped = reshape(Cxxe_curr_mod, n_curr_channels^2, n_epochs);
    upper_mask_1d = upper_mask(:);
    F = Cxxe_curr_reshaped(upper_mask_1d, :);
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
        if exist('pagemtimes', 'builtin') || exist('pagemtimes', 'file')
            Cov_w_curr = pagemtimes(Epochs_cov, w_curr);
            Env = squeeze(sum(w_curr .* Cov_w_curr, 1))';
        else
            Env = zeros(1, n_epochs);
            for ep_idx = 1:n_epochs
                Env(ep_idx) = w_curr' * Epochs_cov(:,:,ep_idx) * w_curr;
            end
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
        if exist('pagemtimes', 'builtin') || exist('pagemtimes', 'file')
            Cxxe_next = pagemtimes(B_accum', pagemtimes(Cxxe_white, B_accum));
        else
            Cxxe_next = zeros(n_sub_channels, n_sub_channels, n_epochs);
            for ep_idx = 1:n_epochs
                Cxxe_next(:,:,ep_idx) = B_accum' * Cxxe_white(:,:,ep_idx) * B_accum;
            end
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

