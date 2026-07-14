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
    if exist('pagemtimes', 'builtin') || exist('pagemtimes', 'file')
        Cov_w_best = pagemtimes(Epochs_cov, w_best);
        env = squeeze(sum(w_best .* Cov_w_best, 1));
    else
        env = zeros(n_epochs, 1);
        for ep_idx = 1:n_epochs
            env(ep_idx) = w_best' * Epochs_cov(:,:,ep_idx) * w_best;
        end
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
    
    if exist('pagemtimes', 'builtin') || exist('pagemtimes', 'file')
        X_current = pagemtimes(X_current, P);
    else
        for ep_idx = 1:n_epochs
            X_current(:,:,ep_idx) = X_current(:,:,ep_idx) * P;
        end
    end
end

end

