function [W, A, Vf, corrs, Feat, Epochs_cov, eigenvalues] = espoc_r2(X_epochs, z, varargin)
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
%
% ALGORITHM OVERVIEW:
%
% 1) For each epoch, compute sensor covariance matrix C(e).
%
% 2) Whiten covariances and map them to an unconstrained linear space
%    by vectorizing the upper triangular part.
%
% 3) Optionally reduce dimensionality of covariance features using PCA.
%
% 4) Apply Canonical Correlation Analysis (CCA) between covariance
%    features and the external regressor z.
%
%    This yields canonical vectors:
%       Vf – in covariance feature space (unconstrained solution)
%       Vz – in regressor space
%
% 5) Transform canonical vectors into covariance patterns and
%    project each solution back to sensor space via rank-1 approximation.
%
%    This produces spatial filters W and corresponding spatial patterns A.
%
% OUTPUT:
%
%   W           - Spatial filters in sensor space
%                 size: [n_sources, n_channels, n_components]
%
%   A           - Corresponding spatial patterns (forward-model patterns)
%                 size: [n_sources, n_channels, n_components]
%
%   Vf          - Canonical vectors in covariance feature space
%
%   Vz          - Canonical vectors in regressor space
%
%   corrs       - Correlation between reconstructed source power
%                 and projected regressor
%
%   F           - Vectorized covariance features (before PCA)
%
%   Epochs_cov  - Epoch-wise covariance matrices
%
%   eigenvalues - Eigenvalues of reconstructed covariance matrices
%                 (used to interpret global vs local source modes)
%
%
% Conceptual interpretation:
%
% - Vf defines "global source modes" in covariance feature space.
% - Eigen-decomposition of reconstructed matrices yields
%   "local spatial modes" (rank-1 components).
% - W and A provide interpretable spatial filters and patterns
%   within the standard EEG/MEG forward model.

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'X_min_var_explained', 1, ...
                  'whitening_reg', 0.0001, ...
                  'cca_mode', 'standard', ...
                  'cca_reg', 0.1);
 
assert(size(z,1) == 1, 'z must have only 1 dimension');

% ---
[Feat, Wm, Cx, Epochs_cov, ~] = get_white_covariance_series(X_epochs, opt);
Cf = cov(Feat');

[Featdr, Uf] = project_to_pc(Feat, opt.X_min_var_explained);

Vfdr = spoc(Featdr,z);

Vf = Uf*Vfdr(:,1);
Af = Cf*Vf;

% Project and normalize EEG/MEG filters
for global_src_idx=1:size(Af,2)
    [w, a, s] = project_to_manifold(Af(:,global_src_idx), Wm, Cx);
        
    % Find correlation of the filters
    n_local = size(w,2);
    n_epochs = size(Epochs_cov,3);

    if exist('pagemtimes', 'builtin') || exist('pagemtimes', 'file')
        Cov_w = pagemtimes(Epochs_cov, w);
        % Env dimensions: [n_epochs, n_local]
        Env = zeros(n_epochs, n_local);
        for ep_idx = 1:n_epochs
            Env(ep_idx, :) = sum(w .* Cov_w(:,:,ep_idx), 1);
        end
    else
        Env = zeros(n_epochs, n_local);
        for local_src_idx=1:n_local
            w_loc = w(:,local_src_idx);
            for ep_idx=1:n_epochs
                Env(ep_idx, local_src_idx) = w_loc' * Epochs_cov(:,:,ep_idx) * w_loc;
            end
        end
    end

    cr = zeros(1, n_local);
    for local_src_idx = 1:n_local
        cr(local_src_idx) = corr(Env(:, local_src_idx), z');
    end
    [cr,idx] = sort(cr,'descend');
    w = w(:,idx);
    a = a(:,idx);
    
    eigenvalues(global_src_idx,:) = s;
    corrs(global_src_idx,:) = cr;
    W(global_src_idx,:,:) = w;
    A(global_src_idx,:,:) = a;
end

if size(W,1)==1
    corrs = squeeze(corrs);
    W = squeeze(W);
    A = squeeze(A);
end

end

