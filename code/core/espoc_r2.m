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
    for local_src_idx=1:size(w,2)
        for ep_idx=1:size(Epochs_cov,3)
            Env(ep_idx) = w(:,local_src_idx)' * Epochs_cov(:,:,ep_idx) * w(:,local_src_idx);
        end
        cr(local_src_idx)=corr(Env',z');
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [v] = cov2upper(C)

upper_triu_mask = triu(true(size(C)),1);
upper_mask = triu(true(size(C)));
C(upper_triu_mask) = C(upper_triu_mask)*sqrt(2);
upper_triangle = C(upper_mask);
v = upper_triangle(:);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function C = upper2cov(v)

n = (-1 + sqrt(1 + 8 * numel(v))) / 2;
assert(mod(n,1) == 0, 'Vector length does not correspond to a triangular matrix.');

C = zeros(n);
upper_mask = triu(true(n));
C(upper_mask) = v;

upper_triu_mask = triu(true(n), 1);
C(upper_triu_mask) = C(upper_triu_mask) / sqrt(2);

C = C + triu(C, 1)';

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [VecCov, Wm, Cxx, Epochs_cov, Epochs_covW] = get_white_covariance_series(X_epochs, opt)

% Function to get upper triangular covarience time series in dimension
% reduced space

[~,n_channels,n_epochs] = size(X_epochs);
n_features = (n_channels^2-n_channels)/2+n_channels;

Epochs_cov = zeros(n_channels,n_channels,n_epochs);
for ep_idx = 1:n_epochs
    Xcov = cov(X_epochs(:,:,ep_idx));
    Epochs_cov(:,:,ep_idx) = Xcov;
end
% Mean covariance matrix
Cxx = mean(Epochs_cov,3);

% Whitening matrix
Cxx_r = Cxx+opt.whitening_reg*eye(size(Cxx))*trace(Cxx)/size(Cxx,1);
iWm = sqrtm(Cxx_r);    
Wm = eye(n_channels) / iWm;

% Whightened covariance series (upper triangular parts)
Epochs_covW = zeros(n_channels,n_channels,n_epochs);
VecCov = zeros(n_features,n_epochs);
for ep_idx = 1:n_epochs
    XcovW = Wm * Epochs_cov(:,:,ep_idx) * Wm';
    Epochs_covW(:,:,ep_idx) = XcovW;
    VecCov(:, ep_idx) = cov2upper(XcovW);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [X_proj, U] = project_to_pc(X, min_var_explained)
% PROJECT_TO_PC  PCA projection with variance threshold.
%
%   [X_proj, U] = project_to_pc(X, min_var_explained)
%
%   INPUT:
%       X  - data matrix [D x N]
%       min_var_explained - fraction of variance to retain (0–1)
%
%   OUTPUT:
%       X_proj - projected data
%       U      - retained principal directions

% Center data (important for PCA!)
X = X - mean(X,2);

% Economy SVD
[U,S,~] = svd(X,"econ");

S = diag(S);

% Numerical rank
tol_rank = max(size(X)) * eps(S(1));
r = sum(S > tol_rank);

% Variance explained
ve = S.^2;
var_explained = cumsum(ve) / sum(ve);
var_explained(end) = 1;

% Select number of components
tol = 1e-12;
n_components = find(var_explained >= min_var_explained - tol, 1);
if isempty(n_components)
    n_components = r;
end
n_components = max(min(n_components, r), 1);

% Truncate basis
U = U(:,1:n_components);

% Project
X_proj = U' * X;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [V] = spoc(F, z)
[D,E] = size(F);

Sfz = zeros(D,1);
Sff = zeros(D,D);

F = F - mean(F,2);
z = (z - mean(z)) / std(z);

for e = 1:E
    Sfz = Sfz + F(:,e) * z(e);
    Sff = Sff + F(:,e) * F(:,e)';
end

Sfz = Sfz / (E-1);
Sff = Sff / (E-1);

[V,S] = eig(Sfz*Sfz', Sff); s=diag(S);[s,idxs]=sort(s,'descend');V=V(:,idxs);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [W, A, s] = project_to_manifold(V, Wm, Cxx)

% Project filters to manifold
WW = upper2cov(V);

[Uw,S] = eig(WW);s=diag(S);[s,idxs]=sort(s,'descend');Uw=Uw(:,idxs);
% stem(s)
% xlabel('number of component')
% ylabel('\lambda value')
% title('Spectrum of eigenvalues of the matrix W')
% Optionally svd() could be used instead of eig() (Result is the same. Order differs)
% [Uw,~,~] = svd(WW_r);

% Normalization and pattern recovery
for local_src_idx=1:size(Uw,2)
    % Return filters from the whightened space
    wi = Wm * Uw(:,local_src_idx);
    % Normalize
    Wprn = wi / sqrt(wi' * Cxx * wi);
    W(:,local_src_idx) = Wprn;
    A(:,local_src_idx) = Cxx * Wprn / (Wprn' * Cxx * Wprn);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OTHER HELPERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function opt = propertylist2struct(varargin)
% PROPERTYLIST2STRUCT - Make options structure from parameter/value list
%
%   OPT= propertylist2struct('param1', VALUE1, 'param2', VALUE2, ...)
%   Generate a structure OPT with fields 'param1' set to value VALUE1, field
%   'param2' set to value VALUE2, and so forth.
%
%   See also set_defaults

opt= [];
if nargin==0,
  return;
end

if isstruct(varargin{1}) | isempty(varargin{1}),
  % First input argument is already a structure: Start with that, write
  % the additional fields
  opt= varargin{1};
  iListOffset= 1;
else
  % First argument is not a structure: Assume this is the start of the
  % parameter/value list
  iListOffset = 0;
end

nFields= (nargin-iListOffset)/2;
if nFields~=round(nFields),
  error('Invalid parameter/value list');
end

for ff= 1:nFields,
  fld = varargin{iListOffset+2*ff-1};
  if ~ischar(fld),
    error('Invalid parameter/value list');
  end
%  prp= varargin{iListOffset+2*ff};
%  opt= setfield(opt, fld, prp);
  opt.(fld)= varargin{iListOffset+2*ff};
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [opt, isdefault]= set_defaults(opt, varargin)
%[opt, isdefault]= set_defaults(opt, field/value list)
%
%Description:
% This functions fills in the given struct opt some new fields with
% default values, but only when these fields DO NOT exist before in opt.
% Existing fields are kept with their original values.
%
%Example:
%   opt= set_defaults(opt, 'color','g', 'linewidth',3);
%
% The second output argument isdefault is a struct with the same fields
% as the returned opt, where each field has a boolean value indicating
% whether or not the default value was inserted in opt for that field.

% blanker@cs.tu-berlin.de

% Set 'isdefault' to ones for the field already present in 'opt'
isdefault= [];
if ~isempty(opt),
  for Fld=fieldnames(opt)',
    isdefault.(Fld{1})= 0;
  end
end

defopt = propertylist2struct(varargin{:});
for Fld= fieldnames(defopt)',
  fld= Fld{1};
  if ~isfield(opt, fld),
    %% if opt is a struct *array*, the fields of all elements need to
    %% be set. This is done with the 'deal' function.
    [opt.(fld)]= deal(defopt.(fld));
    isdefault.(fld)= 1;
  end
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
