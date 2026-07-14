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
