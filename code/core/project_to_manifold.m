function [W, A, s] = project_to_manifold(V, Wm, Cxx)
% Project filters to manifold
WW = upper2cov(V);
[Uw, S, ~] = eig(WW);
[s, idx] = sort(diag(abs(S)), 'descend');
Uw = Uw(:, idx);
n_local = size(Uw, 2);
n_channels = size(Wm, 1);
W = zeros(n_channels, n_local);
A = zeros(n_channels, n_local);
% Нормализация и восстановление паттернов
for local_src_idx = 1:n_local
    wi = Wm * Uw(:, local_src_idx);
    Wprn = wi / sqrt(wi' * Cxx * wi);
    W(:, local_src_idx) = Wprn;
    A(:, local_src_idx) = Cxx * Wprn / (Wprn' * Cxx * Wprn);
end
end
