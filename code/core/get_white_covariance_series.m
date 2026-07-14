function [F, Wm, Cxx, Epochs_cov, Epochs_covW] = get_white_covariance_series(X_epochs, opt)
[n_samples, n_channels, n_epochs] = size(X_epochs);
n_features = (n_channels^2 - n_channels)/2 + n_channels;
X_mean = mean(X_epochs, 1);
X_cen = X_epochs - X_mean;

if exist('pagemtimes', 'builtin') || exist('pagemtimes', 'file')
    Epochs_cov = pagemtimes(X_cen, 'transpose', X_cen, 'none') / (n_samples - 1);
else
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
Epochs_covW = zeros(n_channels, n_channels, n_epochs);

if exist('pagemtimes', 'builtin') || exist('pagemtimes', 'file')
    % Тензорное отбеливание
    Epochs_covW = pagemtimes(Wm, pagemtimes(Epochs_cov, Wm'));
    for ep_idx = 1:n_epochs
        F(:, ep_idx) = cov2upper(Epochs_covW(:,:,ep_idx));
    end
else
    % Fallback
    for ep_idx = 1:n_epochs
        XcovW = Wm * Epochs_cov(:,:,ep_idx) * Wm';
        Epochs_covW(:,:,ep_idx) = XcovW;
        F(:, ep_idx) = cov2upper(XcovW);
    end
end
F = F - mean(F, 2);
end
