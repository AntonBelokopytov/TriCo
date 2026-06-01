close all
clear
clc

ft_path = 'C:\Users\anton\Documents\GitHub\CBI\site-packages\fieldtrip';
if ~exist('ft_defaults','file')
    addpath(ft_path);
end
ft_defaults;

%% Загрузка данных
elec = load("D:\OS(CURRENT)\data\simulation_support_data\eeg\elec.mat").elec;
laycfg = [];
laycfg.elec = elec;
lay = ft_prepare_layout(laycfg);     
G = load('D:\OS(CURRENT)\data\simulation_support_data\eeg\MNE_EEG_FWD_TRPL.mat').MNE_EEG_FWD_TRPL;

%% =================== ПАРАМЕТРЫ СИМУЛЯЦИИ ===================
Nsrc = 101;     
Ndistr = 2;        % Истинное количество целевых источников
Ts = 900;          % Длина симуляции в секундах
Fs = 250;          % Частота дискретизации
Wsize = 2;         % Размер окна для эпох (сек)
Ssize = 0.5;       % Шаг (сек)
SNR = 10^(0.5);    % Отношение сигнал/шум

fprintf('Генерация данных с топологией "Раскручивающаяся спираль"...\n');
[X, z_true, GA] = generate_spiral_sources(G, Nsrc, Ndistr, Ts, Fs, SNR);

%% =================== НАРЕЗКА НА ЭПОХИ ===================
fprintf('Нарезка на эпохи...\n');
X_epo = epoch_data(X', Fs, Wsize, Ssize);
nEpochs = size(X_epo, 3);
nChan = size(X_epo, 2);

% Считаем ковариации
Covs = zeros(nChan, nChan, nEpochs);
for ep_idx = 1:nEpochs
    Covs(:,:,ep_idx) = cov(X_epo(:,:,ep_idx));
end

%% =================== TANGENT SPACE & UMAP ===================
fprintf('Проекция в касательное пространство (Tangent Space)...\n');
Tcovs = Tangent_space_local(Covs);

fprintf('Построение UMAP эмбеддинга...\n');
u = UMAP("n_neighbors", 30, "n_components", 2, "min_dist", 0.3);
R = u.fit_transform(Tcovs'); 
Rmean = R - mean(R,1);

%% =================== eSPoC ===================
fprintf('Запуск eSPoC...\n');
% Подаем 2D координаты UMAP как внешнюю переменную!
[W, A, Vf, Vz, corrs] = espoc(X_epo, Rmean');

%% =================== ВИЗУАЛИЗАЦИЯ ===================
% 1. Визуализация UMAP (Спираль, раскрашенная по времени)
figure('Position', [100 100 600 500], 'Color', 'w');
time_color = linspace(1, Ts, nEpochs);
scatter(Rmean(:,1), Rmean(:,2), 25, time_color, 'filled', 'MarkerFaceAlpha', 0.7);
colormap(jet);
cb = colorbar;
ylabel(cb, 'Time (seconds)', 'FontSize', 11);
xlabel('UMAP Dimension 1', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('UMAP Dimension 2', 'FontSize', 12, 'FontWeight', 'bold');
title('Graph Embedding of Covariance Matrices (Unwinding Spiral)', 'FontSize', 14);
grid on;

% --- НАСТРОЙКИ ДЛЯ FIELDTRIP ---
topo = [];
topo.dimord = 'chan_time';
topo.label  = elec.label; 
topo.time   = 0;
plt_cfg = [];
plt_cfg.layout  = lay;    
plt_cfg.comment = 'no';
plt_cfg.style   = 'fill';
plt_cfg.marker  = 'off';  

% 2. Сравнение найденных компонент с истинными
figure('Position', [100 100 1400 800], 'Color', 'w');
t = tiledlayout(Ndistr, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

z_epo_raw = epoch_data(z_true', Fs, Wsize, Ssize); 
z_epo_true = squeeze(mean(z_epo_raw, 1)); 

for i = 1:Ndistr
    % Достаем результаты eSPoC
    w_est = W(1,:,i)'; 
    a_est = A(1,:,i)'; 
    
    % Выравнивание знака
    [~, m_idx] = max(abs(a_est));
    sign_corr = sign(a_est(m_idx));
    a_est = a_est * sign_corr;
    w_est = w_est * sign_corr;
    
    % Расчет извлеченной огибающей
    S_est = zeros(1, nEpochs);
    for ep = 1:nEpochs
        S_est(ep) = w_est' * Covs(:,:,ep) * w_est;
    end
    S_est = (S_est - mean(S_est)) / std(S_est);
    
    % Жадный поиск соответствия истинному источнику
    [~, true_idx] = max(abs(corr(S_est', z_epo_true')));
    
    % --- Топография истинного паттерна ---
    ax1 = nexttile(t, (i-1)*3 + 1);
    topo.avg = GA(:, true_idx);
    plt_cfg.figure = ax1;
    ft_topoplotER(plt_cfg, topo);
    title(sprintf('True Pattern (Source %d)', true_idx), 'FontSize', 12);
    
    % --- Топография найденного паттерна ---
    ax2 = nexttile(t, (i-1)*3 + 2);
    topo.avg = a_est;
    plt_cfg.figure = ax2;
    ft_topoplotER(plt_cfg, topo);
    title(sprintf('eSPoC Recovered Pattern (Comp %d)', i), 'FontSize', 12);
    
    % --- Динамика ---
    ax3 = nexttile(t, (i-1)*3 + 3);
    z_true_norm = (z_epo_true(true_idx,:) - mean(z_epo_true(true_idx,:))) / std(z_epo_true(true_idx,:));
    
    plot(z_true_norm, 'k', 'LineWidth', 2); hold on;
    plot(S_est, 'r', 'LineWidth', 1.5);
    
    title(sprintf('Power Envelope (Corr: %.2f)', abs(corr(S_est', z_true_norm'))), 'FontSize', 12);
    legend('True Spiral Envelope', 'eSPoC Recovered Power', 'Location', 'northeast');
    xlim([0 nEpochs]);
    grid on;
end
sgtitle('Decoding the UMAP Spiral with eSPoC', 'FontSize', 16, 'FontWeight', 'bold');

%% =================== ЛОКАЛЬНЫЕ ФУНКЦИИ ===================

function [X, z, GA] = generate_spiral_sources(G, Nsrc, Ndistr, Ts, Fs, SNR)
    N = Ts * Fs;
    flanker = 1 * Fs;
    
    [b, a] = butter(5, [8, 12] / (Fs / 2)); 
    [b_lp, a_lp] = butter(5, 0.5 / (Fs / 2)); 
    
    Gx = G(:, 1:3:end); Gy = G(:, 2:3:end); Gz = G(:, 3:3:end);  
    [Nsens, Nsites] = size(Gx);
    
    GA = zeros(Nsens, Nsrc);
    src_indsA = randperm(Nsites);
    for i = 1:Nsrc
        src_idx = src_indsA(i);
        r = rand(3, 1); r = r / norm(r);          
        GA(:, i) = Gx(:, src_idx)*r(1) + Gy(:, src_idx)*r(2) + Gz(:, src_idx)*r(3);
    end
    
    S = filtfilt(b, a, randn(Nsrc, N + 2 * flanker)')';
    S = S(:, flanker + 1 : end - flanker);
    z = zeros(Nsrc, N);
    
    t_sec = (1:N) / Fs;
    f_mod = 1 / (Ts / 4); % 4 полных витка спирали
    
    % Создаем линейно растущий радиус спирали
    radius_grow = linspace(0.1, 1.8, N);
    base_amp = 2.0; % Базовое смещение, чтобы сигнал не уходил ниже нуля
    
    for k = 1:Nsrc
        carrier_env = abs(hilbert(S(k, :)')');
        S_norm = S(k, :) ./ carrier_env;
        
        if k == 1
            % Источник 1: Синус с растущей амплитудой
            amp_mod = base_amp + radius_grow .* sin(2 * pi * f_mod * t_sec);
        elseif k == 2
            % Источник 2: Косинус с растущей амплитудой
            amp_mod = base_amp + radius_grow .* cos(2 * pi * f_mod * t_sec);
        else
            % Фоновые источники: случайный шум
            noise_mod = randn(1, N + 2 * flanker);
            lp_noise = filtfilt(b_lp, a_lp, noise_mod);
            lp_noise = lp_noise(flanker + 1 : end - flanker);
            lp_noise = lp_noise / std(lp_noise);
            amp_mod = lp_noise - min(lp_noise) + 0.1; 
        end
        
        S(k, :) = S_norm .* amp_mod;
        sigma_s = std(S(k, :));
        S(k, :) = S(k, :) / sigma_s;
        z(k, :) = (amp_mod / sigma_s).^2;
    end
    
    X_target = GA(:, 1:Ndistr) * S(1:Ndistr, :);
    X_bg = GA(:, Ndistr+1:end) * S(Ndistr+1:end, :);
    
    X_n = randn(Nsens, N);
    X_n = X_n - mean(X_n, 2);
    X_n = X_n ./ std(X_n, 0, 2);
    
    X = SNR * X_target + X_bg + 0.1 * X_n / norm(X_target,'fro');
end

function Tcovs = Tangent_space_local(Covs)
    [nChan, ~, nEpochs] = size(Covs);
    Cmean = mean(Covs, 3);
    Cmean_inv_half = Cmean^(-0.5);
    
    nFeat = nChan * (nChan + 1) / 2;
    Tcovs = zeros(nFeat, nEpochs);
    
    for i = 1:nEpochs
        C_proj = logm(Cmean_inv_half * Covs(:,:,i) * Cmean_inv_half);
        upper_mask = triu(true(nChan));
        upper_triu_mask = triu(true(nChan), 1);
        C_proj(upper_triu_mask) = C_proj(upper_triu_mask) * sqrt(2);
        Tcovs(:, i) = C_proj(upper_mask);
    end
end