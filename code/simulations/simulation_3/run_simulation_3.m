close all
clear
clc

ft_path = 'C:\Users\anton\Documents\GitHub\CBI\site-packages\fieldtrip'; % Поменяйте на ваш путь, если нужно
if ~exist('ft_defaults','file')
    addpath(ft_path);
end
ft_defaults;

%% Загрузка данных
G = load("D:\OS(CURRENT)\data\simulation_support_data\eeg\MNE_EEG_FWD_TRPL.mat").MNE_EEG_FWD_TRPL;
elec = load("D:\OS(CURRENT)\data\simulation_support_data\eeg\elec.mat").elec;
topo = [];
topo.dimord = 'chan_time';
topo.label  = elec.label;  
topo.time   = 0;
plt_cfg = [];
plt_cfg.layout = ft_prepare_layout(struct('elec', elec));     
plt_cfg.comment = 'no';
plt_cfg.style = 'fill';
plt_cfg.marker = 'off';

%% =================== ПАРАМЕТРЫ СИМУЛЯЦИИ ===================
Fs = 250; 
Ts = 600; 
N = Fs * Ts;
flanker = Fs * 1;
Nsrc = 100;
Ndistr = 2; % ВАЖНО: У нас 2 целевых источника

% --- 1. Генерируем 1 ОБЩУЮ управляющую переменную (сетевой драйвер) ---
% Это может быть координата UMAP или реальное поведение
[b_lp, a_lp] = butter(5, 0.5 / (Fs / 2));
noise_mod = randn(1, N + 2 * flanker);
z_master = filtfilt(b_lp, a_lp, noise_mod);
z_master = z_master(flanker + 1 : end - flanker);
z_master = (z_master - min(z_master)) / std(z_master) + 0.5; % Положительная огибающая

% --- 2. Генерируем несущие сигналы для 2-х пространственных диполей ---
[b, a] = butter(5, [8, 12] / (Fs / 2));
S_carrier = filtfilt(b, a, randn(2, N + 2 * flanker)')';
S_carrier = S_carrier(:, flanker + 1 : end - flanker);

% Нормализуем несущие и модулируем ОБЕ несущие ОДНИМ драйвером z_master
S = zeros(2, N);
for k = 1:2
    carrier_env = abs(hilbert(S_carrier(k, :)')');
    S_norm = S_carrier(k, :) ./ carrier_env;
    S(k, :) = S_norm .* z_master; % Ко-модуляция!
    S(k, :) = S(k, :) / std(S(k, :));
end

% --- 3. Пространственное смешивание ---
Gx = G(:, 1:3:end); Gy = G(:, 2:3:end); Gz = G(:, 3:3:end);
Nsites = size(Gx, 2);
GA = zeros(size(Gx, 1), 2);
src_indsA = randperm(Nsites, 2); % Выбираем 2 случайные локации
for i = 1:2
    r = rand(3, 1); r = r / norm(r);          
    GA(:, i) = Gx(:, src_indsA(i))*r(1) + Gy(:, src_indsA(i))*r(2) + Gz(:, src_indsA(i))*r(3);
end

% Целевой сигнал сети
X_target = GA * S;

% Фон и шум
[~, X_bg, X_n] = generate_distributed_sources(G, Nsrc, 1, 1, Ts, Fs);
SNR = 10;
X = SNR * X_target + X_bg + 0.1 * trace(cov(X_target')) * X_n;

%% =================== ОБРАБОТКА И eSPoC ===================
% Фильтрация
Xfilt = filtfilt(b, a, X')';
% Эпохирование (для ковариаций)
Ws = 2; Ss = 0.5;
epochs = epoch_data(Xfilt', Fs, Ws, Ss);
z_epo_raw = epoch_data(z_master', Fs, Ws, Ss);
z_epo = squeeze(mean(z_epo_raw, 1)); 
z_epo = (z_epo - mean(z_epo)) / std(z_epo);

fprintf('Запуск eSPoC для ОДНОЙ поведенческой переменной...\n');
% Обратите внимание: на вход подается только ОДНА переменная (1D вектор z_epo)
[W_e, A_e, ~, ~, corrs_e] = espoc(epochs, z_epo);

% eSPoC возвращает локальные компоненты. 
% Так как z_epo одномерный, глобальная ось всего одна: W_e(1, :, :)
% СТАЛО (Исправлено):
% Так как z_epo одномерный, espoc уже сделал squeeze до [Каналы x Локальные Компоненты]
if ismatrix(W_e) 
    w_local = W_e;
    a_local = A_e;
    corrs_local = corrs_e;
else
    w_local = squeeze(W_e(1, :, :));
    a_local = squeeze(A_e(1, :, :));
    corrs_local = squeeze(corrs_e(1, :));
end

%% =================== ВИЗУАЛИЗАЦИЯ ===================
figure('Position', [100 100 1200 600], 'Color', 'w');
t = tiledlayout(2, 3, 'TileSpacing', 'compact');
sgtitle('eSPoC Network Discovery: 1 Variable \rightarrow 2 Sources', 'FontSize', 16, 'FontWeight', 'bold');

% --- ИСТИННЫЕ ПАТТЕРНЫ ---
nexttile(1);
topo.avg = GA(:, 1); plt_cfg.figure = gca; ft_topoplotER(plt_cfg, topo);
title('True Source 1', 'FontSize', 14);

nexttile(4);
topo.avg = GA(:, 2); plt_cfg.figure = gca; ft_topoplotER(plt_cfg, topo);
title('True Source 2', 'FontSize', 14);

% --- НАЙДЕННЫЕ eSPoC ПАТТЕРНЫ (ИЗ ОДНОЙ ОСИ!) ---
% Берем первые две локальные компоненты (с самыми высокими корреляциями)
for i = 1:2
    est_a = a_local(:, i);
    % Жадный поиск, чтобы выровнять порядок для красивого плота
    corr1 = abs(corr(est_a, GA(:, 1)));
    corr2 = abs(corr(est_a, GA(:, 2)));
    if corr1 > corr2, plot_pos = 2; else plot_pos = 5; end
    
    % Выравнивание знака для плота
    [~, m_idx] = max(abs(est_a));
    est_a = est_a * sign(est_a(m_idx)) * sign(GA(m_idx, (plot_pos+1)/3));
    
    nexttile(plot_pos);
    topo.avg = est_a; plt_cfg.figure = gca; ft_topoplotER(plt_cfg, topo);
    title(sprintf('eSPoC Local Comp %d\n(Corr = %.2f)', i, corrs_local(i)), 'FontSize', 12);
end

% --- ДИНАМИКА ---
nexttile(3, [2 1]);
% Считаем огибающие найденных фильтров
env1 = zeros(1, size(epochs,3));
env2 = zeros(1, size(epochs,3));
for ep = 1:size(epochs,3)
    env1(ep) = w_local(:,1)' * cov(epochs(:,:,ep)) * w_local(:,1);
    env2(ep) = w_local(:,2)' * cov(epochs(:,:,ep)) * w_local(:,2);
end
env1 = (env1 - mean(env1))/std(env1);
env2 = (env2 - mean(env2))/std(env2);

plot(z_epo, 'k', 'LineWidth', 2.5); hold on;
plot(env1, 'r', 'LineWidth', 1.5);
plot(env2, 'b', 'LineWidth', 1.5);
legend('True Master Driver (z)', 'eSPoC Local Env 1', 'eSPoC Local Env 2');
title('Network Co-modulation', 'FontSize', 14);
xlim([0 200]);
grid on;