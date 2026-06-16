close all; clear; clc;
ft_path = 'C:\Users\ansbel\Documents\GitHub\CBI\site-packages\fieldtrip';
if ~exist('ft_defaults','file'), addpath(ft_path); end
ft_defaults;

%% Загрузка анатомии
FM = load('forward_model.mat');
G = FM.leadfield;
elec = FM.elec;

laycfg = []; laycfg.elec = elec; lay = ft_prepare_layout(laycfg);      

%% 1. Параметры симуляции
Nsrc = 101;     
Ndistr = 3;        
flanker = 1;
Ts = 850;      
Fs = 250;
Ws = 2; Ss = 1;
fprintf('Генерация источников через базовую модель...\n');
[X_s_raw, X_bg, X_n, z_raw, GA, S_raw] = generate_distributed_sources(G, Nsrc, Ndistr, flanker, Ts, Fs);

% --- Сохраняем паттерны для топопланов ---
Ainit = GA(:, 1:Ndistr);

% 2. Создаем "Эстафету" (Гауссовские маски времени)
T_samples = Ts * Fs;
time_vec = 1:T_samples;
% Разносим пики активации по времени эксперимента
center1 = T_samples / 6;
center2 = T_samples / 2;
center3 = 5 * T_samples / 6;
sigma = T_samples / 32; % Ширина окна

mask1 = exp(-((time_vec - center1).^2) / (2*sigma^2));
mask2 = exp(-((time_vec - center2).^2) / (2*sigma^2));
mask3 = exp(-((time_vec - center3).^2) / (2*sigma^2));
masks = [mask1; mask2; mask3];

% Накладываем маски на целевые источники
S = S_raw; z = z_raw;
for i = 1:Ndistr
    carrier_env = abs(hilbert(S(i, :)')');
    S(i,:) = S(i,:) ./ carrier_env;
    S(i,:) = S(i,:) .* masks(i,:);      
    z(i,:) = z(i,:) .* (masks(i,:).^2);
end

% Пересобираем целевой ЭЭГ сигнал с новыми изолированными источниками
X_s = Ainit * S(1:Ndistr,:);

%%
% 3. Смешивание ЭЭГ с заданным SNR
SNR_fixed = 10^(1); 
gamma = 0.1;
X = SNR_fixed * X_s + X_bg + gamma * X_n / norm(X_s,'fro');
X = X - mean(X,1);
% X = bandpass(X', [8 12], Fs)'; % Обязательно фильтруем смешанный шум!

%%%%%%%%%%
% Снижение размерности (PCA)
%%%%%%%%%%
[U,S_pca,~] = svd(X,'econ');          
S_pca = diag(S_pca);
tol = max(size(X)) * eps(S_pca(1));
r = sum(S_pca > tol);
ve = S_pca.^2;
var_explained = cumsum(ve) / sum(ve);
var_explained(end) = 1;
n_components = find(var_explained>=0.99, 1);
n_components = max(min(n_components, r), 1);
U = U(:,1:n_components);               
X = U'*X;

% 4. Нарезка на эпохи
X_epo = epoch_data(X', Fs, Ws, Ss);
z_epo_raw = epoch_data(z(1:Ndistr,:)', Fs, Ws, Ss);
z_epo = squeeze(mean(z_epo_raw, 1));

% Нормализация целевых переменных
for i = 1:Ndistr
    z_epo(i,:) = (z_epo(i,:) - mean(z_epo(i,:))) / std(z_epo(i,:));
end

% === 5. Ковариации с легкой регуляризацией ===
nChan = size(X_epo, 2);
nEpochs = size(X_epo, 3);
Covs = zeros(nChan, nChan, nEpochs);
for i = 1:nEpochs
    C = cov(X_epo(:,:,i));
    Covs(:,:,i) = C; 
end
Tcovs = Tangent_space(Covs);           

% === 6. Создание 3D UMAP ===
clear u
u = UMAP("n_neighbors", 10, "n_components", 3, "min_dist", 0.1);
u.metric = 'euclidean';
R = u.fit_transform(Tcovs');

% === 7. ВИЗУАЛИЗАЦИЯ: UMAP + ОГИБАЮЩИЕ + ТОПОПЛАНЫ ===
% Переводим z_epo строго в диапазон [0, 1]
C_raw = zeros(3, nEpochs);
for i = 1:3
    z_min = min(z_epo(i,:));
    z_max = max(z_epo(i,:));
    C_raw(i,:) = (z_epo(i,:) - z_min) / (z_max - z_min);
end
C_raw = C_raw'; % [nEpochs x 3]

% Вычисляем интенсивность
intensity = max(C_raw, [], 2); 
C_final = C_raw ./ (intensity + eps); 
C_final(intensity < 0.1, :) = 0.8; 
alpha_vec = 0.6 + 0.4 * intensity; 
marker_sizes = 20 + 80 * intensity;

% --- Создание сложного макета фигуры ---
figure('Color', 'w', 'Position', [100, 100, 1500, 600]);
t = tiledlayout(2, 6, 'TileSpacing', 'compact', 'Padding', 'compact');

% График 1: 3D UMAP Траектория (Слева, занимает 2 строки и 3 колонки)
ax_umap = nexttile(t, 1, [2, 3]);
hold(ax_umap, 'on'); % Обязательно включаем hold on для добавления фиктивных точек

% Основной график
h = scatter3(ax_umap, R(:,1), R(:,2), R(:,3), marker_sizes, C_final, 'filled', ...
    'MarkerEdgeColor', 'none');
h.MarkerFaceAlpha = 'flat';
h.AlphaDataMapping = 'none';
h.AlphaData = alpha_vec;

% --- ДОБАВЛЯЕМ ЛЕГЕНДУ НА UMAP (Фиктивные NaN-точки) ---
h1 = scatter3(ax_umap, NaN, NaN, NaN, 80, [1 0 0], 'filled'); % Красный
h2 = scatter3(ax_umap, NaN, NaN, NaN, 80, [0 1 0], 'filled'); % Зеленый
h3 = scatter3(ax_umap, NaN, NaN, NaN, 80, [0 0 1], 'filled'); % Синий
h_bg = scatter3(ax_umap, NaN, NaN, NaN, 30, [0.8 0.8 0.8], 'filled'); % Серый фон

legend(ax_umap, [h1, h2, h3, h_bg], ...
    {'Source 1 active', 'Source 2 active', 'Source 3 active', 'Background Sources only'}, ...
    'Location', 'best', 'FontSize', 11);

title(ax_umap, 'UMAP Trajectory', 'FontSize', 14, 'FontWeight', 'bold');
xlabel(ax_umap, 'UMAP component 1'); ylabel(ax_umap, 'UMAP component 2'); zlabel(ax_umap, 'UMAP component 3');
grid(ax_umap, 'on'); view(ax_umap, 45, 30);

% График 2: Легенда огибающих (Справа сверху, занимает 1 строку и 3 колонки)
ax_env = nexttile(t, 4, [1, 3]); hold(ax_env, 'on'); grid(ax_env, 'on'); box(ax_env, 'on');
t_ep = 1:nEpochs;
plot(ax_env, t_ep, z_epo(1,:), 'r', 'LineWidth', 2);
plot(ax_env, t_ep, z_epo(2,:), 'g', 'LineWidth', 2);
plot(ax_env, t_ep, z_epo(3,:), 'b', 'LineWidth', 2);
title(ax_env, 'Source Envelopes', 'FontSize', 14, 'FontWeight', 'bold');
xlabel(ax_env, 'Time (Epochs)'); ylabel(ax_env, 'Amplitude');
legend(ax_env, 'Source 1', 'Source 2', 'Source 3', 'Location', 'best');
xlim(ax_env, [1 nEpochs]);

% Графики 3-5: Топопланы (Справа снизу, 3 независимые ячейки)
topo = [];
topo.dimord = 'chan_time';
topo.label  = elec.label;
topo.time   = 0;
topo.elec   = elec;

cfg_topo = [];
cfg_topo.layout   = lay;
cfg_topo.comment  = 'no';
cfg_topo.style    = 'fill';
cfg_topo.marker   = 'off';
cfg_topo.colorbar = 'no';

title_colors = {'r', 'g', 'b'};

for i = 1:3
    ax_topo = nexttile(t, 9 + i); 
    topo.avg = Ainit(:, i);       
    cfg_topo.figure = ax_topo;
    
    ft_topoplotER(cfg_topo, topo);
    title(ax_topo, sprintf('Source %d', i), 'FontSize', 12, 'FontWeight', 'bold', 'Color', title_colors{i});
end