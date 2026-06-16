close all
clear
clc

ft_path = 'C:\Users\ansbel\Documents\GitHub\CBI\site-packages\fieldtrip';

if ~exist('ft_defaults','file')
    addpath(ft_path);
end

ft_defaults;

%% =====================================================================
% LOAD DATA
% =====================================================================
% sub_path = 'D:\OS(CURRENT)\data\parkinson\pathology\Patient_1_CenterOut_OFF_EEG_clean_epochs.fif';
sub_path = 'E:\OS(CURRENT)\data\parkinson\control\Control_7_CenterOut_epochs.fif';

cfg = [];
cfg.dataset = sub_path;
Xinf = ft_preprocessing(cfg);    % Load EEG/MEG data
Fs = Xinf.fsample;               % Sampling frequency

% Initialize topography structure
topo = [];
topo.dimord = 'chan_time';
topo.label  = Xinf.elec.label;  
topo.time   = 0;
topo.elec   = Xinf.elec;
topo.time    = 0;

% Prepare FieldTrip layout for topography plotting
laycfg = [];
laycfg.elec = Xinf.elec;
lay = ft_prepare_layout(laycfg);     

cfg.marker       = 'labels';
cfg.layout       = lay;
cfg.comment      = 'no';
cfg.style        = 'fill';
cfg.markersymbol = 'o';
cfg.colorbar     = 'yes'; 

N_trials = numel(Xinf.trial)

%%
fmin = 15;
fmax = 25;
Ws = 0.5;   
Ss = Ws/2; 
% Ws = 0.5;   
% Ss = 0.25; 

[b_band, a_band] = butter(3, [fmin fmax] / (Fs/2), 'bandpass');

X_filt = [];
X_tr_epochs = []; 
Z_tr_epochs = [];
for tr_i = 1:N_trials
    tr_i
    X_tr = Xinf.trial{tr_i};
    X_tr = X_tr(1:38,:) - mean(X_tr(1:38,:),2);
    X_tr_filt = filtfilt(b_band, a_band, X_tr'); 
    X_filt(:,:,tr_i) = X_tr_filt;
    temp_epochs = epoch_data(X_tr_filt, Fs, Ws, Ss);
    X_tr_epochs(:,:,:,tr_i) = temp_epochs;

    Z_tr = Xinf.trial{tr_i}(39:end,:);
    temp_epochs = epoch_data(Z_tr', Fs, Ws, Ss);
    temp_epochs = squeeze(mean(temp_epochs.^2,1));
    Z_tr_epochs(:,:,tr_i) = temp_epochs;
end

%%
% idxs = [  0,   1,   2,   3,   6,   8,  12,  13,  15,  16,  17,  19,  22,...
%         24,  25,  28,  30,  32,  33,  35,  37,  39,  41,  42,  44,  45,...
%         48,  50,  51,  55,  57,  58,  59,  62,  63,  65,  69,  70,  73,...
%         74,  75,  78,  79,  82,  85,  88,  89,  92,  93,  94,  97, 100,...
%        101, 103, 105, 106, 110, 111, 112, 117, 118, 121, 122, 124, 126,...
%        128, 131, 132, 134, 136, 137, 138] + 1;

idxs = [  2,   3,   6,   7,   8,  11,  12,  15,  18,  20,  21,  24,  26,...
        28,  30,  32,  34,  37,  38,  40,  42,  43,  46,  48,  50,  52,...
        54,  57,  58,  61,  63,  65,  67,  68,  70,  71,  73,  77,  78,...
        79,  81,  83,  84,  86,  89,  90,  92,  95,  96,  98, 101, 103,...
       105, 106, 108, 109, 113, 115, 118, 119, 121, 123, 126, 127, 128,...
       131, 133, 134, 137, 139, 141, 142, 145, 146] + 1;

Ntr = 200;
X_eps_cond = X_tr_epochs(:,:,:,idxs);
X_eps = X_eps_cond(:,:,:);
X_eps = X_eps_cond(:,:,1:end);

z_eps_cond = Z_tr_epochs(:,:,idxs);
z_eps = z_eps_cond(6:8,:);
z_eps = z_eps_cond(6:8,1:end);

X_filt_cond = X_filt(:,:,idxs);

pos = Xinf.elec.chanpos; 
Dists = pdist2(pos, pos); 

ex_variable = mean(z_eps,1);
ex_variable = (ex_variable - mean(ex_variable)) / std(ex_variable);

Epochs_cov = [];
for ep_idx = 1:size(X_eps,3)
    Xcov = cov(X_eps(:,:,ep_idx));
    Epochs_cov(:,:,ep_idx) = Xcov;
end

%%
[W, A, Vf, Vz, corrs] = espoc(X_eps, ex_variable);

% figure;
stem(corrs')
hold on 

%%
[W, A] = spoc(X_eps,ex_variable);

corrs_sp = [];
for w_i = 1:size(W,2)
    for ep_i = 1:size(X_eps,3)
        Env(w_i,ep_i) = W(:,w_i)' * Epochs_cov(:,:,ep_i) * W(:,w_i);
    end
    Env(w_i,:) = (Env(w_i,:) - mean(Env(w_i,:))) ./ std(Env(w_i,:));
    corrs_sp(w_i) = corr(Env(w_i,:)',ex_variable');
end

stem(corrs_sp)

%%
comp_idx = 38;

wx = W(:,comp_idx);
ax = A(:,comp_idx);

[~, max_idx] = max(abs(ax));
ax = ax * sign(ax(max_idx));

clear Yenv Yseg 

% Определяем количество триалов после выборки по условию (idxs)
num_cond_trials = size(X_filt_cond, 3);

Yenv = zeros(size(X_filt_cond, 1), num_cond_trials);
Yseg = zeros(size(X_filt_cond, 1), num_cond_trials);

% 1. Корректно извлекаем компонент и его огибающую для каждой эпохи
for tr_i = 1:num_cond_trials
    % Умножаем [time x channels] на [channels x 1]
    comp_signal = X_filt_cond(:,:,tr_i) * wx; 
    
    Yseg(:,tr_i) = comp_signal;                 % Исходный сигнал источника
    Yenv(:,tr_i) = abs(hilbert(comp_signal));   % Его огибающая
end

Filt = wx;

% 3. Ограничиваем структуру электродов до 38
elec_38 = Xinf.elec;
elec_38.label = Xinf.elec.label(1:38);
elec_38.chanpos = Xinf.elec.chanpos(1:38, :);
elec_38.elecpos = Xinf.elec.elecpos(1:38, :);

figure; hold on; grid on
set(gcf,'Color','w');

% 4. Задаем оси времени
tsec_env = linspace(-1, 6, size(Yenv,1));
tsec_seg = linspace(-1, 6, size(Yseg,1));

E        = size(Yenv, 2);
env_mean = mean(Yenv, 2, 'omitnan');
sd       = std (Yenv, 0, 2, 'omitnan');

% Подготовка FieldTrip layout для 38 каналов
lay = ft_prepare_layout(struct('elec', elec_38));

% ==== ГЛАВНАЯ РАСКЛАДКА ====================================================
t = tiledlayout(3,2, 'TileSpacing','compact', 'Padding','compact');

% ---- (Left) Heatmap: все эпохи, огибающая ----
axH = nexttile(t, 1, [3 1]);          
imagesc(axH, tsec_env, 1:E, Yenv');
set(axH,'YDir','normal','Color','w'); grid(axH,'on');
xline(axH, 0, 'k--', 'LineWidth', 1);
xlabel(axH,'time, s'); ylabel(axH,'epoch');
title(axH,'Envelope per epoch');
colorbar(axH);
% caxis(axH, [0 4]); % Закомментировал, чтобы цвета автоподстроились под данные

% ---- (Right-Top) ERP без усреднения (Сигнал + Огибающая) ----
axERP = nexttile(t, 2); hold(axERP,'on'); grid(axERP,'on');
set(axERP,'Color','w');
% Рисуем сырой сигнал источника серым
plot(axERP, tsec_seg, Yseg, 'Color', [0.8 0.8 0.8]); 
% Поверх рисуем огибающие полупрозрачным синим для наглядности
plot(axERP, tsec_env, Yenv, 'Color', [0.1 0.3 0.9 0.3]); 
xline(axERP, 0, 'k--', 'LineWidth', 1);
xlabel(axERP,'time, s'); ylabel(axERP,'amplitude');
title(axERP,'Source activity & Envelopes (all epochs)');

% ---- (Right-Middle) Средняя огибающая ± SD ----
axENV = nexttile(t, 4); hold(axENV,'on'); grid(axENV,'on');
set(axENV,'Color','w');
xfill = [tsec_env, fliplr(tsec_env)];
yfill = [ (env_mean+sd).', fliplr((env_mean-sd).') ];
fill(axENV, xfill, yfill, [0.3 0.5 1.0], 'FaceAlpha',0.2, 'EdgeColor','none');
plot(axENV, tsec_env, env_mean, 'Color', [0.1 0.3 0.9], 'LineWidth', 2);
xline(axENV, 0, 'k--', 'LineWidth', 1);
xlabel(axENV,'time, s'); ylabel(axENV,'envelope (a.u.)');
title(axENV,'Mean envelope \pm SD');

% ==== (Right-Bottom) ДВА ГРАФИКА: ФИЛЬТР и ПАТТЕРН =========================
axTmp = nexttile(t, 6);
pos6  = axTmp.OuterPosition;   
delete(axTmp);

figCol = get(gcf,'Color');     
p6 = uipanel('Parent', gcf, ...
             'Units','normalized', ...
             'Position', pos6, ...
             'BorderType','none', ...
             'BackgroundColor', figCol);   
t6 = tiledlayout(p6, 1, 2, 'TileSpacing','compact', 'Padding','compact');

% --------- Левый подтайл: ТОПОГРАФИЯ ФИЛЬТРА ---------
axFiltTopo = nexttile(t6, 1);
set(axFiltTopo, 'Color','w');
valsFilt = Filt(:);
topoF = [];
topoF.dimord = 'chan_time';
topoF.label  = elec_38.label; 
topoF.time   = 0;
topoF.avg    = valsFilt;
topoF.elec   = elec_38;       
cfg = [];
cfg.figure       = axFiltTopo;
cfg.layout       = lay;
cfg.comment      = 'no';
cfg.style        = 'fill';
cfg.markersymbol = 'o';
cfg.zlim         = 'maxmin';
cfg.layout.pos(:, 1:2) = cfg.layout.pos(:, 1:2) * 1.1; 
cfg.layout.pos(:, 2) = cfg.layout.pos(:, 2) - 0.05;
ft_topoplotER(cfg, topoF);
title(axFiltTopo,'Filter');

% --------- Правый подтайл: ТОПОГРАФИЯ ПАТТЕРНА ---------
axPatTopo = nexttile(t6, 2);
set(axPatTopo, 'Color','w');
valsPat = ax(:); 
topoP = [];
topoP.dimord = 'chan_time';
topoP.label  = elec_38.label; 
topoP.time   = 0;
topoP.avg    = valsPat;
topoP.elec   = elec_38;       
cfg = [];
cfg.figure       = axPatTopo;
cfg.layout       = lay;
cfg.comment      = 'no';
cfg.style        = 'fill';
cfg.markersymbol = 'o';
cfg.zlim         = 'maxmin';
cfg.layout.pos(:, 1:2) = cfg.layout.pos(:, 1:2) * 1.1; 
cfg.layout.pos(:, 2) = cfg.layout.pos(:, 2) - 0.05;
ft_topoplotER(cfg, topoP);
title(axPatTopo,'Pattern');

% Перекрашиваем все оси внутри панели в белый
set(findall(p6, 'type','axes'), 'Color', figCol);

% ==== Синхронизация осей по X у временных графиков =========================
linkaxes([axH axERP axENV], 'x');
