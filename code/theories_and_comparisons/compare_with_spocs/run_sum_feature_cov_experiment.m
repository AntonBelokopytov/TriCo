close all
clear
clc

ft_path = 'C:\Users\ansbel\Documents\2Git\fieldtrip';

if ~exist('ft_defaults','file')
    addpath(ft_path);
end

ft_defaults;

%%
elec = load("elec.mat").elec;

topo = [];
topo.dimord = 'chan_time';
topo.label  = elec.label;  
topo.time   = 0;
topo.elec   = elec;

laycfg = [];
laycfg.elec = elec;
lay = ft_prepare_layout(laycfg);     

cfg = [];
cfg.marker       = '';
cfg.layout       = lay;
cfg.comment      = 'no';
cfg.style        = 'fill';
cfg.markersymbol = '';
cfg.colorbar     = 'no'; 
cfg.layout.pos(:, 1:2) = cfg.layout.pos(:, 1:2) * 1.1; 
cfg.layout.pos(:, 2) = cfg.layout.pos(:, 2) - 0.05;

%%
G = load('MNE_EEG_FWD_TRPL.mat').MNE_EEG_FWD_TRPL;
Nsrc = 100;
Ndistr = 1;

flanker = 1;
Ts = 100;
Fs = 250;

%%
[X_s, X_bg, X_n, z, GA] = generate_low_distributed_sources(G, Nsrc, Ndistr,...
    flanker, Ts, Fs);

SNR = 5;
X = SNR*X_s + X_bg + 0.1*trace(cov(X_s'))*X_n;

[b,a] = butter(3,[8,12]/(Fs/2)); 
Xfilt = filtfilt(b,a,X')';

%%
Ws = 1;
Ss = 1;
X_epo = epoch_data(Xfilt',Fs,Ws,Ss);
z_epo = epoch_data(z',Fs,Ws,Ss);
z_epo = squeeze(mean(z_epo,1));
z_epo = z_epo(1:Ndistr,:);


[W, A, Vf, Vz, corrs_espoc, Feat, Epochs_cov, eigenvalues] = espoc(X_epo, z_epo);
[Ws, As] = spoc(X_epo, z_epo(1,:));

Env = [];
for local_src_idx = 1:size(Ws,2)
    for ep_idx = 1:size(Epochs_cov,3)
        % Compute source power per epoch
        Env(ep_idx) = Ws(:,local_src_idx)' * ...
                      Epochs_cov(:,:,ep_idx) * ...
                      Ws(:,local_src_idx);
    end
    corrs_spoc(local_src_idx) = corr(Env',z_epo(1,:)');
end

% Compare correlations
figure;
stem(corrs_espoc'); hold on
stem(corrs_spoc')
legend('eSPoC','SPoC')

%%
% eSPoC — берём компонент с максимальной корреляцией
[~, idx_e] = max(abs(corrs_espoc));
a_e = A(:,1);

% SPoC — берём компонент с максимальной корреляцией
[~, idx_s] = max(abs(corrs_spoc));
a_s = As(:,1);

% Истинный паттерн
a_true = GA(:,1);

% Align signs to true pattern
if corr(a_e, a_true) < 0
    a_e = -a_e;
end

if corr(a_s, a_true) < 0
    a_s = -a_s;
end

figure;
set(gcf,'Color','w');
t = tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

% ---- TRUE
ax1 = nexttile;
title(ax1,'True pattern');
topo.avg = a_true;
cfg.figure = ax1;
ft_topoplotER(cfg, topo);

% ---- eSPoC
ax2 = nexttile;
title(ax2,'eSPoC pattern');
topo.avg = a_e;
cfg.figure = ax2;
ft_topoplotER(cfg, topo);

% ---- SPoC
ax3 = nexttile;
title(ax3,'SPoC pattern');
topo.avg = a_s;
cfg.figure = ax3;
ft_topoplotER(cfg, topo);

%%
