close all
clear
clc

ft_path = 'C:\Users\ansbel\Documents\2Git\fieldtrip';

if ~exist('ft_defaults','file')
    addpath(ft_path);
end

ft_defaults;

%%
elec = load("electrodes_data.mat").electrodes_data;

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
cfg.colorbar     = 'yes'; 

%%
G = load('MNE_EEG_FWD_TRPL.mat').MNE_EEG_FWD_TRPL;
Nsrc = 100;
Ndistr = 2;

flanker = 1;
Ts = 500;
Fs = 250;

%%
[X_s, X_bg, X_n, z, GA] = generate_low_distributed_sources(G, Nsrc, Ndistr,...
    flanker, Ts, Fs);

%%
SNR = 5;
X = SNR*X_s + X_bg + 0.1*trace(cov(X_s'))*X_n;

Ws = 1;
Ss = 1;
X_epo = epoch_data(X',Fs,Ws,Ss);
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
