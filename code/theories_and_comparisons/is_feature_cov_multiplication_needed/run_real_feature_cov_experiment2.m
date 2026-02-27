close all
clear
clc

ft_path = 'C:\Users\ansbel\Documents\2Git\fieldtrip';

if ~exist('ft_defaults','file')
    addpath(ft_path);
end

ft_defaults;

%% =====================================================================
% LOAD DATA
% =====================================================================

% Path to pre-epoched data (Center-Out experiment)
sub_path = 'sub2_center_out_epochs.fif';

cfg = [];
cfg.dataset = sub_path;

% Load EEG/MEG data
Xinf = ft_preprocessing(cfg);

Fs = Xinf.fsample;   % Sampling frequency

% ---------------------------------------------------------------------
% Prepare structure for topography visualization
% ---------------------------------------------------------------------

topo = [];
topo.dimord = 'chan_time';
topo.label  = Xinf.elec.label;
topo.time   = 0;
topo.elec   = Xinf.elec;

% Prepare layout for topoplot
laycfg = [];
laycfg.elec = Xinf.elec;
lay = ft_prepare_layout(laycfg);

cfg = [];
cfg.marker       = 'labels';
cfg.layout       = lay;
cfg.comment      = 'no';
cfg.style        = 'fill';
cfg.markersymbol = 'o';
cfg.colorbar     = 'yes';

%% =====================================================================
% BANDPASS FILTER (mu rhythm 8–12 Hz)
% =====================================================================

[b,a] = butter(3,[8,12]/(Fs/2));   % 3rd-order Butterworth bandpass

% Epoch window size and step (seconds)
Ws = 0.2;   % window length
Ss = 0.1;   % step size

X_epochs = [];   % EEG epochs (band-pass filtered)
Z_epochs = [];   % Behavioral epochs

X_raw_filt = []; % Continuous filtered EEG
Z_raw = [];      % Continuous behavioral variables

%% =====================================================================
% LOOP OVER TRIALS
% =====================================================================

for i = 1:numel(Xinf.trial)

    disp(i)

    Trial = Xinf.trial{i};

    % --------------------------------------------------------------
    % Filter EEG channels (first 38 channels)
    % --------------------------------------------------------------
    Trial_filt = filtfilt(b,a,Trial(1:38,:)')';

    % Remove edge effects (half second from both ends)
    Trial_eeg_filt = Trial_filt(:,Fs/2:end-Fs/2);

    % Extract behavioral channels (e.g., velocity signals)
    Trial_var = Trial(39:end,Fs/2:end-Fs/2);

    % Concatenate continuous signals
    X_raw_filt = cat(2, X_raw_filt, Trial_eeg_filt);
    Z_raw = cat(2, Z_raw, Trial_var);

    % --------------------------------------------------------------
    % Epoching EEG and behavioral variables
    % --------------------------------------------------------------
    X_epochs = cat(3, X_epochs, epoch_data(Trial_eeg_filt',Fs,Ws,Ss));
    Z_epochs = cat(3, Z_epochs, epoch_data(Trial_var',Fs,Ws,Ss));
end

% Display dimensions
size(X_raw_filt)
size(Z_raw)
size(X_epochs)
size(Z_epochs)

%% =====================================================================
% DEFINE REGRESSOR (movement velocity power)
% =====================================================================

% Select velocity channels
velocity_chs = 6:8;

% Compute squared velocity (power-like measure)
Z = squeeze(mean(Z_epochs(:,velocity_chs,:).^2, 1));

% Average across selected channels
Z = mean(Z,1);

%% =====================================================================
% RUN eSPoC
% =====================================================================

[We, Ae, Vf, Vz, corrs_espoc, Feat, Epochs_cov, eigenvalues] = ...
    espoc(X_epochs, Z);

gl_f = Vf'*Feat;
gl_f = (gl_f - mean(gl_f)) / std(gl_f);

Envs = [];
for i=1:size(We,2)
    for j=1:size(Epochs_cov,3)
        Envs(i,j) = eigenvalues(i)*We(:,i)'*Epochs_cov(:,:,j)*We(:,i);
    end
end

gl_fw = sum(Envs,1);
gl_fw = (gl_fw - mean(gl_fw)) / std(gl_fw);

figure;
plot(gl_f)
hold on
plot(gl_fw)

cr = corr(gl_f', sum(Envs,1)')

%%
Envs = [];
for i=1:size(We,2)
    for j=1:size(Epochs_cov,3)
        Envs(i,j) = We(:,i)'*Epochs_cov(:,:,j)*We(:,i);
    end
end

cr = corr(gl_f', Envs')
figure;
plot(cr)
