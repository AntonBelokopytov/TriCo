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
velocity_chs = 1:11;

% Compute squared velocity (power-like measure)
Z = squeeze(mean(Z_epochs(:,velocity_chs,:).^2, 1));

% Average across selected channels
Zm = mean(Z(6:8,:),1);

%% =====================================================================
% RUN eSPoC
% =====================================================================

[We, Ae, Vf, Vz, corrs_espoc, Feat, Epochs_cov, eigenvalues] = ...
    espoc(X_epochs, Zm);
% [We, Ae, Vf, corrs_espoc, Feat, Epochs_cov, eigenvalues] = ...
%     espoc_r2(X_epochs, Zm);

% =====================================================================
% RUN classical SPoC for comparison
% =====================================================================

[Ws, As] = spoc(X_epochs, Zm);

Env = [];

for local_src_idx = 1:size(Ws,2)
    for ep_idx = 1:size(Epochs_cov,3)
        % Compute source power per epoch
        Env(ep_idx) = Ws(:,local_src_idx)' * ...
                      Epochs_cov(:,:,ep_idx) * ...
                      Ws(:,local_src_idx);
    end
    corrs_spoc(local_src_idx) = corr(Env',Zm');
end

% % Compare correlations
figure;
stem(corrs_espoc(1,:)'); hold on
stem(corrs_spoc')
legend('eSPoC','SPoC')

%%
figure;
plot((Vz'*Z)')

%%
figure;
plot(Vz(:,1)'*Z)

%% =====================================================================
% VISUALIZE FILTERS + PATTERNS (eSPoC vs SPoC)
% =====================================================================
espoc_src_idx = 1;   % индекс сравниваемого источника
spoc_src_idx = 1;   % индекс сравниваемого источника

% -------------------------
% eSPoC
% -------------------------
w_e = squeeze(We(1,:,espoc_src_idx))';
a_e = squeeze(Ae(1,:,espoc_src_idx))';

% Fix sign for consistency
[~, idx] = max(abs(w_e));
w_e = w_e * sign(w_e(idx));
[~, idx] = max(abs(a_e));
a_e = a_e * sign(a_e(idx));

% -------------------------
% SPoC
% -------------------------
w_s = Ws(:,spoc_src_idx);
a_s = As(:,spoc_src_idx);

[~, idx] = max(abs(w_s));
w_s = w_s * sign(w_s(idx));
[~, idx] = max(abs(a_s));
a_s = a_s * sign(a_s(idx));

% -------------------------
% Plot
% -------------------------
figure
set(gcf,'Color','w');
t = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

% eSPoC Filter
ax1 = nexttile(t,1);
title(ax1,'eSPoC Filter');
topo.avg = w_e;
cfg.figure = ax1;
ft_topoplotER(cfg, topo);

% eSPoC Pattern
ax2 = nexttile(t,2);
title(ax2,'eSPoC Pattern');
topo.avg = a_e;
cfg.figure = ax2;
ft_topoplotER(cfg, topo);

% SPoC Filter
ax3 = nexttile(t,3);
title(ax3,'SPoC Filter');
topo.avg = w_s;
cfg.figure = ax3;
ft_topoplotER(cfg, topo);

% SPoC Pattern
ax4 = nexttile(t,4);
title(ax4,'SPoC Pattern');
topo.avg = a_s;
cfg.figure = ax4;
ft_topoplotER(cfg, topo);

% =====================================================================
% VISUALIZE ENVELOPES (eSPoC vs SPoC)
% =====================================================================

% Continuous components
Comp_e = w_e' * X_raw_filt;
Comp_s = w_s' * X_raw_filt;

% Hilbert envelope
Env_e = abs(hilbert(Comp_e));
Env_s = abs(hilbert(Comp_s));

% Normalize for fair comparison
Env_e = (Env_e - mean(Env_e)) / std(Env_e);
Env_s = (Env_s - mean(Env_s)) / std(Env_s);

figure
set(gcf,'Color','w');

plot(Env_e,'LineWidth',1.2); hold on
plot(Env_s,'LineWidth',1.2);

legend('eSPoC','SPoC')
xlabel('Time samples')
ylabel('Normalized envelope')
title('Envelope comparison')
grid on

% =====================================================================
% VISUALIZE ENVELOPES PER TRIAL (eSPoC vs SPoC)
% =====================================================================

% ----- Continuous components -----
Comp_e = w_e' * X_raw_filt;
Comp_s = w_s' * X_raw_filt;

% ----- Envelopes -----
Env_e = abs(hilbert(Comp_e));
Env_s = abs(hilbert(Comp_s));

% ----- Reshape per trial -----
n_trials = numel(Xinf.trial);

Env_e_trials = reshape(Env_e, [], n_trials);
Env_s_trials = reshape(Env_s, [], n_trials);

% ----- Normalize each method separately -----
Env_e_trials = (Env_e_trials - mean(Env_e_trials(:))) / std(Env_e_trials(:));
Env_s_trials = (Env_s_trials - mean(Env_s_trials(:))) / std(Env_s_trials(:));

% ----- Plot -----
figure
set(gcf,'Color','w');
t = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

% eSPoC heatmap
ax1 = nexttile(t,1);
imagesc(Env_e_trials')
axis tight
xlabel('Time (samples)')
ylabel('Trial')
title('eSPoC Envelope')
colorbar

% SPoC heatmap
ax2 = nexttile(t,2);
imagesc(Env_s_trials')
axis tight
xlabel('Time (samples)')
ylabel('Trial')
title('SPoC Envelope')
colorbar
