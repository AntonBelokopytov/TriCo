close all
clear
clc

ft_path = 'D:\OS(CURRENT)\scripts\eSPoC_UMAP\0_2AOS\fieldtrip';

if ~exist('ft_defaults','file')
    addpath(ft_path);
end

ft_defaults;

%% =====================================================================
% LOAD DATA
% =====================================================================
sub_path = 'sub1_center_out_epochs.fif';

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

%%
