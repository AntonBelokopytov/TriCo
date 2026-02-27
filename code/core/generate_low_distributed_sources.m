function [X_s, X_bg, X_n, z, GA] = generate_low_distributed_sources( ...
    G, Nsrc, Ntarget, flanker, Ts, Fs, rho_max)

if nargin < 7, rho_max = 0.9; end

N = Ts * Fs;
flanker_samples = round(flanker * Fs);
N_total = N + 2 * flanker_samples;

freq_bands = [8 12; 13 30];
[bl, al] = butter(3, 0.5/(Fs/2));

% --- Leadfield ---
Gx = G(:,1:3:end);
Gy = G(:,2:3:end);
Gz = G(:,3:3:end);

[Nsens, Nsites] = size(Gx);

GA = zeros(Nsens, Nsrc);
S  = zeros(Nsrc, N_total);

src_inds = randperm(Nsites);

%% -------------------------------------------------
% 1. Latent global factors (induce Cff correlations)
%% -------------------------------------------------

global_state = filtfilt(bl, al, randn(1, N_total));
global_state = (global_state - mean(global_state)) / std(global_state);

shared_factor = filtfilt(bl, al, randn(1, N_total));
shared_factor = (shared_factor - mean(shared_factor)) / std(shared_factor);

%% -------------------------------------------------
% 2. Target regressor
%% -------------------------------------------------

z_full = filtfilt(bl, al, randn(1, N_total));
z_full = (z_full - mean(z_full)) / std(z_full);
z = z_full(flanker_samples+1:end-flanker_samples);

%% -------------------------------------------------
% 3. Generate correlated sources
%% -------------------------------------------------

for k = 1:Nsrc
    
    % --- Spatial projection (add overlap) ---
    idx = src_inds(k);
    r = randn(3,1); r = r / norm(r);
    
    % Add weak spatial mixing to create leakage
    mix = 0.2 * randn(Nsens,1);
    
    GA(:,k) = Gx(:,idx)*r(1) + Gy(:,idx)*r(2) + Gz(:,idx)*r(3) + mix;
    
    % --- Carrier ---
    band = freq_bands(randi(size(freq_bands,1)), :);
    raw = filter(1, [1 -0.98], randn(1, N_total));
    [b,a] = butter(3, band/(Fs/2));
    carrier = filtfilt(b,a,raw);
    carrier = carrier ./ std(carrier);
    
    % --- Independent modulator ---
    m_ind = filtfilt(bl, al, randn(1, N_total));
    m_ind = (m_ind - mean(m_ind)) / std(m_ind);
    
    % --- Signed relation to z ---
    if k == 1
        rho = 1.0;     % strong positive
    elseif k == 2
        rho = -1.0;    % strong negative
    elseif k <= Ntarget
        rho = (2*rand - 1) * rho_max;
    else
        rho = 0;
    end
    
    % Correlated modulator
    m = rho*z_full ...
        + 0.5*shared_factor ...   % induces inter-source correlation
        + 0.3*global_state ...
        + sqrt(1-rho^2)*m_ind;
    
    % Linear amplitude model
    alpha = 0.5;
    amp = 1 + alpha*m;
    amp = max(amp, 0.1);
    
    S(k,:) = amp .* carrier;
    S(k,:) = S(k,:) / std(S(k,:));
end

S = S(:, flanker_samples+1:end-flanker_samples);

%% -------------------------------------------------
% 4. Structured brain noise (correlated)
%% -------------------------------------------------

N_noise = 300;
noise_sites = randi(Nsites, N_noise, 1);
G_noise = Gx(:, noise_sites);

latent_noise = filter(1, [1 -0.95], randn(5, N));  % low-rank noise
mix_noise = randn(N_noise,5);

S_noise = mix_noise * latent_noise;

X_brain_noise = G_noise * S_noise;
X_brain_noise = X_brain_noise / sqrt(trace(cov(X_brain_noise')));

%% -------------------------------------------------
% 5. Sensor white noise
%% -------------------------------------------------

X_sensor_white = randn(Nsens, N);
X_sensor_white = X_sensor_white / sqrt(trace(cov(X_sensor_white')));

%% -------------------------------------------------
% 6. Assemble
%% -------------------------------------------------

X_s  = GA(:,1:Ntarget) * S(1:Ntarget,:);
X_bg = GA(:,Ntarget+1:end) * S(Ntarget+1:end,:);

X_n = 0.6 * X_brain_noise + 0.4 * X_sensor_white;
X_n = X_n / sqrt(trace(cov(X_n')));

end
