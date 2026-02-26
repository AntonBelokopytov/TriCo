function [X_s, X_bg, X_n, z, GA] = generate_low_distributed_sources(G, Nsrc, Ndistr,...
    flanker, Ts, Fs, rho_max)
    % rho_max - максимальная амплитуда корреляции для дистракторов
    if nargin < 7, rho_max = 0.9; end

    N = Ts * Fs;
    flanker_samples = round(flanker * Fs);
    N_total = N + 2 * flanker_samples;
    freq_bands = [4, 8; 8, 13; 13, 30; 30, 45];
    [bl, al] = butter(3, 0.5 / (Fs / 2)); 
    
    Gx = G(:,1:3:end); Gy = G(:,2:3:end); Gz = G(:,3:3:end);
    [Nsens, Nsites] = size(Gx);
    GA = zeros(Nsens, Nsrc);
    src_indsA = randperm(Nsites);
    S = zeros(Nsrc, N_total);
    
    % --- 1. Глобальный драйв (состояние) ---
    common_drive = filtfilt(bl, al, randn(1, N_total));
    common_drive = (common_drive - min(common_drive)) / std(common_drive);

    % --- 2. Таргетная модуляция z ---
    mk_z = filtfilt(bl, al, randn(1, N_total));
    % Центрируем, чтобы удобно было делать отрицательную корреляцию
    target_modulator = (mk_z - mean(mk_z)) / std(mk_z); 
    z = target_modulator(flanker_samples+1:end-flanker_samples);

    % --- 3. Генерация источников ---
    for k = 1:Nsrc
        src_idx = src_indsA(k);
        r = rand(3,1); r = r / norm(r);          
        GA(:,k) = Gx(:,src_idx)*r(1) + Gy(:,src_idx)*r(2) + Gz(:,src_idx)*r(3);
        
        % Частотный диапазон
        if k <= Ndistr
            band = [8, 12];
        else
            band_idx = randi(size(freq_bands,1));
            band = freq_bands(band_idx, :);
        end
        
        % Несущая
        raw_noise = filter(1, [1, -0.99], randn(1, N_total));
        [b_src, a_src] = butter(3, band / (Fs / 2));
        carrier = filtfilt(b_src, a_src, raw_noise);
        S(k, :) = carrier ./ (abs(hilbert(carrier)) + eps);
        
        % Огибающая
        mk_indep = (filtfilt(bl, al, randn(1, N_total)));
        mk_indep = (mk_indep - mean(mk_indep)) / std(mk_indep);
        
        if k <= Ndistr
            % Главный таргет (всегда положительная корреляция 1.0)
            rho = 1.0;
            mk = target_modulator; 
        elseif k <= Ndistr + 10 % 10 источников-дистракторов
            % Случайная корреляция от -rho_max до +rho_max
            rho = (rand*2 - 1) * rho_max; 
            mk = (1 - abs(rho)) * mk_indep + rho * target_modulator;
        else
            % Остальной фон почти не коррелирует
            rho = 0.05;
            mk = (1 - rho) * mk_indep + rho * target_modulator;
        end
        
        % Добавляем общий драйв и переводим в положительную область (мощность)
        mk_final = mk + 0.2 * common_drive;
        % Экспонента или max(0) гарантирует, что мощность > 0
        S(k, :) = S(k, :) .* exp(mk_final); 
        S(k, :) = S(k, :) / std(S(k, :));
    end

    S = S(:, flanker_samples+1:end-flanker_samples);

    % --- 4. Пространственный шум мозга ---
    N_noise_dipoles = 400;
    noise_sites = randi(Nsites, N_noise_dipoles, 1);
    G_noise = Gx(:, noise_sites);
    S_noise = filter(1, [1, -0.95], randn(N_noise_dipoles, N));
    X_brain_noise = G_noise * S_noise; 
    X_brain_noise = X_brain_noise / sqrt(trace(cov(X_brain_noise')));

    % --- 5. Сборка ---
    X_s = GA(:, 1:Ndistr) * S(1:Ndistr, :);
    X_bg = GA(:, Ndistr+1:end) * S(Ndistr+1:end, :);
    X_sensor_white = randn(Nsens, N);
    X_sensor_white = X_sensor_white / sqrt(trace(cov(X_sensor_white')));
    
    X_n = 0.8 * X_brain_noise + 0.2 * X_sensor_white;
    X_n = X_n / sqrt(trace(cov(X_n')));
end
