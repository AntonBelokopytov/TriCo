function [X_s, X_bg, X_n, z, GA, S] = generate_distributed_sources(G, Nsrc, Ndistr, flanker, Ts, Fs)
    N = Ts * Fs;
    flanker = flanker * Fs;
    
    % Установка фильтров
    [b, a] = butter(5, [8, 12] / (Fs / 2)); % Для несущей (8-12 Гц)
    [b_lp, a_lp] = butter(5, 0.5 / (Fs / 2)); % ФНЧ < 0.5 Гц для амплитудной модуляции
    
    % Инициализация forward model
    Gx = G(:, 1:3:end);  
    Gy = G(:, 2:3:end);  
    Gz = G(:, 3:3:end);  
    [Nsens, Nsites] = size(Gx);
    
    % Создание случайных источников со случайным направлением
    GA = zeros(Nsens, Nsrc);
    src_indsA = randperm(Nsites);
    for i = 1:Nsrc
        src_idx = src_indsA(i);
        r = rand(3, 1); 
        r = r / norm(r);          
        GA(:, i) = Gx(:, src_idx)*r(1) + Gy(:, src_idx)*r(2) + Gz(:, src_idx)*r(3);
    end
    
    % Генерация временных рядов источников (несущие сигналы)
    S = filtfilt(b, a, randn(Nsrc, N + 2 * flanker)')';
    S = S(:, flanker + 1 : end - flanker);
    
    z = zeros(Nsrc, N);
    
    % Формирование огибающей и модуляция согласно Dähne et al. (2014)
    for k = 1:Nsrc
        % 1. Нормализация огибающей несущего сигнала к 1
        carrier_env = abs(hilbert(S(k, :)')');
        S_norm = S(k, :) ./ carrier_env;
        
        % 2. Создание функции амплитудной модуляции (отфильтрованный белый шум < 0.5 Гц)
        noise_mod = randn(1, N + 2 * flanker);
        lp_noise = filtfilt(b_lp, a_lp, noise_mod);
        lp_noise = lp_noise(flanker + 1 : end - flanker);
        
        % Восстанавливаем дисперсию модулирующего шума
        lp_noise = lp_noise / std(lp_noise);
        
        % 3. Добавление смещения
        amp_mod = lp_noise - min(lp_noise) + 0.05; 
        
        % 4. Применение амплитудной модуляции
        S(k, :) = S_norm .* amp_mod;
        
        % Приводим сигнал источника к единичной дисперсии 
        sigma_s = std(S(k, :));
        S(k, :) = S(k, :) / sigma_s;
        
        % 5. Целевая переменная z — это модуляция мощности.
        % Так как мы поделили амплитуду S на sigma_s, мощность уменьшилась на квадрат sigma_s
        z(k, :) = (amp_mod / sigma_s).^2;
    end

    % Генерация чистых сенсорных данных (целевые источники)
    X_s = GA(:, 1:Ndistr) * S(1:Ndistr, :);
    
    % Генерация фоновой активности (остальные источники)
    X_bg = GA(:, Ndistr + 1:end) * S(Ndistr + 1:end, :);
    
    % Генерация сенсорного шума (некоррелированный белый шум, нулевое среднее, единичная дисперсия)
    X_n = randn(Nsens, N);
    X_n = X_n - mean(X_n, 2);
    X_n = X_n ./ std(X_n, 0, 2);
end