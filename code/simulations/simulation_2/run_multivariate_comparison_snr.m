close all
clear
clc

ft_path = 'C:\Users\anton\Documents\GitHub\CBI\site-packages\fieldtrip';
if ~exist('ft_defaults','file')
    addpath(ft_path);
end
ft_defaults;

%% Загрузка данных
elec = load("C:\Users\anton\Documents\GitHub\TriCo\data\support\elec.mat").elec;
laycfg = [];
laycfg.elec = elec;
lay = ft_prepare_layout(laycfg);     
G = load('C:\Users\anton\Documents\GitHub\TriCo\data\support\MNE_EEG_FWD_TRPL.mat').MNE_EEG_FWD_TRPL;

%% =================== ПАРАМЕТРЫ СИМУЛЯЦИИ ===================
Nsrc = 101;     
Ndistr = 2;        % Истинное количество целевых нейрональных источников
Nmix = 2;          % Размерность внешней (поведенческой) переменной
Nextract = Ndistr;      % Сколько компонент (источников) алгоритмы будут пытаться извлечь

if Nmix < Ndistr
    warning('Обычно количество внешних переменных (Nmix) должно быть >= числу источников (Ndistr).');
end
if Nextract > Nmix
    error('Число извлекаемых компонент (Nextract) не может быть больше размерности внешней переменной (Nmix)!');
end

noise_level = 0.1; 
flanker = 1;
Ts = 850;      
Fs = 250;
Ws = 1;
Ss = 1;
nMC = 500;        
n_train_epochs = 250; 
SNR_range = 10.^(-1.4:0.2:1); 
nSNR = length(SNR_range);
labels = {'eSPoC', 'mSPoC'};
nMethods = length(labels);

% 4D-Массивы для метрик: [nMC, nSNR, nMethods, Ndistr]
filcorr_test = zeros(nMC, nSNR, nMethods, Ndistr); 
patcorr      = zeros(nMC, nSNR, nMethods, Ndistr); 
zcorr_test   = zeros(nMC, nSNR, nMethods, Ndistr); 

%% ================= ИНИЦИАЛИЗАЦИЯ ПОТОКОВОГО ПУЛА =================
% Проверяем текущий пул. Если он не потоковый, удаляем и создаем нужный на 4 потока.
poolobj = gcp('nocreate');
if isempty(poolobj) || ~isa(poolobj, 'parallel.ThreadPool')
    if ~isempty(poolobj)
        delete(poolobj);
    end
    parpool('Threads', 4);
end

for mc_idx = 1:nMC
    fprintf('Monte-Carlo iteration: %d / %d\n', mc_idx, nMC);
    
    % 1. Генерируем "чистые" источники
    [X_s, X_bg, X_n, z, GA, S] = generate_distributed_sources(G, Nsrc, Ndistr, flanker, Ts, Fs);
    
    Ainit = GA(:, 1:Ndistr); 
    
    z_epo_raw = epoch_data(z(1:Ndistr,:)', Fs, Ws, Ss); 
    z_epo = squeeze(mean(z_epo_raw, 1)); 
    if Ndistr == 1, z_epo = z_epo(:)'; end 
    
    % Нормализуем целевые источники
    for src_i = 1:Ndistr
        z_epo(src_i,:) = (z_epo(src_i,:) - mean(z_epo(src_i,:))) / std(z_epo(src_i,:));
    end
    
    z_train = z_epo(:, 1:n_train_epochs);
    z_test  = z_epo(:, n_train_epochs+1 : end);
    
    % ================= СИМУЛЯЦИЯ ВНЕШНИХ СЕНСОРОВ =================
    % Случайная матрица смешивания на каждой итерации Монте-Карло
    ext_weights = randn(Nmix, Ndistr); 
    
    z_multidim = ext_weights * z_epo + noise_level * randn(Nmix, size(z_epo, 2));
    z_multidim_train = z_multidim(:, 1:n_train_epochs);
    z_multidim_test  = z_multidim(:, n_train_epochs+1 : end); 
    
    % Временные матрицы для parfor
    f_test_local = zeros(nSNR, nMethods, Ndistr); 
    p_corr_local = zeros(nSNR, nMethods, Ndistr);
    z_corr_local = zeros(nSNR, nMethods, Ndistr);
    
    % 2. Перебираем разные уровни SNR
    parfor (snr_idx = 1:nSNR, 4)
        current_SNR = SNR_range(snr_idx);
        
        X = current_SNR * X_s + X_bg + 0.1 * X_n / norm(X_s,'fro');
        X_epo = epoch_data(X', Fs, Ws, Ss);
        
        X_epo_train = X_epo(:,:, 1:n_train_epochs);
        X_epo_test  = X_epo(:,:, n_train_epochs+1 : end);
        
        nTrain = size(X_epo_train, 3);
        nTest  = size(X_epo_test, 3);
        nChan  = size(X_epo_test, 2);
        
        Covs_train = zeros(nChan, nChan, nTrain);
        for ep_idx = 1:nTrain
            Covs_train(:,:,ep_idx) = cov(X_epo_train(:,:,ep_idx));
        end
        
        Covs_test = zeros(nChan, nChan, nTest);
        for ep_idx = 1:nTest
            Covs_test(:,:,ep_idx) = cov(X_epo_test(:,:,ep_idx));
        end
        
        % Временные буферы для одной итерации SNR
        f_tmp   = zeros(nMethods, Ndistr);
        p_tmp   = zeros(nMethods, Ndistr);
        z_tmp   = zeros(nMethods, Ndistr);
        
        % ================= Оценка методов =================
        w_all  = zeros(nChan, Nextract, nMethods);
        a_all  = zeros(nChan, Nextract, nMethods);
        vz_all = zeros(Nmix, Nextract, nMethods); 
        
        % 1. eSPoC
        [W_e, A_e, ~, Vz, corrs_e] = espoc(X_epo_train, z_multidim_train);
        
        n_extracted_e = min(Nextract, size(W_e, 1));
        for f_idx = 1:n_extracted_e
            [~, idx] = sort(abs(corrs_e(f_idx,:)), 'descend');
            w_all(:, f_idx, 1)  = squeeze(W_e(f_idx, :, idx(1)))'; 
            a_all(:, f_idx, 1)  = squeeze(A_e(f_idx, :, idx(1)))';
            vz_all(:, f_idx, 1) = Vz(:, f_idx); 
        end
        
        % 2. mSPoC
        mspoc_opts = struct('tau_vector', 0, 'n_component_sets', Nextract, 'verbose', 0);
        [W_m, Wy, ~, A_m, ~] = mspoc(X_epo_train, z_multidim_train, mspoc_opts);
        
        n_extracted_m = min(Nextract, size(W_m, 2));
        for f_idx = 1:n_extracted_m
            w_all(:, f_idx, 2)  = W_m(:, f_idx); 
            a_all(:, f_idx, 2)  = A_m(:, f_idx); 
            vz_all(:, f_idx, 2) = Wy(:, f_idx); 
        end
        
        % ================= Проверка на тесте и Жадный Матчинг =================
        for m_idx = 1:nMethods
            w  = w_all(:,:,m_idx);
            a  = a_all(:,:,m_idx);
            vz = vz_all(:,:,m_idx);
            
            % Расчет тестовой огибающей ЭЭГ
            env_test = zeros(Nextract, nTest);
            for c = 1:Nextract
                for ep_idx = 1:nTest
                    env_test(c, ep_idx) = w(:,c)' * Covs_test(:,:,ep_idx) * w(:,c);
                end
            end
            
            % Раскрутка поведенческой смеси на тестовых данных
            z_rec_test = vz' * z_multidim_test; 
            
            % Матрица кросс-корреляций: [Nextract(найденные) x Ndistr(истинные)]
            corr_mat = abs(corr(env_test', z_test')); 
            
            % Универсальный ЖАДНЫЙ алгоритм сопоставления 
            for t = 1:min(Nextract, Ndistr)
                % Ищем глобальный максимум в оставшейся матрице
                [max_val, max_idx] = max(corr_mat(:));
                
                % Защита, если матрица полностью замаскирована
                if max_val == -1, break; end 
                
                [best_ext, best_tar] = ind2sub(size(corr_mat), max_idx);
                
                f_tmp(m_idx, best_tar) = max_val;
                p_tmp(m_idx, best_tar) = abs(corr(a(:,best_ext), Ainit(:,best_tar)));
                z_tmp(m_idx, best_tar) = abs(corr(z_rec_test(best_ext,:)', z_test(best_tar,:)'));
                
                % Маскируем (удаляем) уже привязанный фильтр и истинный источник
                corr_mat(best_ext, :) = -1;
                corr_mat(:, best_tar) = -1;
            end
        end
        
        % Запись в локальные буферы parfor
        f_test_local(snr_idx, :, :) = f_tmp;
        p_corr_local(snr_idx, :, :) = p_tmp;
        z_corr_local(snr_idx, :, :) = z_tmp;
    end
    
    % Запись результатов итерации MC в глобальные тензоры
    filcorr_test(mc_idx, :, :, :) = f_test_local;
    patcorr(mc_idx, :, :, :)      = p_corr_local;
    zcorr_test(mc_idx, :, :, :)   = z_corr_local;
end

%% ================= Вычисление статистики =================
% Используем reshape, чтобы измерения не схлопнулись при nMethods=1
mean_f = reshape(mean(filcorr_test, 1), [nSNR, nMethods, Ndistr]);
ci_f   = reshape(1.96 * std(filcorr_test, 0, 1) / sqrt(nMC), [nSNR, nMethods, Ndistr]);
mean_p = reshape(mean(patcorr, 1), [nSNR, nMethods, Ndistr]);
ci_p   = reshape(1.96 * std(patcorr, 0, 1) / sqrt(nMC), [nSNR, nMethods, Ndistr]);
mean_z = reshape(mean(zcorr_test, 1), [nSNR, nMethods, Ndistr]);
ci_z   = reshape(1.96 * std(zcorr_test, 0, 1) / sqrt(nMC), [nSNR, nMethods, Ndistr]);

%% ================= Визуализация (Динамическая сетка) =================
x = SNR_range; 
% Высота фигуры адаптируется под количество целевых источников
figure('Position', [100 100 1600 350*Ndistr], 'Color', 'w'); 
colors = [0.8 0 0;    % Красный для eSPoC
          0 0 0.8];   % Синий для mSPoC
markers = {'o', 's'};

for src_i = 1:Ndistr
    % 1. Power Correlation
    subplot(Ndistr, 3, (src_i-1)*3 + 1); hold on; box on;
    for m = 1:nMethods
        y = mean_f(:, m, src_i)'; ci = ci_f(:, m, src_i)';
        fill([x fliplr(x)], [y-ci fliplr(y+ci)], colors(m,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');   
        semilogx(x, y, ['-', markers{m}], 'Color', colors(m,:), 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'DisplayName', labels{m});
    end
    title(sprintf('Src %d: Power Correlation', src_i), 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Correlation, r', 'FontSize', 11);
    ylim([0 1.05]); xlim([min(x) max(x)]);
    set(gca, 'XScale', 'log', 'GridAlpha', 0.3, 'MinorGridAlpha', 0.4, 'TickDir', 'out'); grid on;
    if src_i == 1, legend('Location', 'northwest', 'FontSize', 10); end
    
    % 2. Pattern Correlation
    subplot(Ndistr, 3, (src_i-1)*3 + 2); hold on; box on;
    for m = 1:nMethods
        y = mean_p(:, m, src_i)'; ci = ci_p(:, m, src_i)';
        fill([x fliplr(x)], [y-ci fliplr(y+ci)], colors(m,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');   
        semilogx(x, y, ['-', markers{m}], 'Color', colors(m,:), 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'DisplayName', labels{m});
    end
    title(sprintf('Src %d: Spatial Pattern Corr', src_i), 'FontSize', 12, 'FontWeight', 'bold');
    ylim([0 1.05]); xlim([min(x) max(x)]);
    set(gca, 'XScale', 'log', 'GridAlpha', 0.3, 'MinorGridAlpha', 0.4, 'TickDir', 'out'); grid on;
    
    % 3. Behavioral Unmixing Correlation
    subplot(Ndistr, 3, (src_i-1)*3 + 3); hold on; box on;
    for m = 1:nMethods
        y = mean_z(:, m, src_i)'; ci = ci_z(:, m, src_i)';
        fill([x fliplr(x)], [y-ci fliplr(y+ci)], colors(m,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');   
        semilogx(x, y, ['-', markers{m}], 'Color', colors(m,:), 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'DisplayName', labels{m});
    end
    title(sprintf('Src %d: Behavioral Var Corr', src_i), 'FontSize', 12, 'FontWeight', 'bold');
    ylim([0 1.05]); xlim([min(x) max(x)]);
    set(gca, 'XScale', 'log', 'GridAlpha', 0.3, 'MinorGridAlpha', 0.4, 'TickDir', 'out'); grid on;
    
    % Подписи оси X только для самого нижнего ряда
    if src_i == Ndistr
        subplot(Ndistr, 3, (src_i-1)*3 + 1); xlabel('Signal-to-Noise Ratio (SNR)', 'FontSize', 11);
        subplot(Ndistr, 3, (src_i-1)*3 + 2); xlabel('Signal-to-Noise Ratio (SNR)', 'FontSize', 11);
        subplot(Ndistr, 3, (src_i-1)*3 + 3); xlabel('Signal-to-Noise Ratio (SNR)', 'FontSize', 11);
    end
end

%% ================= Сохранение итоговой картинки =================
drawnow;
savefig(gcf, 'mspoc_vs_espoc_multi.fig');
exportgraphics(gcf, 'mspoc_vs_espoc_multi.jpg', 'Resolution', 300);
fprintf('Saved figures: mspoc_vs_espoc_multi.fig and mspoc_vs_espoc_multi.jpg\n');