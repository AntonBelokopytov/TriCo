close all
clear
clc

ft_path = 'C:\Users\anton\Documents\GitHub\site-packages\fieldtrip';
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
Nsrc = 101;     % 1 целевой + 100 фоновых
Ndistr = 1;
flanker = 1;
Ts = 850;      
Fs = 250;
Ws = 1;
Ss = 1;
nMC = 500;
SNR_range = 10.^(-1.4:0.2:1);
gamma = 0.1;
nSNR = length(SNR_range);

% Обновлено для 3 методов
methods = {@espoc, @spoc, @spoc_r2}; 
nMethods = length(methods);
labels = {'eSPoC', 'SPoC_{\lambda}', 'SPoC_{r^2}'};

filcorr_train = zeros(nMC, nSNR, nMethods); 
filcorr_test  = zeros(nMC, nSNR, nMethods);
patcorr       = zeros(nMC, nSNR, nMethods);

%% ================= ИНИЦИАЛИЗАЦИЯ ПУЛА ПРОЦЕССОВ =================
% Проверяем текущий пул. Если он не процессный, удаляем и создаем на 4 процесса.
poolobj = gcp('nocreate');
if isempty(poolobj) || ~isa(poolobj, 'parallel.ProcessPool')
    if ~isempty(poolobj)
        delete(poolobj);
    end
    parpool('Processes', 4);
end

for mc_idx = 1:nMC
    fprintf('Monte-Carlo iteration: %d / %d\n', mc_idx, nMC);
    
    [X_s, X_bg, X_n, z, GA, S] = generate_distributed_sources(G, Nsrc, Ndistr, flanker, Ts, Fs);
    Ainit = GA(:,1); 
    
    % Временные матрицы для parfor
    filcorr_train_local = zeros(nSNR, nMethods); 
    filcorr_test_local  = zeros(nSNR, nMethods);
    patcorr_local       = zeros(nSNR, nMethods);
    
    parfor (snr_idx = 1:nSNR, 4)
        % ================= Data generation =================
        current_SNR = SNR_range(snr_idx);
        X = current_SNR*X_s + X_bg + gamma * X_n / norm(X_s,'fro');
        
        X_epo = epoch_data(X', Fs, Ws, Ss);        
        z_epo_raw = epoch_data(z(1,:)', Fs, Ws, Ss); 
        z_epo = squeeze(mean(z_epo_raw, 1));         
        
        % ================= Train / Test split =================
        X_epo_train = X_epo(:,:,1:250);
        z_epo_train = z_epo(1:250);
        
        X_epo_test = X_epo(:,:,251:250+600);
        z_epo_test = z_epo(251:250+600);
        
        % ================= Covariance matrices =================
        nTrain = size(X_epo_train, 3);
        nTest  = size(X_epo_test, 3);
        nChan  = size(X_epo_test, 2);
        
        % Ковариации для тренировочной выборки
        Covs_train = zeros(nChan, nChan, nTrain);
        for ep_idx = 1:nTrain
            Covs_train(:,:,ep_idx) = cov(X_epo_train(:,:,ep_idx));
        end
        
        % Ковариации для тестовой выборки
        Covs_test = zeros(nChan, nChan, nTest);
        for ep_idx = 1:nTest
            Covs_test(:,:,ep_idx) = cov(X_epo_test(:,:,ep_idx));
        end
        
        % Временные буферы для слайсинга
        temp_train = zeros(1, nMethods);
        temp_test  = zeros(1, nMethods);
        temp_pat   = zeros(1, nMethods);
        
        % ================= Methods Evaluation =================
        for m_idx = 1:nMethods
            alg = methods{m_idx};
            
            [W, A] = alg(X_epo_train, z_epo_train);
            w = W(:,1);
            
            env_train = zeros(nTrain, 1);
            for ep_idx = 1:nTrain
                env_train(ep_idx) = w' * Covs_train(:,:,ep_idx) * w;
            end
            temp_train(m_idx) = corr(env_train(:), z_epo_train(:));
            
            env_test = zeros(nTest, 1);
            for ep_idx = 1:nTest
                env_test(ep_idx) = w' * Covs_test(:,:,ep_idx) * w;
            end
            temp_test(m_idx) = corr(env_test(:), z_epo_test(:));
            
            temp_pat(m_idx) = abs(corr(A(:,1), Ainit));
        end
        
        % Запись результатов итерации в локальные матрицы
        filcorr_train_local(snr_idx, :) = temp_train;
        filcorr_test_local(snr_idx, :)  = temp_test;
        patcorr_local(snr_idx, :)       = temp_pat;
    end
    
    filcorr_train(mc_idx, :, :) = filcorr_train_local;
    filcorr_test(mc_idx, :, :)  = filcorr_test_local;
    patcorr(mc_idx, :, :)       = patcorr_local;
end

%% ================= Вычисление статистики =================
mean_filt_test  = squeeze(mean(filcorr_test, 1));
mean_pat        = squeeze(mean(patcorr, 1));   

ci_filt_test  = squeeze(1.96 * std(filcorr_test, 0, 1) / sqrt(nMC));
ci_pat        = squeeze(1.96 * std(patcorr, 0, 1) / sqrt(nMC));

x = SNR_range;
xticks_vals = 10.^[-1 -0.4 0 0.4 1];
xticks_lbls = {'10^{-1}','10^{-0.4}','10^{0}','10^{0.4}','10^{1}'};

% Фигура теперь под 2 графика (сделал чуть уже)
figure('Position', [100 100 900 400], 'Color', 'w'); 

% Цвета для трех алгоритмов (Красный, Синий, Зеленый)
colors = [0.8 0 0;    
          0 0 0.8;
          0 0.6 0];   

% Маркеры для трех алгоритмов
markers = {'o', 's', '^'}; 

% ---------------- Plot 1: TEST Power Time Course Correlation ----------------
subplot(1,2,1); hold on; box on;
for m = 1:nMethods
    y  = mean_filt_test(:,m)';
    ci = ci_filt_test(:,m)';
    
    fill([x fliplr(x)], [y-ci fliplr(y+ci)], colors(m,:), ...
         'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');   
     
    semilogx(x, y, 'Color', colors(m,:), 'LineWidth', 2, ...
             'Marker', markers{m}, 'MarkerSize', 5, 'MarkerFaceColor', 'w', ...
             'DisplayName', labels{m});
end
title('Power Time Course Correlation', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Signal-to-noise ratio \gamma', 'FontSize', 11);
ylabel('Correlation, r', 'FontSize', 11);
ylim([0 1]); xlim([min(x) max(x)]);
set(gca, 'XScale', 'log', 'XMinorGrid', 'off', 'XGrid', 'on', 'YGrid', 'on', ...
         'XTick', xticks_vals, 'XTickLabel', xticks_lbls, 'TickDir', 'out', 'FontSize', 10);
legend('Location', 'southeast', 'Interpreter', 'tex', 'FontSize', 10); % Легенда только здесь

% ---------------- Plot 2: Pattern Correlation ----------------
subplot(1,2,2); hold on; box on;
for m = 1:nMethods
    y  = mean_pat(:,m)';
    ci = ci_pat(:,m)';
    
    fill([x fliplr(x)], [y-ci fliplr(y+ci)], colors(m,:), ...
         'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
     
    semilogx(x, y, 'Color', colors(m,:), 'LineWidth', 2, ...
             'Marker', markers{m}, 'MarkerSize', 5, 'MarkerFaceColor', 'w', ...
             'DisplayName', labels{m});
end
title('Pattern Correlation', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Signal-to-noise ratio \gamma', 'FontSize', 11);
ylabel('Correlation, r', 'FontSize', 11);
ylim([0.3 1]); xlim([min(x) max(x)]); 
set(gca, 'XScale', 'log', 'XMinorGrid', 'off', 'XGrid', 'on', 'YGrid', 'on', ...
         'XTick', xticks_vals, 'XTickLabel', xticks_lbls, 'TickDir', 'out', 'FontSize', 10);

% ================= Сохранение итоговой картинки =================
% Обновляем холст, чтобы убедиться в корректности рендера перед сохранением
drawnow;

% Сохранение в .fig (для последующего редактирования в MATLAB)
savefig(gcf, 'spoc_snr_comparison.fig');

% Сохранение в .jpg с разрешением 300 DPI (отлично подойдет для вставки в документы/статьи)
exportgraphics(gcf, 'spoc_snr_comparison.jpg', 'Resolution', 300);

fprintf('Saved figures: spoc_snr_comparison.fig and spoc_snr_comparison.jpg\n');