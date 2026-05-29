close all
clear
clc

ft_path = 'C:\Users\ansbel\Documents\GitHub\site-packages\fieldtrip';
if ~exist('ft_defaults','file')
    addpath(ft_path);
end
ft_defaults;

%% Загрузка данных
elec = load("C:\Users\ansbel\Documents\GitHub\TriCo\data\support\elec.mat").elec;
laycfg = [];
laycfg.elec = elec;
lay = ft_prepare_layout(laycfg);     
G = load('C:\Users\ansbel\Documents\GitHub\TriCo\data\support\MNE_EEG_FWD_TRPL.mat').MNE_EEG_FWD_TRPL;

%% =================== ПАРАМЕТРЫ СИМУЛЯЦИИ ===================
Nsrc = 101;     % 1 целевой + 100 фоновых
Ndistr = 1;
flanker = 1;
Ts = 850;      
Fs = 250;
Ws = 1;
Ss = 1;
nMC = 100;
SNR_range = 10.^(-1.4:0.2:1);
gamma = 0.1;
nSNR = length(SNR_range);
methods = {@espoc, @spoc}; 
nMethods = length(methods);
labels = {'eSPoC', 'SPoC'};

[b_pipe, a_pipe] = butter(3, [8, 12]/(Fs/2));

filcorr_train = zeros(nMC, nSNR, nMethods); 
filcorr_test  = zeros(nMC, nSNR, nMethods);
patcorr       = zeros(nMC, nSNR, nMethods);

parfor mc_idx = 1:nMC
    fprintf('Monte-Carlo iteration: %d / %d\n', mc_idx, nMC);
    
    [X_s, X_bg, X_n, z, GA, S] = generate_distributed_sources(G, Nsrc, Ndistr, flanker, Ts, Fs);
    Ainit = GA(:,1); 
    
    % Временные матрицы для parfor
    filcorr_train_local = zeros(nSNR, nMethods); 
    filcorr_test_local  = zeros(nSNR, nMethods);
    patcorr_local       = zeros(nSNR, nMethods);
    
    for snr_idx = 1:nSNR
        % ================= Data generation =================
        SNR = SNR_range(snr_idx);
        X = SNR*X_s + X_bg + gamma * X_n / norm(X_s,'fro');
        % X = filtfilt(b_pipe, a_pipe, X')';
        
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
        
        % ================= Methods Evaluation =================
        for m_idx = 1:nMethods
            alg = methods{m_idx};
            
            [W, A] = alg(X_epo_train, z_epo_train);
            w = W(:,1);
            
            env_train = zeros(nTrain, 1);
            for ep_idx = 1:nTrain
                env_train(ep_idx) = w' * Covs_train(:,:,ep_idx) * w;
            end
            filcorr_train_local(snr_idx, m_idx) = corr(env_train(:), z_epo_train(:));
            
            env_test = zeros(nTest, 1);
            for ep_idx = 1:nTest
                env_test(ep_idx) = w' * Covs_test(:,:,ep_idx) * w;
            end
            filcorr_test_local(snr_idx, m_idx) = corr(env_test(:), z_epo_test(:));
            
            patcorr_local(snr_idx, m_idx) = abs(corr(A(:,1), Ainit));
        end
    end
    
    filcorr_train(mc_idx, :, :) = filcorr_train_local;
    filcorr_test(mc_idx, :, :)  = filcorr_test_local;
    patcorr(mc_idx, :, :)       = patcorr_local;
end

%% ================= Вычисление статистики =================
mean_filt_train = squeeze(mean(filcorr_train, 1));
mean_filt_test  = squeeze(mean(filcorr_test, 1));
mean_pat        = squeeze(mean(patcorr, 1));   

ci_filt_train = squeeze(1.96 * std(filcorr_train, 0, 1) / sqrt(nMC));
ci_filt_test  = squeeze(1.96 * std(filcorr_test, 0, 1) / sqrt(nMC));
ci_pat        = squeeze(1.96 * std(patcorr, 0, 1) / sqrt(nMC));

x = SNR_range;
xticks_vals = 10.^[-1 -0.4 0 0.4 1];
xticks_lbls = {'10^{-1}','10^{-0.4}','10^{0}','10^{0.4}','10^{1}'};

% Увеличили ширину фигуры для трех графиков
figure('Position', [100 100 1350 400], 'Color', 'w'); 

% Цвета: Красный и Синий (Зеленый убрали, так как метода два)
colors = [0.8 0 0;    
          0 0 0.8];   
% Разные маркеры для разных алгоритмов (помогает при ч/б печати)
markers = {'o', 's'}; 

% ---------------- Plot 1: TRAIN Power Time Course Correlation ----------------
subplot(1,3,1); hold on; box on;
for m = 1:nMethods
    y  = mean_filt_train(:,m)';
    ci = ci_filt_train(:,m)';
    
    % Полупрозрачная заливка (современный стандарт)
    fill([x fliplr(x)], [y-ci fliplr(y+ci)], colors(m,:), ...
         'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');   
     
    % Утолщенная линия с белыми внутри маркерами
    semilogx(x, y, 'Color', colors(m,:), 'LineWidth', 2, ...
             'Marker', markers{m}, 'MarkerSize', 5, 'MarkerFaceColor', 'w', ...
             'DisplayName', labels{m});
end
title('TRAIN: Power Time Course Correlation', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Signal-to-noise ratio \gamma', 'FontSize', 11);
ylabel('Correlation, r', 'FontSize', 11);
ylim([0 1]); xlim([min(x) max(x)]);
set(gca, 'XScale', 'log', 'XMinorGrid', 'off', 'XGrid', 'on', 'YGrid', 'on', ...
         'XTick', xticks_vals, 'XTickLabel', xticks_lbls, 'TickDir', 'out', 'FontSize', 10);
legend('Location', 'southeast', 'Interpreter', 'tex', 'FontSize', 10);

% ---------------- Plot 2: TEST Power Time Course Correlation ----------------
subplot(1,3,2); hold on; box on;
for m = 1:nMethods
    y  = mean_filt_test(:,m)';
    ci = ci_filt_test(:,m)';
    
    fill([x fliplr(x)], [y-ci fliplr(y+ci)], colors(m,:), ...
         'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');   
     
    semilogx(x, y, 'Color', colors(m,:), 'LineWidth', 2, ...
             'Marker', markers{m}, 'MarkerSize', 5, 'MarkerFaceColor', 'w', ...
             'DisplayName', labels{m});
end
title('TEST: Power Time Course Correlation', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Signal-to-noise ratio \gamma', 'FontSize', 11);
ylabel('Correlation, r', 'FontSize', 11);
ylim([0 1]); xlim([min(x) max(x)]);
set(gca, 'XScale', 'log', 'XMinorGrid', 'off', 'XGrid', 'on', 'YGrid', 'on', ...
         'XTick', xticks_vals, 'XTickLabel', xticks_lbls, 'TickDir', 'out', 'FontSize', 10);
legend('Location', 'southeast', 'Interpreter', 'tex', 'FontSize', 10);

% ---------------- Plot 3: Pattern Correlation ----------------
subplot(1,3,3); hold on; box on;
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
legend('Location', 'southeast', 'Interpreter', 'tex', 'FontSize', 10);