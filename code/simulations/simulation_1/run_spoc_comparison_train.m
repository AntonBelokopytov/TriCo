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
Nsrc = 101;     
Ndistr = 1;
flanker = 1;
% Увеличиваем Ts до 900, чтобы хватило на 300 обучающих и 600 тестовых эпох
Ts = 900;      
Fs = 250;
Ws = 1;
Ss = 1;
nMC = 100; 

% Фиксированный SNR, как в статье
SNR_fixed = 10^(0.4); 
gamma = 0.1;

% Сетка количества тренировочных эпох
train_epochs_range = 10:10:300; 
nSteps = length(train_epochs_range);

methods = {@espoc, @spoc}; 
nMethods = length(methods);
labels = {'eSPoC', 'SPoC'};

% Предварительное выделение памяти
filcorr_train = zeros(nMC, nSteps, nMethods); 
filcorr_test  = zeros(nMC, nSteps, nMethods);
patcorr       = zeros(nMC, nSteps, nMethods);

parfor mc_idx = 1:nMC
    fprintf('Monte-Carlo iteration: %d / %d\n', mc_idx, nMC);
    
    % Временные матрицы для parfor
    filcorr_train_local = zeros(nSteps, nMethods); 
    filcorr_test_local  = zeros(nSteps, nMethods);
    patcorr_local       = zeros(nSteps, nMethods);
    
    % 1. ГЕНЕРАЦИЯ ДАННЫХ (Один раз на MC-итерацию)
    [X_s, X_bg, X_n, z, GA, S] = generate_distributed_sources(G, Nsrc, Ndistr, flanker, Ts, Fs);
    Ainit = GA(:,1); 
    
    % Смешиваем с фиксированным SNR
    X = SNR_fixed*X_s + X_bg + gamma * X_n / norm(X_s,'fro');
    
    X_epo = epoch_data(X', Fs, Ws, Ss);        
    z_epo_raw = epoch_data(z(1,:)', Fs, Ws, Ss); 
    z_epo = squeeze(mean(z_epo_raw, 1));         
    
    % 2. ПРЕДРАСЧЕТ КОВАРИАЦИЙ (Ускоряет работу в десятки раз)
    nChan = size(X_epo, 2);
    nTotalEpochs = size(X_epo, 3);
    Covs_all = zeros(nChan, nChan, nTotalEpochs);
    for ep_idx = 1:nTotalEpochs
        Covs_all(:,:,ep_idx) = cov(X_epo(:,:,ep_idx));
    end
    
    % Фиксируем тестовую выборку (последние 600 эпох)
    X_epo_test = X_epo(:,:,301:900);
    z_epo_test = z_epo(301:900);
    Covs_test  = Covs_all(:,:,301:900);
    nTest = 600;
    
    % 3. ЦИКЛ ПО ОБЪЕМУ ОБУЧАЮЩЕЙ ВЫБОРКИ
    for step_idx = 1:nSteps
        nTrain = train_epochs_range(step_idx);
        
        % Берем только первые nTrain эпох
        X_epo_train = X_epo(:,:,1:nTrain);
        z_epo_train = z_epo(1:nTrain);
        Covs_train  = Covs_all(:,:,1:nTrain);
        
        % Оценка методов
        for m_idx = 1:nMethods
            alg = methods{m_idx};
            
            [W, A] = alg(X_epo_train, z_epo_train);
            w = W(:,1);
            
            % Корреляция на Train
            env_train = zeros(nTrain, 1);
            for ep_idx = 1:nTrain
                env_train(ep_idx) = w' * Covs_train(:,:,ep_idx) * w;
            end
            filcorr_train_local(step_idx, m_idx) = corr(env_train(:), z_epo_train(:));
            
            % Корреляция на Test
            env_test = zeros(nTest, 1);
            for ep_idx = 1:nTest
                env_test(ep_idx) = w' * Covs_test(:,:,ep_idx) * w;
            end
            filcorr_test_local(step_idx, m_idx) = corr(env_test(:), z_epo_test(:));
            
            % Корреляция паттернов
            patcorr_local(step_idx, m_idx) = abs(corr(A(:,1), Ainit));
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

% ================= Визуализация =================
x = train_epochs_range;

figure('Position', [100 100 1350 400], 'Color', 'w'); 
colors = [0.8 0 0;    
          0 0 0.8];   
markers = {'o', 's'}; 

% ---------------- Plot 1: TRAIN Power Time Course Correlation ----------------
subplot(1,3,1); hold on; box on;
for m = 1:nMethods
    y  = mean_filt_train(:,m)';
    ci = ci_filt_train(:,m)';
    
    fill([x fliplr(x)], [y-ci fliplr(y+ci)], colors(m,:), ...
         'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');   
     
    % Ось X теперь линейная (plot вместо semilogx)
    plot(x, y, 'Color', colors(m,:), 'LineWidth', 2, ...
             'Marker', markers{m}, 'MarkerSize', 5, 'MarkerFaceColor', 'w', ...
             'DisplayName', labels{m});
end
title('TRAIN: Power Time Course Correlation', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Number of training epochs', 'FontSize', 11);
ylabel('Correlation, r', 'FontSize', 11);
ylim([0 1]); xlim([min(x) max(x)]);
set(gca, 'XMinorGrid', 'off', 'XGrid', 'on', 'YGrid', 'on', ...
         'TickDir', 'out', 'FontSize', 10);
legend('Location', 'southeast', 'Interpreter', 'tex', 'FontSize', 10);

% ---------------- Plot 2: TEST Power Time Course Correlation ----------------
subplot(1,3,2); hold on; box on;
for m = 1:nMethods
    y  = mean_filt_test(:,m)';
    ci = ci_filt_test(:,m)';
    
    fill([x fliplr(x)], [y-ci fliplr(y+ci)], colors(m,:), ...
         'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');   
     
    plot(x, y, 'Color', colors(m,:), 'LineWidth', 2, ...
             'Marker', markers{m}, 'MarkerSize', 5, 'MarkerFaceColor', 'w', ...
             'DisplayName', labels{m});
end
title('TEST: Power Time Course Correlation', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Number of training epochs', 'FontSize', 11);
ylabel('Correlation, r', 'FontSize', 11);
ylim([0 1]); xlim([min(x) max(x)]);
set(gca, 'XMinorGrid', 'off', 'XGrid', 'on', 'YGrid', 'on', ...
         'TickDir', 'out', 'FontSize', 10);
legend('Location', 'southeast', 'Interpreter', 'tex', 'FontSize', 10);

% ---------------- Plot 3: Pattern Correlation ----------------
subplot(1,3,3); hold on; box on;
for m = 1:nMethods
    y  = mean_pat(:,m)';
    ci = ci_pat(:,m)';
    
    fill([x fliplr(x)], [y-ci fliplr(y+ci)], colors(m,:), ...
         'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
     
    plot(x, y, 'Color', colors(m,:), 'LineWidth', 2, ...
             'Marker', markers{m}, 'MarkerSize', 5, 'MarkerFaceColor', 'w', ...
             'DisplayName', labels{m});
end
title('Pattern Correlation', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Number of training epochs', 'FontSize', 11);
ylabel('Correlation, r', 'FontSize', 11);
ylim([0.3 1]); xlim([min(x) max(x)]); 
set(gca, 'XMinorGrid', 'off', 'XGrid', 'on', 'YGrid', 'on', ...
         'TickDir', 'out', 'FontSize', 10);
legend('Location', 'southeast', 'Interpreter', 'tex', 'FontSize', 10);














