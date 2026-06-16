close all
clear
clc

ft_path = 'C:\Users\anton\Documents\GitHub\site-packages\fieldtrip';
if ~exist('ft_defaults','file')
    addpath(ft_path);
end
ft_defaults;

%% Загрузка данных
FM = load('forward_model.mat');
G = FM.leadfield;
elec = FM.elec;

laycfg = [];
laycfg.elec = elec;
lay = ft_prepare_layout(laycfg);     

%% =================== ПАРАМЕТРЫ СИМУЛЯЦИИ ===================
Nsrc = 101;     
Ndistr = 1;
flanker = 1;
% Увеличиваем Ts до 900, чтобы хватило на 300 обучающих и 600 тестовых эпох
Ts = 900;      
Fs = 250;
Ws = 1;
Ss = 1;
nMC = 500; 

% Фиксированный SNR, как в статье
SNR_fixed = 10^(0.4); 
gamma = 0.1;

% Сетка количества тренировочных эпох
train_epochs_range = 10:10:300; 
nSteps = length(train_epochs_range);
methods = {@espoc, @spoc, @spoc_r2}; 
nMethods = length(methods);
labels = {'eSPoC', 'SPoC_{\lambda}', 'SPoC_{r^2}'};

% Предварительное выделение памяти
filcorr_train = zeros(nMC, nSteps, nMethods); 
filcorr_test  = zeros(nMC, nSteps, nMethods);
patcorr       = zeros(nMC, nSteps, nMethods);

%% ================= ИНИЦИАЛИЗАЦИЯ ПУЛА ПРОЦЕССОВ =================
poolobj = gcp('nocreate');
if isempty(poolobj) || ~isa(poolobj, 'parallel.ProcessPool')
    if ~isempty(poolobj)
        delete(poolobj);
    end
    parpool('Processes', 4);
end

for mc_idx = 1:nMC
    fprintf('Monte-Carlo iteration: %d / %d\n', mc_idx, nMC);
    
    % Временные матрицы для записи результатов
    filcorr_train_local = zeros(nSteps, nMethods); 
    filcorr_test_local  = zeros(nSteps, nMethods);
    patcorr_local       = zeros(nSteps, nMethods);
    
    % 1. ГЕНЕРАЦИЯ ДАННЫХ (Один раз на MC-итерацию)
    [X_s, X_bg, X_n, z, GA, S] = generate_distributed_sources(G, Nsrc, Ndistr, flanker, Ts, Fs);
    Ainit = GA(:,1); 
    
    % Смешиваем с фиксированным SNR
    X = SNR_fixed*X_s + X_bg + gamma * X_n / norm(X_s,'fro');
    X = X - mean(X,1);

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
    parfor (step_idx = 1:nSteps, 4)
        nTrain = train_epochs_range(step_idx);
        
        % Берем только первые nTrain эпох
        X_epo_train = X_epo(:,:,1:nTrain);
        z_epo_train = z_epo(1:nTrain);
        Covs_train  = Covs_all(:,:,1:nTrain);
        
        % Временные переменные для корректного слайсинга в parfor
        temp_train = zeros(1, nMethods);
        temp_test  = zeros(1, nMethods);
        temp_pat   = zeros(1, nMethods);
        
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
            temp_train(m_idx) = corr(env_train(:), z_epo_train(:));
            
            % Корреляция на Test
            env_test = zeros(nTest, 1);
            for ep_idx = 1:nTest
                env_test(ep_idx) = w' * Covs_test(:,:,ep_idx) * w;
            end
            temp_test(m_idx) = corr(env_test(:), z_epo_test(:));
            
            % Корреляция паттернов
            temp_pat(m_idx) = abs(corr(A(:,1), Ainit));
        end
        
        % Запись результатов итерации
        filcorr_train_local(step_idx, :) = temp_train;
        filcorr_test_local(step_idx, :)  = temp_test;
        patcorr_local(step_idx, :)       = temp_pat;
    end
    
    filcorr_train(mc_idx, :, :) = filcorr_train_local;
    filcorr_test(mc_idx, :, :)  = filcorr_test_local;
    patcorr(mc_idx, :, :)       = patcorr_local;
end

%% ================= Вычисление статистики =================
% Если nMC = 1, std выдаст нули, доверительный интервал будет 0 (полоса не отрисуется)
mean_filt_test  = squeeze(mean(filcorr_test, 1));
mean_pat        = squeeze(mean(patcorr, 1));   

ci_filt_test  = squeeze(1.96 * std(filcorr_test, 0, 1) / sqrt(nMC));
ci_pat        = squeeze(1.96 * std(patcorr, 0, 1) / sqrt(nMC));

% ================= Визуализация =================
x = train_epochs_range;

% Адаптировали размер окна под два графика
figure('Position', [100 100 900 400], 'Color', 'w'); 

% Цвета для трех алгоритмов
colors = [0.8 0 0;    
          0 0 0.8;
          0 0.6 0];   
markers = {'o', 's', '^'}; 

% ---------------- Plot 1: TEST Power Time Course Correlation ----------------
subplot(1,2,1); hold on; box on;
for m = 1:nMethods
    y  = mean_filt_test(:,m)';
    ci = ci_filt_test(:,m)';
    
    fill([x fliplr(x)], [y-ci fliplr(y+ci)], colors(m,:), ...
         'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');   
     
    plot(x, y, 'Color', colors(m,:), 'LineWidth', 2, ...
             'Marker', markers{m}, 'MarkerSize', 5, 'MarkerFaceColor', 'w', ...
             'DisplayName', labels{m});
end
title('Power Time Course Correlation', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Number of training epochs', 'FontSize', 11);
ylabel('Correlation, r', 'FontSize', 11);
ylim([0 1]); xlim([min(x) max(x)]);
set(gca, 'XMinorGrid', 'off', 'XGrid', 'on', 'YGrid', 'on', ...
         'TickDir', 'out', 'FontSize', 10);
legend('Location', 'southeast', 'Interpreter', 'tex', 'FontSize', 10);

% ---------------- Plot 2: Pattern Correlation ----------------
subplot(1,2,2); hold on; box on;
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

% ================= Сохранение итоговой картинки =================
drawnow;
savefig(gcf, 'spoc_train_comparison.fig');
exportgraphics(gcf, 'spoc_train_comparison.jpg', 'Resolution', 300);
fprintf('Saved figures: spoc_train_comparison.fig and spoc_train_comparison.jpg\n');