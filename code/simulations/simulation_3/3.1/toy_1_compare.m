close all
clear
clc

ft_path = 'C:\Users\ansbel\Documents\GitHub\CBI\site-packages\fieldtrip';
if ~exist('ft_defaults','file')
    addpath(ft_path);
end
ft_defaults;

%% Загрузка анатомии
FM = load('forward_model.mat');
G = FM.leadfield;
elec = FM.elec;

laycfg = [];
laycfg.elec = elec;
lay = ft_prepare_layout(laycfg);     

%% =================== ПАРАМЕТРЫ СИМУЛЯЦИИ ===================
Nsrc = 101;     
Ndistr = 3;        % Истинное количество целевых источников
Nmix = Ndistr;     % Размерность UMAP вложения
Nextract = Ndistr; % Извлекаемые компоненты

flanker = 1;
Ts = 850;      
Fs = 250;
Ws = 2;            
Ss = 1;

nMC = 20;          % Количество итераций
SNR_range = 10.^(-1.4:0.2:1); 
nSNR = length(SNR_range);
gamma = 0.1;

labels = {'eSPoC', 'mSPoC'};
nMethods = length(labels);
nSensors = size(G, 1);

% 4D-Массивы для метрик: [nMC, nSNR, nMethods, Ndistr]
filcorr_eval = zeros(nMC, nSNR, nMethods, Ndistr); 
patcorr_eval = zeros(nMC, nSNR, nMethods, Ndistr); 
zcorr_eval   = zeros(nMC, nSNR, nMethods, Ndistr); 

%% ================= ИНИЦИАЛИЗАЦИЯ ПУЛА =================
poolobj = gcp('nocreate');
if isempty(poolobj) || ~isa(poolobj, 'parallel.ProcessPool')
    if ~isempty(poolobj)
        delete(poolobj);
    end
    parpool('Processes', 4); % Threads
end

for mc_idx = 1:nMC
    fprintf('Monte-Carlo iteration: %d / %d\n', mc_idx, nMC);
    
    % 1. Генерируем "сырые" источники
    [X_s_raw, X_bg, X_n, z_raw, GA, S_raw] = generate_distributed_sources(G, Nsrc, Ndistr, flanker, Ts, Fs);
    Ainit = GA(:, 1:Ndistr); 
    
    % 2. Накладываем Гауссовские маски времени (Симуляция "эстафеты")
    T_samples = Ts * Fs;
    time_vec = 1:T_samples;
    center1 = T_samples / 6;
    center2 = T_samples / 2;
    center3 = 5 * T_samples / 6;
    sigma = T_samples / 32; 
    
    masks = [exp(-((time_vec - center1).^2) / (2*sigma^2));
             exp(-((time_vec - center2).^2) / (2*sigma^2));
             exp(-((time_vec - center3).^2) / (2*sigma^2))];
         
    S = S_raw; z = z_raw;
    for i = 1:Ndistr
        carrier_env = abs(hilbert(S(i, :)')');
        S(i,:) = S(i,:) ./ carrier_env;
        S(i,:) = S(i,:) .* masks(i,:);      
        z(i,:) = z(i,:) .* (masks(i,:).^2);
    end
    X_s = Ainit * S(1:Ndistr,:);
    
    % 3. Эпохирование и нормализация истинных огибающих z
    z_epo_raw = epoch_data(z(1:Ndistr,:)', Fs, Ws, Ss); 
    z_epo = squeeze(mean(z_epo_raw, 1)); 
    for src_i = 1:Ndistr
        z_epo(src_i,:) = (z_epo(src_i,:) - mean(z_epo(src_i,:))) / std(z_epo(src_i,:));
    end
    
    % Временные матрицы для parfor
    f_eval_local = zeros(nSNR, nMethods, Ndistr); 
    p_eval_local = zeros(nSNR, nMethods, Ndistr);
    z_eval_local = zeros(nSNR, nMethods, Ndistr);
    
    % ================= ЦИКЛ ПО SNR =================
    parfor (snr_idx = 1:nSNR, 4) 
    % for snr_idx = 1:nSNR
        current_SNR = SNR_range(snr_idx);
        
        % Смешивание, CAR и Аппаратная фильтрация
        X = current_SNR * X_s + X_bg + gamma * X_n / norm(X_s,'fro');
        X = X - mean(X,1);
        
        % Снижение размерности (PCA)
        [U_svd, S_svd, ~] = svd(X, 'econ');          
        S_svd = diag(S_svd);
        tol = max(size(X)) * eps(S_svd(1));
        r = sum(S_svd > tol);
        ve = S_svd.^2;
        var_explained = cumsum(ve) / sum(ve);
        var_explained(end) = 1;
        n_components = find(var_explained >= 0.99, 1);
        n_components = max(min(n_components, r), 1);
        
        U_pca = U_svd(:, 1:n_components);               
        X_pca = U_pca' * X;
        
        % Нарезка на эпохи (БЕЗ РАЗДЕЛЕНИЯ НА TRAIN/TEST)
        X_epo = epoch_data(X_pca', Fs, Ws, Ss);
        nEpochs = size(X_epo, 3);
        nChanPCA = size(X_epo, 2);
        
        % Ковариации
        Covs = zeros(nChanPCA, nChanPCA, nEpochs);
        for ep_idx = 1:nEpochs
            Covs(:,:,ep_idx) = cov(X_epo(:,:,ep_idx));
        end
        
        % ================= UMAP как целевая переменная =================
        Tcovs = Tangent_space(Covs);
        
        u = UMAP("n_neighbors", 15, "n_components", Nmix, "min_dist", 0.1, "verbose", false);
        % Обучаем и получаем вложения по всей выборке
        R = u.fit_transform(Tcovs'); 
        
        z_multidim = R'; % Размерность: [Nmix x nEpochs]
        % ================================================================

        % Буферы итерации
        f_tmp = NaN(nMethods, Ndistr);
        p_tmp = NaN(nMethods, Ndistr);
        z_tmp = NaN(nMethods, Ndistr);
        
        w_all  = zeros(nChanPCA, Nextract, 2); 
        a_all  = zeros(nSensors, Nextract, 2); 
        vz_all = zeros(Nmix, Nextract, 2); 
        
        % === 1. eSPoC ===
        [W_e, A_e, ~, Vz, corrs_e] = espoc_free(X_epo, z_multidim);
        n_ext_e = min(Nextract, size(W_e, 1));
        for f_idx = 1:n_ext_e
            % [~, idx] = sort(abs(corrs_e(f_idx,:)), 'descend');
            % w_all(:, f_idx, 1)  = squeeze(W_e(f_idx, :, idx(1)))'; 
            % a_all(:, f_idx, 1)  = U_pca * squeeze(A_e(f_idx, :, idx(1)))'; 
            % vz_all(:, f_idx, 1) = Vz(:, f_idx); 

            w_all(:, f_idx, 1)  = W_e(:, f_idx);
            a_all(:, f_idx, 1)  = U_pca * A_e(:, f_idx);
            vz_all(:, f_idx, 1) = Vz(:, f_idx);
        end
        
        % === 2. mSPoC ===
        mspoc_opts = struct('tau_vector', 0, 'n_component_sets', Nextract, 'verbose', 0);
        [W_m, Wy, ~, A_m, ~] = my_mspoc(X_epo, z_multidim, mspoc_opts);
        n_ext_m = min(Nextract, size(W_m, 2));
        for f_idx = 1:n_ext_m
            w_all(:, f_idx, 2)  = W_m(:, f_idx); 
            a_all(:, f_idx, 2)  = U_pca * A_m(:, f_idx); 
            vz_all(:, f_idx, 2) = Wy(:, f_idx); 
        end
        
        % === Оценка метрик для SPoC алгоритмов ===
        for m_idx = 1:2
            w  = w_all(:,:,m_idx);
            a  = a_all(:,:,m_idx);
            vz = vz_all(:,:,m_idx);
            
            % Расчет огибающих ЭЭГ
            env_eval = zeros(Nextract, nEpochs);
            for c = 1:Nextract
                for ep_idx = 1:nEpochs
                    env_eval(c, ep_idx) = w(:,c)' * Covs(:,:,ep_idx) * w(:,c);
                end
            end
            
            % Восстановление (ПОВОРОТ) многообразия UMAP
            z_rec_eval = vz' * z_multidim; 
            
            corr_mat = abs(corr(env_eval', z_epo')); 
            
            for t = 1:min(Nextract, Ndistr)
                [max_val, max_idx] = max(corr_mat(:));
                if max_val == -1, break; end 
                [best_ext, best_tar] = ind2sub(size(corr_mat), max_idx);
                
                f_tmp(m_idx, best_tar) = max_val;
                p_tmp(m_idx, best_tar) = abs(corr(a(:,best_ext), Ainit(:,best_tar)));
                
                % Сравниваем восстановленную (повернутую) координату UMAP с истиной
                z_tmp(m_idx, best_tar) = abs(corr(z_rec_eval(best_ext,:)', z_epo(best_tar,:)'));
                
                corr_mat(best_ext, :) = -1; corr_mat(:, best_tar) = -1;
            end
        end
               
        % Сохранение локальных результатов
        f_eval_local(snr_idx, :, :) = f_tmp;
        p_eval_local(snr_idx, :, :) = p_tmp;
        z_eval_local(snr_idx, :, :) = z_tmp;
    end
    
    filcorr_eval(mc_idx, :, :, :) = f_eval_local;
    patcorr_eval(mc_idx, :, :, :) = p_eval_local;
    zcorr_eval(mc_idx, :, :, :)   = z_eval_local;
end

%% ================= Вычисление статистики =================
mean_f = reshape(mean(filcorr_eval, 1, 'omitnan'), [nSNR, nMethods, Ndistr]);
ci_f   = reshape(1.96 * std(filcorr_eval, 0, 1, 'omitnan') / sqrt(nMC), [nSNR, nMethods, Ndistr]);

mean_p = reshape(mean(patcorr_eval, 1, 'omitnan'), [nSNR, nMethods, Ndistr]);
ci_p   = reshape(1.96 * std(patcorr_eval, 0, 1, 'omitnan') / sqrt(nMC), [nSNR, nMethods, Ndistr]);

mean_z = reshape(mean(zcorr_eval, 1, 'omitnan'), [nSNR, nMethods, Ndistr]);
ci_z   = reshape(1.96 * std(zcorr_eval, 0, 1, 'omitnan') / sqrt(nMC), [nSNR, nMethods, Ndistr]);

% ================= Визуализация =================
x = SNR_range; 
figure('Position', [100 100 1600 350*Ndistr], 'Color', 'w'); 
colors = [0.8 0 0; 0 0 0.8; 0 0.6 0]; % eSPoC (Красный), mSPoC (Синий), UMAP (Зеленый)
markers = {'o', 's', '^'};

xticks_vals = 10.^[-1 -0.4 0 0.4 1];
xticks_lbls = {'10^{-1}','10^{-0.4}','10^{0}','10^{0.4}','10^{1}'};

for src_i = 1:Ndistr
    % 1. Envelope Correlation (Отрисовываем все 3 метода)
    subplot(Ndistr, 3, (src_i-1)*3 + 1); hold on; box on;
    for m = 1:nMethods
        y = mean_f(:, m, src_i)'; ci = ci_f(:, m, src_i)';
        fill([x fliplr(x)], [y-ci fliplr(y+ci)], colors(m,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');   
        semilogx(x, y, ['-', markers{m}], 'Color', colors(m,:), 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'DisplayName', labels{m});
    end
    title(sprintf('Src %d: Envelope Correlation', src_i), 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Correlation, r', 'FontSize', 11);
    ylim([0 1.05]); xlim([min(x) max(x)]);
    set(gca, 'XScale', 'log', 'XMinorGrid', 'off', 'XGrid', 'on', 'YGrid', 'on', ...
             'XTick', xticks_vals, 'XTickLabel', xticks_lbls, 'TickDir', 'out', 'FontSize', 10);
    if src_i == 1, legend('Location', 'northwest', 'FontSize', 10); end
    
    % 2. Pattern Correlation (Только SPoC, m=1:2)
    subplot(Ndistr, 3, (src_i-1)*3 + 2); hold on; box on;
    for m = 1:2
        y = mean_p(:, m, src_i)'; ci = ci_p(:, m, src_i)';
        fill([x fliplr(x)], [y-ci fliplr(y+ci)], colors(m,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');   
        semilogx(x, y, ['-', markers{m}], 'Color', colors(m,:), 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'DisplayName', labels{m});
    end
    title(sprintf('Src %d: Spatial Pattern Corr', src_i), 'FontSize', 12, 'FontWeight', 'bold');
    ylim([0 1.05]); xlim([min(x) max(x)]);
    set(gca, 'XScale', 'log', 'XMinorGrid', 'off', 'XGrid', 'on', 'YGrid', 'on', ...
             'XTick', xticks_vals, 'XTickLabel', xticks_lbls, 'TickDir', 'out', 'FontSize', 10);
    
    % 3. Behavioral Unmixing Correlation (Только SPoC, m=1:2)
    subplot(Ndistr, 3, (src_i-1)*3 + 3); hold on; box on;
    for m = 1:2
        y = mean_z(:, m, src_i)'; ci = ci_z(:, m, src_i)';
        fill([x fliplr(x)], [y-ci fliplr(y+ci)], colors(m,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');   
        semilogx(x, y, ['-', markers{m}], 'Color', colors(m,:), 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'DisplayName', labels{m});
    end
    title(sprintf('Src %d: UMAP "Rotation" Corr', src_i), 'FontSize', 12, 'FontWeight', 'bold');
    ylim([0 1.05]); xlim([min(x) max(x)]);
    set(gca, 'XScale', 'log', 'XMinorGrid', 'off', 'XGrid', 'on', 'YGrid', 'on', ...
             'XTick', xticks_vals, 'XTickLabel', xticks_lbls, 'TickDir', 'out', 'FontSize', 10);
    
    % Подписи оси X только для самого нижнего ряда
    if src_i == Ndistr
        subplot(Ndistr, 3, (src_i-1)*3 + 1); xlabel('Signal-to-Noise Ratio (SNR)', 'FontSize', 11);
        subplot(Ndistr, 3, (src_i-1)*3 + 2); xlabel('Signal-to-Noise Ratio (SNR)', 'FontSize', 11);
        subplot(Ndistr, 3, (src_i-1)*3 + 3); xlabel('Signal-to-Noise Ratio (SNR)', 'FontSize', 11);
    end
end

%% ================= Сохранение итоговой картинки =================
drawnow;
savefig(gcf, 'umap_guided_snr_comparison.fig');
exportgraphics(gcf, 'umap_guided_snr_comparison.jpg', 'Resolution', 300);
fprintf('Saved figures: umap_guided_snr_comparison.fig and umap_guided_snr_comparison.jpg\n');