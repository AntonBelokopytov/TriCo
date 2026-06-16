%% =====================================================================
% BATCH VISUALIZATION TOOL FOR eSPoC & UMAP RESULTS
% =====================================================================
close all; clear; clc;

% --- 1. НАСТРОЙКИ (ЧТО СМОТРИМ) ---
% Просто поменяй 'part1' на 'part2' в пути, и скрипт сам всё перенастроит!
base_dir       = 'C:\Users\ansbel\Documents\GitHub\TriCo\data\external\music_listening\part1\';

data_dir       = fullfile(base_dir, 'eeg');
base_emb_dir   = fullfile(base_dir, 'embeddings');
base_stats_dir = fullfile(base_dir, 'stats');
base_fig_dir   = fullfile(base_dir, 'figures'); 
ft_path        = 'C:\Users\ansbel\Documents\GitHub\site-packages\fieldtrip';

freq_name = 'beta';           % Выбери: theta, alpha, beta, gamma

% --- АВТОМАТИЧЕСКАЯ НАСТРОЙКА УСЛОВИЙ (PART 1 vs PART 2) ---
if contains(base_dir, 'part1')
    target_conditions = {
        'EC1', 'EO1', '2Hz', '05Hz', '4Hz', '1Hz', '3Hz', ...
        'NoRy 1', 'Waltz 1', 'Waltz 2', 'NoRy 2', 'NoRy 3', ...
        'Waltz 3', 'NoRy 4', 'Waltz 4', 'NoRy 5', 'Waltz 5', ...
        'EC2', 'EO2'
    };
    nEpochs_expected = 19;
    disp('Режим работы: Часть 1 (19 условий, 110 сек)');
elseif contains(base_dir, 'part2')
    target_conditions = {
        'EC1', 'EO1', '2Hz', '05Hz', '4Hz', '1Hz', '3Hz', ...
        'NoRy 1', 'Waltz 1', 'Waltz 2', 'NoRy 2', 'NoRy 3', ...
        'Waltz 3', 'NoRy 4', 'Waltz 4', 'NoRy 5', 'Waltz 5', ...
        'EC2', 'EO2', 'Waltz 6', 'Waltz 7', 'Waltz 8'
    };
    nEpochs_expected = 22;
    disp('Режим работы: Часть 2 (22 условия, 120 сек)');
else
    error('Не удалось определить часть эксперимента. В пути base_dir должно быть "part1" или "part2".');
end

% Маршрутизация частотных папок
emb_dir   = fullfile(base_emb_dir, freq_name);
stats_dir = fullfile(base_stats_dir, freq_name);
fig_dir   = fullfile(base_fig_dir, freq_name);

if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
if ~exist('ft_defaults','file'), addpath(ft_path); end
ft_defaults;

% Ищем всех обработанных испытуемых
mat_files = dir(fullfile(emb_dir, ['UMAP_*_', freq_name, '.mat']));
disp(['Найдено файлов для визуализации: ', num2str(length(mat_files))]);

%% --- 2. ГЛАВНЫЙ ЦИКЛ ПО ИСПЫТУЕМЫМ ---
for f = 1:length(mat_files)
    
    mat_name = mat_files(f).name;
    base_name = strrep(mat_name, 'UMAP_', '');
    base_name = strrep(base_name, ['_', freq_name, '.mat'], '');
    
    disp('==================================================');
    disp(['Построение графиков для: ', base_name, ' [', freq_name, ']']);
    
    % Персональная папка
    subj_fig_dir = fullfile(fig_dir, base_name);
    if ~exist(subj_fig_dir, 'dir'), mkdir(subj_fig_dir); end
    
    emb_file   = fullfile(emb_dir, mat_name);
    stats_file = fullfile(stats_dir, sprintf('STATS_%s_%s.mat', base_name, freq_name));
    
    if ~exist(stats_file, 'file')
        warning(['Файл статистики не найден для ', base_name, '. Пропускаем.']);
        continue;
    end
    
    load(emb_file);   
    load(stats_file); 
    
    % Защита размерности
    if isvector(corrs_true), corrs_true = corrs_true(:)'; end
    if isvector(eigenvalues_true), eigenvalues_true = eigenvalues_true(:)'; end
    
    % Загрузка лейблов
    cfg = []; cfg.dataset = fullfile(data_dir, [base_name, '_raw.fif']); cfg.continuous = 'yes';
    Xinf = ft_preprocessing(cfg);
    laycfg = []; laycfg.elec = Xinf.elec; lay = ft_prepare_layout(laycfg);     
    topo = []; topo.dimord = 'chan_time'; topo.label = Xinf.elec.label(1:38); topo.time = 0;
    
    % БАЗОВАЯ КОНФИГУРАЦИЯ (без запрета colorbar)
    cfg_topo = []; cfg_topo.marker = 'labels'; cfg_topo.layout = lay; 
    cfg_topo.comment = 'no'; cfg_topo.style = 'fill'; 
    
    %% --- 3. ПРЕДВАРИТЕЛЬНАЯ ПОДГОТОВКА ПЕРЕМЕННЫХ ---
    Rmean = R - mean(R, 1);
    Covs_valid = Covs(:, :, valid_windows);
    nEpochs = min(nEpochs, nEpochs_expected); 
    
    n_chan = size(Covs_valid, 1);
    n_feat = (n_chan^2 - n_chan)/2 + n_chan;
    F = zeros(n_feat, size(Covs_valid, 3));
    for i = 1:size(Covs_valid, 3)
        C = Covs_valid(:,:,i);
        upper_mask = triu(true(size(C)));
        upper_triu_mask = triu(true(size(C)), 1);
        C(upper_triu_mask) = C(upper_triu_mask) * sqrt(2);
        v = C(upper_mask);
        F(:, i) = v(:);
    end
    F = F - mean(F, 2);
    
    gl_src = Vf_true' * F;              
    emb_can_pr = Vz_true' * Rmean';     
    
    boundaries = find(diff(valid_cond_idx) ~= 0);
    ticks = [0; boundaries; length(valid_cond_idx)];
    cmap = jet(nEpochs);
    
    ccx = NaN(1, nEpochs); ccy = NaN(1, nEpochs); ccz = NaN(1, nEpochs);
    for i = 1:nEpochs
        idx = (valid_cond_idx == i);
        if any(idx)
            ccx(i) = mean(R(idx, 1)); ccy(i) = mean(R(idx, 2)); ccz(i) = mean(R(idx, 3));
        end
    end
    
    x = emb_can_pr(1,:); y = emb_can_pr(2,:); z = emb_can_pr(3,:);
    cx_can = NaN(1, nEpochs); cy_can = NaN(1, nEpochs); cz_can = NaN(1, nEpochs);
    for i = 1:nEpochs
        idx = (valid_cond_idx == i);
        if any(idx)
            cx_can(i) = mean(x(idx)); cy_can(i) = mean(y(idx)); cz_can(i) = mean(z(idx));
        end
    end
    
    %% --- ПОСТРОЕНИЕ И СОХРАНЕНИЕ ГРАФИКОВ ---
    
    % РИСУНОК 1: ВРЕМЕННАЯ ДИНАМИКА UMAP
    f1 = figure('Visible', 'off', 'Name', 'UMAP Temporal Dynamics', 'Color', 'w', 'Position', [50, 50, 900, 400]);
    plot(R, 'LineWidth', 1.2); hold on; grid on;
    for k = 1:length(ticks), xline(ticks(k), '--', 'Color', [0.7 0.7 0.7]); end
    xlim([0, size(R,1)]); xticks(ticks(1:end-1)); 
    xticklabels(arrayfun(@(val) ['(', num2str(val), ')'], 1:nEpochs, 'UniformOutput', false));
    ylabel('UMAP Coordinate'); xlabel('Conditions'); legend({'Dim 1', 'Dim 2', 'Dim 3'}, 'Location', 'best');
    title(['Temporal Evolution of UMAP | ', strrep(base_name, '_', '\_')]);
    saveas(f1, fullfile(subj_fig_dir, sprintf('%s_01_UMAP_Dynamics.jpg', base_name))); close(f1);
    
    % РИСУНОК 2: ПОРОГИ ЗНАЧИМОСТИ (STEM)
    f2 = figure('Visible', 'off', 'Name', 'eSPoC Significance', 'Color', 'w', 'Position', [100, 100, 800, 400]);
    stem(corrs_true', 'LineWidth', 1.5); hold on; grid on;
    if ~isnan(max_val), yline(max_val, 'r', ['Max (', num2str(max_val, '%.2f'), ')'], 'LineWidth', 1.5); end
    if ~isnan(min_val), yline(min_val, 'b', ['Min (', num2str(min_val, '%.2f'), ')'], 'LineWidth', 1.5); end
    xlabel('Local component index'); ylabel('Correlation'); title(['eSPoC Correlations & Significance | ', strrep(base_name, '_', '\_')]);
    leg_labels = arrayfun(@(val) sprintf('Global %d', val), 1:size(corrs_true, 1), 'UniformOutput', false);
    legend(leg_labels, 'Location', 'best'); xlim([0.5 size(corrs_true, 2)+0.5]);
    saveas(f2, fullfile(subj_fig_dir, sprintf('%s_02_Significance.jpg', base_name))); close(f2);
    
    % РИСУНОК 3: СРАВНЕНИЕ ПРОЕКЦИЙ (CCA FIT)
    n_plot_comps = min(3, size(gl_src, 1));
    f3 = figure('Visible', 'off', 'Name', 'Canonical Projections', 'Color', 'w', 'Position', [150, 150, 1200, 400]);
    t_cca = tiledlayout(n_plot_comps, 1, 'TileSpacing', 'compact');
    sgtitle('Global Source Signal vs UMAP Canonical Target', 'FontSize', 14, 'FontWeight', 'bold');
    for i = 1:n_plot_comps
        nexttile; hold on; grid on;
        gl_n = (gl_src(i,:) - mean(gl_src(i,:))) / std(gl_src(i,:)); can_n = (emb_can_pr(i,:) - mean(emb_can_pr(i,:))) / std(emb_can_pr(i,:));
        plot(gl_n, 'LineWidth', 1.2, 'Color', [0.2 0.4 0.8]); plot(can_n, 'LineWidth', 1.2, 'Color', [0.9 0.3 0.3]);
        for k = 1:length(ticks), xline(ticks(k), '--', 'Color', [0.8 0.8 0.8]); end
        title(sprintf('Global %d | Corr = %.2f', i, corr(gl_n', can_n')));
        xticks(ticks(1:end-1)); xticklabels(arrayfun(@(val) num2str(val), 1:nEpochs, 'UniformOutput', false)); xlim([0 ticks(end)]);
        if i == 1, legend({'Unconstrained EEG Covariance', 'UMAP Canonical'}, 'Location', 'best'); end
    end
    saveas(f3, fullfile(subj_fig_dir, sprintf('%s_03_CCA_Match.jpg', base_name))); close(f3);
    
    % РИСУНОК 4: 3D КАНОНИЧЕСКОЕ ПРОСТРАНСТВО (СОХРАНЯЕМ И КАК .FIG)
    f4 = figure('Visible', 'off', 'Name', 'Canonical Space', 'Color', 'w', 'Position', [200, 200, 800, 600]);
    plot3(cx_can(~isnan(cx_can)), cy_can(~isnan(cy_can)), cz_can(~isnan(cz_can)), 'k', 'LineWidth', 1.5); hold on; grid on;
    for i = 1:nEpochs
        idx = (valid_cond_idx == i);
        if any(idx)
            scatter3(x(idx), y(idx), z(idx), 10, repmat(cmap(i,:), sum(idx), 1), 'filled', 'MarkerFaceAlpha', 0.3);
            scatter3(cx_can(i), cy_can(i), cz_can(i), 150, cmap(i,:), 'filled', 'MarkerEdgeColor', 'k');
            text(cx_can(i), cy_can(i), cz_can(i), num2str(i), 'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', [0.95 0.95 0.95], 'Margin', 1, 'HorizontalAlignment', 'center');
        end
    end
    xlabel('Canonical Axis 1'); ylabel('Canonical Axis 2'); zlabel('Canonical Axis 3'); title(['eSPoC Canonical Space | ', strrep(base_name, '_', '\_')]); view(-45, 30);
    f4_name = fullfile(subj_fig_dir, sprintf('%s_04_Canonical_Space', base_name));
    saveas(f4, [f4_name '.jpg']);
    set(f4, 'Visible', 'on'); saveas(f4, [f4_name '.fig'], 'fig'); close(f4);
    
    % РИСУНОК 5: СПЕКТР СОБСТВЕННЫХ ЗНАЧЕНИЙ
    f5 = figure('Visible', 'off', 'Name', 'Eigenvalue Spectrum', 'Color', 'w', 'Position', [250, 250, 600, 400]);
    plot(eigenvalues_true', '-o', 'LineWidth', 2, 'MarkerSize', 6); hold on; grid on;
    xlabel('Local Component Index'); ylabel('\lambda value'); title('Eigenvalue Spectrum of the matrix W');
    legend(leg_labels, 'Location', 'best'); xlim([0.5, size(eigenvalues_true, 2)+0.5]);
    saveas(f5, fullfile(subj_fig_dir, sprintf('%s_05_Eigenvalues.jpg', base_name))); close(f5);
    
    % РИСУНОК 6: МАТРИЦА РАССТОЯНИЙ
    f6 = figure('Visible', 'off', 'Name', 'Distance Matrix', 'Color', 'w', 'Position', [300, 300, 600, 500]);
    valid_c = ~isnan(cx_can);
    D = pdist([cx_can(valid_c); cy_can(valid_c); cz_can(valid_c)]'); D_sq = squareform(D);
    imagesc(D_sq); colormap(flipud(hot)); colorbar; title('Euclidean Distances (Canonical Space)');
    valid_labels = target_conditions(valid_c); xticks(1:sum(valid_c)); yticks(1:sum(valid_c));
    xticklabels(valid_labels); yticklabels(valid_labels); xtickangle(45); axis square;
    saveas(f6, fullfile(subj_fig_dir, sprintf('%s_06_DistMatrix.jpg', base_name))); close(f6);
    
    %% =====================================================================
    % РИСУНОК 7: ДЕТАЛЬНЫЙ АНАЛИЗ ВСЕХ ЗНАЧИМЫХ КОМПОНЕНТ
    % =====================================================================
    % Находим все компоненты, пробившие пороги значимости
    [sig_gl, sig_lcl] = find((corrs_true > max_val) | (corrs_true < min_val));
    
    % Защита: если значимых нет, рисуем максимальную по модулю
    if isempty(sig_gl)
        disp('  -> Значимых компонент нет. Строю анализ для максимальной по модулю.');
        [~, max_lin_idx] = max(abs(corrs_true(:)));
        [max_gl, max_lcl] = ind2sub(size(corrs_true), max_lin_idx);
        sig_gl = max_gl; sig_lcl = max_lcl;
    end
    
    for k = 1:length(sig_gl)
        gl_idx = sig_gl(k);
        lcl_idx = sig_lcl(k);
        
        if ndims(W_true) == 3
            wx_pca = squeeze(W_true(gl_idx, :, lcl_idx)); 
            ax_pca = squeeze(A_true(gl_idx, :, lcl_idx));
        else
            wx_pca = W_true(:, lcl_idx); 
            ax_pca = A_true(:, lcl_idx);
        end
        wx_pca = wx_pca(:); ax_pca = ax_pca(:); % Гарантия столбца
        
        wx = U * wx_pca; ax = U * ax_pca;
        [~, m_idx] = max(abs(wx)); wx = wx .* sign(wx(m_idx));
        [~, m_idx] = max(abs(ax)); ax = ax .* sign(ax(m_idx));
        fig_str = sprintf('SigComp_G%d_L%d', gl_idx, lcl_idx);
        f7 = figure('Visible', 'off', 'Name', fig_str, 'Color', 'w', 'Position', [350, 350, 1000, 700]);
        t = tiledlayout(2,2, 'TileSpacing', 'compact', 'Padding', 'compact');
        
        c_val = corrs_true(gl_idx, lcl_idx);
        sgtitle(sprintf('Global %d | Local %d | Corr: %.3f', gl_idx, lcl_idx, c_val), 'FontSize', 14, 'FontWeight', 'bold', 'Color', [0 0.5 0]);
        
        ax1 = nexttile(t,1); title(ax1, 'Spatial Filter (W)'); 
        topo.avg = wx; cfg_topo.figure = ax1; 
        % cmax_w = max(abs(wx)); cfg_topo.zlim = [-cmax_w, cmax_w]; 
        cfg_topo.colorbar = 'EastOutside';
        % ft_topoplotER(cfg_topo, topo); colormap(ax1, jet);
        ft_topoplotER(cfg_topo, topo); colormap(ax1);
        
        ax2 = nexttile(t,2); title(ax2, 'Spatial Pattern (A)'); 
        topo.avg = ax; cfg_topo.figure = ax2; 
        % cmax_a = max(abs(ax)); cfg_topo.zlim = [-cmax_a, cmax_a]; 
        cfg_topo.colorbar = 'EastOutside';
        % ft_topoplotER(cfg_topo, topo); colormap(ax2, jet);
        ft_topoplotER(cfg_topo, topo); colormap(ax2);
        
        ax3 = nexttile(t,3,[1,2]); hold on; grid on; title(ax3, 'Source Power Envelope vs Canonical Target');
        S_plot = zeros(1, size(Covs_valid, 3)); 
        for i = 1:size(Covs_valid, 3), S_plot(i) = wx_pca' * Covs_valid(:, :, i) * wx_pca; end
        S_plot = (S_plot - mean(S_plot)) / std(S_plot);
        
        zz = emb_can_pr(gl_idx, :); 
        zz = (zz - mean(zz)) / std(zz) * sign(c_val); 
        plot(S_plot, 'LineWidth', 1.5, 'Color', [0.2 0.6 0.8]); plot(zz, 'LineWidth', 1.5, 'Color', [0.9 0.3 0.3]);
        for tk = 1:length(ticks), xline(ax3, ticks(tk), '--', 'Color', [0.8 0.8 0.8]); end
        xticks(ax3, ticks(1:end-1)); xlim([0 ticks(end)]); xticklabels(ax3, target_conditions(1:nEpochs)); xtickangle(45);
        ylabel('Normalized Amplitude'); legend('Source Envelope (eSPoC)', 'Canonical Target (UMAP)', 'Location', 'best');
        
        f7_name = fullfile(subj_fig_dir, sprintf('%s_07_%s', base_name, fig_str));
        saveas(f7, [f7_name '.jpg']);
        set(f7, 'Visible', 'on'); saveas(f7, [f7_name '.fig'], 'fig'); close(f7);
    end
    
    %% =====================================================================
    % РИСУНОК 8: МУЛЬТИ-КОМПОНЕНТНАЯ ПАНЕЛЬ (ТОП-4)
    % =====================================================================
    % Берем 4 самые сильные компоненты (по модулю корреляции)
    [~, sort_idx] = sort(abs(corrs_true(:)), 'descend');
    top_N = min(4, length(sort_idx));
    
    f8 = figure('Visible', 'off', 'Name', 'Top 4 Components', 'Color', 'w', 'Position', [400, 400, 1400, 800]);
    t2 = tiledlayout(top_N, 5, 'TileSpacing', 'compact', 'Padding', 'compact');
    all_right_axes = [];       
    
    for i = 1:top_N
        [gl_idx, lcl_idx] = ind2sub(size(corrs_true), sort_idx(i));
        
        if ndims(W_true) == 3
            wx_pca = squeeze(W_true(gl_idx, :, lcl_idx)); ax_pca = squeeze(A_true(gl_idx, :, lcl_idx));
        else
            wx_pca = W_true(:, lcl_idx); ax_pca = A_true(:, lcl_idx);
        end
        wx_pca = wx_pca(:); ax_pca = ax_pca(:);
        wx = U * wx_pca; ax = U * ax_pca;
        [~, m_idx] = max(abs(wx)); wx = wx .* sign(wx(m_idx)); [~, m_idx] = max(abs(ax)); ax = ax .* sign(ax(m_idx));
        
        row = (i-1)*5;
        
        ax_wx = nexttile(t2, row + 1); 
        topo.avg = wx; cfg_topo.figure = ax_wx; 
        % cmax_w = max(abs(wx)); cfg_topo.zlim = [-cmax_w, cmax_w]; 
        cfg_topo.colorbar = 'SouthOutside';
        ft_topoplotER(cfg_topo, topo); colormap(ax_wx);
        if i == 1, title(ax_wx, 'Filter (W)'); end
        
        ax_ax = nexttile(t2, row + 2); 
        topo.avg = ax; cfg_topo.figure = ax_ax; 
        % cmax_a = max(abs(ax)); cfg_topo.zlim = [-cmax_a, cmax_a]; 
        cfg_topo.colorbar = 'SouthOutside';
        ft_topoplotER(cfg_topo, topo); colormap(ax_ax);
        if i == 1, title(ax_ax, 'Pattern (A)'); end
        
        ax_plot = nexttile(t2, row + 3, [1 3]); all_right_axes = [all_right_axes ax_plot];
        S_plot = zeros(1, size(Covs_valid, 3)); for j = 1:size(Covs_valid, 3), S_plot(j) = wx_pca' * Covs_valid(:, :, j) * wx_pca; end
        S_plot = (S_plot - mean(S_plot)) / std(S_plot);
        zz = emb_can_pr(gl_idx, :); zz = (zz - mean(zz)) / std(zz) * sign(corrs_true(gl_idx, lcl_idx)); 
        
        plot(ax_plot, S_plot, 'LineWidth', 1.2, 'Color', [0.2 0.6 0.8]); hold on; plot(ax_plot, zz, 'LineWidth', 1, 'Color', [0.9 0.3 0.3]); grid on;
        
        c_val = corrs_true(gl_idx, lcl_idx); t_color = 'k'; if c_val > max_val || c_val < min_val, t_color = [0 0.5 0]; end 
        title(ax_plot, sprintf('G%d L%d | Corr = %.2f', gl_idx, lcl_idx, c_val), 'Color', t_color);
        xticks(ax_plot, ticks(1:end-1)); xticklabels(ax_plot, []); xlim(ax_plot, [0, ticks(end)]);
        for k = 1:length(ticks), xline(ax_plot, ticks(k), '--', 'Color', [0.8 0.8 0.8]); end
    end
    
    if ~isempty(all_right_axes)
        xticklabels(all_right_axes(end), target_conditions(1:nEpochs)); xtickangle(all_right_axes(end), 45); xlabel(all_right_axes(end), 'Experimental Conditions');
        legend(all_right_axes(1), {'Source Envelope', 'Canonical Target'}, 'Location', 'northeastoutside');
    end
    saveas(f8, fullfile(subj_fig_dir, sprintf('%s_08_Top4Comp.jpg', base_name))); close(f8);
    
    %% =====================================================================
    % РИСУНОК 9: UMAP, РАСКРАШЕННЫЙ ПО МОЩНОСТИ ЛУЧШЕГО ИСТОЧНИКА
    % =====================================================================
    [~, max_lin_idx] = max(abs(corrs_true(:)));
    [best_gl, best_lcl] = ind2sub(size(corrs_true), max_lin_idx);
    
    if ndims(W_true) == 3
        wx_pca = squeeze(W_true(best_gl, :, best_lcl));
    else
        wx_pca = W_true(:, best_lcl);
    end
    wx_pca = wx_pca(:);
    
    S_best = zeros(1, size(Covs_valid, 3)); 
    for i = 1:size(Covs_valid, 3), S_best(i) = wx_pca' * Covs_valid(:, :, i) * wx_pca; end
    S_best = (S_best - mean(S_best)) / std(S_best); 
    
    f9 = figure('Visible', 'off', 'Name', 'UMAP Source Power', 'Color', 'w', 'Position', [500, 500, 800, 600]);
    scatter3(R(:, 1), R(:, 2), R(:, 3), 30, S_best, 'filled', 'MarkerEdgeColor', [0.2 0.2 0.2], 'LineWidth', 0.1);
    colormap(f9, 'parula');
    cb = colorbar; cb.Label.String = sprintf('Normalized Source Power (G%d-L%d)', best_gl, best_lcl); cb.Label.FontSize = 12;
    xlabel('UMAP Axis 1'); ylabel('UMAP Axis 2'); zlabel('UMAP Axis 3');
    title(sprintf('UMAP manifold colored by Best eSPoC Source Power | Corr: %.2f', corrs_true(best_gl, best_lcl)));
    view(-45, 30); grid on;
    
    f9_name = fullfile(subj_fig_dir, sprintf('%s_09_UMAP_SourcePower', base_name));
    saveas(f9, [f9_name '.jpg']);
    set(f9, 'Visible', 'on'); saveas(f9, [f9_name '.fig'], 'fig'); close(f9);
    
    disp(['Все графики сохранены в: ', subj_fig_dir]);
end
disp('==================================================');
disp('БАТЧ-ВИЗУАЛИЗАЦИЯ УСПЕШНО ЗАВЕРШЕНА!');