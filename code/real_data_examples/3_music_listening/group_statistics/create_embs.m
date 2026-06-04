%% =====================================================================
% BATCH UMAP PIPELINE FOR CONTINUOUS EEG DATA (STRICT BOUNDARIES & NaN PROTECTION)
% =====================================================================
close all; clear; clc;

% --- 1. Настройка параметров ---
data_dir  = 'C:\Users\ansbel\Documents\GitHub\TriCo\data\external\music_listening\part2\eeg\';
base_out_dir = 'C:\Users\ansbel\Documents\GitHub\TriCo\data\external\music_listening\part2\embeddings\';
bads_file = fullfile(data_dir, 'BADS.mat');
ft_path   = 'C:\Users\ansbel\Documents\GitHub\site-packages\fieldtrip';

freq_name = 'beta';        % Имя диапазона 
freq_band = [15, 25];      % Границы фильтра

% АВТОМАТИЧЕСКОЕ СОЗДАНИЕ ПАПКИ ДЛЯ ТЕКУЩЕГО ДИАПАЗОНА
out_dir = fullfile(base_out_dir, freq_name);
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

n_channels = 38;            % Только ЭЭГ
Wsize     = 2;              % Окно нарезки (сек)
Ssize     = 0.5;            % Шаг нарезки (сек)
cond_len  = 120;            % Длительность одного условия (сек)

% target_conditions = {
%    'EC1', 'EO1', '2Hz', '05Hz', '4Hz', '1Hz', '3Hz', ...
%    'NoRy 1', 'Waltz 1', 'Waltz 2', 'NoRy 2', 'NoRy 3', ...
%    'Waltz 3', 'NoRy 4', 'Waltz 4', 'NoRy 5', 'Waltz 5', ...
%    'EC2', 'EO2'
% };

% Список текстовых названий условий
target_conditions = {
    'EC1', 'EO1', '2Hz', '05Hz', '4Hz', '1Hz', '3Hz', ...
    'NoRy 1', 'Waltz 1', 'Waltz 2', 'NoRy 2', 'NoRy 3', ...
    'Waltz 3', 'NoRy 4', 'Waltz 4', 'NoRy 5', 'Waltz 5', ...
    'EC2', 'EO2', 'Waltz 6', 'Waltz 7', 'Waltz 8'
};

% Инициализация FieldTrip
if ~exist('ft_defaults','file'), addpath(ft_path); end
ft_defaults;

% Загрузка матрицы с BADS 
load(bads_file); 

% Поиск всех _raw.fif файлов
fif_files = dir(fullfile(data_dir, '*_raw.fif'));
disp(['Найдено файлов для обработки: ', num2str(length(fif_files))]);

%% --- 2. Главный цикл по файлам ---
for f = 1:length(fif_files)
    
    fname = fif_files(f).name;
    base_name = strrep(fname, '_raw.fif', '');
    disp(['==================================================']);
    disp(['Обработка файла: ', base_name, ' [', freq_name, ']']);
    
    % Загрузка данных
    cfg = [];
    cfg.dataset = fullfile(data_dir, fname);
    cfg.continuous = 'yes';
    Xinf = ft_preprocessing(cfg);
    Fs = Xinf.fsample;
    
    Xraw = Xinf.trial{1}(1:n_channels, :);
    total_samples = size(Xraw, 2);
    
    total_duration_sec = total_samples / Fs;
    nEpochs = round(total_duration_sec / cond_len);
    disp(['Распознано условий: ', num2str(nEpochs)]);
    
    %% --- 3. Создание масок (BADS и Условия) ---
    mask_ts = true(1, total_samples);
    cond_ts = zeros(1, total_samples);
    samples_per_cond = round(cond_len * Fs);
    
    for k = 1:nEpochs
        idx_st = (k-1) * samples_per_cond + 1;
        idx_en = min(k * samples_per_cond, total_samples);
        cond_ts(idx_st:idx_en) = k;
    end
    
    mat_field_name = ['s_', base_name];
    if exist(mat_field_name, 'var')
        bad_intervals = eval(mat_field_name);
        for i = 1:size(bad_intervals, 1)
            bad_st_samp = max(1, round(bad_intervals(i, 1) * Fs) + 1);
            bad_en_samp = min(total_samples, round((bad_intervals(i, 1) + bad_intervals(i, 2)) * Fs));
            mask_ts(bad_st_samp : bad_en_samp) = false;
        end
    else
        warning(['Для файла ', fname, ' не найдено BADS. Используется 100% маска.']);
    end
    
    %% --- 4. Фильтрация и Предварительная разметка окон ---
    [b, a] = butter(3, freq_band / (Fs/2));   
    Xfilt = filtfilt(b, a, Xraw')'; 
    
    Xfilt = Xfilt(:, round(Fs/2) : end - round(Fs/2));
    mask_ts = mask_ts(:, round(Fs/2) : end - round(Fs/2));
    cond_ts = cond_ts(:, round(Fs/2) : end - round(Fs/2)); 
    
    % Оборачиваем вызов epoch_data в squeeze, чтобы убрать лишнее измерение канала
    m_wins = squeeze(epoch_data(double(mask_ts'), Fs, Wsize, Ssize));
    c_wins = squeeze(epoch_data(cond_ts', Fs, Wsize, Ssize));
    
    valid_artifacts = all(m_wins, 1); 
    valid_boundaries = (c_wins(1, :) == c_wins(end, :));
    valid_windows = (valid_artifacts & valid_boundaries)';
    
    cond_idx_epochs = c_wins(1, :)'; 
    valid_cond_idx = cond_idx_epochs(valid_windows);    
    
    %% --- 5. Базовый SVD ---
    [U_full, S_full, ~] = svd(Xfilt(:, mask_ts), 'econ');
    S_diag = diag(S_full);
    ve = S_diag.^2;
    var_explained = cumsum(ve) / sum(ve);
    var_explained(end) = 1; 
    
    %% --- 6. Динамический расчет SVD и Касательного пространства ---
    var_thresh = 1.00;
    reduce_step = 0.005; 
    has_nans = true;
    
    while has_nans && var_thresh > 0.5
        n_components = find(var_explained >= var_thresh, 1); 
        if isempty(n_components), n_components = length(S_diag); end
        
        U = U_full(:, 1:n_components);               
        Xfiltpca = U' * Xfilt;
        
        ep_wins = epoch_data(Xfiltpca', Fs, Wsize, Ssize);
        
        Covs = zeros(size(ep_wins, 2), size(ep_wins, 2), size(ep_wins, 3)); 
        for i = 1:size(ep_wins, 3)
            Covs(:,:,i) = cov(ep_wins(:,:,i));
        end
        
        Covs_valid = Covs(:, :, valid_windows);
        Tcovs_valid = Tangent_space(Covs_valid);  
        
        if any(isnan(Tcovs_valid(:)))
            disp(['[!] Обнаружены NaN в Tangent Space (var = ', num2str(var_thresh), ...
                  ', comps = ', num2str(n_components), '). Понижаем размерность...']);
            var_thresh = var_thresh - reduce_step;
        else
            has_nans = false;
            disp(['[+] Tangent Space стабильно. Использовано дисперсии: ', num2str(var_thresh), ...
                  ' (Компонент: ', num2str(n_components), ')']);
        end
    end
    
    if has_nans
        error(['Не удалось избавиться от NaN в файле ', fname, ' даже при сильном снижении размерности.']);
    end
    
    %% --- 7. UMAP Embedding ---
    disp('Расчет UMAP...');
    u = UMAP("n_neighbors", 20, "n_components", 3, "min_dist", 0);
    u.metric = 'euclidean';
    u.target_metric = 'euclidean';
    
    R = u.fit_transform(Tcovs_valid');
    
    %% --- 8. ВИЗУАЛИЗАЦИЯ (С ЦЕНТРОИДАМИ И ТРАЕКТОРИЕЙ) ---
    disp('Создание и сохранение графиков...');
    
    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1000, 800]);
    cmap = jet(nEpochs);
    hold on; grid on;
    
    ccx = zeros(1, nEpochs); ccy = zeros(1, nEpochs); ccz = zeros(1, nEpochs);
    
    % Перебираем условия и рисуем облака точек
    for i = 1:nEpochs
        idx = (valid_cond_idx == i);
        sc_x = R(idx, 1); sc_y = R(idx, 2); sc_z = R(idx, 3);
        
        if ~isempty(sc_x)
            ccx(i) = mean(sc_x); ccy(i) = mean(sc_y); ccz(i) = mean(sc_z);
            
            % Рисуем облако точек кластера
            scatter3(sc_x, sc_y, sc_z, 15, repmat(cmap(i,:), length(sc_x), 1), 'filled', 'MarkerFaceAlpha', 0.3);
            
            % Рисуем крупный центроид
            scatter3(ccx(i), ccy(i), ccz(i), 150, cmap(i,:), 'filled', 'MarkerEdgeColor', 'k');
            
            % Плашка с номером условия 
            text(ccx(i), ccy(i), ccz(i), num2str(i), 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k', ...
                'BackgroundColor', [0.95 0.95 0.95], 'Margin', 1, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
        else
            ccx(i) = NaN; ccy(i) = NaN; ccz(i) = NaN;
        end
    end
    
    % Рисуем черную траекторию между центроидами
    valid_c = ~isnan(ccx);
    plot3(ccx(valid_c), ccy(valid_c), ccz(valid_c), 'k', 'LineWidth', 1.5);
    
    % --- Настройка Colorbar с названиями условий ---
    colormap(cmap);
    caxis([0.5, nEpochs + 0.5]);
    cb = colorbar;
    cb.Ticks = 1:nEpochs;
    
    % Динамическая сборка подписей
    cb_labels = cell(1, nEpochs);
    for i = 1:nEpochs
        if i <= length(target_conditions)
            cb_labels{i} = sprintf('%d: %s', i, target_conditions{i});
        else
            cb_labels{i} = sprintf('%d: Cond %d', i, i);
        end
    end
    cb.TickLabels = cb_labels;
    cb.Label.String = 'Condition ID';
    cb.Label.FontSize = 12;
    cb.Label.FontWeight = 'bold';
    
    % Заголовок
    title_str = sprintf('UMAP: %s | Band: %s\nVar Thresh: %.3f | SVD Components: %d', ...
                        strrep(base_name, '_', '\_'), freq_name, var_thresh, n_components);
    title(title_str, 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('UMAP 1'); ylabel('UMAP 2'); zlabel('UMAP 3');
    view(-45, 30);
    
    % --- СОХРАНЕНИЕ ГРАФИКОВ ---
    img_name = sprintf('UMAP_%s_%s.jpg', base_name, freq_name);
    saveas(fig, fullfile(out_dir, img_name));
    
    set(fig, 'Visible', 'on'); 
    fig_name = sprintf('UMAP_%s_%s.fig', base_name, freq_name);
    saveas(fig, fullfile(out_dir, fig_name), 'fig');
    
    close(fig); 
    
    %% --- 9. Сохранение данных в .mat ---
    out_name = sprintf('UMAP_%s_%s.mat', base_name, freq_name);
    out_path = fullfile(out_dir, out_name);
    
    save(out_path, 'R', 'U', 'Covs', 'Tcovs_valid', 'valid_windows', 'valid_cond_idx', 'n_components', 'freq_band', 'freq_name', 'nEpochs');
    
    disp(['Данные и графики сохранены в: ', out_dir]);
end
disp('==================================================');
disp('ВСЕ ФАЙЛЫ УСПЕШНО ОБРАБОТАНЫ!');