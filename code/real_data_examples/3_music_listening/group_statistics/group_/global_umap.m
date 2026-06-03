%% =====================================================================
% GROUP-LEVEL MANIFOLD ALIGNMENT (TANGENT SPACE -> MEAN DISTANCE -> UMAP)
% =====================================================================
close all; clear; clc;

% --- 1. НАСТРОЙКИ ---
base_dirs = {
    'C:\Users\ansbel\Documents\GitHub\TriCo\data\external\music_listening\part1\', ...
    'C:\Users\ansbel\Documents\GitHub\TriCo\data\external\music_listening\part2\'
};
freq_name = 'beta';
N_common = 19; 

disp('>>> ШАГ 1: СБОР МАТРИЦ РАССТОЯНИЙ (TANGENT SPACE) <<<');
All_Dist_Matrices = []; 
subj_count = 0;

for p = 1:length(base_dirs)
    emb_dir = fullfile(base_dirs{p}, 'embeddings', freq_name);
    mat_files = dir(fullfile(emb_dir, ['UMAP_*_', freq_name, '.mat']));
    
    for f = 1:length(mat_files)
        % Мы берем файл UMAP, чтобы получить доступ к ковариациям (Covs)
        load(fullfile(emb_dir, mat_files(f).name), 'Tcovs_valid', 'valid_cond_idx');
        
        % Tcovs_valid сейчас [N_features x N_windows]
        n_features = size(Tcovs_valid, 1);
        
        % cc теперь [N_conditions x N_features]
        cc = NaN(N_common, n_features);
        
        for c = 1:N_common
            idx = find(valid_cond_idx == c);
            if ~isempty(idx)
                % Усредняем векторы Тангенциального пространства по всем окнам условия
                cc(c, :) = mean(Tcovs_valid(:, idx), 2);
            end
        end        

        % Если субъект прошел все 19 условий
        if ~any(isnan(cc(:)))
            % Считаем матрицу расстояний для одного субъекта (19x19)
            % Используем фробениусову норму для матриц (Riemannian distance)
            dist_mat = zeros(N_common, N_common);
            for i = 1:N_common
                for j = 1:N_common
                    C1 = squeeze(cc(i,:,:)); C2 = squeeze(cc(j,:,:));
                    dist_mat(i,j) = norm(C1 - C2, 'fro');
                end
            end
            All_Dist_Matrices(:, :, end+1) = dist_mat;
            subj_count = subj_count + 1;
        end
    end
end

% --- 2. УСРЕДНЕНИЕ МАТРИЦЫ И UMAP ---
Mean_Dist_Matrix = mean(All_Dist_Matrices, 3);

disp('>>> ШАГ 2: UMAP-ВЛОЖЕНИЕ ГРУППОВОЙ МАТРИЦЫ <<<');
% UMAP на матрице расстояний (через 'precomputed' metric)
% Требуется установленный UMAP для MATLAB (от Lutz)
[Y_global, ~] = run_umap(Mean_Dist_Matrix, 'metric', 'precomputed', 'n_neighbors', 5, 'n_components', 2);

% --- 3. ВИЗУАЛИЗАЦИЯ ---
target_conditions = {
    'EC1', 'EO1', '2Hz', '05Hz', '4Hz', '1Hz', '3Hz', ...
    'NoRy 1', 'Waltz 1', 'Waltz 2', 'NoRy 2', 'NoRy 3', ...
    'Waltz 3', 'NoRy 4', 'Waltz 4', 'NoRy 5', 'Waltz 5', ...
    'EC2', 'EO2'
};

figure('Name', 'Group-level UMAP Embedding', 'Color', 'w', 'Position', [100, 100, 800, 700]);
hold on; grid on;
scatter(Y_global(:,1), Y_global(:,2), 200, 'filled', 'MarkerFaceAlpha', 0.7);

for i = 1:N_common
    text(Y_global(i,1), Y_global(i,2), target_conditions{i}, 'FontSize', 12, 'FontWeight', 'bold');
end

title(['Групповой UMAP (на основе усредненной матрицы расстояний, N=', num2str(subj_count), ')']);
xlabel('UMAP Dim 1'); ylabel('UMAP Dim 2');