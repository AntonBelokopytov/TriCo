function X_epo = epoch_data(X, Fs, Ws, Ss)
% EPOCH_DATA  Segment continuous data into overlapping epochs (Optimized).

[T, D] = size(X);

% Переводим секунды в отсчеты
W = fix(Ws * Fs);   % размер окна
S = fix(Ss * Fs);   % шаг

% Защита от слишком короткого сигнала
if T < W
    X_epo = [];
    return;
end

% 1. Математически вычисляем итоговое количество эпох (E)
E = floor((T - W) / S) + 1;

% 2. ПРЕДАЛЛОКАЦИЯ: сразу выделяем память под итоговый массив.
% class(X) сохраняет тип данных (например, 'single'), экономя память для ЭЭГ.
X_epo = zeros(W, D, E, class(X));

% 3. Быстрый цикл for (MATLAB JIT-компилятор идеально его оптимизирует)
for ep = 1:E
    % Вычисляем начальный и конечный индексы напрямую
    start_idx = (ep - 1) * S + 1;
    end_idx = start_idx + W - 1;
    
    % Записываем данные в заранее подготовленную ячейку памяти
    X_epo(:, :, ep) = X(start_idx:end_idx, :);
end

% Удаляем лишнее измерение, если эпоха всего одна
if E == 1
    X_epo = squeeze(X_epo);
end

end