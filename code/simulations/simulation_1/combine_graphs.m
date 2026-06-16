close all
clear
clc

% 1. Открываем исходные фигуры невидимыми
h1 = openfig('spoc_snr_comparison.fig', 'invisible');
h2 = openfig('spoc_train_comparison.fig', 'invisible');

% 2. Создаем новую фигуру для объединенного результата (делаем её более квадратной)
new_fig = figure('Position', [100, 100, 1000, 820], 'Color', 'w');

% 3. Находим все оси в исходных фигурах
axes1 = findobj(h1, 'Type', 'axes');
axes2 = findobj(h2, 'Type', 'axes');

% Сортируем оси слева направо по их исходной координате X,
% чтобы точно знать, где Power Course (левый), а где Pattern (правый)
[~, idx1] = sort(arrayfun(@(ax) ax.Position(1), axes1));
axes1 = axes1(idx1); 

[~, idx2] = sort(arrayfun(@(ax) ax.Position(1), axes2));
axes2 = axes2(idx2);

% 4. Копируем объекты осей в новую фигуру
ax_snr_power  = copyobj(axes1(1), new_fig); % Слева из фигуры SNR
ax_snr_pattern = copyobj(axes1(2), new_fig);

ax_train_power  = copyobj(axes2(1), new_fig); % Справа из фигуры Train
ax_train_pattern = copyobj(axes2(2), new_fig);

% 5. Задаем координаты для сетки 2x2 [лево, низ, ширина, высота]
% Оставляем отступы для подписей осей и заголовков
pos_top_left     = [0.08, 0.54, 0.38, 0.36];
pos_bottom_left  = [0.08, 0.10, 0.38, 0.36];
pos_top_right    = [0.56, 0.54, 0.38, 0.36];
pos_bottom_right = [0.56, 0.10, 0.38, 0.36];

% Применяем новые позиции
set(ax_snr_power,   'Position', pos_top_left);
set(ax_snr_pattern, 'Position', pos_bottom_left);
set(ax_train_power,  'Position', pos_top_right);
set(ax_train_pattern,'Position', pos_bottom_right);

% 6. Корректируем заголовки
title(ax_snr_power,   'Power Time Course Correlation', 'FontSize', 11, 'FontWeight', 'bold');
title(ax_snr_pattern, 'Pattern Correlation', 'FontSize', 11, 'FontWeight', 'bold');
title(ax_train_power,  'Power Time Course Correlation', 'FontSize', 11, 'FontWeight', 'bold');
title(ax_train_pattern,'Pattern Correlation', 'FontSize', 11, 'FontWeight', 'bold');

% 7. ВОССТАНАВЛИВАЕМ ЛЕГЕНДУ
% Оставляем легенду только на первом (левом верхнем) графике
legend(ax_snr_power, 'Location', 'southeast', 'Interpreter', 'tex', 'FontSize', 10);

% 8. Общий заголовок для всей композиции
% annotation('textbox', [0, 0.93, 1, 0.07], 'String', 'eSPoC vs SPoC Simulation Framework Evaluation', ...
%     'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');

% 9. Закрываем временные скрытые фигуры
close(h1);
close(h2);

% Обновляем холст и выводим итоговое окно
drawnow;
figure(new_fig);

% 10. Сохраняем объединенную панель графиков
savefig(gcf, 'spoc_combined_matrix.fig');
exportgraphics(gcf, 'spoc_combined_matrix.jpg', 'Resolution', 300);

fprintf('Успешно создана и сохранена объединенная фигура 2х2.\n');