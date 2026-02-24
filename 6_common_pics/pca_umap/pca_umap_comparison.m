rng(1)

n = 3200;
t = linspace(0,1,n)';

x = randn(n,1);
y = randn(n,1);
z = randn(n,1);
g = randn(n,1);

random_vals = randn(n,1);

N = 1000;
idx = 1:N;
x(idx) = 15*random_vals(idx);

idx = N+1:N*2;
y(idx) = 25*random_vals(idx);

idx = N*2+1:N*3;
z(idx) = 9*random_vals(idx);

idx = 3001:3200;
g(idx) = 5*random_vals(idx);

X = [x y z g];
X = abs(hilbert(X));

color = burst;

figure
plot(X)

labels = [ones(N,1);
          2*ones(N,1);
          3*ones(N,1);
          4*ones(200,1)];

%%
figure
scatter3(x,y,z,20,labels)

%% PCA
[coeff,score] = pca(X);

figure
scatter(score(:,1),score(:,2),20,labels,'filled')
title('PCA')
colorbar

%% UMAP
clear u
u = UMAP("n_neighbors",50,"n_components",2);
u.metric = 'euclidean';

R = u.fit_transform(X);

figure
scatter(R(:,1),R(:,2),20,labels,'filled')
title('UMAP')
colorbar

legend('Source 1', 'Source 2', 'Source 3', 'Source 4')

%%
figure
tiledlayout(3,2,'Padding','compact','TileSpacing','compact')

cmap = lines(4);

% ===== Верхняя панель: временные ряды =====
nexttile([1,2])
plot(X,'LineWidth',1)
title('Simulated time series')
xlabel('Time')
ylabel('Power')
grid on

xline([1000,2000,3000], 'red--', 'LineWidth',4)
xlim([0,3200])

legend(       {'Source 1','Source 2','Source 3','Source 4'}, ...
       'Location','bestoutside')
% ===== PCA =====
nexttile
hold on

h1 = scatter(score(labels==1,1),score(labels==1,2),20,cmap(1,:),'filled');
h2 = scatter(score(labels==2,1),score(labels==2,2),20,cmap(2,:),'filled');
h3 = scatter(score(labels==3,1),score(labels==3,2),20,cmap(3,:),'filled');
h4 = scatter(score(labels==4,1),score(labels==4,2),20,cmap(4,:),'filled');

title('PCA')
xlabel('PC1')
ylabel('PC2')
axis equal
grid on

% ===== UMAP =====
nexttile
hold on

u1 = scatter(R(labels==1,1),R(labels==1,2),20,cmap(1,:),'filled');
u2 = scatter(R(labels==2,1),R(labels==2,2),20,cmap(2,:),'filled');
u3 = scatter(R(labels==3,1),R(labels==3,2),20,cmap(3,:),'filled');
u4 = scatter(R(labels==4,1),R(labels==4,2),20,cmap(4,:),'filled');

title('UMAP 2D embedding')
xlabel('Dim 1')
ylabel('Dim 2')
axis equal
grid on

legend([u1 u2 u3 u4], ...
       {'Source 1','Source 2','Source 3','Source 4'}, ...
       'Location','bestoutside')

set(gcf,'Color','w')
set(gca,'FontSize',12)

nexttile(5)
hold on
ax = gca;
ax.ColorOrderIndex = 5;

h = plot(score);   % сохраняем хендлы линий

xline([1000,2000,3000], ...
      'red--', ...
      'LineWidth',4, ...
      'HandleVisibility','off');  % не в легенду

title('PCA components')
xlabel('Time')
ylabel('Score')
grid on

% создаём подписи автоматически
nComp = size(score,2);
labels_pca = arrayfun(@(k) sprintf('PC%d',k), ...
                      1:nComp, ...
                      'UniformOutput',false);

legend(h, labels_pca, 'Location','best')



nexttile(6)
hold on
ax = gca;
ax.ColorOrderIndex = 5;

h = plot(R);

xline([1000,2000,3000], ...
      'red--', ...
      'LineWidth',4, ...
      'HandleVisibility','off');

title('UMAP coordinates')
xlabel('Time')
ylabel('Coordinate')
grid on

nDim = size(R,2);
labels_umap = arrayfun(@(k) sprintf('Dim %d',k), ...
                       1:nDim, ...
                       'UniformOutput',false);

legend(h, labels_umap, 'Location','best')


%%
