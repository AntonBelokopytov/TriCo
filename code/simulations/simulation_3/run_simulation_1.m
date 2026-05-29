close all
clear
clc

ft_path = 'C:\Users\ansbel\Documents\2Git\fieldtrip';

if ~exist('ft_defaults','file')
    addpath(ft_path);
end

ft_defaults;

%%
G = load("MNE_EEG_FWD_TRPL.mat").MNE_EEG_FWD_TRPL;
elec = load("electrodes_data.mat").electrodes_data;

topo = [];
topo.dimord = 'chan_time';
topo.label  = elec.label;  
topo.time   = 0;
topo.elec   = elec;

laycfg = [];
laycfg.elec = elec;
lay = ft_prepare_layout(laycfg);     

cfg = [];
cfg.marker       = '';
cfg.layout       = lay;
cfg.comment      = 'no';
cfg.style        = 'fill';
cfg.markersymbol = '';
cfg.colorbar     = 'yes'; 

%% =================== ПАРАМЕТРЫ СИМУЛЯЦИИ ===================
Nsrc = 100;     % количество источников
Ndistr = 1;     % количество активных распределённых источников
flanker = 1;    % параметр генерации
Ts = 900;       % длительность (сек)
Fs = 250;       % частота дискретизации

clear u
u = UMAP("n_neighbors",30,"n_components",2);
u.metric = 'euclidean';
u.target_metric = 'euclidean';

%%
Ecorrs_spoc=[];Acorrs_spoc=[];
Ecorrs_pca=[];Acorrs_pca=[];
Ecorrs_ica=[];Acorrs_ica=[];
for mc=1:10
    mc
    % Генерация фоновой активности
    [~,X_bg,X_n] = generate_distributed_sources(G, Nsrc, Ndistr,...
        flanker, Ts, Fs);

    % Генерация двух независимых источников
    [Xs1,~,~,z1,GA1] = generate_distributed_sources(G, Nsrc, Ndistr,...
        flanker, Ts, Fs);
    [Xs2,~,~,z2,GA2] = generate_distributed_sources(G, Nsrc, Ndistr,...
        flanker, Ts, Fs);
    
    % Матрица "истинных" огибающих
    z = zeros(2,size(Xs1,2));

    % Истинные паттерны источников
    Ainit = [GA1(:,1),GA2(:,1)];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % =================== ФОРМИРОВАНИЕ ВРЕМЕННОЙ СТРУКТУРЫ ===================
    % Источник 1 активен во второй трети,
    % источник 2 — в третьей трети
    
    N = fix(Fs*Ts / 3);
    Xs = zeros(size(Xs1));
    
    t = N:N*2;
    Xs(:,t) = Xs1(:,t);
    z(1,t) = z1(1,t);
 
    t = N*2:N*3;
    Xs(:,t) = Xs2(:,t);
    z(2,t) = z2(1,t);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % =================== ПЕРЕБОР SNR ===================
    SNR_raw = 1:1:10;
    
    for snr_i=1:size(SNR_raw,2)
        SNR = SNR_raw(snr_i);
        SNR     

        % Формирование итогового сигнала:
        % источник + фон + шум
        X = SNR*Xs + X_bg + 0.1*trace(cov(Xs'))*X_n;
        
        %
        [b,a] = butter(5,[8,12]/(Fs/2)); 
        Xfilt = filtfilt(b,a,X')';
        Cxx = cov(Xfilt');
                    
        %%%%
        % =================== PCA ===================
        [U,S,~] = svd(Xfilt,'econ');
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        S = diag(S);
        % compute an estimate of the rank of the data
        tol = max(size(X)) * eps(S(1));
        r = sum(S > tol);
        % compute cumulative variance explained
        ve = S.^2;
        var_explained = cumsum(ve) / sum(ve);
        var_explained(end) = 1;
        n_components = find(var_explained>=0.99, 1);
        n_components = max(min(n_components, r), 1);
        
        Xpca = U(:,1:n_components)'*Xfilt;
        
        % =================== Нарезка на эпохи ===================
        Ws = 5;
        Ss = 1;

        epochs = epoch_data(Xpca',Fs,Ws,Ss);
        
        covs = []; Covs_vec = []; Covs_pca = [];
        for i = 1:size(epochs,3)
            C = cov(epochs(:,:,i));
            covs(:,:,i) = C;
            Covs_vec(i,:) = C(triu(true(size(C))));
        end
        
        % =================== UMAP ===================
        % PCA
        Tcovs = Tangent_space(covs);
                
        R = u.fit_transform(Tcovs');

        % =================== eSPoC ===================
        % R — целевая переменная (UMAP координаты)
        [W, A, Vx, Vz, corrs, VecCov, z, Epochs_cov] = espoc(epochs, R(:,1:2)');
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Берём первую и последнюю канонические оси
        Wst = W(:,:,1);
        Wen = W(:,:,end);
        Ast = A(:,:,1);
        Aen = A(:,:,end);

        % Возвращаем фильтры в сенсорное пространство
        w = U(:,1:n_components)*cat(1,Wst,Wen)'; 
        a = U(:,1:n_components)*cat(1,Ast,Aen)'; 
        
        % =================== ОГИБАЮЩИЕ ===================
        env=[];
        for i=1:4
            env(i,:) = abs(hilbert(w(:,i)'*Xfilt));
        end

        % =================== КОРРЕЛЯЦИИ ===================
        env_corrs = abs(corr(z',env')); Ecorrs=[]; Acorrs=[]; found_f=[];
        for i=1:2
            [maxVal, linIdx] = max(env_corrs(:));     
            [init_w, rec_w] = ind2sub(size(env_corrs), linIdx);
            
            Ecorrs(init_w) = env_corrs(init_w, rec_w);
            Acorrs(init_w) = abs(corr(a(:,rec_w), Ainit(:,init_w)));
            found_f(i) = rec_w;
            env_corrs(init_w,:) = 0;
            env_corrs(:,rec_w) = 0;
        end

        Ecorrs_spoc(mc,snr_i,:) = Ecorrs;
        Acorrs_spoc(mc,snr_i,:) = Acorrs;
        
        % a_spoc = a(:,found_f);
        % w_spoc = w(:,found_f);
                
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % =================== PCA ===================
        nPC=64;
        w = U(:,1:nPC); 
        a = Cxx*w;
        
        env=[];
        for i=1:nPC
            env(i,:) = abs(hilbert(w(:,i)'*Xfilt));
        end

        pat_corrs = abs(corr(Ainit,a)); 
        env_corrs = abs(corr(z',env')); Ecorrs=[]; Acorrs=[]; found_f=[];
        for i=1:2
            [maxVal, linIdx] = max(env_corrs(:));     
            [init_w, rec_w] = ind2sub(size(env_corrs), linIdx);
            
            Ecorrs(init_w) = env_corrs(init_w, rec_w);
            Acorrs(init_w) = abs(corr(a(:,rec_w), Ainit(:,init_w)));
            found_f(i) = rec_w;
            env_corrs(init_w,:) = 0;
            env_corrs(:,rec_w) = 0;
        end
        
        Ecorrs_pca(mc,snr_i,:) = Ecorrs;
        Acorrs_pca(mc,snr_i,:) = Acorrs;
        
        % a_pca = a(:,found_f);
        % w_pca = w(:,found_f);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % =================== ICA ===================
        nIC=64;

        [icasig, Aica, Wica] = fastica( ...
        Xfilt, ...
        'numOfIC', nIC, ...
        'approach', 'symm', ...
        'g', 'tanh', ...
        'verbose', 'off');

        w = Wica'; 
        a = Cxx*w;

        env=[];
        for i=1:nIC
            env(i,:) = abs(hilbert(w(:,i)'*Xfilt));
        end

        pat_corrs = abs(corr(Ainit,a)); 
        env_corrs = abs(corr(z',env')); Ecorrs=[]; Acorrs=[]; found_f=[];
        for i=1:2
            [maxVal, linIdx] = max(env_corrs(:));     
            [init_w, rec_w] = ind2sub(size(env_corrs), linIdx);

            Ecorrs(init_w) = env_corrs(init_w, rec_w);
            Acorrs(init_w) = abs(corr(a(:,rec_w), Ainit(:,init_w)));
            found_f(i) = rec_w;
            env_corrs(init_w,:) = 0;
            env_corrs(:,rec_w) = 0;
        end

        Ecorrs_ica(mc,snr_i,:) = Ecorrs;
        Acorrs_ica(mc,snr_i,:) = Acorrs;
        
        % a_ica = a(:,found_f);
        % w_ica = w(:,found_f);
    end
end

%% ================= ENVELOPE =================
n = size(Ecorrs_spoc,1);

% --- SPoC
z1 = atanh(Ecorrs_spoc);
m1 = squeeze(mean(z1,1));
s1 = squeeze(std(z1,0,1));
sem1 = s1/sqrt(n);
meanE1 = tanh(m1);
ciE1_low  = tanh(m1 - 1.96*sem1);
ciE1_high = tanh(m1 + 1.96*sem1);

% --- PCA
z2 = atanh(Ecorrs_pca);
m2 = squeeze(mean(z2,1));
s2 = squeeze(std(z2,0,1));
sem2 = s2/sqrt(n);
meanE2 = tanh(m2);
ciE2_low  = tanh(m2 - 1.96*sem2);
ciE2_high = tanh(m2 + 1.96*sem2);

% --- ICA
z3 = atanh(Ecorrs_ica);
m3 = squeeze(mean(z3,1));
s3 = squeeze(std(z3,0,1));
sem3 = s3/sqrt(n);
meanE3 = tanh(m3);
ciE3_low  = tanh(m3 - 1.96*sem3);
ciE3_high = tanh(m3 + 1.96*sem3);

% -------- Row 1: Src 1 (Envelope)
nexttile(1)
title('Envelope correlation – Src 1')
hold on

fill([SNR_raw fliplr(SNR_raw)], ...
     [ciE1_low(:,1)' fliplr(ciE1_high(:,1)')], ...
     [0.6 0.8 1],'EdgeColor','none','FaceAlpha',0.3)

fill([SNR_raw fliplr(SNR_raw)], ...
     [ciE2_low(:,1)' fliplr(ciE2_high(:,1)')], ...
     [1 0.6 0.6],'EdgeColor','none','FaceAlpha',0.3)

fill([SNR_raw fliplr(SNR_raw)], ...
     [ciE3_low(:,1)' fliplr(ciE3_high(:,1)')], ...
     [0.6 1 0.6],'EdgeColor','none','FaceAlpha',0.3)

plot(SNR_raw,meanE1(:,1),'b','LineWidth',2)
plot(SNR_raw,meanE2(:,1),'r','LineWidth',2)
plot(SNR_raw,meanE3(:,1),'g','LineWidth',2)

xlabel('SNR')
ylabel('Correlation')
legend('eSPoC CI','PCA CI','ICA CI','eSPoC','PCA','ICA')
grid on


% -------- Row 2: Src 2 (Envelope)
nexttile(3)
title('Envelope correlation – Src 2')
hold on

fill([SNR_raw fliplr(SNR_raw)], ...
     [ciE1_low(:,2)' fliplr(ciE1_high(:,2)')], ...
     [0.6 0.8 1],'EdgeColor','none','FaceAlpha',0.3)

fill([SNR_raw fliplr(SNR_raw)], ...
     [ciE2_low(:,2)' fliplr(ciE2_high(:,2)')], ...
     [1 0.6 0.6],'EdgeColor','none','FaceAlpha',0.3)

fill([SNR_raw fliplr(SNR_raw)], ...
     [ciE3_low(:,2)' fliplr(ciE3_high(:,2)')], ...
     [0.6 1 0.6],'EdgeColor','none','FaceAlpha',0.3)

plot(SNR_raw,meanE1(:,2),'b','LineWidth',2)
plot(SNR_raw,meanE2(:,2),'r','LineWidth',2)
plot(SNR_raw,meanE3(:,2),'g','LineWidth',2)

xlabel('SNR')
ylabel('Correlation')
grid on


% ================= PATTERN =================
n = size(Acorrs_spoc,1);

% --- SPoC
z1 = atanh(Acorrs_spoc);
m1 = squeeze(mean(z1,1));
s1 = squeeze(std(z1,0,1));
sem1 = s1/sqrt(n);
meanP1 = tanh(m1);
ciP1_low  = tanh(m1 - 1.96*sem1);
ciP1_high = tanh(m1 + 1.96*sem1);

% --- PCA
z2 = atanh(Acorrs_pca);
m2 = squeeze(mean(z2,1));
s2 = squeeze(std(z2,0,1));
sem2 = s2/sqrt(n);
meanP2 = tanh(m2);
ciP2_low  = tanh(m2 - 1.96*sem2);
ciP2_high = tanh(m2 + 1.96*sem2);

% --- ICA
z3 = atanh(Acorrs_ica);
m3 = squeeze(mean(z3,1));
s3 = squeeze(std(z3,0,1));
sem3 = s3/sqrt(n);
meanP3 = tanh(m3);
ciP3_low  = tanh(m3 - 1.96*sem3);
ciP3_high = tanh(m3 + 1.96*sem3);

% -------- Row 1: Src 1 (Pattern)
nexttile(2)
title('Pattern correlation – Src 1')
hold on

fill([SNR_raw fliplr(SNR_raw)], ...
     [ciP1_low(:,1)' fliplr(ciP1_high(:,1)')], ...
     [0.6 0.8 1],'EdgeColor','none','FaceAlpha',0.3)

fill([SNR_raw fliplr(SNR_raw)], ...
     [ciP2_low(:,1)' fliplr(ciP2_high(:,1)')], ...
     [1 0.6 0.6],'EdgeColor','none','FaceAlpha',0.3)

fill([SNR_raw fliplr(SNR_raw)], ...
     [ciP3_low(:,1)' fliplr(ciP3_high(:,1)')], ...
     [0.6 1 0.6],'EdgeColor','none','FaceAlpha',0.3)

plot(SNR_raw,meanP1(:,1),'b','LineWidth',2)
plot(SNR_raw,meanP2(:,1),'r','LineWidth',2)
plot(SNR_raw,meanP3(:,1),'g','LineWidth',2)

xlabel('SNR')
ylabel('Correlation')
grid on


% -------- Row 2: Src 2 (Pattern)
nexttile(4)
title('Pattern correlation – Src 2')
hold on

fill([SNR_raw fliplr(SNR_raw)], ...
     [ciP1_low(:,2)' fliplr(ciP1_high(:,2)')], ...
     [0.6 0.8 1],'EdgeColor','none','FaceAlpha',0.3)

fill([SNR_raw fliplr(SNR_raw)], ...
     [ciP2_low(:,2)' fliplr(ciP2_high(:,2)')], ...
     [1 0.6 0.6],'EdgeColor','none','FaceAlpha',0.3)

fill([SNR_raw fliplr(SNR_raw)], ...
     [ciP3_low(:,2)' fliplr(ciP3_high(:,2)')], ...
     [0.6 1 0.6],'EdgeColor','none','FaceAlpha',0.3)

plot(SNR_raw,meanP1(:,2),'b','LineWidth',2)
plot(SNR_raw,meanP2(:,2),'r','LineWidth',2)
plot(SNR_raw,meanP3(:,2),'g','LineWidth',2)

xlabel('SNR')
ylabel('Correlation')
grid on

%%
