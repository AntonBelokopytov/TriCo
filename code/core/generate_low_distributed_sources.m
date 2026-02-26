function [X_s, X_bg, X_n, z, GA] = generate_low_distributed_sources(G, Nsrc, Ndistr,...
    flanker, Ts, Fs)

N = Ts*Fs;
flanker = flanker*Fs;

%set filters
[b,a] = butter(3,[8,12]/(Fs/2)); % for sources
[bn,an] = butter(3,[1,35]/(Fs/2)); % for noise
[bl,al] = butter(3, 0.5/(Fs/2)); % for modulation < 0.5 Гц
[be,ae] = butter(3,5/(Fs/2)); % for envelope noise

% init forward model
Gx = G(:,1:3:end);  
Gy = G(:,2:3:end);  
Gz = G(:,3:3:end);  

[Nsens, Nsites] = size(Gx);

% Create random sources with random direction
GA = zeros(Nsens, Nsrc);
src_indsA = randperm(Nsites);
for i = 1:Nsrc
    src_idx = src_indsA(i);
    r = rand(3,1); r = r / norm(r);          
    GA(:,i) = Gx(:,src_idx)*r(1) + Gy(:,src_idx)*r(2) + Gz(:,src_idx)*r(3);
end

% Generate source timeseries
S = filtfilt(b,a,randn(Nsrc,N+2*flanker)')';
S = S(:,flanker+1:end-flanker);

% Create random envelopes for every source
for k=1:Nsrc
    S(k,:) = (S(k,:) - mean(S(k,:)));
    env = abs(hilbert(S(k,:)')');
    S(k,:) = S(k,:) ./ env;


    mk = filtfilt(bl,al,randn(1,N+2*flanker));
    mk = mk(:,flanker+1:end-flanker);
    
    mk_clean = mk - min(mk) + eps;
    
    % Add some envelope noise
    en = filtfilt(be,ae,randn(1,N+2*flanker));
    en = en(:,flanker+1:end-flanker);
    en_noize = en./norm(en);

    mk_noise = mk + 0.1*norm(mk)*en_noize;
    mk_noise = mk_noise - min(mk_noise) + eps;


    S(k,:) = S(k,:).*mk_noise; 
    s_std = std(S(k,:));
    S(k,:) = S(k,:) ./ s_std;  

    
    z(k,:) = mk_clean;
end

% generate clean sensor data
X_s = GA(:,1:Ndistr)*S(1:Ndistr,:);

% generate other constant sources
X_bg = GA(:,Ndistr+1:end)*S(Ndistr+1:end,:);

% generate white noise
X_n = filtfilt(bn,an,randn(Nsens,N+2*flanker)')';
X_n = X_n(:,flanker+1:end-flanker);

% normalize all the parts
Pn = trace(cov(X_n'));
X_n = X_n./sqrt(Pn);

end
