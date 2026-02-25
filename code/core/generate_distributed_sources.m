function [X_s, X_bg, X_n, z, GA] = generate_distributed_sources(G, Nsrc, Ndistr,...
    flanker, Ts, Fs)

N = Ts*Fs;
flanker = flanker*Fs;

%set filters
[b,a] = butter(5,[8,12]/(Fs/2)); % for sources
[bn,an] = butter(5,[1,35]/(Fs/2)); % for noise

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
    S(k,:) = (S(k,:)-mean(S(k,:))) / std(S(k,:));
    env = abs(hilbert(S(k,:)')');
    z(k,:) = env;
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