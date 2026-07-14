function [V] = spoc(F, z)
[D,E] = size(F);

Sfz = zeros(D,1);
Sff = zeros(D,D);

F = F - mean(F,2);
z = (z - mean(z)) / std(z);

for e = 1:E
    Sfz = Sfz + F(:,e) * z(e);
    Sff = Sff + F(:,e) * F(:,e)';
end

Sfz = Sfz / (E-1);
Sff = Sff / (E-1);

[V,S] = eig(Sfz*Sfz', Sff); s=diag(S);[s,idxs]=sort(s,'descend');V=V(:,idxs);
end
