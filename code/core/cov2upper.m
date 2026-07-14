function [v] = cov2upper(C)
upper_mask = triu(true(size(C)));
upper_triu_mask = triu(true(size(C)), 1);
C(upper_triu_mask) = C(upper_triu_mask) * sqrt(2);
v = C(upper_mask);
v = v(:);
end
