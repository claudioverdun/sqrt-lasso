%% This is a helper function to compute the supprot recovery ratio (SRR)
% It is given by |S \cap S_st| / |S| where S is the support of the
% ground-truth and S_st are the st indices correspondign to the largest in
% magnitude absolute values. 
%
% Inputs:
% x : n x 1 a vector of dimension
% s : integer, 1 <= s <= n, the number of entries in the support. In the ground-truth they are the
% first s entries
% st : integer, 1 <= s <= n, the number of entries from x to use for SRR computation. 
function ratio = s_ratio(x,s,st)
n = length(x);
idx_0 = abs(x) >= 10^-6;
if sum(idx_0) == 0
    ratio = 0;
elseif sum(idx_0) < st
    ratio = 1.0*sum(idx_0(1:s)) / s;
else
    x_tr = quantile(abs(x), 1.0*(n-st) / n);
    idx = abs(x) >= x_tr;
    ratio = 1.0*sum(idx(1:s)) / s;
end
end