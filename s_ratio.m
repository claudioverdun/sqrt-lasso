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