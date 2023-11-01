function [x, f_vals] = IRLS_eps_decay(A,b,lambda,eps0,x0,N,Nlsp,decay,s)
n = size(A,2);
m = length(b);
x = x0;
threshold = 10.^-10;
 
f_min = realmax;

if strcmp(decay,'exp')
    eps2 = eps0;
else
    eps2 = realmax;
end

f_vals = zeros(N,1);

for it=1:N
    x_old = x;
     
    switch decay
        case 'sqrt' 
            eps2 = eps0 / sqrt(it);
        case 'harm' 
            eps2 = eps0 / it;
        case 'fn_sqrt'
            f_min = min(norm(A*x - b)/sqrt(m) + lambda* norm(x,1),f_min);
            eps2 = f_min / (2*lambda*sqrt((n+1)*it));
        case 'sigma'
            mags = sort(abs(x),'descend');
            sigma = sum(mags((s+1):n));
            eps2 = min(eps2, eps0*(norm(A*x - b)/sqrt(m) + lambda* sigma)/lambda / (n+1));
        case 'Rn'
            mags = sort(abs(x),'descend');
            eps2 = min(eps2, eps0*(norm(A*x - b)/sqrt(m) + lambda* mags(s+1))/lambda / (n+1));
        case 'exp'
            eps2 = eps2*0.5;
    end
    eps1 = lambda*eps2;
    
    f_vals(it) = objective(A,x,b,lambda,eps1,eps2);
    
    %z0 = sqrt(norm(A*x-b)^2/m + eps1);
    %z = sqrt(abs(lambda*x).^2 + eps2);
    z0 = max(norm(A*x-b)/sqrt(m),eps1);
    z = max(abs(x),eps2);
    A_expanded = [A/sqrt(z0*m);diag(sqrt(lambda)./sqrt(z))];
    b_expanded = [b/sqrt(z0*m);zeros(n,1)];
    [x, flag] = lsqr(A_expanded,b_expanded, 10^-10, Nlsp,[],[],x);
   
%     if flag ==  1
%         fprintf('lsqr did not converged');
%         return;
%     end
%     x = A_expanded \ b_expanded;
    
%     if norm(x_old-x) < threshold
%         break;
%     end
end
end

function f_val = objective(A,x,b,lambda,eps1,eps2)
m = length(b);
f_val = norm(A*x - b)/sqrt(m);
if f_val < eps1
    f_val = 0.5*(f_val.^2/ eps1 + eps1); 
end
xabs = abs(x);
idx = xabs < eps2;
xabs(idx) = 0.5*(xabs(idx).^2/ eps2 + eps2);
f_val = f_val + lambda*sum(xabs);
end