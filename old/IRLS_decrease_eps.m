function [x,eps_track] = IRLS(A,b,s,lambda,eps1,eps2,x0,N,Nlsp,epsmin)
n = size(A,2);
m = length(b);
x = x0;
threshold = 10^-10;

eps_track={};

for it=1:N
    x_old = x;
    
    
         %   epsilon decay rule for ||x||_p^p
        sX_c=sort(abs(x),'descend');
        sX_c_complement=sX_c((s+1):end);
%     eps_rule = norm(sX_c_complement,q)/(n-s)^(1);
        eps_rule = norm(sX_c_complement,1)/n;
%      eps_rule = eps2*10^(-(it/N)^(2-q)*10);
%     eps2 = max(min(eps2,eps_rule),epsmin);
        if norm(x_old-x)/norm(x) < sqrt(eps2)/100
            eps2 = eps2/10;
            eps2 = max(eps2,epsmin)
        eps_track{it}=eps2;
        end
        
        
    z0 = sqrt(norm(A*x-b)^2/m + eps1);
    z = sqrt(abs(lambda*x).^2 + eps2);
    A_expanded = [A/sqrt(2*z0*m);diag(lambda*sqrt(1./(2*z)))];
    b_expanded = [b/sqrt(2*z0*m);zeros(n,1)];
    [x, flag] = lsqr(A_expanded,b_expanded, [], Nlsp,[],[],x);
    

    if flag ==  1
        fprintf('lsqr did not converged');
        return;
    end
%     x = A_expanded \ b_expanded;
    
    if norm(x_old-x) < threshold
        break;
    end
end
end