function x = IRLS(A,b,lambda,eps0,K0,x0,Nlsp,residual)
n = size(A,2);
m = length(b);
x = x0;
threshold = 10^-10;

eps = eps0;
K = K0/sqrt(eps);
t = 1;
while true
    for k=1:K
        x_old = x;
        z0 = sqrt(norm(A*x-b)^2/m + eps);
        z = sqrt(abs(lambda*x).^2 + eps);
        A_expanded = [A/sqrt(2*z0*m);diag(lambda*sqrt(1./(2*z)))];
        b_expanded = [b/sqrt(2*z0*m);zeros(n,1)];
        [x, flag] = lsqr(A_expanded,b_expanded, 10^-10, Nlsp,[],[],x);

    %     if flag ==  1
    %         fprintf('lsqr did not converged');
    %         return;
    %     end
    %     x = A_expanded \ b_expanded;
        %norm(A*x - b)/sqrt(m)
        if norm(A*x - b)/sqrt(m) <= residual
            break;
        end
    end
    eps = eps - eps0 *6 / (t^2*pi^2);
%     norm(A*x - b)/sqrt(m)
    t = t+1;
    K = ceil(K0 / sqrt(eps) / t^(6/8));
    
    if norm(A*x - b)/sqrt(m) <= residual
        break;
    end
end
   
end