function x = IRLS(A,b,lambda,eps1,eps2,x0,Nlsp,residual)
n = size(A,2);
m = length(b);
x = x0;
threshold = 10^-10;

while true
    x_old = x;
    z0 = sqrt(norm(A*x-b)^2/m + eps1);
    z = sqrt(abs(lambda*x).^2 + eps2);
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
end