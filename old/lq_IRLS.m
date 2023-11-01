function x = IRLS(A,b,lambda,q,eps1,eps2,x0,N,Nlsp)
n = size(A,2);
m = length(b);
x = x0;
threshold = 10^-10;

for it=1:N
    x_old = x;
    z0 = (2 - q)^2 * (norm(A*x-b)^2/m + eps1)^((2-q)/2) / q^2;
    z = (2 - q)^2 *(abs(lambda*x).^2 + eps2).^((2-q)/2) / q^2;
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