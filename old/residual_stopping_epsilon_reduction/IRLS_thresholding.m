function x = IRLS(A,b,lambda,eps0,K0,x0,Nlsp,residual)
n = size(A,2);
m = length(b);
x = x0;
threshold = 10^-10;

eps = eps0;
K = K0/sqrt(eps);
t = 1;

norms = sum(abs(A).^2,1);

while true
    for k=1:K
        x_old = x;
%         res = A*x-b;
%         res_norm_sq = norm(res)^2;
%         l1_norm = sum(sqrt(abs(lambda*x).^2 + eps));
%         obj = sqrt(res_norm_sq/m + eps) + l1_norm
        
        z0 = sqrt(norm(A*x-b)^2/m + eps);
        z = sqrt(abs(lambda*x).^2 + eps);
        A_expanded = [A/sqrt(2*z0*m);diag(lambda*sqrt(1./(2*z)))];
        b_expanded = [b/sqrt(2*z0*m);zeros(n,1)];
        [x, flag] = lsqr(A_expanded,b_expanded, 10^-10, Nlsp,[],[],x);
        
        res = A*x-b;
        res_norm_sq = norm(res)^2;
        l1_norm = sum(sqrt(abs(lambda*x).^2 + eps));
        obj = sqrt(res_norm_sq/m + eps) + l1_norm;
        sigma = max(eps,sqrt(res_norm_sq/m));
        
        for j = 1:n
            col = A(:,j);
            qu = true;
            
            while qu
                update = col' * res;
                xnew = x(j) - update/norms(j);
                xnew = sign(xnew) .* max(abs(xnew) - m*sigma*lambda/norms(j),0 );

                % update residual
                res_norm_sq_new = res_norm_sq + 2*(xnew - x(j))*update + abs(xnew - x(j))^2*norms(j);

                l1_norm_new = l1_norm - sqrt(abs(lambda*x(j))^2 + eps) + sqrt(abs(lambda*xnew)^2 + eps);
                obj_new = sqrt(res_norm_sq_new/m + eps) + l1_norm_new;
                qu = false;
                if (obj_new < obj)
                    res = res + col*(xnew - x(j));
                    res_norm_sq = res_norm_sq_new;
                    l1_norm = l1_norm_new;
                    obj = obj_new;
                    x(j) = xnew;
                    sigma = max(eps,sqrt(res_norm_sq/m));
                    qu = true;
                end
            end
        end
        
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