function x = pathwise_smooth_concomitant_lasso_v2(A, b, threshold, N, T, F, sigma0, lambda_max)
m = length(b);
n = size(A,2);
lambda = min(norm(A' * b, 'Inf')/(norm(b) * sqrt(m)), lambda_max);
x = zeros(n,1);
sigma = norm(b)/sqrt(m);

norms = sum(abs(A).^2,1);
res = A*x-b;
res_norm = norm(res);
% lambda reduction loops
for t= 1:T
    for it=1:N
        if mod(it,F) == 1
            theta = -res/max([lambda*m*sigma0 norm(A' * res,'Inf') lambda*sqrt(m)*res_norm]);
            P = 0.5*res_norm/sqrt(m) + lambda*norm(x,1);
            D = 0.5*norm(b)^2/m - 0.5*norm(b - lambda*m*theta)^2/m;
            G = P-D;
            
            if G < threshold 
                break;
            end
        end
        
        for j = 1:n
            col = A(:,j);
            xj_old = x(j);
            update = col' * res;
            xnew = x(j) - update/norms(j);
            x(j) = sign(xnew) .* max(abs(xnew) - m*sigma*lambda/norms(j),0 );
            
            % update residual
            res_norm = sqrt(res_norm^2 + 2*(x(j) - xj_old)*update + (x(j) - xj_old)^2*norms(j));
            res = res + col*(x(j) - xj_old);
            
            sigma = max(sigma0,res_norm/sqrt(m));
        end
    end
    
    lambda = 0.5*lambda;
end
end
