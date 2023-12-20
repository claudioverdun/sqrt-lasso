function [x,time] = smooth_concomitant_lasso_v2_to(A, b, N, sigma0, lambda, x0, x_tr, threshold, timeout)
time = 0;
tic;

m = length(b);
n = size(A,2);
x = x0;
sigma = norm(A*x - b)/sqrt(m);

norms = sum(abs(A).^2,1);
res = A*x-b;
res_norm = norm(res);

while true
%     if mod(it,F) == 1
%         theta = -res/max([lambda*m*sigma0 norm(A' * res,'Inf') lambda*sqrt(m)*res_norm]);
%         P = 0.5*res_norm/sqrt(m) + lambda*norm(x,1);
%         D = 0.5*norm(b)^2/m - 0.5*norm(b - lambda*m*theta)^2/m;
%         G = P-D;
% 
%         if G < threshold 
%             break;
%         end
%     end

    for j = 1:n
        col = A(:,j);
        xj_old = x(j);
        update = col' * res;
        xnew = x(j) - update/norms(j);
        x(j) = sign(xnew) .* max(abs(xnew) - m*sigma*lambda/norms(j),0 );

        % update residual
        res_norm = sqrt(max(res_norm^2 + 2*(x(j) - xj_old)*update + (x(j) - xj_old)^2*norms(j),0));
        res = res + col*(x(j) - xj_old);

        sigma = max(sigma0,res_norm/sqrt(m));
    end

    if norm(x_tr-x) < threshold * norm(x_tr)
        dt = toc;
        time = time + dt;
        break;
    end

    dt = toc;
    time = time + dt;
    if time >= timeout
        break;
    end
    tic;
end
end
