function x = smooth_concomitant_lasso(A, b, threshold, N, T, F, sigma0, lambda_max)
m = length(b);
n = size(A,2);
lambda = min(norm(A' * b, 'Inf')/(norm(b) * sqrt(m)), lambda_max);
x = zeros(n,1);
sigma = norm(b)/sqrt(m);

norms = sum(abs(A).^2,1);

% lambda reduction loops
for t= 1:T
    for it=1:N
        if mod(it,F) == 1
            theta = (b - A*x)/max([lambda*m*sigma0 norm(A' * (A*x-b),'Inf') lambda*sqrt(m)*norm(A*x - b)]);
            P = 0.5*norm(A*x - b)/sqrt(m) + lambda*norm(x,1);
            D = 0.5*norm(b)^2/m - 0.5*norm(b - lambda*m*theta)^2/m;
            G = P-D;
            
            if G < threshold 
                break;
            end
        end
        
        for j = 1:n
            xnew = x(j) - A(:,j)' * (A*x - b)/norms(j);
            x(j) = sign(xnew) .* max(abs(xnew) - m*sigma*lambda/norms(j),0 );
            sigma = max(sigma0,norm(A*x - b)/sqrt(m));
        end
    end
    
    lambda = 0.5*lambda;
end
end
