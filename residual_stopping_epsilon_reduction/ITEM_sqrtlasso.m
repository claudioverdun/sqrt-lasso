function z = ITEM_sqrtlasso(A,b,lambda,mu,L,eps0,K0,x0,residual)
m = length(b);
eps = eps0;
K = K0/sqrt(eps);
t = 1;

while true
    gradf= @(x) A' * (A*x-b)/ sqrt( norm(A*x-b)^2 + eps)/sqrt(m) + lambda * x ./ sqrt( abs(x).^2 + eps);
    condition =@(x) norm(A*x-b)/sqrt(m) <= residual;
    z = ITEM(gradf,mu,L,K,x0,condition);
    eps = eps - eps0 *6 / (t^2*pi^2);
%     norm(A*z - b)/sqrt(m)
    t = t+1;
    K = ceil(K0 / sqrt(eps) / t^(6/8));
    
    if norm(A*z - b)/sqrt(m) <= residual
        break;
    end
    
    if t == 100
            break;
        end
end

end