% Algorithm 1 from 
% On Fast Convergence of Proximal Algorithms for
% SQRT-Lasso Optimization: Don’t Worry About
% its Nonsmooth Loss Function

function x = proximal_gradient(A,b,eps0,K0,lambda,Lmax,x0,residual)
threshold = 10^-10;
m = length(b);

L = Lmax;
Ltilde = L;
x = x0;

eps = eps0;
K = K0/sqrt(eps);
t = 1;
while true 
    obj = @(x) sqrt(norm(A * x - b)^2 + eps)/sqrt(m);
    grad = @(x) A' * (A*x-b)/ sqrt( norm(A*x-b)^2 + eps)/ sqrt(m);
    Q = @(x,y,lambda,L) obj(x) + grad(x)' * (y-x) + 0.5*L*norm(y-x)^2 + lambda* norm(y,1);
    F = @(x,lambda) obj(x) + lambda* norm(x,1);
    
    for k=1:K
    
        %it, L
        x_old = x;
        gr = grad(x);
        while true
            thresh = step(x,gr,Ltilde,lambda);
            obj_new = F(thresh,lambda);
            q_new = Q(x,thresh,lambda,Ltilde);
            if obj_new >= q_new
                break;
            end
            Ltilde = 0.5 * Ltilde;
        end

        L = min(2*Ltilde,Lmax);
        Ltilde = L;

        x = step(x,gr,L,lambda);
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
    
    if t == 100
        break;
    end
end
end

function thresh = step(x,gr,L,lambda)
xnew = x - gr / L;
thresh = sign(xnew) .* max(abs(xnew) - lambda/L,0 );
end