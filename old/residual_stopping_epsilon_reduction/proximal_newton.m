function x = proximal_newton(A,b,eps0,K0,lambda,x0, Ninner, residual)
m = length(b);
threshold = 10^-10;


x= x0;
eps = eps0;
K = K0/sqrt(eps);
t = 1;
while true
    obj = @(x) sqrt(norm(A * x - b)^2 + eps)/sqrt(m);
    grad = @(x) A' * (A*x-b)/ sqrt( norm(A*x-b)^2 + eps)/ sqrt(m);
    H = @(x) Hessian(x, A,b,eps);
    Q = @(x,y,lambda) obj(y) + grad(y)' * (x-y) + 0.5*(x-y)'*H(y)*(x-y) + lambda* norm(x,1);
    F = @(x,lambda) obj(x) + lambda* norm(x,1);
    for k=1:K
        x_new = coordinate_descent(x, grad(x), H(x), x, lambda, Ninner);%, @(t) Q(t,x,lambda));
    %     [Q(x,x,lambda), Q(x_new,x,lambda)]
        diff = x_new - x;
        gamma = grad(x)' * diff + lambda*(norm(x_new,1) - norm(x,1));
        eta = 1;
        prev= F(x,lambda);

        while F(x + eta*diff,lambda) > prev + 0.25 * gamma* eta
            eta = eta*0.9;
    %         [F(x + eta*diff,lambda) - prev - 0.25 * gamma* eta]
        end

    %     eta
        x = x + eta*diff;

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

function H = Hessian(x, A, b, eps)
m = length(b);
res = A*x  - b;
nr = norm(res)^2;
ATres = A' * res;
H = (A' * A/sqrt( nr + eps ) + ATres * ATres'/(nr+eps)^(3/2)  )/sqrt(m);
end