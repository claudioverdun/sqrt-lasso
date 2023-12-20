function [x, time] = proximal_newton_to_rie(A,b,eps1,lambda,x0,N, Ninner, x_tr, threshold, timeout)
time = 0;
tic ;

Aha = A' * A;
m = length(b);
%eta_threshold = 10^-10;%10^-16;%
obj = @(x) sqrt(norm(A * x - b)^2 + eps1)/sqrt(m);
grad = @(x) A' * (A*x-b)/ sqrt( norm(A*x-b)^2 + eps1)/ sqrt(m);
H = @(x) Hessian(x, A,Aha,b,eps1);
Q = @(x,y,lambda) obj(y) + grad(y)' * (x-y) + 0.5*(x-y)'*H(y)*(x-y) + lambda* norm(x,1);
F = @(x,lambda) obj(x) + lambda* norm(x,1);

x= x0;

while true
    x_old = x;
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
    
%     if eta*norm(diff) < eta_threshold % eta < threshold
%         break;
%     end
%     eta
    x = x + eta*diff;

    if norm(x_old-x) < threshold * norm(x_old)
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

function H = Hessian(x, A, AhA, b, eps)
m = length(b);
res = A*x  - b;
nr = norm(res)^2;
ATres = A' * res;
H = (AhA/sqrt( nr + eps ) + ATres * ATres'/(nr+eps)^(3/2)  )/sqrt(m);
end