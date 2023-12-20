% Algorithm 1 from 
% On Fast Convergence of Proximal Algorithms for
% SQRT-Lasso Optimization: Don’t Worry About
% its Nonsmooth Loss Function

function [x,time] = proximal_gradient_to_rie(A,b,eps1,lambda,Lmax,x0,N, x_tr, threshold, timeout)
%threshold = 10^-10;%10^-16;
time = 0;
tic;

m = length(b);
obj = @(x) sqrt(norm(A * x - b)^2 + eps1)/sqrt(m);
grad = @(x) A' * (A*x-b)/ sqrt( norm(A*x-b)^2 + eps1)/ sqrt(m);
Q = @(x,y,lambda,L) obj(x) + grad(x)' * (y-x) + 0.5*L*norm(y-x)^2 + lambda* norm(y,1);
F = @(x,lambda) obj(x) + lambda* norm(x,1);
L = Lmax;
Ltilde = L;
x = x0;

while true
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
%     if (norm(x_old - x) < threshold)
%         break;
%     end
    if norm(x_old-x) < threshold * norm(x_old)
        dt = toc;
        time = time + dt;
        break;
    end

    dt = toc;
    time = time + dt;
    if time >= timeout  
        break
    end
    tic;
end

end

function thresh = step(x,gr,L,lambda)
xnew = x - gr / L;
thresh = sign(xnew) .* max(abs(xnew) - lambda/L,0 );
end