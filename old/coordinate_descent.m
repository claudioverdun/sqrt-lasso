function x = coordinate_descent(x0, grad, Hessian, eta, lambda, N)%, obj)
threshold = 10^-10;
n = length(x0);
lin_0 = grad - Hessian * eta;
x = x0;
for it=1:N
    
    x_old = x;
    for j=1:n
        lin_plus = Hessian(j,:) * x - Hessian(j,j) * x(j);
        lin = lin_0(j) + lin_plus;
        sq = 0.5*Hessian(j,j);
        
        % minimizing quadratic equation 
        % sq * x_j^2 + lin* x_j + lambda*|x_j|
        
        % case sq = 0
        if abs(sq) < eps
            if abs(lin) <= lambda
                x(j) = 0;
            else
                x(j) = Inf;
                fprintf('Reached infinity');
                return
            end
        % case sq >0
        elseif sq >0
            x(j) = - 0.5* max(abs(lin) - lambda,0)*sign(lin)/sq;
        end
        % case sq <0 due to convexity of the function
    end
    
%     [obj(x_old) obj(x)]
    if norm(x - x_old) < threshold
        break;
    end
end
end