function x = Accelerated_IRLS_v3(A,b,lambda,eps1,eps2,x0,N,Nlsp)
n = size(A,2);
m = length(b);
x = x0;
threshold = 10^-10;
v = x;
Lk = lambda/ sqrt(eps2) + 1.0/sqrt(eps1);
ak1 = 0;
obj = @(x,z0,z) 0.5*(norm(A*x-b)^2/m + eps1)/z0 + 0.5*z0 + 0.5*sum((abs(lambda*x).^2 + eps2)./z + z);
grad = @(x,z0,z) A'*(A*x - b)/m/z0 + lambda^2*sum(x ./ z);


z0 = sqrt(norm(A*x-b)^2/m + eps1);
z = sqrt(abs(lambda*x).^2 + eps2);

for it=1:N
    x_old = x;
    Lk1 = 0.5*Lk;
    while true
        % x update
        
        ak1= 0.5/Lk1 + sqrt(0.25 / Lk1^2 + ak1^2 * Lk/Lk1 );
        tauk = 1.0/(ak1*Lk1);
        yk = tauk *v + (1- tauk)*x;
        
        z0k = sqrt(norm(A*yk-b)^2/m + eps1);
        zk = sqrt(abs(lambda*yk).^2 + eps2);
        A_expanded = [A/sqrt(2*z0k*m);diag(lambda*sqrt(1./(2*zk)))];
        b_expanded = [b/sqrt(2*z0k*m);zeros(n,1)];
        [x, flag] = lsqr(A_expanded,b_expanded, [], Nlsp,[],[],yk);

        if flag ==  1
            fprintf('lsqr did not converged');
            return;
        end
        z0 = sqrt(norm(A*x-b)^2/m + eps1);
        z = sqrt(abs(lambda*x).^2 + eps2);
        gr = grad(yk,z0k,zk);
        if obj(x,z0,z) <= obj(yk,z0k,zk) - 0.5*norm(gr)^2/ Lk1
            v = v - ak1*gr;
            break;
        end
        Lk1 = 2 * Lk1;
        x= x_old;
    end
    if norm(x_old-x) < threshold
        break;
    end
end
end


