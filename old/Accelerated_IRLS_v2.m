function x = Accelerated_IRLS_v2(A,b,lambda,eps1,eps2,x0,N,Nlsp)
n = size(A,2);
m = length(b);
x = x0;
threshold = 10^-10;
Ak = 0;
v = x;
beta = 0;
obj = @(x,z0,z) 0.5*(norm(A*x-b)^2/m + eps1)/z0 + 0.5*z0 + 0.5*sum((abs(lambda*x).^2 + eps2)./z + z);
grad = @(x,z0,z) A'*(A*x - b)/m/z0 + lambda^2*sum(x ./ z);


z0 = sqrt(norm(A*x-b)^2/m + eps1);
z = sqrt(abs(lambda*x).^2 + eps2);

for it=1:N
    % x update
    x_old = x;
    beta = fminbnd(@(beta) obj(x-beta*(v-x),z0,z),0,1);
    
    yk = x + beta*(v-x);
    z0k = sqrt(norm(A*yk-b)^2/m + eps1);
    zk = sqrt(abs(lambda*yk).^2 + eps2);
    A_expanded = [A/sqrt(2*z0k*m);diag(lambda*sqrt(1./(2*zk)))];
    b_expanded = [b/sqrt(2*z0k*m);zeros(n,1)];
    [x, flag] = lsqr(A_expanded,b_expanded, [], Nlsp,[],[],yk);
    
    if flag ==  1
        fprintf('lsqr did not converged');
        return;
    end
    
    if norm(x_old-x) < threshold
        break;
    end
    
    diff = obj(yk,z0k,zk) - obj(x,z0k,zk);
    gr = grad(yk,z0k,zk);
    gradnsq = norm(gr)^2;
    
    akp = (diff + sqrt(diff*(diff + 4*Ak*gradnsq )))/(2*gradnsq);
    Ak = Ak+akp;
    v = v - akp*gr(1:n);
end
end


