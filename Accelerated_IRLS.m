function x = Accelerated_IRLS(A,b,lambda,eps1,eps2,x0,N,Nlsp)
n = size(A,2);
m = length(b);
x = x0;
threshold = 10^-10;
Ak = 0;
v = x;
beta = 0;
obj = @(x,z0,z) (norm(A*x-b)^2/m + eps1)/z0 + z0 + sum((abs(lambda*x).^2 + eps2)./z + z);
grad = @(x,z0,z) [2*A'*(A*x - b)/m/z0 + 2 * lambda^2*sum(x ./ z);
    1 - (norm(A*x - b)^2/m + eps1)/z0^2;
    1 - lambda^2 *(x.^2 + eps2) ./ z.^2];

z0 = sqrt(norm(A*x-b)^2/m + eps1);
z = sqrt(abs(lambda*x).^2 + eps2);
vz = z;
vz0 = z0;

for it=1:N
    % x update
    x_old = x;
    max_beta = min([1, min([(z0 - sqrt(eps1)/2)/(vz0-z0);(z - sqrt(eps2)/2)./(vz-z)])]);
    if (max_beta >0)
        beta = fminbnd(@(beta) obj(x-beta*(v-x),z0 -beta*(vz0-z0),z -beta*(vz-z)),0,max_beta);
    else
        beta =0;
    end
    yk = x + beta*(v-x);
    z0k = z0 -beta*(vz0-z0);
    zk = z -beta*(vz-z);
    A_expanded = [A/sqrt(z0k*m);diag(lambda*sqrt(1./zk))];
    b_expanded = [b/sqrt(z0k*m);zeros(n,1)];
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
    v = v + akp*gr(1:n);
    vz0 = vz0 + akp*gr(n+1);
    vz = vz + akp*gr(n+2:end);
    
    % z update
    if (max_beta >0)
        beta = fminbnd(@(beta) obj(x-beta*(v-x),z0 -beta*(vz0-z0),z -beta*(vz-z)),0,max_beta);
    else
        beta =0;
    end
    yk = x + beta*(v-x);
    z0k = z0 -beta*(vz0-z0);
    zk = z -beta*(vz-z);
    
    z0 = sqrt(norm(A*yk-b)^2/m + eps1);
    z = sqrt(abs(lambda*yk).^2 + eps2);
    
    diff = obj(yk,z0k,zk) - obj(yk,z0,z);
    gr = grad(yk,z0k,zk);
    gradnsq = norm(gr)^2;
    
    akp = (diff + sqrt(diff*(diff + 4*Ak*gradnsq )))/(2*gradnsq);
    Ak = Ak+akp;
    v = v + akp*gr(1:n);
    vz0 = vz0 + akp*gr(n+1);
    vz = vz + akp*gr(n+2:end);
end
end


