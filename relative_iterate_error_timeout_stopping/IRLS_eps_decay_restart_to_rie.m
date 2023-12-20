function [x,time] = IRLS_eps_decay_restart_to_rie(A,b,lambda,eps0,x0,Nlsp,N_it,decay,s,solver, x_tr, threshold, timeout)
time = 0;
tic;

n = size(A,2);
m = length(b);
x = x0;
%threshold = 10.^-3;%10.^-6;

f_min = realmax;

if strcmp(decay,'exp')
    eps2 = eps0;
else
    eps2 = realmax;
end

% f_vals = zeros(N,1);

% N_it = floor(N / Nrestart);
epst = eps0;
rit = 1;
while true
    offset = (rit-1) * N_it;
    
    for it=1:N_it
        x_old = x;     
        switch decay
            case 'sqrt' 
                eps2 = epst / sqrt(it);
            case 'harm' 
                eps2 = eps0 / it;
            case 'fn_sqrt'
                f_min = min(norm(A*x - b)/sqrt(m) + lambda* norm(x,1),f_min);
                eps2 = f_min / (2*lambda*sqrt((n+1)*it));
            case 'sigma'
                mags = sort(abs(x),'descend');
                sigma = sum(mags((s+1):n));
                eps2 = min(eps2, eps0*(norm(A*x - b)/sqrt(m) + lambda* sigma)/lambda / (n+1));
            case 'Rn'
                mags = sort(abs(x),'descend');
                eps2 = min(eps2, eps0*(norm(A*x - b)/sqrt(m) + lambda* mags(s+1))/lambda / (n+1));
            case 'exp'
                eps2 = eps2*0.5;
        end
        eps1 = lambda*eps2;
        
%         f_vals(offset + it) = objective(A,x,b,lambda,eps1,eps2);
        
        %z0 = sqrt(norm(A*x-b)^2/m + eps1);
        %z = sqrt(abs(lambda*x).^2 + eps2);
        z0 = max(norm(A*x-b)/sqrt(m),eps1);
        z = max(abs(x),eps2);
        switch solver
            case 'lsqr'
                A_expanded = [A/sqrt(z0*m);diag(sqrt(lambda)./sqrt(z))];
                b_expanded = [b/sqrt(z0*m);zeros(n,1)];
                [x, flag] = lsqr(A_expanded,b_expanded, [], Nlsp,[],[],x);
            case 'kaczmarz'
                A_expanded = [A/sqrt(z0*m);diag(sqrt(lambda)./sqrt(z))];
                b_expanded = [b/sqrt(z0*m);zeros(n,1)];
                x = reshuffling_kaczmarz(A_expanded,b_expanded, x, 10^-6, Nlsp);
            case 'cd'
                A_expanded = [A/sqrt(z0*m);diag(sqrt(lambda)./sqrt(z))];
                b_expanded = [b/sqrt(z0*m);zeros(n,1)];
                x = coordinate_descent_reshuffling(A_expanded,b_expanded, x, 10^-6, Nlsp);
            case 'lsqr_fun'
                A_expanded = @(x,flag) afun(x,flag,A,z0,z,lambda,m);
                b_expanded = [b/sqrt(z0*m);zeros(n,1)];
                warning off
                [x, flag] = lsqr(A_expanded,b_expanded, 10^-10, Nlsp,[],[],x);
                warning on
        end
    %     if flag ==  1
    %         fprintf('lsqr did not converged');
    %         return;
    %     end
    %     x = A_expanded \ b_expanded;
        
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
    if norm(x_old-x) < threshold * norm(x_old)
        break;
    end

    if time >= timeout
        break;
    end
    epst = eps2;
    rit = rit + 1;
%     threshold = threshold*0.1;
end
% i = 100;
end

function f_val = objective(A,x,b,lambda,eps1,eps2)
m = length(b);
f_val = norm(A*x - b)/sqrt(m);
if f_val < eps1
    f_val = 0.5*(f_val.^2/ eps1 + eps1); 
end
xabs = abs(x);
idx = xabs < eps2;
xabs(idx) = 0.5*(xabs(idx).^2/ eps2 + eps2);
f_val = f_val + lambda*sum(xabs);
end

function y = afun(x,flag,A,z0,z,lambda,m)
if strcmp(flag,'notransp') % Compute A*x
    y_1 = A*x /sqrt(z0*m);
    y_2 = sqrt(lambda)*x./sqrt(z);
    y = [y_1 ; y_2];
elseif strcmp(flag,'transp') % Compute A'*x
    x_1 = x(1:m);
    x_2 = x((m+1):length(x));
    y = A'*x_1 /sqrt(z0*m);
    y = y + sqrt(lambda)*x_2./sqrt(z);
end
end