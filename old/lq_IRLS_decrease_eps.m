function [x,eps1_track,eps2_track] = IRLS(A,b,s,lambda,q,eps1,eps2,x0,N,Nlsp,epsmin)
n = size(A,2);
m = length(b);
x = x0;
threshold = 10^-14;

eps1_track={};
eps2_track={};

for it=1:N
    
     x_old = x;
     
     %   epsilon decay rule for ||x||_p^p
        sX_c=sort(abs(x),'descend');
        sX_c_complement=sX_c((s+1):end);
%     eps_rule = norm(sX_c_complement,q)/(n-s)^(1);
        eps_rule = norm(sX_c_complement,1)/n;
%      eps_rule = eps2*10^(-(it/N)^(2-q)*10);
%     eps2 = max(min(eps2,eps_rule),epsmin);
        if norm(x_old-x)/norm(x) < sqrt(eps2)/100
            eps2 = eps2/10;
            eps2 = max(eps2,epsmin)
        eps2_track{it}=eps2;
        end
    %   epsilon decay rule for ||Ax-b||_2^p
%     sX_c=sort(abs(x),'descend');
%     sX_c_complement=sX_c((s+1):end);
%     eps_rule = norm(sX_c_complement,q)/(n-s)^(1);
        eps_rule = norm(A*x-b,1)/n;
%      eps_rule = eps2*10^(-(it/N)^(2-q)*10);
        eps1 = max(min(eps1,eps_rule),epsmin);
        eps1_track{it}=eps1;
    
%   write the correct variables for least-square   
%     z0 = max(sqrt(norm(A*x-b)^2/m),eps1);
%     z = max(sqrt(abs(lambda*x).^2),eps2);
    z0 = (2 - q)^2 * (norm(A*x-b)^2/m + eps1)^((2-q)/2) / q^2;
    z = (2 - q)^2 *(abs(lambda*x).^2 + eps2).^((2-q)/2) / q^2;
    A_expanded = [A/sqrt(2*z0*m);diag(lambda*sqrt(1./(2*z)))];
    b_expanded = [b/sqrt(2*z0*m);zeros(n,1)];
%    Check the constants
%     z0 = (2 - q)^2 * (norm(A*x-b)^2/m + eps1)^((2-q)/2) / q^2;
%     z = (2 - q)^2 *(abs(lambda*x).^2 + eps2).^((2-q)/2) / q^2;
%     A_expanded = [A/sqrt(2*z0*m);diag(lambda*sqrt(1./(2*z)))];
%     b_expanded = [b/sqrt(2*z0*m);zeros(n,1)];
    
%   least-squares solution via LSQR
    [x, flag] = lsqr(A_expanded,b_expanded, [], Nlsp,[],[],x);
    

        


    if flag ==  1
        fprintf('lsqr did not converged');
        return;
    end
%     x = A_expanded \ b_expanded;

    

    
    if norm(x_old-x) < threshold
        fprintf('IRLS converged');
        break;
    end
end
end