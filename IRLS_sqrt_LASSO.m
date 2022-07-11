function [x,x_track,eps1_track,eps2_track] = IRLS_sqrt_LASSO(A,b,s,lambda,eps1,eps2,x0,N,Nlsp,epsmin,eps_mode1,eps_mode2)

n = size(A,2);
m = length(b);
x = x0;
threshold = 10^-6;

eps1_track = {};
eps2_track = {}; 
x_track = {};
z0_track = {};
z_track = {};
A_expanded_track = {};
b_expanded = {};



for it=1:N
    
    x_old = x;
     
    %epsilon decay rule for ||x||_1
    switch eps_mode2
        case 'KMS'
            sX_c=sort(abs(x),'descend');
            sX_c_complement=sX_c((s+1):end);
            eps_rule2 = norm(sX_c_complement,1)/n;
            fprintf('Epsilon 2 is %.16f \n', eps_rule2)  
        case 'DDFG'
            sX_c=sort(abs(x),'descend');
            sX_c_complement=sX_c((s+1):end);
            eps_rule2 = max(sX_c_complement)/n;
            
        case 'automatic'
            if norm(x_old-x)/norm(x) < sqrt(eps2)/100
                eps2 = eps2/10; 
            end
            
    end
     
    eps2 = max(min(eps2,eps_rule2),epsmin);  
    eps2_track{it} = eps2;
     
    %epsilon decay rule for ||Ax-b||_2
    % THERE IS A PROBLEM HERE. THIS EPSILON DECAY STILL DOES NOT MAKE SENSE
    switch eps_mode1
        case 'KMS'
            eps_rule1 = norm(A*x-b,2);
            fprintf('Epsilon 1 is %f \n', eps_rule1) 
        case 'DDFG'
            eps_rule1 = max(A*x-b)/n;            
        case 'automatic'
            if norm(A*x-b) < sqrt(eps1)/100
                eps1 = eps1/10; 
                fprintf('Epsilon 1 is %f \n', eps1)
            end  
            eps1 = eps1/2; 
            eps1 = max(eps1,epsmin);
            fprintf('Epsilon 1 is %.16f \n', eps1)
    end
     
%     eps1 = max(min(eps1,eps_rule1),epsmin);
    eps1_track{it} = eps1;

% %   write the correct variables for least-square   
%     z0 = max(sqrt(norm(A*x-b)^2/m),eps1);
%     z = max(sqrt(abs(lambda*x).^2),eps2);

    z0 = max(norm(A*x-b,2), eps1);
    z = max(abs(x), eps2);
    
     fprintf('z0 is %f \n', z0)
%     fprintf('z LSQR %f \n', flag)
       
    %z = (abs(lambda*x).^2 + eps2);

    %z0 = (2 - q)^2 * (norm(A*x-b)^2/m + eps1)^((2-q)/2) / q^2;
    %z = (2 - q)^2 *(abs(lambda*x).^2 + eps2).^((2-q)/2) / q^2;
    
    A_expanded = [A/(m^(1/4)*sqrt(z0));diag(sqrt(lambda./(z)))];
    b_expanded = [b/(m^(1/4)*sqrt(z0));zeros(n,1)];
    
%     dim1 = size(A_expanded,1)
%     dim2 = size(A_expanded,2)
    
    z0_track{it} = z0;
    z_track{it} = z;
    A_expanded_track{it} = norm(A_expanded);
    
    b_expanded_track{it} = b_expanded;

%   least-squares solution via LSQR
    [x, flag] = lsqr(A_expanded,b_expanded, [], Nlsp,[],[]);
%     [x_cg, flag] = pcg(A_expanded,b_expanded, [], Nlsp,[],[]);
    BLA=A_expanded*x-b_expanded;
    EQNORMAL_RHS= norm(A_expanded.'*b_expanded);
    fprintf('Norm of A^T*b %f \n', EQNORMAL_RHS)
    fprintf('flag LSQR %f \n', flag)
    fprintf('Norm of x %f \n', norm(x))
    fprintf('Norm of b %f \n', norm(b_expanded))
    fprintf('Norm of Ax-b %f \n', norm(BLA))
%     disp('lsqr did not converged 'num2str(norm(x)))
%     norm(x_cg)
    fprintf('\n')
    x_track{it} = x;
       
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