% x0 = [0;0;1;0];
% f = @(x) 0.5*norm(x - x0)^2;
% gradf = @(x) x - x0;
% x = ITEM(gradf,0,1,[1;0;0;0],1000)

SNR_level = 10:10:100;
trials = 5;

m = 100;
n = 256;
s = 5;

norms=3;
errors = zeros(7,length(SNR_level), trials, norms);
times = zeros(7,length(SNR_level), trials);

lambda_opt=sqrt(log(n)/m);
% lambda_opt = 2*sqrt(2)*s*sqrt( log(n)/m);
% lambda_opt_slow = 2*sqrt(2)*s* log(n)/m %sqrt( log(n)/m);%s * sqrt( log(n)/m);
% lambda_opt_fast = 2*sqrt(2)*s*sqrt( log(n)/m)

% What is there a division by 1.5?
lambda = lambda_opt/1.5;
eps1 = 1;
eps2 = 1;

% eps1 = 10^-4;
% eps2 = 10^-4;

epsmin=10^-8;

mu = 0;
% what is L? is it for ITEM and proximal?
L = lambda/ sqrt(eps2) + 1.0/sqrt(eps1);

for snri = 1:length(SNR_level)
    SNR = SNR_level(snri);
    fprintf('SNR %d:',SNR)
    
    for t=1:trials
        fprintf('*')
        A = randn(m,n);

        x = [ones(s,1) ; zeros(n-s,1)];
        x = x (randperm(length(x)));
        b = A * x;
        noise = randn(m,1);
        scale = sqrt(norm(b)^2 / 10^(SNR/10));
        noise = scale * noise / norm(noise);
        
        b = b + noise;
        
        x0 = randn(n,1);
        Lmax = norm(A,2);

        
%% single run methods
        
        tic
%       xr = smooth_concomitant_lasso_v2(A, b, 10^-6, 1000, 10, eps1, lambda, x0);
%       times(1,snri,t) = toc;
%       errors(1,snri,t) = norm(xr - x)/norm(x);
%       tic
% 
%         xr2 = IRLS_v2(A,b,lambda,eps1,eps2,x0,1000,10000);
%         times(2,snri,t) = toc;
%         errors(2,snri,t,1) = norm(xr2 - x)/norm(x);
%         errors(2,snri,t,2) = norm(xr2 - x,1)/norm(x,1);
%         errors(2,snri,t,3) = norm(xr2 - x,0.5)/norm(x,0.5);
%         tic
        
        [xr2,eps_track] = IRLS(A,b,s,lambda,eps1,eps2,x0,2000,10000,epsmin);
        times(2,snri,t) = toc;
        errors(2,snri,t,1) = norm(xr2 - x)/norm(x);
        errors(2,snri,t,2) = norm(xr2 - x,1)/norm(x,1);
        errors(2,snri,t,3) = norm(xr2 - x,0.5)/norm(x,0.5);
        tic
        
        % function x = IRLS_v2(A,b,s,lambda,eps1,eps2,x0,N,Nlsp,epsmin)
        [xr3,eps1_track3,eps2_track3] = lq_IRLS(A,b,s,lambda^0.8,0.8,eps1,eps2,xr2,2000,10000,epsmin);
        times(3,snri,t) = toc;
%        Why do you need to add the time? 
        times(3,snri,t) = times(3, snri,t) + times(2, snri,t);
        errors(3,snri,t,1) = norm(xr3 - x)/norm(x);
        errors(3,snri,t,2) = norm(xr3 - x,1)/norm(x,1);
        errors(3,snri,t,3) = norm(xr3 - x,0.5)/norm(x,0.5);
        tic
        [xr4,eps1_track4,eps2_track4]  = lq_IRLS(A,b,s,lambda^0.6,0.6,eps1,eps2,xr2,2000,10000,epsmin);
        times(4,snri,t) = toc;
        times(4, snri,t) = times(4, snri,t) + times(2, snri,t); 
        errors(4,snri,t,1) = norm(xr4 - x)/norm(x);
        errors(4,snri,t,2) = norm(xr4 - x,1)/norm(x,1);
        errors(4,snri,t,3) = norm(xr4 - x,0.5)/norm(x,0.5);
        tic
        [xr5,eps1_track5,eps2_track5]  = lq_IRLS(A,b,s,lambda^0.4,0.4,eps1,eps2,xr2,2000,10000,epsmin);
        times(5,snri,t) = toc;
        times(5, snri,t) = times(5, snri,t) + times(2, snri,t); 
        errors(5,snri,t,1) = norm(xr5 - x)/norm(x);
        errors(5,snri,t,2) = norm(xr5 - x,1)/norm(x,1);
        errors(5,snri,t,3) = norm(xr5 - x,0.5)/norm(x,0.5);
        tic   
        [xr6,eps1_track6,eps2_track6]  = lq_IRLS(A,b,s,lambda^0.2,0.2,eps1,eps2,xr2,2000,10000,epsmin);
        times(6,snri,t)  = toc;
        times(6, snri,t) = times(6, snri,t) + times(2, snri,t); 
        errors(6,snri,t,1) = norm(xr6 - x)/norm(x);
        errors(6,snri,t,2) = norm(xr6 - x,1)/norm(x,1);
        errors(6,snri,t,3) = norm(xr6 - x,0.5)/norm(x,0.5);
        tic        
        [xr7,eps1_track7,eps2_track7]  = lq_IRLS(A,b,s,lambda^0.1,0.1,eps1,eps2,xr2,2000,10000,epsmin);
        times(7,snri,t) = toc;
        times(7, snri,t) = times(7, snri,t) + times(2, snri,t); 
        errors(7,snri,t,1) = norm(xr7 - x)/norm(x);
        errors(7,snri,t,2) = norm(xr7 - x,1)/norm(x,1);
        errors(7,snri,t,3) = norm(xr7 - x,0.5)/norm(x,0.5);
        toc
        
%         xr5 = smooth_concomitant_lasso_v2(A, b, 10^-6, 1000, 10, eps1, lambda, x0);
%         times(5,snri,t) = toc;
%         errors(5,snri,t) = norm(xr5 - x)/norm(x);      
    end
    
    fprintf('\n')
end

avg_error = median(errors,3);
avg_time = median(times,3);

avg_error = median(errors,3);
avg_time = median(times,3);

errors = zeros(7,length(SNR_level), trials, norms);

figure
semilogy(SNR_level, avg_error(2:7,:,1))
title('Relative errors in the 2-norm')
legend({'l1 IRLS','l0.8 IRLS','l0.6 IRLS','l0.4 IRLS','l0.2 IRLS','l0.1 IRLS'})%,'IRLS(1000)','Half Accelerated IRLS','Half Accelerated IRLS v2'});%,'Half Accelerated IRLS'})
% 'concomitant lasso',
figure
semilogy(SNR_level, avg_error(2:7,:,2))
title('Relative errors in the 1-norm')
legend({'l1 IRLS','l0.8 IRLS','l0.6 IRLS','l0.4 IRLS','l0.2 IRLS','l0.1 IRLS'})%,'IRLS(1000)','Half Accelerated IRLS','Half Accelerated IRLS v2'});%,'Half Accelerated IRLS'})
% 'concomitant lasso',
figure
semilogy(SNR_level, avg_error(2:7,:,3))
title('Relative errors in the 0.5-norm')
legend({'l1 IRLS','l0.8 IRLS','l0.6 IRLS','l0.4 IRLS','l0.2 IRLS','l0.1 IRLS'})%,'IRLS(1000)','Half Accelerated IRLS','Half Accelerated IRLS v2'});%,'Half Accelerated IRLS'})
% 'concomitant lasso',
figure
plot(SNR_level,avg_time(1:7,:))
title('Runtime')
legend({'l1 IRLS','l0.8 IRLS','l0.6 IRLS','l0.4 IRLS','l0.2 IRLS','l0.1 IRLS'})