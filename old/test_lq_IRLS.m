% x0 = [0;0;1;0];
% f = @(x) 0.5*norm(x - x0)^2;
% gradf = @(x) x - x0;
% x = ITEM(gradf,0,1,[1;0;0;0],1000)

SNR_level = 10:10:90;
trials = 5;

m = 400;
n = 1000;
s = 5;

errors = zeros(7,length(SNR_level), trials);
times = zeros(7,length(SNR_level), trials);


lambda_opt = sqrt( log(n)/m);%s * sqrt( log(n)/m);
lambda = lambda_opt/1.5;
eps1 = 10^-8;
eps2 = 10^-8;

mu = 0;
L = lambda/ sqrt(eps2) + 1.0/sqrt(eps1);

for snri = 1:length(SNR_level)
    SNR = SNR_level(snri);
    fprintf('SNR %d:',SNR)
    
    for t=1:trials
        fprintf('*')
        A = randn(m,n);

        x = [ones(s,1) ; zeros(n-s,1)];
        b = A * x;
        noise = randn(m,1);
        scale = sqrt(norm(b)^2 / 10^(SNR/10));
        noise = scale * noise / norm(noise);
        
        b = b + noise;
        
        x0 = randn(n,1);
        Lmax = norm(A,2);

        % % single run methods
        tic
        xr = smooth_concomitant_lasso_v2(A, b, 10^-6, 1000, 10, eps1, lambda, x0);
        times(1,snri,t) = toc;
        errors(1,snri,t) = norm(xr - x)/norm(x);
        tic
        xr2 = IRLS(A,b,lambda,eps1,eps2,x0,100,10000);
        times(2,snri,t) = toc;
        errors(2,snri,t) = norm(xr2 - x)/norm(x);
        tic
        xr3 = lq_IRLS(A,b,lambda^0.8,0.8,eps1,eps2,xr2,1000,10000);
        times(3,snri,t) = toc;
        times(3,snri,t) = times(3, snri,t) + times(2, snri,t);
        errors(3,snri,t) = norm(xr3 - x)/norm(x);
        tic
        xr4 = lq_IRLS(A,b,lambda^0.6,0.6,eps1,eps2,xr2,100,10000);
        times(4,snri,t) = toc;
        times(4, snri,t) = times(4, snri,t) + times(2, snri,t); 
        errors(4,snri,t) = norm(xr4 - x)/norm(x);
        tic
        xr5 = lq_IRLS(A,b,lambda^0.4,0.4,eps1,eps2,xr2,100,10000);
        times(5,snri,t) = toc;
        times(5, snri,t) = times(5, snri,t) + times(2, snri,t); 
        errors(5,snri,t) = norm(xr5 - x)/norm(x);
        tic
        
        xr6 = lq_IRLS(A,b,lambda^0.2,0.2,eps1,eps2,xr2,100,10000);
        times(6,snri,t) = toc;
        times(6, snri,t) = times(6, snri,t) + times(2, snri,t); 
        errors(6,snri,t) = norm(xr6 - x)/norm(x);
        tic
        
        xr7 = lq_IRLS(A,b,lambda^0.1,0.1,eps1,eps2,xr2,100,10000);
        times(7,snri,t) = toc;
        times(7, snri,t) = times(7, snri,t) + times(2, snri,t); 
        errors(7,snri,t) = norm(xr7 - x)/norm(x);
        tic
        
%         xr5 = smooth_concomitant_lasso_v2(A, b, 10^-6, 1000, 10, eps1, lambda, x0);
%         times(5,snri,t) = toc;
%         errors(5,snri,t) = norm(xr5 - x)/norm(x);      
    end
    
    fprintf('\n')
end

avg_error = median(errors,3);
avg_time = median(times,3);

figure
semilogy(SNR_level, avg_error(1:7,:))
title('Relative errors')
legend({'concomitant lasso','l1 IRLS','l0.8 IRLS','l0.6 IRLS','l0.4 IRLS','l0.2 IRLS','l0.1 IRLS'})%,'IRLS(1000)','Half Accelerated IRLS','Half Accelerated IRLS v2'});%,'Half Accelerated IRLS'})

figure
plot(SNR_level,avg_time(1:7,:))
title('Runtime')
legend({'concomitant lasso','l1 IRLS','l0.8 IRLS','l0.6 IRLS','l0.4 IRLS','l0.2 IRLS','l0.1 IRLS'})