% x0 = [0;0;1;0];
% f = @(x) 0.5*norm(x - x0)^2;
% gradf = @(x) x - x0;
% x = ITEM(gradf,0,1,[1;0;0;0],1000)

SNR_level = 10.^(-9:1);
trials = 100;

m = 300;
n = 1000;
s = 50;

errors = zeros(5,5, iterations);
times = zeros(5,length(SNR_level), trials);

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
        noise = SNR * norm(b) * noise / norm(noise);
        
        b = b + noise;
        
        x0 = randn(n,1);
        Lmax = norm(A,2);

        % % single run methods
        tic
        xr = ITEM_sqrtlasso(A,b,lambda,mu,L,eps1,eps2,x0,100000);
        times(1,snri,t) = toc;
        errors(1,snri,t) = norm(xr - x)/norm(x);
        tic
        xr2 = proximal_gradient(A,b,eps1,lambda,L,x0,100000);
        times(2,snri,t) = toc;
        errors(2,snri,t) = norm(xr2 - x)/norm(x);
        tic
        xr3 = proximal_newton(A,b,eps1,lambda,x0,100,1000);
        times(3,snri,t) = toc;
        errors(3,snri,t) = norm(xr3 - x)/norm(x);
        tic
        xr4 = IRLS(A,b,lambda,eps1,eps2,x0,100,10000);
        times(4,snri,t) = toc;
        errors(4,snri,t) = norm(xr4 - x)/norm(x);
        tic
        xr5 = smooth_concomitant_lasso_v2(A, b, 10^-6, 1000, 10, eps1, lambda, x0);
        times(5,snri,t) = toc;
        errors(5,snri,t) = norm(xr5 - x)/norm(x);      
    end
    
    fprintf('\n')
end

avg_error = mean(errors,3);
avg_time = mean(times,3);

figure
loglog(SNR_level, avg_error(1:5,:))
title('Relative errors')
legend({'ITEM','proximal gradient','proximal newton','IRLS(100)','concomitant lasso'})%,'IRLS(1000)','Half Accelerated IRLS','Half Accelerated IRLS v2'});%,'Half Accelerated IRLS'})

figure
semilogx(SNR_level,avg_time(1:5,:))
title('Runtime')
legend({'ITEM','proximal gradient','proximal newton','IRLS(100)','concomitant lasso'})%,'IRLS(1000)','Half Accelerated IRLS','Half Accelerated IRLS v2'});%,'Half Accelerated IRLS'})
