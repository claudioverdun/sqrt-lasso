% x0 = [0;0;1;0];
% f = @(x) 0.5*norm(x - x0)^2;
% gradf = @(x) x - x0;
% x = ITEM(gradf,0,1,[1;0;0;0],1000)

factors = 1:2:2^4;
trials = 5;

m_init = 25;
n_init = 100;
s_init = 5;

errors = zeros(6,length(factors), trials);
times = zeros(6,length(factors), trials);

for fi = 1:length(factors)
    factor = factors(fi);
    m = m_init * factor;
    n = n_init * factor;
    s = s_init * factor;
    fprintf('Factor %d:',factor)
    
    for t=1:trials
        fprintf('*')
        A = randn(m,n);

        x = [ones(s,1) ; zeros(n-s,1)];
        b = A * x;

    %     lambda = 10^-0.5;
    %     lambda_opt = sqrt(m)*norminv(1 - 0.05/s);
        lambda_opt = sqrt( log(n)/m);%s * sqrt( log(n)/m);
        lambda = lambda_opt/1.5;
        eps1 = 10^-10;
        K0 = 100 * sqrt(eps1);
        eps2 = 10^-10;
        x0 = randn(n,1);

        mu = 0;
        L = lambda/ sqrt(eps2) + 1.0/sqrt(eps1);
        Lmax = norm(A,2);

        % % single run methods
        tic
        xr = ITEM_sqrtlasso(A,b,lambda,mu,L,eps1,1000*K0,x0,10^-4);
        times(1,fi,t) = toc;
        errors(1,fi,t) = norm(xr - x)/norm(x);
        tic
        xr2 = proximal_gradient(A,b,eps1,100*K0,lambda,L,x0,10^-4);
        times(2,fi,t) = toc;
        errors(2,fi,t) = norm(xr2 - x)/norm(x);
        tic
        xr3 = proximal_newton(A,b,eps1,K0,lambda,x0,10000,10^-4);
        times(3,fi,t) = toc;
        errors(3,fi,t) = norm(xr3 - x)/norm(x);
        tic
        xr4 = IRLS(A,b,lambda,eps1,K0,x0,10000,10^-4);
        times(4,fi,t) = toc;
        errors(4,fi,t) = norm(xr4 - x)/norm(x);
        tic
        xr5 = smooth_concomitant_lasso_v2(A, b, 10^-6, 1000, eps1, lambda, x0,10^-4);
        times(5,fi,t) = toc;
        errors(5,fi,t) = norm(xr5 - x)/norm(x);
        tic
        xr6 = IRLS_thresholding(A,b,lambda,eps1,K0,x0,10000,10^-4);
        times(6,fi,t) = toc;
        errors(6,fi,t) = norm(xr6 - x)/norm(x);
        
%         tic
%         xr6 = IRLS(A,b,lambda,eps1,eps2,xr4,900,10000);
%         times(6,fi,t) = toc + times(4,fi,t);
%         errors(6,fi,t) = norm(xr6 - x)/norm(x);
        % pipeline
%         tic
%         xr6 = IRLS(A,b,lambda,eps1,eps2,xr,100,10000);
%         times(6,fi,t) = toc + times(1,fi,t);
%         errors(6,fi,t) = norm(xr6 - x)/norm(x);
%         tic
%         xr7 = Accelerated_IRLS_v2(A,b,lambda,eps1,eps2,x0,100,10000);
%         times(7,fi,t) = toc;
%         errors(7,fi,t) = norm(xr7 - x)/norm(x);
%         tic
%         xr8 = Accelerated_IRLS_v3(A,b,lambda,eps1,eps2,x0,100,10000);
%         times(8,fi,t) = toc;
%         errors(8,fi,t) = norm(xr8 - x)/norm(x);
%         lambda_max = 1000000;
%         % pathwise methods
%         tic
%         xr5 = smooth_concomitant_lasso_v2(A, b, 10^-6, 10000, 10, 10, eps1, lambda_max);
%         times(5,fi,t) = toc;
%         errors(5,fi,t) = norm(xr5 - x)/norm(x);
%         tic
%         xr6 = pathwise_proximal_gradient(A,b,eps1,lambda_max,L,100000,10);
%         times(6,fi,t) = toc;
%         errors(6,fi,t) = norm(xr6 - x)/norm(x);
%         tic
%         xr7 = pathwise_proximal_newton(A,b,eps1,lambda_max,100,10000,10);
%         times(7,fi,t) = toc;
%         errors(7,fi,t) = norm(xr7 - x)/norm(x);
%         tic
%         xr8 = pathwise_IRLS(A,b,eps1,eps2,lambda_max,100,10000,10);
%         times(8,fi,t) = toc;
%         errors(8,fi,t) = norm(xr8 - x)/norm(x);
        
        
    end
    
    fprintf('\n')
end

avg_error = median(errors,3);%mean(errors,3);
avg_time = median(times,3);%mean(times,3);

figure
semilogy(factors, avg_error(1:6,:))
semilogy(factors, avg_error(1:6,:))
title('Relative errors')
legend({'ITEM','proximal gradient','proximal newton','IRLS','concomitant lasso','IRLS + thresholding'})%,'IRLS(1000)','Half Accelerated IRLS','Half Accelerated IRLS v2'});%,'Half Accelerated IRLS'})

% figure
% semilogy(factors, avg_error(5:8,:))
% title('Relative errors')
% legend({'concomitant lasso','pathwise proximal gradient','pathwise proximal newton','pathwise IRLS'})

figure
plot(factors,avg_time(1:6,:))
title('Runtime')
legend({'ITEM','proximal gradient','proximal newton','IRLS(100)','concomitant lasso','IRLS + thresholding'})%,'IRLS(1000)','Half Accelerated IRLS','Half Accelerated IRLS v2'});%,'Half Accelerated IRLS'})

% figure
% plot(factors, avg_time(5:8,:))
% title('Runtime')
% legend({'concomitant lasso','pathwise proximal gradient','pathwise proximal newton','pathwise IRLS'})