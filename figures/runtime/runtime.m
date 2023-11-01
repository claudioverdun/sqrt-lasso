addpath('/../../')

rng(123);

SNR_level = 10*(3:5);
trials = 1;

m = 200;
n = 1000;
s = 50;

errors = zeros(6,length(SNR_level), trials);
ratios = zeros(6,length(SNR_level), trials);
times = zeros(6,length(SNR_level), trials);
x_rec = zeros(6,length(SNR_level), trials, n);

lambda_opt = sqrt( log(n)/m);%s * sqrt( log(n)/m);
lambda = lambda_opt/1.5;

eps2 = 10^-8;
eps1 = 10^-8;
eps0 = 10^-3;

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
        sigma = sqrt(norm(b)^2 * 10.0^(-SNR/10.0));
        b = b + sigma*randn(size(b)); 
        
        x0 = randn(n,1);
        Lmax = norm(A,2);

        % % single run methods
        tic
        x_rec(1,snri,t,:) = IRLS_eps_decay(A,b,lambda,eps0,x0,1000,10000,'sqrt',s,'lsqr');
        times(1,snri,t) = toc;
        errors(1,snri,t) = norm(squeeze(x_rec(1,snri,t,:)) - x)/norm(x);
        ratios(1,snri,t) = s_ratio(squeeze(x_rec(1,snri,t,:)),s);
        tic
        x_rec(2,snri,t,:) = IRLS_eps_decay(A,b,lambda,eps0,x0,1000,10000,'sigma',s,'lsqr');
        times(2,snri,t) = toc;
        errors(2,snri,t) = norm(squeeze(x_rec(2,snri,t,:)) - x)/norm(x);
        ratios(2,snri,t) = s_ratio(squeeze(x_rec(2,snri,t,:)),s);
        tic
        x_rec(3,snri,t,:) = IRLS_eps_decay_restart(A,b,lambda,eps0,x0,1000,1000,10,'sqrt',s,'lsqr');
        times(3,snri,t) = toc;
        errors(3,snri,t) = norm(squeeze(x_rec(3,snri,t,:)) - x)/norm(x);
        ratios(3,snri,t) = s_ratio(squeeze(x_rec(3,snri,t,:)),s);
        tic
        x_rec(4,snri,t,:) = IRLS_eps_decay(A,b,lambda,eps0,x0,1000,100,'sqrt',s,'lsqr_fun');
        times(4,snri,t) = toc;
        errors(4,snri,t) = norm(squeeze(x_rec(4,snri,t,:)) - x)/norm(x); 
        ratios(4,snri,t) = s_ratio(squeeze(x_rec(4,snri,t,:)),s);
        tic
        x_rec(5,snri,t,:) = IRLS_eps_decay(A,b,lambda,eps0,x0,1000,100,'sigma',s,'lsqr_fun');
        times(5,snri,t) = toc;
        errors(5,snri,t) = norm(squeeze(x_rec(5,snri,t,:)) - x)/norm(x);
        ratios(5,snri,t) = s_ratio(squeeze(x_rec(5,snri,t,:)),s);
        tic
        x_rec(6,snri,t,:) = IRLS_eps_decay_restart(A,b,lambda,eps0,x0,1000,100,10,'sqrt',s,'lsqr_fun');
        times(6,snri,t) = toc;
        errors(6,snri,t) = norm(squeeze(x_rec(6,snri,t,:)) - x)/norm(x); 
        ratios(6,snri,t) = s_ratio(squeeze(x_rec(6,snri,t,:)),s); 
    end
    
    fprintf('\n')
end

avg_error = mean(errors,3);
avg_time = mean(times,3);
avg_ratios = mean(ratios,3);

figure
semilogy(SNR_level, avg_error(1:6,:))
title('Relative errors')
legend({'ITEM','proximal gradient','proximal newton','concomitant lasso','IRLS(sqrt)','IRLS(sigma)','IRLS(sqrt+restart)'})%,'IRLS(1000)','Half Accelerated IRLS','Half Accelerated IRLS v2'});%,'Half Accelerated IRLS'})

figure
semilogy(SNR_level,avg_time(1:6,:))
title('Runtime')
legend({'ITEM','proximal gradient','proximal newton','concomitant lasso','IRLS(sqrt)','IRLS(sigma)','IRLS(sqrt+restart)'})%,'IRLS(1000)','Half Accelerated IRLS','Half Accelerated IRLS v2'});%,'Half Accelerated IRLS'})

figure
semilogy(SNR_level,avg_ratios(1:6,:))
title('Support recovery')
legend({'ITEM','proximal gradient','proximal newton','concomitant lasso','IRLS(sqrt)','IRLS(sigma)','IRLS(sqrt+restart)'})%,'IRLS(1000)','Half Accelerated IRLS','Half Accelerated IRLS v2'});%,'Half Accelerated IRLS'})


function ratio = s_ratio(x,s)
n = length(x);
x_tr = quantile(abs(x), 1.0*s / n);
idx = abs(x) >= x_tr;
ratio = 1.0*sum(idx(1:s)) / s;
end