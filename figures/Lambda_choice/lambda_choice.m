addpath('/../../')

rng(123);

Lmult = (-5:0.5:1);
trials = 30;

m = 300;
n = 1000;
s = 50;

errors = zeros(7,length(Lmult), trials);
ratios = zeros(7,length(Lmult), trials);
ratio2s = zeros(7,length(Lmult), trials);
times = zeros(7,length(Lmult), trials);
objective = zeros(7,length(Lmult), trials);

lambda_opt = sqrt(log(n)/m);%s * sqrt( log(n)/m);

eps2 = 10^-8;

eps0 = 10^-3;

mu = 0;


for mi = 1:length(Lmult)
%     SNR = SNR_level(snri);
    mult = Lmult(mi);
    fprintf('Multiplier %d:',mult);
    
    lambda = 10^mult *lambda_opt/1.5;
    eps1 = lambda*10^-8;
    L = lambda/ sqrt(eps2) + 1.0/sqrt(eps1);

    for t=1:trials
        fprintf('*')
        A = randn(m,n);

        x = [ones(s,1) ; zeros(n-s,1)];
        b = A * x;
%         sigma = sqrt(norm(b)^2 * 10.0^(-SNR/10.0));
%         b = b + sigma*randn(size(b)); 
        
        x0 = randn(n,1);
        Lmax = norm(A,2);

        % % single run methods
        tic
        x_1 = ITEM_sqrtlasso(A,b,lambda,mu,L,eps1,eps2,x0,100000);
        times(1,mi,t) = toc;
        errors(1,mi,t) = norm(x_1 - x)/norm(x);
        ratios(1,mi,t) = s_ratio(x_1,s,s);
        ratio2s(1,mi,t) = s_ratio(x_1,s,2*s);
        objective(1,mi,t) = norm(A*x_1 - b)/sqrt(m) + lambda*norm(x_1,1);
        tic
        x_2 = proximal_gradient(A,b,0.0,lambda,L,x0,100000);
        times(2,mi,t) = toc;
        errors(2,mi,t) = norm(x_2 - x)/norm(x);
        ratios(2,mi,t) = s_ratio(x_2,s,s);
        ratio2s(2,mi,t) = s_ratio(x_2,s,2*s);
        objective(2,mi,t) = norm(A*x_2 - b)/sqrt(m) + lambda*norm(x_2,1);
        tic
        if t <= 5
            x_3 = proximal_newton(A,b,0.0,lambda,x0,100,1000);
            times(3,mi,t) = toc;
            errors(3,mi,t) = norm(x_3 - x)/norm(x);
            ratios(3,mi,t) = s_ratio(x_3,s,s);
            ratio2s(3,mi,t) = s_ratio(x_3,s,2*s);
            objective(3,mi,t) = norm(A*x_3 - b)/sqrt(m) + lambda*norm(x_3,1);
        end
        tic
        x_4 = smooth_concomitant_lasso_v2(A, b, 10^-6, 1000, 10, eps1, lambda, x0); %10^-16
        times(4,mi,t) = toc;
        errors(4,mi,t) = norm(x_4 - x)/norm(x); 
        ratios(4,mi,t) = s_ratio(x_4,s,s);
        ratio2s(4,mi,t) = s_ratio(x_4,s,2*s);
        objective(4,mi,t) = norm(A*x_4 - b)/sqrt(m) + lambda*norm(x_4,1);
        tic
        x_5 = IRLS_eps_decay(A,b,lambda,eps0,x0,100,1000,'sqrt',s,'lsqr_fun');
        times(5,mi,t) = toc;
        errors(5,mi,t) = norm(x_5 - x)/norm(x);
        ratios(5,mi,t) = s_ratio(x_5,s,s);
        ratio2s(5,mi,t) = s_ratio(x_5,s,2*s);
        objective(5,mi,t) = norm(A*x_5 - b)/sqrt(m) + lambda*norm(x_5,1);
        tic
        x_6 = IRLS_eps_decay(A,b,lambda,eps0,x0,100,1000,'sigma',s,'lsqr_fun');
        times(6,mi,t) = toc;
        errors(6,mi,t) = norm(x_6 - x)/norm(x); 
        ratios(6,mi,t) = s_ratio(x_6,s,s);
        ratio2s(6,mi,t) = s_ratio(x_6,s,2*s);
        objective(6,mi,t) = norm(A*x_6 - b)/sqrt(m) + lambda*norm(x_6,1);
        tic
        x_7 = IRLS_eps_decay_restart(A,b,lambda,eps0,x0,100,1000,10,'sqrt',s,'lsqr_fun');
        times(7,mi,t) = toc;
        errors(7,mi,t) = norm(x_7 - x)/norm(x);
        ratios(7,mi,t) = s_ratio(x_7,s,s);
        ratio2s(7,mi,t) = s_ratio(x_7,s,2*s);
        objective(7,mi,t) = norm(A*x_7 - b)/sqrt(m) + lambda*norm(x_7,1);
    end
    
    fprintf('\n')
end

avg_error = mean(errors,3);
avg_time = mean(times,3);
avg_ratios = mean(ratios,3);
avg_ratio2s = mean(ratio2s,3);

avg_error(3,:) = avg_error(3,:)*trials / 5;
avg_time(3,:) = avg_time(3,:)*trials / 5;
avg_ratios(3,:) = avg_ratios(3,:)*trials / 5;
avg_ratio2s(3,:) = avg_ratio2s(3,:)*trials / 5;

avg_ratios = 1 - avg_ratios;
avg_ratio2s = 1 - avg_ratio2s;

figure
loglog(10.^Lmult*lambda_opt/1.5, avg_error(1:7,:))
title('Relative errors')
legend({'ITEM','proximal gradient','proximal newton','concomitant lasso','IRLS(sqrt)','IRLS(sigma)','IRLS(sqrt+restart)'})%,'IRLS(1000)','Half Accelerated IRLS','Half Accelerated IRLS v2'});%,'Half Accelerated IRLS'})

figure
loglog(10.^Lmult*lambda_opt/1.5, avg_time(1:7,:))
title('Runtime')
legend({'ITEM','proximal gradient','proximal newton','concomitant lasso','IRLS(sqrt)','IRLS(sigma)','IRLS(sqrt+restart)'})%,'IRLS(1000)','Half Accelerated IRLS','Half Accelerated IRLS v2'});%,'Half Accelerated IRLS'})

figure
semilogx(10.^Lmult*lambda_opt/1.5,avg_ratios(1:7,:))
title('Missing support percentage')
legend({'ITEM','proximal gradient','proximal newton','concomitant lasso','IRLS(sqrt)','IRLS(sigma)','IRLS(sqrt+restart)'})%,'IRLS(1000)','Half Accelerated IRLS','Half Accelerated IRLS v2'});%,'Half Accelerated IRLS'})

figure
semilogx(10.^Lmult*lambda_opt/1.5,avg_ratio2s(1:7,:))
title('Missing support percentage (2s)')
legend({'ITEM','proximal gradient','proximal newton','concomitant lasso','IRLS(sqrt)','IRLS(sigma)','IRLS(sqrt+restart)'})%,'IRLS(1000)','Half Accelerated IRLS','Half Accelerated IRLS v2'});%,'Half Accelerated IRLS'})

labels = {'mult','ITEM', 'prox_grad',...
          'prox_newton', ...
          'concomitant', ...
          'irls_sqrt', ...
          'irls_sigma', ...
          'irls_sqrt_restart'};
results_err = [Lmult.' avg_error.'];
results_err = array2table(results_err,'VariableNames',labels);
writetable(results_err,'lambda_err.csv','Delimiter',' ')

results_time = [Lmult.' avg_time.'];
results_time = array2table(results_time,'VariableNames',labels);
writetable(results_time,'lambda_time.csv','Delimiter',' ')

results_ratio = [Lmult.' avg_ratios.'];
results_ratio = array2table(results_ratio,'VariableNames',labels);
writetable(results_ratio,'lambda_ratio.csv','Delimiter',' ')

results_ratio = [Lmult.' avg_ratio2s.'];
results_ratio = array2table(results_ratio,'VariableNames',labels);
writetable(results_ratio,'lambda_ratio2s.csv','Delimiter',' ')

