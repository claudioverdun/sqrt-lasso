addpath('/../../')

rng(123);

SNR_level = 10*(0:8);
trials = 30;
timeout = 30;
threshold = 10^-5;
threshold_ITEM = 10^-8;

m = 300;
n = 1000;
s = 50;

errors = zeros(7,length(SNR_level), trials);
ratios = zeros(7,length(SNR_level), trials);
ratio2s = zeros(7,length(SNR_level), trials);
times = zeros(7,length(SNR_level), trials);
objective = zeros(7,length(SNR_level), trials);

lambda_opt = sqrt( log(n)/m);%s * sqrt( log(n)/m);
lambda = lambda_opt/1.5;%/10;

eps2 = 10^-8;
eps1 = lambda*10^-8;
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
%         x = x(randperm(n));

        b = A * x;
        sigma = sqrt(norm(b)^2 * 10.0^(-SNR/10.0));
        b = b + sigma*randn(size(b)); 
        
        x0 = randn(n,1);
        Lmax = norm(A,2);

        % % single run methods
        [x_1, t_1] = ITEM_sqrtlasso_to_rie(A,b,lambda,mu,L,eps1,eps2,x0,1000000, x, threshold_ITEM, timeout);
        times(1,snri,t) = t_1;
        errors(1,snri,t) = norm(x_1 - x)/norm(x);
        ratios(1,snri,t) = s_ratio(x_1,s,s);
        ratio2s(1,snri,t) = s_ratio(x_1,s,2*s);
        objective(1,snri,t) = norm(A*x_1 - b)/sqrt(m) + lambda*norm(x_1,1);
        
        [x_2, t_2] = proximal_gradient_to_rie(A,b,0.0,lambda,L,x0,1000000, x, threshold, timeout);
        times(2,snri,t) = t_2;
        errors(2,snri,t) = norm(x_2 - x)/norm(x);
        ratios(2,snri,t) = s_ratio(x_2,s,s);
        ratio2s(2,snri,t) = s_ratio(x_2,s,2*s);
        objective(2,snri,t) = norm(A*x_2 - b)/sqrt(m) + lambda*norm(x_2,1);
        
        [x_3, t_3] = proximal_newton_to_rie(A,b,0.0,lambda,x0,1000,1000, x, threshold, timeout);
        times(3,snri,t) = t_3;
        errors(3,snri,t) = norm(x_3 - x)/norm(x);
        ratios(3,snri,t) = s_ratio(x_3,s,s);
        ratio2s(3,snri,t) = s_ratio(x_3,s,2*s);
        objective(3,snri,t) = norm(A*x_3 - b)/sqrt(m) + lambda*norm(x_3,1);
        
        [x_4, t_4] = smooth_concomitant_lasso_v2_to_rie(A, b, 10000, eps1, lambda, x0, x, threshold, timeout); %10^-16
        times(4,snri,t) = t_4;
        errors(4,snri,t) = norm(x_4 - x)/norm(x); 
        ratios(4,snri,t) = s_ratio(x_4,s,s);
        ratio2s(4,snri,t) = s_ratio(x_4,s,2*s);
        objective(4,snri,t) = norm(A*x_4 - b)/sqrt(m) + lambda*norm(x_4,1);
        
        [x_5, t_5] = IRLS_eps_decay_to_rie(A,b,lambda,eps0,x0,10000,1000,'sqrt',s,'lsqr_fun', x, threshold, timeout);
        times(5,snri,t) = t_5;
        errors(5,snri,t) = norm(x_5 - x)/norm(x);
        ratios(5,snri,t) = s_ratio(x_5,s,s);
        ratio2s(5,snri,t) = s_ratio(x_5,s,2*s);
        objective(5,snri,t) = norm(A*x_5 - b)/sqrt(m) + lambda*norm(x_5,1);
        
        [x_6, t_6] = IRLS_eps_decay_to_rie(A,b,lambda,0.5,x0,10000,1000,'sigma',s,'lsqr_fun', x, threshold, timeout);
        times(6,snri,t) = t_6;
        errors(6,snri,t) = norm(x_6 - x)/norm(x); 
        ratios(6,snri,t) = s_ratio(x_6,s,s);
        ratio2s(6,snri,t) = s_ratio(x_6,s,2*s);
        objective(6,snri,t) = norm(A*x_6 - b)/sqrt(m) + lambda*norm(x_6,1);
        
        [x_7, t_7] = IRLS_eps_decay_restart_to_rie(A,b,lambda,eps0,x0,1000,1000,'sqrt',s,'lsqr_fun', x, threshold, timeout);
        times(7,snri,t) = t_7;
        errors(7,snri,t) = norm(x_7 - x)/norm(x);
        ratios(7,snri,t) = s_ratio(x_7,s,s);
        ratio2s(7,snri,t) = s_ratio(x_7,s,2*s);
        objective(7,snri,t) = norm(A*x_7 - b)/sqrt(m) + lambda*norm(x_7,1);
    end
    
    fprintf('\n')
end

avg_error = mean(errors,3);
avg_time = mean(times,3);
avg_ratios = mean(ratios,3);
avg_ratio2s = mean(ratio2s,3);


med_time = median(times,3);

% avg_error(3,:) = avg_error(3,:)*trials / 5;
% avg_time(3,:) = avg_time(3,:)*trials / 5;
% avg_ratios(3,:) = avg_ratios(3,:)*trials / 5;
% avg_ratio2s(3,:) = avg_ratio2s(3,:)*trials / 5;

avg_ratios = 1 - avg_ratios;
avg_ratio2s = 1 - avg_ratio2s;

figure
semilogy(SNR_level, avg_error(1:7,:))
title('Relative errors')
legend({'ITEM','proximal gradient','proximal newton','concomitant lasso','IRLS(sqrt)','IRLS(sigma)','IRLS(sqrt+restart)'})%,'IRLS(1000)','Half Accelerated IRLS','Half Accelerated IRLS v2'});%,'Half Accelerated IRLS'})

figure
semilogy(SNR_level, avg_time(1:7,:))
title('Runtime')
legend({'ITEM','proximal gradient','proximal newton','concomitant lasso','IRLS(sqrt)','IRLS(sigma)','IRLS(sqrt+restart)'})%,'IRLS(1000)','Half Accelerated IRLS','Half Accelerated IRLS v2'});%,'Half Accelerated IRLS'})

figure
semilogy(SNR_level, med_time(1:7,:))
title('Runtime (med)')
legend({'ITEM','proximal gradient','proximal newton','concomitant lasso','IRLS(sqrt)','IRLS(sigma)','IRLS(sqrt+restart)'})%,'IRLS(1000)','Half Accelerated IRLS','Half Accelerated IRLS v2'});%,'Half Accelerated IRLS'})


figure
plot(SNR_level,avg_ratios(1:7,:))
title('Missing support percentage')
legend({'ITEM','proximal gradient','proximal newton','concomitant lasso','IRLS(sqrt)','IRLS(sigma)','IRLS(sqrt+restart)'})%,'IRLS(1000)','Half Accelerated IRLS','Half Accelerated IRLS v2'});%,'Half Accelerated IRLS'})

figure
plot(SNR_level,avg_ratio2s(1:7,:))
title('Missing support percentage (2s)')
legend({'ITEM','proximal gradient','proximal newton','concomitant lasso','IRLS(sqrt)','IRLS(sigma)','IRLS(sqrt+restart)'})%,'IRLS(1000)','Half Accelerated IRLS','Half Accelerated IRLS v2'});%,'Half Accelerated IRLS'})

labels = {'snr','ITEM', 'prox_grad',...
          'prox_newton', ...
          'concomitant', ...
          'irls_sqrt', ...
          'irls_sigma', ...
          'irls_sqrt_restart'};
results_err = [SNR_level.' avg_error.'];
results_err = array2table(results_err,'VariableNames',labels);
writetable(results_err,'noise_stab_1_err.csv','Delimiter',' ')

results_time = [SNR_level.' avg_time.'];
results_time = array2table(results_time,'VariableNames',labels);
writetable(results_time,'noise_stab_1_time.csv','Delimiter',' ')

results_ratio = [SNR_level.' avg_ratios.'];
results_ratio = array2table(results_ratio,'VariableNames',labels);
writetable(results_ratio,'noise_stab_1_ratio.csv','Delimiter',' ')

results_ratio = [SNR_level.' avg_ratio2s.'];
results_ratio = array2table(results_ratio,'VariableNames',labels);
writetable(results_ratio,'noise_stab_1_ratio2s.csv','Delimiter',' ')

