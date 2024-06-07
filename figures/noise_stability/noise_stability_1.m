%% This script generates data for the lower half of Figure 2 in 
% [1] Claudio Mayrink Verdun, Oleh Melnyk, Felix Krahmer, Peter Jung,   
% Fast, noise-blind, and accurate: Tuning-free sparse regression with
% global linear convergence, COLT 2024

% Include path to algorithms and fix randomness 
addpath ../../ ../../algorithms/relative_iterate_error_timeout_stopping/
rng(123);

% Trial setup
SNR_level = 10*(1:6);
trials = 30;
timeout = 60;
threshold = 10^-5;
threshold_ITEM = 10^-8;

m = 200;
n = 5000;
s = 20;

n_algs = 10;

errors = zeros(n_algs,length(SNR_level), trials);
ratios = zeros(n_algs,length(SNR_level), trials);
ratio2s = zeros(n_algs,length(SNR_level), trials);
times = zeros(n_algs+1,length(SNR_level), trials);
objective = zeros(n_algs,length(SNR_level), trials);
m_errors = zeros(n_algs,length(SNR_level), trials);
rm_errors = zeros(n_algs,length(SNR_level), trials);
effective_spartity = zeros(n_algs,length(SNR_level), trials);

% Algorithmic parameters
lambda = 1/7;

eps2 = 10^-8;
eps1 = lambda*10^-8;
eps2_huge = eps2 * 10^10;
eps1_huge = eps1 * 10^10;

eps0 = 10^-4;
eps0_restart = 10^-3;

mu = 0;
L = lambda/ sqrt(eps2) + 1.0/sqrt(eps1);

for snri = 1:length(SNR_level)
    % For each SNR level
    SNR = SNR_level(snri);
    fprintf('SNR %d:',SNR)
    
    for t=1:trials
        % For each trial 
        fprintf('*')

        % Generate the problem
        A = randn(m,n);
        x = [ones(s,1) ; zeros(n-s,1)];
        b = A * x;
        sigma = sqrt(norm(b)^2 * 10.0^(-SNR/10.0));
        noise = sigma*randn(size(b));
        b = b + noise;
        normb = norm(b);

        % Compute eigendecomposition of A^T A used in Sherman-Morrison-Woodbury 
        % version of IRLS and store its time
        tic
        [U,Sigma] = eigs(A' * A,m);
        Sigma = diag(Sigma);
        svd_time = toc;
        fprintf('SVD time %d:',svd_time)

        times(n_algs+1,snri,t) = svd_time;
        
        x0 = randn(n,1);
        Lmax = norm(A,2);

        % Solve square-root LASSO with each algorithm
        
        % ITEM
        [x_1, t_1] = ITEM_sqrtlasso_to_rie(A,b,lambda,mu,L,eps1,eps2,x0, threshold_ITEM, timeout);
        times(1,snri,t) = t_1;
        errors(1,snri,t) = norm(x_1 - x)/norm(x);
        ratios(1,snri,t) = s_ratio(x_1,s,s);
        ratio2s(1,snri,t) = s_ratio(x_1,s,2*s);
        objective(1,snri,t) = norm(A*x_1 - b)/sqrt(m) + lambda*norm(x_1,1);
        m_errors(1,snri,t) = norm(A*x_1 - b);
        rm_errors(1,snri,t) = m_errors(1,snri,t)/normb;
        effective_spartity(1,snri,t) = norm(x_1,1)^2 / norm(x_1)^2;
         
        % Proximal gradient
        [x_2, t_2] = proximal_gradient_decay_to_rie(A,b,eps1_huge/10^2,lambda,1000/sqrt(eps1_huge),x0, threshold, timeout);
        times(2,snri,t) = t_2;
        errors(2,snri,t) = norm(x_2 - x)/norm(x);
        ratios(2,snri,t) = s_ratio(x_2,s,s);
        ratio2s(2,snri,t) = s_ratio(x_2,s,2*s);
        objective(2,snri,t) = norm(A*x_2 - b)/sqrt(m) + lambda*norm(x_2,1);
        m_errors(2,snri,t) = norm(A*x_2 - b);
        rm_errors(2,snri,t) = m_errors(2,snri,t)/normb;
        effective_spartity(2,snri,t) = norm(x_2,1)^2 / norm(x_2)^2;

        % Proximal Newton
        [x_3, t_3] = proximal_newton_decay_to_rie(A,b,10^-3* eps1_huge,lambda,x0,1000, threshold, timeout);
        times(3,snri,t) = t_3;
        errors(3,snri,t) = norm(x_3 - x)/norm(x);
        ratios(3,snri,t) = s_ratio(x_3,s,s);
        ratio2s(3,snri,t) = s_ratio(x_3,s,2*s);
        objective(3,snri,t) = norm(A*x_3 - b)/sqrt(m) + lambda*norm(x_3,1);
        m_errors(3,snri,t) = norm(A*x_3 - b);
        rm_errors(3,snri,t) = m_errors(3,snri,t)/normb;
        effective_spartity(3,snri,t) = norm(x_3,1)^2 / norm(x_3)^2;

        % Smooth concomitant LASSO
        [x_4, t_4] = smooth_concomitant_lasso_v2_decay_to_rie(A, b, 5, 10^-3*eps1_huge, lambda, x0, threshold, timeout);
        times(4,snri,t) = t_4;
        errors(4,snri,t) = norm(x_4 - x)/norm(x); 
        ratios(4,snri,t) = s_ratio(x_4,s,s);
        ratio2s(4,snri,t) = s_ratio(x_4,s,2*s);
        objective(4,snri,t) = norm(A*x_4 - b)/sqrt(m) + lambda*norm(x_4,1);
        m_errors(4,snri,t) = norm(A*x_4 - b);
        rm_errors(4,snri,t) = m_errors(4,snri,t)/normb;
        effective_spartity(4,snri,t) = norm(x_4,1)^2 / norm(x_4)^2;
        
        % IRLS with 'sqrt' smoothing parameters decay rule
        [x_5, t_5] = IRLS_eps_decay_given_to_rie(A,U,Sigma,b,lambda,eps0,x0,100,10^-8,'sqrt',s,'lsqr', threshold, timeout-svd_time);
        times(5,snri,t) = t_5;
        errors(5,snri,t) = norm(x_5 - x)/norm(x);
        ratios(5,snri,t) = s_ratio(x_5,s,s);
        ratio2s(5,snri,t) = s_ratio(x_5,s,2*s);
        objective(5,snri,t) = norm(A*x_5 - b)/sqrt(m) + lambda*norm(x_5,1);
        m_errors(5,snri,t) = norm(A*x_5 - b);
        rm_errors(5,snri,t) = m_errors(5,snri,t)/normb;
        effective_spartity(5,snri,t) = norm(x_5,1)^2 / norm(x_5)^2;
        
        % IRLS with 0.5*'sigma', i.e., best-s-\ell_1-alt update rule in [1]
        [x_6, t_6] = IRLS_eps_decay_given_to_rie(A,U,Sigma,b,lambda,0.5,x0,100,10^-8,'sigma',s,'lsqr', threshold, timeout-svd_time);
        times(6,snri,t) = t_6;
        errors(6,snri,t) = norm(x_6 - x)/norm(x); 
        ratios(6,snri,t) = s_ratio(x_6,s,s);
        ratio2s(6,snri,t) = s_ratio(x_6,s,2*s);
        objective(6,snri,t) = norm(A*x_6 - b)/sqrt(m) + lambda*norm(x_6,1);
        m_errors(6,snri,t) = norm(A*x_6 - b);
        rm_errors(6,snri,t) = m_errors(6,snri,t)/normb;
        effective_spartity(6,snri,t) = norm(x_6,1)^2 / norm(x_6)^2;
        
        % IRLS with 'sqrt' smoothing parameters decay rule and restarts
        [x_7, t_7] = IRLS_eps_decay_restart_given_to_rie(A,U,Sigma,b,lambda,eps0_restart,x0,100,10^-8,100,'sqrt',s,'lsqr', threshold, timeout-svd_time);
        times(7,snri,t) = t_7;
        errors(7,snri,t) = norm(x_7 - x)/norm(x);
        ratios(7,snri,t) = s_ratio(x_7,s,s);
        ratio2s(7,snri,t) = s_ratio(x_7,s,2*s);
        objective(7,snri,t) = norm(A*x_7 - b)/sqrt(m) + lambda*norm(x_7,1);
        m_errors(7,snri,t) = norm(A*x_7 - b);
        rm_errors(7,snri,t) = m_errors(7,snri,t)/normb;
        effective_spartity(7,snri,t) = norm(x_7,1)^2 / norm(x_7)^2;

        % Oracle
        tic
        x_8 = oracle(A,b,x0,s,n);
        times(8,snri,t) = toc;
        errors(8,snri,t) = norm(x_8 - x)/norm(x);
        ratios(8,snri,t) = s_ratio(x_8,s,s);
        ratio2s(8,snri,t) = s_ratio(x_8,s,2*s);
        objective(8,snri,t) = norm(A*x_8 - b)/sqrt(m) + lambda*norm(x_8,1);
        m_errors(8,snri,t) = norm(A*x_8 - b);
        rm_errors(8,snri,t) = m_errors(8,snri,t)/normb; 
        effective_spartity(8,snri,t) = norm(x_8,1)^2 / norm(x_8)^2;

        % Frank-Wolfe for constrained problem
        jnorm = sqrt(norm(b)^2/m + eps1);        
        norm_est = jnorm/lambda; 

        [x_9, t_9] = Frank_Wolfe_to_rie(A, b, norm_est, threshold, timeout);
        times(9,snri,t) = t_9;
        errors(9,snri,t) = norm(x_9 - x)/norm(x);
        ratios(9,snri,t) = s_ratio(x_9,s,s);
        ratio2s(9,snri,t) = s_ratio(x_9,s,2*s);
        objective(9,snri,t) = norm(A*x_9 - b)/sqrt(m) + lambda*norm(x_9,1);
        m_errors(9,snri,t) = norm(A*x_9 - b);
        rm_errors(9,snri,t) = m_errors(9,snri,t)/normb;
        effective_spartity(9,snri,t) = norm(x_9,1)^2 / norm(x_9)^2;

        % Frank-Wolfe with epigraphic lifting
        [x_10, t_10] = Frank_Wolfe_epi_to_rie(A, b, lambda,eps1, threshold, timeout);
        times(10,snri,t) = t_10;
        errors(10,snri,t) = norm(x_10 - x)/norm(x);
        ratios(10,snri,t) = s_ratio(x_10,s,s);
        ratio2s(10,snri,t) = s_ratio(x_10,s,2*s);
        objective(10,snri,t) = norm(A*x_10 - b)/sqrt(m) + lambda*norm(x_10,1);
        m_errors(10,snri,t) = norm(A*x_10 - b);
        rm_errors(10,snri,t) = m_errors(10,snri,t)/normb;
        effective_spartity(10,snri,t) = norm(x_10,1)^2 / norm(x_10)^2;
    end
    
    fprintf('\n')
end

%% Agregate the data

avg_error = mean(errors,3);
avg_time = mean(times,3);
avg_ratios = mean(ratios,3);
avg_ratio2s = mean(ratio2s,3);
avg_m_error = mean(m_errors,3);
avg_rm_error = mean(rm_errors,3);
effective_spartity(isnan(effective_spartity)) = 0;
avg_esp = mean(effective_spartity,3);

med_time = median(times,3);

avg_ratios = 1 - avg_ratios;
avg_ratio2s = 1 - avg_ratio2s;

%% Plot the results

alg_names = {'ITEM','proximal gradient','proximal newton','concomitant lasso','IRLS(sqrt)','IRLS(sigma)','IRLS(sqrt+restart)','Oracle','Frank Wolfe constrained', 'Frank Wolfe epi lift'};

figure
semilogy(SNR_level, avg_error(1:n_algs,:))
title('Relative errors')
legend(alg_names)

figure
semilogy(SNR_level, avg_time(1:n_algs,:))
title('Runtime')
legend(alg_names)

figure
semilogy(SNR_level, med_time(1:n_algs,:))
title('Runtime (med)')
legend(alg_names)

figure
plot(SNR_level,avg_ratios(1:n_algs,:))
title('Missing support percentage')
legend(alg_names)

figure
plot(SNR_level,avg_ratio2s(1:n_algs,:))
title('Missing support percentage (2s)')
legend(alg_names)

figure
plot(SNR_level,avg_rm_error(1:n_algs,:))
title('Relative Measurements Error')
legend(alg_names)

figure
plot(SNR_level,avg_esp(1:n_algs,:))
title('Effective Sparsity')
legend(alg_names)

%% Save the results

labels = {'snr','ITEM', 'prox_grad',...
          'prox_newton', ...
          'concomitant', ...
          'irls_sqrt', ...
          'irls_sigma', ...
          'irls_sqrt_restart',...
          'oracle', ...
          'frank_wolfe_constr',...
          'frank_wolfe_epi'};
results_err = [SNR_level.' avg_error.'];
results_err = array2table(results_err,'VariableNames',labels);
writetable(results_err,'noise_stab_1_err.csv','Delimiter',' ')

labels_time = {'snr','ITEM', 'prox_grad',...
          'prox_newton', ...
          'concomitant', ...
          'irls_sqrt', ...
          'irls_sigma', ...
          'irls_sqrt_restart',...
          'oracle',...
          'frank_wolfe_constr',...
          'frank_wolfe_epi',...
          'svd'};

results_time = [SNR_level.' avg_time.'];
results_time = array2table(results_time,'VariableNames',labels_time);
writetable(results_time,'noise_stab_1_time.csv','Delimiter',' ')

results_ratio = [SNR_level.' avg_ratios.'];
results_ratio = array2table(results_ratio,'VariableNames',labels);
writetable(results_ratio,'noise_stab_1_ratio.csv','Delimiter',' ')

results_ratio = [SNR_level.' avg_ratio2s.'];
results_ratio = array2table(results_ratio,'VariableNames',labels);
writetable(results_ratio,'noise_stab_1_ratio2s.csv','Delimiter',' ')

results_me = [SNR_level.' avg_m_error.'];
results_me  = array2table(results_me ,'VariableNames',labels);
writetable(results_me,'noise_stab_1_me.csv','Delimiter',' ')

results_rme = [SNR_level.' avg_rm_error.'];
results_rme = array2table(results_rme,'VariableNames',labels);
writetable(results_rme,'noise_stab_1_rme.csv','Delimiter',' ')

results_esp = [SNR_level.' avg_esp.'];
results_esp = array2table(results_esp,'VariableNames',labels);
writetable(results_esp,'noise_stab_1_esp.csv','Delimiter',' ')

function x = oracle(A,b,x0,s,n)
x = lsqr(A(:,1:s),b, 10^-10, 1000,[],[],x0(1:s));
x = [x; zeros(n-s,1)];
end