addpath('/../../')

rng(123);

SNR_level = 10*(1:6);
trials = 1;
timeout = 60;
threshold = 10^-5;
threshold_ITEM = 10^-8;

m = 200;
n = 5000;
s = 20;

errors = zeros(8,length(SNR_level), trials);
ratios = zeros(8,length(SNR_level), trials);
ratio2s = zeros(8,length(SNR_level), trials);
times = zeros(8,length(SNR_level), trials);
objective = zeros(8,length(SNR_level), trials);
m_errors = zeros(8,length(SNR_level), trials);
rm_errors = zeros(8,length(SNR_level), trials);
effective_spartity = zeros(8,length(SNR_level), trials);

% lambda_opt = sqrt( log(n)/m);%s * sqrt( log(n)/m);
% lambda = lambda_opt/1.5;
lambda = 0.01; 
% lambda =1.0/7;

eps2 = 10^-8;
eps1 = lambda*10^-8;
eps2_huge = eps2 * 10^10;
eps1_huge = eps1 * 10^10;

eps0 = 10^-4;
eps0_restart = 10^-3;

mu = 0;
L = lambda/ sqrt(eps2) + 1.0/sqrt(eps1);

for snri = 1:length(SNR_level)
    SNR = SNR_level(snri);
    fprintf('SNR %d:',SNR)
    
    for t=1:trials
        fprintf('*')
        A = randn(m,n);
%         A = exprnd(1.0,m,n) .* (3 - 2*randi(2,m,n));

        x = [ones(s,1) ; zeros(n-s,1)];
%         x = x(randperm(n));

        b = A * x;
        sigma = sqrt(norm(b)^2 * 10.0^(-SNR/10.0));
        noise = sigma*randn(size(b));
%         fprintf('%.5f \n', norm(noise)/norm(b))
%         fprintf('SNR Cont. %.5f \n', norm(x)^2/sigma^2)
        b = b + noise;
        normb = norm(b);
        
        tic
%         [U,Sigma] = eigs(A' * A,m);
%         Sigma = diag(Sigma);
        svd_time = toc;
        fprintf('SVD time %d:',svd_time)

        x0 = randn(n,1);
        x01 = zeros(n,1);
        Lmax = norm(A,2);

        % % single run methods
%         [x_1, t_1] = ITEM_sqrtlasso_to_rie(A,b,lambda,mu,L,eps1,eps2,x0,1000000, x, threshold_ITEM, timeout);
%         times(1,snri,t) = t_1;
%         errors(1,snri,t) = norm(x_1 - x)/norm(x);
%         ratios(1,snri,t) = s_ratio(x_1,s,s);
%         ratio2s(1,snri,t) = s_ratio(x_1,s,2*s);
%         objective(1,snri,t) = norm(A*x_1 - b)/sqrt(m) + lambda*norm(x_1,1);
%         m_errors(1,snri,t) = norm(A*x_1 - b);
%         rm_errors(1,snri,t) = m_errors(1,snri,t)/normb;
%         effective_spartity(1,snri,t) = norm(x_1,1)^2 / norm(x_1)^2;
        
%         [x_2, t_2] = proximal_gradient_decay_to_rie(A,b,10^-1*eps1_huge,0.1*lambda,10^1*10/sqrt(eps1_huge),x0,1000000, x, threshold, 60);
%         times(2,snri,t) = t_2;
%         errors(2,snri,t) = norm(x_2 - x)/norm(x);
%         ratios(2,snri,t) = s_ratio(x_2,s,s);
%         ratio2s(2,snri,t) = s_ratio(x_2,s,2*s);
%         objective(2,snri,t) = norm(A*x_2 - b)/sqrt(m) + lambda*norm(x_2,1);
%         m_errors(2,snri,t) = norm(A*x_2 - b);
%         rm_errors(2,snri,t) = m_errors(2,snri,t)/normb;
%         effective_spartity(2,snri,t) = norm(x_2,1)^2 / norm(x_2)^2;
        
        [x_3, t_3] = Frank_Wolfe_epi_to_rie(A, b, lambda,eps1, threshold, timeout);
%         [x_3, t_3] = proximal_newton_decay_to_rie(A,b,10^-3* eps1_huge,lambda,x0,1000, threshold, timeout);
%         [x_3, t_3] = proximal_newton_decay_to_rie(A,b,10^-1* eps1_huge,lambda,x0,1000, threshold, timeout);
        times(3,snri,t) = t_3;
        errors(3,snri,t) = norm(x_3 - x)/norm(x);
        ratios(3,snri,t) = s_ratio(x_3,s,s);
        ratio2s(3,snri,t) = s_ratio(x_3,s,2*s);
        objective(3,snri,t) = norm(A*x_3 - b)/sqrt(m) + lambda*norm(x_3,1);
        m_errors(3,snri,t) = norm(A*x_3 - b);
        rm_errors(3,snri,t) = m_errors(3,snri,t)/normb;
        effective_spartity(3,snri,t) = norm(x_3,1)^2 / norm(x_3)^2;
        
%         [x_4, t_4] = smooth_concomitant_lasso_v2_decay_to_rie(A, b, 10000, 10, 10^-3*eps1_huge, lambda, x0, x, threshold, timeout);
%         [x_4, t_4] = smooth_concomitant_lasso_v2_decay_to_rie(A, b, 10000, 5,10^-1*eps1_huge, lambda*0.1, x0, x, threshold, timeout);
%         times(4,snri,t) = t_4;
%         errors(4,snri,t) = norm(x_4 - x)/norm(x); 
%         ratios(4,snri,t) = s_ratio(x_4,s,s);
%         ratio2s(4,snri,t) = s_ratio(x_4,s,2*s);
%         objective(4,snri,t) = norm(A*x_4 - b)/sqrt(m) + lambda*norm(x_4,1);
%         m_errors(4,snri,t) = norm(A*x_4 - b);
%         rm_errors(4,snri,t) = m_errors(4,snri,t)/normb;
%         effective_spartity(4,snri,t) = norm(x_4,1)^2 / norm(x_4)^2;
%         
%         [x_5, t_5] = IRLS_eps_decay_given_to_rie(A,U,Sigma,b,lambda,0.5,x0,10000,200,10^-2,'sigma',s,'pcg', x, threshold, timeout - svd_time);
%         times(5,snri,t) = t_5;
%         errors(5,snri,t) = norm(x_5 - x)/norm(x);
%         ratios(5,snri,t) = s_ratio(x_5,s,s);
%         ratio2s(5,snri,t) = s_ratio(x_5,s,2*s);
%         objective(5,snri,t) = norm(A*x_5 - b)/sqrt(m) + lambda*norm(x_5,1);
%         m_errors(5,snri,t) = norm(A*x_5 - b);
%         rm_errors(5,snri,t) = m_errors(5,snri,t)/normb;
%         effective_spartity(5,snri,t) = norm(x_5,1)^2 / norm(x_5)^2;
% %         
%         [x_6, t_6] = IRLS_eps_decay_given_to_rie(A,U,Sigma,b,lambda,0.5,x0,10000,100,10^-8,'sigma',s,'rand_solver', x, threshold, timeout-svd_time);
%         times(6,snri,t) = t_6;
%         errors(6,snri,t) = norm(x_6 - x)/norm(x); 
%         ratios(6,snri,t) = s_ratio(x_6,s,s);
%         ratio2s(6,snri,t) = s_ratio(x_6,s,2*s);
%         objective(6,snri,t) = norm(A*x_6 - b)/sqrt(m) + lambda*norm(x_6,1);
%         m_errors(6,snri,t) = norm(A*x_6 - b);
%         rm_errors(6,snri,t) = m_errors(6,snri,t)/normb;
%         effective_spartity(6,snri,t) = norm(x_6,1)^2 / norm(x_6)^2;
% %                  IRLS_eps_decay_restart_given_to_rie(A,U,Sigma,b,lambda,eps0,x0,Nlsp,tol,N_it,decay,s,solver, x_tr, threshold, timeout)
%         [x_7, t_7] = IRLS_eps_decay_restart_given_to_rie(A,U,Sigma,b,lambda,eps0_restart,x0,100,10^-8,100,'sqrt',s,'rand_solver', x, threshold, timeout);
%         times(7,snri,t) = t_7;
%         errors(7,snri,t) = norm(x_7 - x)/norm(x);
%         ratios(7,snri,t) = s_ratio(x_7,s,s);
%         ratio2s(7,snri,t) = s_ratio(x_7,s,2*s);
%         objective(7,snri,t) = norm(A*x_7 - b)/sqrt(m) + lambda*norm(x_7,1);
%         m_errors(7,snri,t) = norm(A*x_7 - b);
%         rm_errors(7,snri,t) = m_errors(7,snri,t)/normb;
%         effective_spartity(7,snri,t) = norm(x_7,1)^2 / norm(x_7)^2;

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
    end
    
    fprintf('\n')
end

avg_error = mean(errors,3);
avg_time = mean(times,3);
avg_ratios = mean(ratios,3);
avg_ratio2s = mean(ratio2s,3);
avg_m_error = mean(m_errors,3);
avg_rm_error = mean(rm_errors,3);
avg_esp = mean(effective_spartity,3);

med_time = median(times,3);

% avg_error(3,:) = avg_error(3,:)*trials / 5;
% avg_time(3,:) = avg_time(3,:)*trials / 5;
% avg_ratios(3,:) = avg_ratios(3,:)*trials / 5;
% avg_ratio2s(3,:) = avg_ratio2s(3,:)*trials / 5;

avg_ratios = 1 - avg_ratios;
avg_ratio2s = 1 - avg_ratio2s;

labels = {'Newton (old)','Newton (new)'};

ER = readtable('noise_stab_01_err.csv','Delimiter',' ');

figure
semilogy(SNR_level, [ER.frank_wolfe_epi avg_error(3,:).'])
title('Relative error')
legend(labels)

ER.frank_wolfe_epi = avg_error(3,:).';
writetable(ER,'noise_stab_01_err.csv','Delimiter',' ')

% % % 
TI = readtable('noise_stab_01_time.csv','Delimiter',' ');

figure
semilogy(SNR_level, [TI.frank_wolfe_epi avg_time(3,:).'])
title('Runtime')
legend(labels)
% 
TI.frank_wolfe_epi = avg_time(3,:).';
writetable(TI,'noise_stab_01_time.csv','Delimiter',' ')

SFR = readtable('noise_stab_01_ratio.csv','Delimiter',' ');

figure
plot(SNR_level, [SFR.frank_wolfe_epi avg_ratios(3,:).'])
title('Missing support percentage')
legend(labels)
% 
SFR.frank_wolfe_epi = avg_ratios(3,:).';
writetable(SFR,'noise_stab_01_ratio.csv','Delimiter',' ')

SFR2 = readtable('noise_stab_01_ratio2s.csv','Delimiter',' ');

figure
plot(SNR_level, [SFR2.frank_wolfe_epi avg_ratio2s(3,:).'])
title('Missing support percentage (2s)')
legend(labels)
% 
SFR2.frank_wolfe_epi = avg_ratio2s(3,:).';
writetable(SFR2,'noise_stab_01_ratio2s.csv','Delimiter',' ')
% % 

ME = readtable('noise_stab_01_me.csv','Delimiter',' ');

figure
plot(SNR_level,[ME.frank_wolfe_epi avg_m_error(3,:).'])
title('Measurement Error')
legend(labels)
% 
ME.frank_wolfe_epi = avg_m_error(3,:).';
writetable(ME,'noise_stab_01_me.csv','Delimiter',' ')
% %

RME = readtable('noise_stab_01_rme.csv','Delimiter',' ');

figure
semilogy(SNR_level,[RME.frank_wolfe_epi avg_rm_error(3,:).'])
title('Relative Measurement Error')
legend(labels)
ylim([10^-4.5, 10^-1])
% 
RME.frank_wolfe_epi = avg_rm_error(3,:).';
writetable(RME,'noise_stab_01_rme.csv','Delimiter',' ')
% 
ESP = readtable('noise_stab_01_esp.csv','Delimiter',' ');

figure
plot(SNR_level,[ESP.frank_wolfe_epi avg_esp(3,:).'])
title('Effective Sparsity')
legend(labels)
% % 
ESP.frank_wolfe_epi = avg_esp(3,:).';
writetable(ESP,'noise_stab_01_esp.csv','Delimiter',' ')



function x = oracle(A,b,x0,s,n)
x = lsqr(A(:,1:s),b, 10^-10, 1000,[],[],x0(1:s));
x = [x; zeros(n-s,1)];
end