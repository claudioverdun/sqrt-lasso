%% This script generates data for Figure 1 in 
% [1] Claudio Mayrink Verdun, Oleh Melnyk, Felix Krahmer, Peter Jung,
% Fast, noise-blind, and accurate: Tuning-free sparse regression with
% global linear convergence, COLT 2024

% Include path to algorithms and fix randomness 
addpath ../../algorithms/other
rng(123);

% Trial setup
m = 200;
n = 5000;
s = 20;

% Generate the problem
A = randn(m,n);
x = [ones(s,1) ; zeros(n-s,1)];
x0 = randn(n,1);
b = A * x;

% Algorithmic parameters
lambda = 1.0/7;

eps0 = 10^-4;
eps0_restart = 10^-3;

Nit = 1000;
lsqr_it = 10000;

f_vals = zeros(7,Nit);
x_rec = zeros(7,n);

% IRLS with 'sqrt' smoothing parameters decay rule, see Theorem 19 in [1])
[x_rec(1,:), f_vals(1,:)] = IRLS_eps_decay_fvals(A,b,lambda,eps0,x0,Nit,lsqr_it,'sqrt',s);

% IRLS with 'sqrt'-like smoothing parameters decay rule based on the function values, see Theorem 6 in [1]
[x_rec(2,:), f_vals(2,:)] = IRLS_eps_decay_fvals(A,b,lambda,eps0,x0,Nit,lsqr_it,'fn_sqrt',s);

% IRLS with 0.5*'sigma', i.e., best-s-\ell_1-alt update rule, see Remark 27 in [1]
[x_rec(3,:), f_vals(3,:)] = IRLS_eps_decay_fvals(A,b,lambda,0.5,x0,Nit,lsqr_it,'sigma',s);

% IRLS with 0.5*'Rn', i.e., best-s-\ell_\infty update rule
[x_rec(4,:), f_vals(4,:)] = IRLS_eps_decay_fvals(A,b,lambda,0.5,x0,Nit,lsqr_it,'Rn',s);

% IRLS with exponential decay of smoothing parameters, division by half
[x_rec(5,:), f_vals(5,:)] = IRLS_eps_decay_fvals(A,b,lambda,eps0,x0,Nit,lsqr_it,'exp',s);

% IRLS with 'sigma', i.e., best-s-\ell_1 update rule, see Theorem 9 in [1]
[x_rec(6,:), f_vals(6,:)] = IRLS_eps_decay_fvals(A,b,lambda,1,x0,Nit,lsqr_it,'sigma',s);

% IRLS with 'sqrt' smoothing parameters decay rule and restarts
[x_rec(7,:), f_vals(7,:)] = IRLS_eps_decay_restart_fvals(A,b,lambda,eps0_restart,x0,Nit,lsqr_it,10,'sqrt',s);

% With high probability, ground-truth object x is the global minimizer.
% Hence, we can compute the optimal value of the function in the noiseless case.
f_min = min(lambda * norm(x,1), min(min(f_vals)) - 10^-16);

% Plot the results
figure
semilogy(1:Nit, f_vals - f_min)
title('Objective')
legend({'sqrt','fn','0.5*sigma','0.5*Rn','0.5*','sigma', 'sqrt restart'})

% Save the results
labels = {'Nit', 'sqrt',...
          'fn', ...
          'sigma_half', ...
          'Rn', ...
          'exp', ...
          'sigma', ...
          'sqrt_restart'};
results = [(1:Nit).' (f_vals -f_min).'];
results = array2table(results,'VariableNames',labels);
writetable(results,'convergence_noiseless.csv','Delimiter',' ')


