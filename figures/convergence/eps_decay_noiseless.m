rng(123);

% 1 1/sqrt(N)
% 2 f(x_N) / sqrt(N)
% 3 sigma(N)
% 4 s(N)
% 5 10^{-N}

m = 300;
n = 1000;
s = 50;
A = randn(m,n);
x = [ones(s,1) ; zeros(n-s,1)];
x0 = randn(n,1);
b = A * x;

snr = 40;
sigma = sqrt(norm(b)^2 * 10^(-snr/10));
%b = b + sigma*randn(size(b)); 

lambda_opt = sqrt( log(n)/m); %s*sqrt( log(n)/m);
lambda = lambda_opt/1.5;
%lambda = lambda / sqrt(m);

eps0 = 10^-3;

Nit = 1000;
lsqr_it = 10000;

f_vals = zeros(7,Nit);
x_rec = zeros(7,n);

[x_rec(1,:), f_vals(1,:)] = IRLS_eps_decay(A,b,lambda,eps0,x0,Nit,lsqr_it,'sqrt',s);
[x_rec(2,:), f_vals(2,:)] = IRLS_eps_decay(A,b,lambda,eps0,x0,Nit,lsqr_it,'fn_sqrt',s);
[x_rec(3,:), f_vals(3,:)] = IRLS_eps_decay(A,b,lambda,0.5,x0,Nit,lsqr_it,'sigma',s);
[x_rec(4,:), f_vals(4,:)] = IRLS_eps_decay(A,b,lambda,0.5,x0,Nit,lsqr_it,'Rn',s);
[x_rec(5,:), f_vals(5,:)] = IRLS_eps_decay(A,b,lambda,eps0,x0,Nit,lsqr_it,'exp',s);
[x_rec(6,:), f_vals(6,:)] = IRLS_eps_decay(A,b,lambda,eps0,x0,Nit,lsqr_it,'harm',s);
[x_rec(7,:), f_vals(7,:)] = IRLS_eps_decay_restart(A,b,lambda,eps0,x0,Nit,lsqr_it,10,'sqrt',s);


%  = f_vals1; 
% f_vals(2,:) = f_vals2;
% f_vals(3,:) = f_vals3;
% f_vals(4,:) = f_vals4;
% f_vals(5,:) = f_vals5;
% f_vals(6,:) = f_vals6;
% 
% 
% x_rec(1,:) = xr1; 
% x_rec(2,:) = xr2;
% x_rec(3,:) = xr3;
% x_rec(4,:) = xr4;
% x_rec(5,:) = xr5;
% x_rec(6,:) = xr6;

f_min = min(lambda * norm(x,1), min(min(f_vals)) - 10^-16);

figure
semilogy(1:Nit, f_vals - f_min)
title('Objective')
legend({'sqrt','fn','0.5*sigma','0.5*Rn','0.5*','1/k', 'sqrt restart'})%,'IRLS(1000)','Half Accelerated IRLS','Half Accelerated IRLS v2'});%,'Half Accelerated IRLS'})


labels = {'Nit', 'sqrt',...
          'fn', ...
          'sigma', ...
          'Rn', ...
          'exp', ...
          'harm', ...
          'sqrt_restart'};
results = [(1:Nit).' (f_vals -f_min).'];
results = array2table(results,'VariableNames',labels);
writetable(results,'convergence_noiseless.csv','Delimiter',' ')


% figure
% semilogy(factors, avg_error(5:8,:))
% title('Relative errors')
% legend({'concomitant lasso','pathwise proximal gradient','pathwise proximal newton','pathwise IRLS'})

% figure
% plot(factors,avg_time(1:5,:))
% title('Runtime')
% legend({'ITEM','proximal gradient','proximal newton','IRLS(100)','concomitant lasso'})%,'IRLS(1000)','Half Accelerated IRLS','Half Accelerated IRLS v2'});%,'Half Accelerated IRLS'})

% figure
% plot(factors, avg_time(5:8,:))
% title('Runtime')
% legend({'concomitant lasso','pathwise proximal gradient','pathwise proximal newton','pathwise IRLS'})