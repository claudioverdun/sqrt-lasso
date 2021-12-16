% x0 = [0;0;1;0];
% f = @(x) 0.5*norm(x - x0)^2;
% gradf = @(x) x - x0;
% x = ITEM(gradf,0,1,[1;0;0;0],1000)


m = 30;
n = 100;
s = 5;
A = randn(m,n);
x = [ones(5,1) ; zeros(95,1)];
b = A * x;

q = 0.1;
lambda = 10^-0.5;
lambda_opt = sqrt(m)*norminv(1 - 0.05/s);
lambda_opt = sqrt( log(n)/m); %s*sqrt( log(n)/m);
lambda = lambda_opt/1.5;
lambda_q = lambda^q;
%lambda_q = (1 + 0.5^q)/(2 * 100^q);

eps1 = 10^-5;
eps2 = 10^-5;
x0 = randn(n,1);
% x0 = x + [0.1*randn(5,1); zeros(95,1)];
mu = 0;
L = lambda/ sqrt(eps2) + 1.0/sqrt(eps1);

% gradf= @(x) A' * (A*x-b)/ sqrt( norm(A*x-b)^2 + eps1)/sqrt(m) + lambda * x ./ sqrt( abs(x).^2 + eps2);
Lmax = norm(A,2);

% % single run methods
% xr = ITEM_sqrtlasso(A,b,lambda,mu,L,eps1,eps2,x0,10000);
% xr2 = proximal_gradient(A,b,eps1,lambda,L,x0,10000);
% xr3 = proximal_newton(A,b,eps1,lambda,x0,1000,10000);
xr4 = IRLS(A,b,lambda,eps1,eps2,x0,1000,10000);
xr5 = lq_IRLS(A,b,lambda_q,q,eps1,eps2,xr4,1000,10000);
% xr5 = smooth_concomitant_lasso_v2(A, b, 10^-6, 1000, 10, eps1, lambda, x0);
% xr6 = Accelerated_IRLS(A,b,lambda,eps1,eps2,x0,1000,10000);
% xr7 = Restarted_IRLS(A,b,lambda,eps1,eps2,x0,25,10000,40);
% xr8 = Accelerated_IRLS_v2(A,b,lambda,eps1,eps2,x0,1000,10000);
% xr9 = Accelerated_IRLS_v2(A,b,lambda,eps1,eps2,x0,1000,10000);
[xr4';
    xr5';]
% [xr.';
%  xr2';
%  xr3';
%  xr4';
%  xr5';
%  xr6';
%  xr7';
%  xr8';
%  xr9']

lambda_max = 1000000;
% pathwise methods
% xr5 = smooth_concomitant_lasso(A, b, 10^-10, 1000, 10, 10, eps1, lambda_max);
% xr6 = pathwise_proximal_gradient(A,b,eps1,lambda_max,L,10000,10);
% xr7 = pathwise_proximal_newton(A,b,eps1,lambda_max,100,10000,10);
% xr8 = pathwise_IRLS(A,b,eps1,eps2,lambda_max,100,10000,10);

% [xr5';
%  xr6';
%  xr7';
%  xr8']

% bn = b; % + sigma*normrnd(0,1,m,1);
% lambda = 10^1;
% OPTIONS = set_parameters;
% OPTIONS.tol_I = 1.0e-6;
% [xr3,beta,runhist_I,runhist_II]= square_root_PMM(A,b,m,n,lambda,OPTIONS,'l1');
% xr3

% norm(xr - x)/norm(x)
% norm(xr2 - x)/norm(x)