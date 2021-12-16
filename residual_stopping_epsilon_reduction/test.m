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

lambda = 10^-0.5;
lambda_opt = sqrt(m)*norminv(1 - 0.05/s);
lambda_opt = sqrt( log(n)/m); %s*sqrt( log(n)/m);
lambda = lambda_opt/1.5;

eps1 = 10^-6;
eps2 = 10^-10;
x0 = randn(n,1);
% x0 = x + [0.1*randn(5,1); zeros(95,1)];
mu = 0;
L = lambda/ sqrt(eps2) + 1.0/sqrt(eps1);

% gradf= @(x) A' * (A*x-b)/ sqrt( norm(A*x-b)^2 + eps1)/sqrt(m) + lambda * x ./ sqrt( abs(x).^2 + eps2);
Lmax = norm(A,2);

% % single run methods
% xr = ITEM_sqrtlasso(A,b,lambda,mu,L,eps1,10,x0,10^-4);
% xr2 = proximal_gradient(A,b,eps1,10,lambda,L,x0,10^-4);
% xr3 = proximal_newton(A,b,eps1,1,lambda,x0,10000,10^-4);
% xr4 = IRLS(A,b,lambda,eps1,1,x0,10000,10^-4);
% xr5 = smooth_concomitant_lasso_v2(A, b, 10^-6, 1000, eps1, lambda, x0,10^-4);
xr6 = IRLS_thresholding(A,b,lambda,eps1,1,x0,10000,10^-4);

[xr.';
 xr2';
 xr3';
 xr4';
 xr5';
 xr6']

