% x0 = [0;0;1;0];
% f = @(x) 0.5*norm(x - x0)^2;
% gradf = @(x) x - x0;
% x = ITEM(gradf,0,1,[1;0;0;0],1000)


m = 300;
n = 1000;
s = 50;
A = randn(m,n);
x = [ones(s,1) ; zeros(n-s,1)];
x0 = randn(n,1);
b = A * x;

lambda_opt = sqrt( log(n)/m); %s*sqrt( log(n)/m);
lambda = lambda_opt;

eps1 = 10^-5;
eps2 = 10^-5;

lambda_max = 1000000;
% pathwise methods
% xr5 = smooth_concomitant_lasso(A, b, 10^-10, 1000, 10, 10, eps1, lambda_max);
% xr6 = smooth_concomitant_lasso_v2(A, b, 10^-10, 1000, 10, 10, eps1, lambda_max);

mu = 0;
L = lambda/ sqrt(eps2) + 1.0/sqrt(eps1);
xr0 = ITEM_sqrtlasso(A,b,lambda,mu,L,eps1,eps2,x0,100000);
xr = IRLS(A,b,lambda,eps1,eps2,xr0,100,10000);

[xr0';
 xr']
% [xr5';
%  xr6']