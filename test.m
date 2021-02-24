% x0 = [0;0;1;0];
% f = @(x) 0.5*norm(x - x0)^2;
% gradf = @(x) x - x0;
% x = ITEM(gradf,0,1,[1;0;0;0],1000)


m = 30;
n = 100;
A = randn(m,n);
x = [ones(5,1) ; zeros(95,1)];
b = A * x;

lambda = 10^0;
eps1 = 10^-5;
eps2 = 10^-5;
x0 = randn(n,1);
mu = 0;
L = lambda/ sqrt(eps2) + 1.0/sqrt(eps1);

gradf= @(x) A' * (A*x-b)/ sqrt( norm(A*x-b)^2 + eps1) + lambda * x ./ sqrt( abs(x).^2 + eps2);

xr = ITEM(gradf,mu,L,x0,10000)

Lmax = norm(A,2);
lambda = 10^-0.5;
xr2 = proximal_gradient(A,b,eps1,lambda,L,x0,100000)


% bn = b; % + sigma*normrnd(0,1,m,1);
% lambda = 10^1;
% OPTIONS = set_parameters;
% OPTIONS.tol_I = 1.0e-6;
% [xr2,beta,runhist_I,runhist_II]= square_root_PMM(A,bn,m,n,lambda,OPTIONS,'l1');
% 
% norm(xr - x)/norm(x)
% norm(xr2 - x)/norm(x)