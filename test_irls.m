
m=100;
n=200;
s=10;

A=randn(m,n);
x0=zeros(n,1);
x0(1:s)=randn(s,1);
x0=x0(randperm(length(x0)));
x_init=randn(size(x0));

lambda =sqrt(log(n)/m);

%epsilon decay rule for ||Ax-b||_2
eps1=10;

%epsilon decay rule for ||x||_1
eps2=10;

% eps min in oder to avoid numerical problems
epsmin = 10^-10;

%outer iterations
N = 100;

%inner iterations
Nlsp=10000;

b=A*x0+normrnd(0,1,[100,1]);


[x,x_track,eps1_track,eps2_track] = IRLS_sqrt_LASSO(A,b,s,lambda,eps1,eps2,x_init,N,Nlsp,epsmin,'automatic','KMS');


%CVX solution

cvx_begin quiet
cvx_precision low
variable y(n)
minimize(1/sqrt(m)*norm(A*y - b,2) + lambda*norm(y,1))
cvx_end


%Ground truth vs CVX
norm(y-x0,2)

%Ground truth vs IRLS
norm(x-x0,2) 
