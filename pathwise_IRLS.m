function x = pathwise_IRLS(A,b,eps1,eps2,lambda_max,N,Nlsp,T)
m = length(b);
n = size(A,2);
lambda = min(norm(A' * b, 'Inf')/(norm(b) * sqrt(m)), lambda_max);
x = zeros(n,1);

for t= 1:T
    x = IRLS(A,b,lambda,eps1,eps2,x,N,Nlsp);
    lambda = 0.5*lambda;
end
end