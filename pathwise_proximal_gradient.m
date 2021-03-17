function x = pathwise_proximal_gradient(A,b,eps1,lambda_max,Lmax,N,T)
m = length(b);
n = size(A,2);
lambda = min(norm(A' * b, 'Inf')/(norm(b) * sqrt(m)), lambda_max);
x = zeros(n,1);

for t= 1:T
    x = proximal_gradient(A,b,eps1,lambda,Lmax,x,N);
    lambda = 0.5*lambda;
end
end