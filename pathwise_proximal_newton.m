function x = pathwise_proximal_newton(A,b,eps1,lambda_max,N,Ninner,T)
m = length(b);
n = size(A,2);
lambda = min(norm(A' * b, 'Inf')/(norm(b) * sqrt(m)), lambda_max);
x = zeros(n,1);

for t= 1:T
    x = proximal_newton(A,b,eps1,lambda,x,N, Ninner);
    lambda = 0.5*lambda;
end
end