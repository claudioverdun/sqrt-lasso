function x = pathwise(A,b,lambda_max, solver,T)
m = length(b);
n = size(A,2);
lambda = min(norm(A' * b, 'Inf')/(norm(b) * sqrt(m)), lambda_max);
x = zeros(n,1);

for t= 1:T
    x = solver(x,lambda);  
    lambda = 0.5*lambda;
end
end