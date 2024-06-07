%% An implementation of the coordinate descent for finging the minimizer of 
% the majorant constructed in the proximal Newton algorithm

% See also:
% Tuo Zhao. Han Liu. Tong Zhang. 
% Pathwise coordinate optimization for sparse learning: Algorithm and theory. 
% Ann. Statist. 46 (1) 180 - 218, February 2018. https://doi.org/10.1214/17-AOS1547
%
%% Input:
%
% x0 : n x 1, initial guess.
%
% grad: n x 1 vector containing gradient of a function. 
%
% Hessian: n x n matrix containing the Hessian of a function
%
% lambda: >0, regularization parameter for square-root LASSO objective.
%
%
% N: integer >0, the number of coordinate descent iterations
%
% threshold: >0, threshold value for relative error between the current
%                iterate and ground-truth.
%
% timeout: >0, time limit (in seconds) for the runtime of the algorithm.
%
%% Output:
%
% x: n x 1, reconstructed vector
%
% time: >0, total runtime

function [x, time] = coordinate_descent(x0, grad, Hessian, lambda, N, threshold, timeout)
% Initialization
tic
time = 0;
n = length(x0);

% We aim to minimize grad^T x + 0.5 (x - x0)^T H (x - x0) + lambda ||x||_1 with
% coordinate descent. When minimizing the above objective with respect to
% x_j, it simplifies to
% grad_j x_j + 0.5 x_j^2 H_j,j + x_\j^T H_\j,j x_j - x0^T H_.,j x_j + \lambda |x_j|,
% where x_\j denotes all indices except j. 
% The first and the fourth term together are called below as lin_0
% The third term can be written as x^T H_.,j - H_j,j x_j, denoted by
% lin_plus

% Compute residual
lin_0 = grad - Hessian * x0;
x = x0;

% Initial active set
idx = (abs(grad) >= 0.95*lambda) | (abs(x0) >= eps);
I = 1:n;
I = I(idx);

for it=1:N
    % For a single coordinate descent loop over all indices

    % Missing here: safe screening rule to avoid going over all coordinates
    x_old = x;
    for k=1:length(I)
        j = I(k);

        % Check if in the active set
        if idx(j) == 0
            continue;
        end

        lin_plus = Hessian(j,:) * x - Hessian(j,j) * x(j);
        lin = lin_0(j) + lin_plus;
        sq = 0.5*Hessian(j,j);
        
        % minimizing quadratic equation 
        % sq * x_j^2 + lin* x_j + lambda*|x_j|
        
        % case sq = 0
        if abs(sq) < eps
            if abs(lin) <= lambda
                x(j) = 0;
            else
                x(j) = Inf;
                fprintf('Reached infinity');
                return
            end
        % case sq >0
        elseif sq >0
            x(j) = - 0.5* max(abs(lin) - lambda,0)*sign(lin)/sq;
        end
        % case sq <0 due to convexity of the function
    end
    
    % Check runtime stopping criterion
    dt = toc;
    time = time + dt;
    if time >= timeout
        break;
    end
    tic;
    
    % Check relative error stopping criterion
    if norm(x - x_old) < threshold * norm(x_old)
        dt = toc;
        time = time + dt;
        break;
    end

    % Update active set

    idx = abs(x) >= eps;
    [~,K] = max((1-idx).*abs(lin_0+ Hessian *x));
    idx(K(1)) = 1;
    I = 1:n;
    I = I(idx);
end
end