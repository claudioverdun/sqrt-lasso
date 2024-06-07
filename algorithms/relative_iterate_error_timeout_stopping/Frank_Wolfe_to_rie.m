%% This is an implementation of Frank-Wolfe algorithm for \ell_1 constrained minimization of least squares
%
% Based on:
% Farah Cherfaoui, Valentin Emiya, Liva Ralaivola, and Sandrine Anthoine. 
% Recovery and convergence rate of the Frank–Wolfe algorithm for the m-exact-sparse problem. 
% IEEE Transactions on Information Theory, 65(11):7407–7414, 2019
%
%% Input:
%
% A: m x n design matrix. 
%
% b: m x 1 vector of measurements.
%
% lambda: >0, regularization parameter for square-root LASSO objective.
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

function [x,time] = Frank_Wolfe_to_rie(A, b, beta, threshold, timeout)
% Initialization
time = 0;
tic;

threshold_sgn = 10^-8;

x = zeros(size(A,2),1);

% Precompute residual and frequently used variables
ATA = A' * A;
Ax = A* x;
res = b - Ax;
Ares =  A' * res;

while true
    % Frank-Wolfe iteration
    x_old = x;
    nx_old = norm(x_old);
    
    % Find the descent direction by selecting an extereme point from 
    % {(+-beta e_j, beta)} where e_j are standard basis vectors
    
    % Compute the largest in absolute value element of the gradient
    [r_max, I] = max(abs(Ares));
    I = I(randperm(length(I)));
    
    % If gradient is very small, then stop
    if r_max < threshold_sgn
        dt = toc;
        time = time + dt;
        break;
    end

    for ii =1:length(I)
        % Else, take standard basis vector corresponding to the largest
        % entry with appropriate sign

        j = I(ii);
        Aj = A(:,j);
        s = Ares(j)/abs(Ares(j)) * beta;
        delta = Aj * s - Ax;

        % Find optimal step size
        gamma_opt = res' * delta / norm(delta)^2;
    
        if (gamma_opt < 0 || gamma_opt > 1)
            if norm(b - Aj * s)^2 < norm(res)^2
                gamma = 1;
            else
                continue;
            end
        else
            gamma = gamma_opt;
        end

        % Compute new x
        x_new = (1 - gamma) * x(j) + gamma* s;
        if abs(x_new - x(j)) > threshold *nx_old
            % Step is large enough 
            % Update precomputed variables with minimal computations
            change = Aj * (x_new - x(j));
            Ax = Ax + change;
            res = res - change;
            Ares = Ares - ATA(:,j) * (x_new - x(j));
            % Update x
            x(j) = x_new;
            break;
        end
    end

    % Check relative error stopping criterion
    if norm(x_old-x) < threshold * nx_old
        dt = toc;
        time = time + dt;
        break;
    end

    % Check runtime stopping criterion
    dt = toc;
    time = time + dt;
    if time >= timeout
        break;
    end
    tic;
end
end