%% This is an implementation of Smooth concomitant LASSO algorithm
% with decaying smoothing parameter
%
% Based on:
% Eugene Ndiaye, Olivier Fercoq, Alexandre Gramfort, Vincent Lecl`ere, and Joseph Salmon. 
% Efficient smoothed concomitant lasso estimation for high dimensional regression. 
% Journal of Physics: Conference Series, 904:012006,  2017.
%
% See also:
% Appendix F.5 of Claudio Mayrink Verdun, Oleh Melnyk, Felix Krahmer, Peter Jung,
% Fast, noise-blind, and accurate: Tuning-free sparse regression with
% global linear convergence, COLT 2024
%
%
%% Input:
%
% A: m x n design matrix. 
%
% b: m x 1 vector of measurements.
%
% F: integer >=1, the frequency of the safe screening update, i.e. every F
%                 iterations
%
% sigma0: >0, initial smoothing parameter for \ell-2 data fidelity term.
%
% lambda: >0, regularization parameter for square-root LASSO objective.
%
% x0: n x 1, Initial guess.
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

function [x,time] = smooth_concomitant_lasso_v2_decay_to_rie(A, b, F, sigma0, lambda, x0, threshold, timeout)
% Initialization
time = 0;
tic;

m = length(b);
n = size(A,2);
x = x0;
sigma = norm(A*x - b)/sqrt(m);

% Precompute residual and frequently used variables
norms = sum(abs(A).^2,1);
res = A*x-b;
res_norm = norm(res);

sigma_tmp = sigma0;
it = 1;
I = 1:n;

while true
    % For each smoothing parameter

    while true
        % Coordinate descent iteration

        % Perform safe screening
        if mod(it,F) == 1 || F == 1
            theta = -res/max([lambda*m*sigma0 norm(A' * res,'Inf') lambda*sqrt(m)*res_norm]);
            P = 0.5*res_norm/m/sigma + 0.5*sigma + lambda*norm(x, 1); 
            D = lambda * b' * theta + sigma0*(0.5 - 0.5*lambda^2 * m * norm(theta).^2);
            G = P-D;
            r = sqrt(2*G / lambda^2 / sigma0 /m);
            I = 1:n;
            I = I(abs( A' * theta) + r * sqrt(norms)' >= 1);
        end
        x_old = x;
    
        for k = 1:length(I)
            j = I(k);

            % Perform proximal coordinate descent step
            col = A(:,j);
            xj_old = x(j);
            update = col.' * res;
            xnew = x(j) - update/norms(j);
            x(j) = sign(xnew) .* max(abs(xnew) - m*sigma*lambda/norms(j),0 );
    
            % Update precomputed variables with minimal computations
            
            res_norm = sqrt(max(res_norm^2 + 2*(x(j) - xj_old)*update + (x(j) - xj_old)^2*norms(j),0));
            res = res + col*(x(j) - xj_old);

            % Update weights
            sigma = max(sigma_tmp,res_norm/sqrt(m));
        end
    
        % Check relative error stopping criterion
        if norm(x_old-x) < threshold * norm(x_old)
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

        it = it + 1;
    end

    % Check relative error stopping criterion in the outer loop
    if norm(x_old-x) < threshold * norm(x_old)
        dt = toc;
        time = time + dt;
        break;
    end

    % Check runtime stopping criterion in the outer loop
    dt = toc;
    time = time + dt;
    if time >= timeout
        break;
    end
    tic;

    % Update of the smoothing parameter
    sigma_tmp = sigma_tmp * sqrt(0.1);
end
end
