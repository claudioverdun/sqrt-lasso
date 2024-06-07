%% This is an implementation of Information-Theoretic Exact Method (ITEM) 
% for smoothed square-root LASSO with decaying smoothing parameter
%
% Original ITEM paper:
% Adrien Taylor and Yoel Drori. An optimal gradient method for smooth 
% strongly convex minimization. Mathematical Programming, 199(1-2):557–594, 2023. 
%
% Square-root LASSO explanation:
% Appendix F.2 of Claudio Mayrink Verdun, Oleh Melnyk, Felix Krahmer, Peter Jung,
% Fast, noise-blind, and accurate: Tuning-free sparse regression with
% global linear convergence, COLT 2024
%
%
%% Input:
%
% A: m x n design matrix. 
%
% U: n x m matrix with rows being the eigenvectors of A^T A.
% 
% Sigma: m x 1 vector containing the eigenvalues of A^T A. 
%
% b: m x 1 vector of measurements.
%
% lambda: >0, regularization parameter for square-root LASSO objective.
%
% mu: >=0, strong convexity constant
%
% eps1: >0, Initial smoothing parameter for \ell-2 data fidelity term.
%
% eps2: >0, Initial smoothing parameter for \ell-1 regularization term.
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

function [x,time] = ITEM_sqrtlasso_decay_to_rie(A,b,lambda,mu,eps1,eps2,x0, threshold, timeout)
% Initialize
m = length(b);

time = 0;

eps1_tmp = eps1;
eps2_tmp = eps2;

x = x0;
while true
    % For each value of the smoothing parameters
    x_old = x;

    % Compute current smoothness constant
    L_temp = 1 / sqrt(eps1_tmp) + lambda/sqrt(eps2_tmp);

    % Constuct gradient of the smoothed square-root LASSO loss function 
    gradf= @(x) A' * (A*x-b)/ sqrt( norm(A*x-b)^2 + eps1_tmp)/sqrt(m) + lambda * x ./ sqrt( abs(x).^2 + eps2_tmp);
    
    % Call ITEM solver
    [x,time_new] = ITEM_to_rie(gradf,mu,L_temp,x, threshold, timeout-time);
    time = time + time_new;

    % Check relative error stopping criterion
    if norm(x_old-x) < threshold * norm(x_old)  
        break;
    end

    % Check runtime stopping criterion
    if time >= timeout
        break
    end

    % Update of the smoothing parameters
    eps1_tmp = eps1_tmp * sqrt(0.1);
    eps2_tmp = eps2_tmp * sqrt(0.1);
end
end