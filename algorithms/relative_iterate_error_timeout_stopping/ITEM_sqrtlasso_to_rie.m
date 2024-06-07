%% This is an implementation of Information-Theoretic Exact Method (ITEM) 
% for smoothed square-root LASSO
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
% L: > mu, smoothness constant or, equivalently, Lipschitz constant of the
%           gradient
%
% eps1: >0, Smoothing parameter for \ell-2 data fidelity term.
%
% eps2: >0, Smoothing parameter for \ell-1 regularization term.
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

function [x,time] = ITEM_sqrtlasso_to_rie(A,b,lambda,mu,L,eps1,eps2,x0, threshold, timeout)
% Constuct gradient of the smoothed square-root LASSO loss function 
m = length(b);
gradf= @(x) A' * (A*x-b)/ sqrt( norm(A*x-b)^2 + eps1)/sqrt(m) + lambda * x ./ sqrt( abs(x).^2 + eps2);

% Call ITEM solver
[x,time] = ITEM_to_rie(gradf,mu,L,x0, threshold, timeout);
end