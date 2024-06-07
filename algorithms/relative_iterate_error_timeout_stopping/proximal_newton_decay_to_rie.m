%% This is an implementation of proximal Newton method for smoothed square-root LASSO
% with decaying smoothing paramater 
%
% Algorithm 2 from:
% Xinguo Li, Haoming Jiang, Jarvis Haupt, Raman Arora, Han Liu, Mingyi Hong, and Tuo Zhao.
% On fast convergence of proximal algorithms for SQRT-Lasso optimization: Don't worry about
% its nonsmooth loss function. In Proceedings of The 35th Uncertainty in Artificial Intelligence Conference, 
% volume 115 of Proceedings of Machine Learning Research, pages 49–59, Tel Aviv, Israel, 22–25 Jul 2020. PMLR.
%
% See also:
% Appendix F.4 of Claudio Mayrink Verdun, Oleh Melnyk, Felix Krahmer, Peter Jung,
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
% eps1: >0, initial smoothing parameter for \ell-2 data fidelity term.
%
% lambda: >0, regularization parameter for square-root LASSO objective.
%
% x0: n x 1, Initial guess.
%
% Ninner: integer >0, the number of coordinate descent iterations to be
%                      performed for solving the majorant minimization step
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

function [x, time] = proximal_newton_decay_to_rie(A,b,eps1,lambda,x0, Ninner, threshold, timeout)
% Initialization
time = 0;
tic ;

Aha = A' * A;
m = length(b);

x= x0;
eps = eps1;

while true
    % Data fidelity objective function
    obj = @(x) sqrt(norm(A * x - b)^2 + eps)/sqrt(m);
    % Its gradient
    grad = @(x) A' * (A*x-b)/ sqrt( norm(A*x-b)^2 + eps)/ sqrt(m);
    % Its Hessian
    H = @(x) Hessian(x, A,Aha,b,eps);
    % Quadratic majorant
%     Q = @(x,y,lambda) obj(y) + grad(y)' * (x-y) + 0.5*(x-y)'*H(y)*(x-y) + lambda* norm(x,1);
    % Loss function with regularizer
    F = @(x,lambda) obj(x) + lambda* norm(x,1);
    
    x_sub = x;
    
    while true
        x_old = x;

        % Find the minimizer of the quadratic majorant via coordinate descent

        dt = toc;
        time = time + dt;
        [x_new, dt] = coordinate_descent(x, grad(x), H(x), lambda, Ninner, threshold, timeout -  time);
        tic 
        time = time + dt;

        % Line search step
        diff = x_new - x;
        gamma = grad(x)' * diff + lambda*(norm(x_new,1) - norm(x,1));
        eta = 1;
        prev= F(x,lambda);
        
        while F(x + eta*diff,lambda) > prev + 0.25 * gamma* eta
            eta = eta*0.9;
        end
        
        x = x + eta*diff;
    
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
    end

    % Check relative error stopping criterion in the outer loop
    if norm(x_sub-x) < threshold * norm(x_sub)
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
    eps = eps* sqrt(0.1);
end
end

% Helper function, which computes the Hessian matrix of the smoothed data
% fidelity term
function H = Hessian(x, A, AhA, b, eps)
m = length(b);
res = A*x  - b;
nr = norm(res)^2;
ATres = A' * res;
H = (AhA/sqrt( nr + eps ) + ATres * ATres'/(nr+eps)^(3/2)  )/sqrt(m);
end