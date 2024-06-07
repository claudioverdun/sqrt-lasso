%% This is an implementation of proximal gradient descent for smoothed square-root LASSO
% with decaying smoothing parameter
%
% Algorithm 1 from:
% Xinguo Li, Haoming Jiang, Jarvis Haupt, Raman Arora, Han Liu, Mingyi Hong, and Tuo Zhao.
% On fast convergence of proximal algorithms for SQRT-Lasso optimization: Don't worry about
% its nonsmooth loss function. In Proceedings of The 35th Uncertainty in Artificial Intelligence Conference, 
% volume 115 of Proceedings of Machine Learning Research, pages 49–59, Tel Aviv, Israel, 22–25 Jul 2020. PMLR.
%
% See also:
% Appendix F.3 of Claudio Mayrink Verdun, Oleh Melnyk, Felix Krahmer, Peter Jung,
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
% eps1: >0, Initial smoothing parameter for \ell-2 data fidelity term.
%
% lambda: >0, regularization parameter for square-root LASSO objective.
%
% Lmax: >0, An upper bound on the smoothness constant or, equivalently, 
%           Lipschitz constant of the gradient
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

function [x,time] = proximal_gradient_decay_to_rie(A,b,eps1,lambda,Lmax,x0, threshold, timeout)
% Initialization
time = 0;
tic;


x = x0;
eps = eps1;

while true
    x_sub = x;

    % Compute current smoothness constant
    L = 1/sqrt(eps1);
    Ltilde = L;
    m = length(b);

    % Function handles

    % Data fidelity objective function
    obj = @(x) sqrt(norm(A * x - b)^2 + eps)/sqrt(m);
    % Its gradient
    grad = @(x) A' * (A*x-b)/ sqrt( norm(A*x-b)^2 + eps)/ sqrt(m);
    % Quadratic majorant
    Q = @(x,y,lambda,L) obj(x) + grad(x)' * (y-x) + 0.5*L*norm(y-x)^2 + lambda* norm(y,1);
    % Loss function with regularizer
    F = @(x,lambda) obj(x) + lambda* norm(x,1);
    
    while true
        % For each iteration
        x_old = x;

        % Find an appropriate step size with line search
        gr = grad(x);
        while true
            thresh = step(x,gr,Ltilde,lambda);
            obj_new = F(thresh,lambda);
            q_new = Q(x,thresh,lambda,Ltilde);
            if obj_new >= q_new
                break;
            end
            Ltilde = 0.5 * Ltilde;
        end
        
        L = min(2*Ltilde,Lmax);
        Ltilde = L;

        % Perform soft thresholding proximal step
        x = step(x,gr,L,lambda);

        % Check relative error stopping criterion
        if norm(x_old-x) < threshold * norm(x_old)
            dt = toc;
            tic
            time = time + dt;
            break;
        end
    
        % Check runtime stopping criterion
        dt = toc;
        time = time + dt;
        if time >= timeout  
            break
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
        break
    end
    tic;

    % Update of the smoothing parameter
    eps = eps* sqrt(0.1);
end

end

% Perform soft thresholding proximal step
function thresh = step(x,gr,L,lambda)
xnew = x - gr / L;
thresh = sign(xnew) .* max(abs(xnew) - lambda/L,0 );
end