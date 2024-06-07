%% This is an implementation of epigraphic lifting Frank-Wolfe algorithm for square-root LASSO
%
% Based on:
% [1] Appendix F.6 of Claudio Mayrink Verdun, Oleh Melnyk, Felix Krahmer, Peter Jung,
% Fast, noise-blind, and accurate: Tuning-free sparse regression with
% global linear convergence, COLT 2024
%
%% Input:
%
% A: m x n design matrix. 
%
% b: m x 1 vector of measurements.
%
% lambda: >0, regularization parameter for square-root LASSO objective.
%
% eps1: >0, initial smoothing parameter for \ell-2 data fidelity term.
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

function [x,time] = Frank_Wolfe_epi_to_rie(A, b, lambda, eps1, threshold, timeout)
% Initialization
time = 0;
tic;

m = length(b);

x = zeros(size(A,2),1);
t = norm(x,1);

% The radius of the \ell-1 ball containing the solution  
jnorm = sqrt(norm(b)^2/m + eps1);
beta = jnorm/lambda;

% Precompute residual and frequently used variables
ATA = A' * A;
Ax = A* x;
res = Ax-b;
ATb  = A' *b;
ATAx = A'*Ax;
Ares =  ATAx - ATb; 

while true
    % Frank-Wolfe iteration
    x_old = x;
%     t_old = t;
%     nx_old = norm([x_old;t_old]);
    nx_old = norm(x_old);

    % Find the descent direction by selecting an extereme point from 
    % {(0,0), (+-beta e_j, beta)} where e_j are standard basis vectors
    
    % Compute the largest in absolute value element of the gradient
    [r_max, I] = max(abs(Ares),[],'all');
    idx = randperm(length(I));
    I = I(idx);

    % Select descent direction according to Eq. (44) in [1]
    if -r_max / sqrt(m^2 * eps1 + norm(res)^2*m)  + lambda >= 0
        % If it is insuffiently large
        % (0,0) is the descent direction
        s = 0;
        r = 0; 
        Aj = zeros(m,1);
        j=1;
    else
        % Else, take standard basis vector corresponding to the largest
        % entry with appropriate sign
        j = I(1);
        Aj = A(:,j);
        s = -Ares(j)/ abs(Ares(j)) *beta;
        r = beta;
    end

    % Find optimal step size
    
    % Compute the leading coefficient of the quadratic equation (46) in [1]
    diff = -x;
    diff(j) = diff(j) + s; 
    diff = A * diff;
    ndiff = norm(diff)/ sqrt(m);
    mult= lambda^2*(r-t)^2;
    ca = ndiff^2*(mult-ndiff^2);

    
    if abs(ca) < eps1
        % If it is zero, just take gamma 0 or 1
        gamma_pull = [0 1];
    else
        % Else compute the rest of the coefficients and the discriminant
        pr = res' * diff/ m;
        cb = 2*pr*(mult - ndiff^2);
        cc = mult*(norm(res)^2/m + eps1) - pr^2;
        d = cb^2 - 4*ca*cc;

        % Numerical stability measure 
        if abs(d) < 10^-3
            d = 0;
        end

        if d < 0
            % If it is negative, no real roots of (46) and we only consider
            % gamma 0 or 1
            gamma_pull = [0,1];
        else
            % Otherwise, we add roots as a candidates for the minimizers
            gamma_pull = [(- cb - sqrt(d))/2/ca (- cb + sqrt(d))/2/ca 0 1];
        end
        % Check that roots are in the interval (0,1)
        gamma_pull(gamma_pull < 0) = 0;
        gamma_pull(gamma_pull > 1) = 1;
    end

    % Evaluate the loss function in Eq. (45) for gamma candidates and pick
    % the global minimizer
    gamma_pull = unique(gamma_pull);
    f_vals = zeros(length(gamma_pull),1);

    for k=1:length(gamma_pull)
        gamma = gamma_pull(k);
        f_vals(k) = sqrt(norm(res + gamma*diff)^2/m + eps1) + lambda * ((1-gamma)*t + gamma*r);
    end

    [~,J] = min(f_vals);
    gamma = gamma_pull(J(1));

    % Update x and t
    x_new = (1 - gamma) * x; 
    x_new(j) = x_new(j) + gamma* s;
    t = (1 - gamma) * t + gamma* r; 
    
    % Update precomputed variables with minimal computations
    change = (x_new(j) - (1 - gamma) * x(j));
    Ax = (1 - gamma) * Ax + Aj *change;
    res = Ax - b;
    ATAx = (1 - gamma) *ATAx + ATA(:,j) * change;
    Ares = ATAx  - ATb;
    x = x_new;

    % Check relative error stopping criterion
%     if norm([x_old;t_old]-[x;t]) <= threshold * nx_old
    if norm(x_old-x) <= threshold * nx_old
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