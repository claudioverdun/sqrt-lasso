%% This is a version of IRLS algorithm with restarts for square-root LASSO which performs a fixed
% number of iterations and stores intermediate loss function values
%
% Authors: Oleh Melnyk and Claudio Mayrink Verdun
%
% Based on:
% [1] Claudio Mayrink Verdun, Oleh Melnyk, Felix Krahmer, Peter Jung,
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
% eps0: >0, Inital smoothing parameter for \ell-1 regularization term or 
%           may be used as a parameter for some of the decay rules below.
% 
% x0: n x 1, Initial guess.
%
% N: integer >=0, number of IRLS iterations to be performed. 
% 
% Nlsp: integer >=0, number of LSQR iterations per single IRLS iteration.
%
% Nrestart: Number of restarts. IRLS iterations are equally split as 
% round(N/Nrestart) between restarts.
%
% decay: one of 'sqrt', 'harm', 'fn_sqrt', 'sigma', 'Rn', 'exp',
%        Update rules for the smoothing parameter. In the following, t is
%        the iteration counter. See [1] for a more detailed description of
%        each of them.
%       'sqrt' - eps0/sqrt(t), see Theorem 19 in [1] for sublinear convergence guarantee.
%       'harm' - eps0/t, no convergence guarantees.
%       'fn_sqrt' - special version of 'sqrt' based on the function loss at the current iterate, 
%                   see Theorem 6 in [1] for sublinear convergence
%                   guarantee. eps0 is ignored for this parameter.
%       'sigma' - best s-term approximation in \ell-1 norm update rule, see Theorem 9 in [1] for
%                 linear convergence. In this case, eps0 > 0 is the
%                 multiplier in front of the residual term, see Remark 28 in [1]. 
%       'Rn' - best s-term approximation in \ell-infty norm update rule. In this case, eps0 > 0 is also the
%                 multiplier in front of the residual term.
%       'exp' - exponential decay rule, decreasing the parameter by half every iteration.
%
% s: exact or expected sparsity: only used for 'sigma' and 'Rn' update rules.
%
%
%% Output:
%
% x: n x 1, reconstructed vector
% 
% f_vals: N x 1, vector containing intermediate values of the function 

function [x, f_vals] = IRLS_eps_decay_restart_fvals(A,b,lambda,eps0,x0,N,Nlsp,Nrestart,decay,s)
% Initialization
n = size(A,2);
m = length(b);
x = x0;
 
f_min = realmax;

if strcmp(decay,'exp')
    eps2 = eps0;
else
    eps2 = realmax;
end

f_vals = zeros(N,1);

% Number of IRLS iteration per restart
N_it = floor(N / Nrestart);
epst = eps0;
for rit=1:Nrestart
    % Each restart
    
    for it=1:N_it
        % Each IRLS iteration

        % Update of the smoothing parameter
        switch decay
            case 'sqrt' 
                eps2 = epst / sqrt(it);
            case 'harm' 
                eps2 = eps0 / it;
            case 'fn_sqrt'
                f_min = min(norm(A*x - b)/sqrt(m) + lambda* norm(x,1),f_min);
                eps2 = 2* f_min / (lambda*sqrt((n+1)*it));
            case 'sigma'
                mags = sort(abs(x),'descend');
                sigma = sum(mags((s+1):n));
                eps2 = min(eps2, eps0*(norm(A*x - b)/sqrt(m) + lambda* sigma)/lambda / (n+1));
            case 'Rn'
                mags = sort(abs(x),'descend');
                eps2 = min(eps2, eps0*(norm(A*x - b)/sqrt(m) + lambda* mags(s+1))/lambda / (n+1));
            case 'exp'
                eps2 = eps2*0.5;
        end
        eps1 = lambda*eps2;
        
        % Compute the function value with x^t and eps^t
        offset = (rit-1) * N_it;
        f_vals(offset + it) = objective(A,x,b,lambda,eps1,eps2);
        
        % Compute weights
        z0 = max(norm(A*x-b)/sqrt(m),eps1);
        z = max(abs(x),eps2);

        % Constuct weighted least squares problem and solve it
        A_expanded = @(x,flag) afun(x,flag,A,z0,z,lambda,m);
        b_expanded = [b/sqrt(z0*m);zeros(n,1)];
        warning off
        [x, flag] = lsqr(A_expanded,b_expanded, 10^-10, Nlsp,[],[],x);
        warning on
    end

    % Store last smoothing parameter as an itializatio for the next restart
    epst = eps2;
end
end

% Helper fuction to compute value of the smoothed objective as given in Eq. (5)
function f_val = objective(A,x,b,lambda,eps1,eps2)
m = length(b);
f_val = norm(A*x - b)/sqrt(m);
if f_val < eps1
    f_val = 0.5*(f_val.^2/ eps1 + eps1); 
end
xabs = abs(x);
idx = xabs < eps2;
xabs(idx) = 0.5*(xabs(idx).^2/ eps2 + eps2);
f_val = f_val + lambda*sum(xabs);
end

% Helper function to compute the multiplication with the matix and its 
% adjoint in the weighted linear system, see Lemma 3 and p.41 (shortly
% before Eq. (39)) of [1]
function y = afun(x,flag,A,z0,z,lambda,m)
if strcmp(flag,'notransp') % Compute A*x
    y_1 = A*x /sqrt(z0*m);
    y_2 = sqrt(lambda)*x./sqrt(z);
    y = [y_1 ; y_2];
elseif strcmp(flag,'transp') % Compute A'*x
    x_1 = x(1:m);
    x_2 = x((m+1):length(x));
    y = A'*x_1 /sqrt(z0*m);
    y = y + sqrt(lambda)*x_2./sqrt(z);
end
end