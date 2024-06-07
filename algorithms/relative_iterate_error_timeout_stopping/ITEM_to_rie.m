%% This is an implementation of Information-Theoretic Exact Method (ITEM) for arbitrary given function
% Based on:
% Adrien Taylor and Yoel Drori. An optimal gradient method for smooth 
% strongly convex minimization. Mathematical Programming, 199(1-2):557–594, 2023. 
%
%
%% Input:
%
% gradf: a function handle, mapping a vector n x 1 to n x 1 gradient of a
%       function to be optimized
%
% mu: >=0, strong convexity constant
%
% L: > mu, smoothness constant or, equivalently, Lipschitz constant of the
%           gradient
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
% z: n x 1, reconstructed vector
%
% time: >0,  total runtime

function [z,time] = ITEM_to_rie(gradf,mu,L,x0, threshold, timeout)
% Initialization
time = 0;
tic;

% Separate stongly convex (mu>0) case and convex case (mu=0) 
if (mu ~= 0)
    % Stongly convex case
    
    % Initializate parameters
    q = mu/L;
    Ak = 0;
    zk = x0;
    xk = x0;

    while true
        z_old = zk;

        % Compute acceleration parameters
        Ak1 = ( (1+q)*Ak+2*(1+sqrt((1+Ak)*(1+q*Ak)) ))/(1-q)^2; 
        betak  	= Ak/(1-q)/Ak1;
        deltak 	= 0.5 * ( (1-q)^2*Ak1 -(1+q)*Ak)/(1+q+q*Ak);

        % Momentum step
        yk  = betak * xk + (1-betak) * zk;
        grk = gradf(yk);

        % Gradient step
        xk = yk - grk/L;

        % Second momentum
        zk = (1 - q*deltak)*zk + q* deltak*yk - deltak*grk/L;
        
        % Check relative error stopping criterion
        if norm(z_old -zk) < threshold * norm(z_old)
            dt = toc;
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
    z = zk;
else
    thetak = 1;

    yk = x0;
    xk = x0;

    while true
        x_old = xk;

        % Gradient step
        grk = gradf(xk);
        yk1  =  xk - grk/L;

        % Momentum step
        thetak1 = 0.5*(1+sqrt(1 + 4*thetak^2));
        xk = yk1 + (thetak - 1)*(yk1 - yk)/thetak1 - thetak * grk /thetak1 / L;
        yk = yk1;
        thetak = thetak1;
        
        % Check relative error stopping criterion
        if norm(x_old-xk) < threshold * norm(x_old)  
            dt = toc;
            time = time + dt;
            break;
        end
    
        % Check runtime stopping criterion
        dt = toc;
        time = time + dt;
        if time >= timeout
            break
        end
        tic
    end
    z = xk;
end
end