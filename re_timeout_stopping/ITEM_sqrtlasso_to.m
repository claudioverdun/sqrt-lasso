function [x,time] = ITEM_sqrtlasso_to(A,b,lambda,mu,L,eps1,eps2,x0,N, x_tr, threshold, timeout)
m = length(b);
gradf= @(x) A' * (A*x-b)/ sqrt( norm(A*x-b)^2 + eps1)/sqrt(m) + lambda * x ./ sqrt( abs(x).^2 + eps2);

[x,time] = ITEM_to(gradf,mu,L,x0,N, x_tr, threshold, timeout);

% 
% if (mu ~= 0)
%     q = mu/L;
% 
%     Ak = 0;
%     zk = x0;
%     xk = x0;
% 
%     for i = 1:N
%         z_old = zk;
%         Ak1 = ( (1+q)*Ak+2*(1+sqrt((1+Ak)*(1+q*Ak)) ))/(1-q)^2; 
%         betak  	= Ak/(1-q)/Ak1;
%         deltak 	= 0.5 * ( (1-q)^2*Ak1 -(1+q)*Ak)/(1+q+q*Ak);
%         yk  = betak * xk + (1-betak) * zk;
%         grk = gradf(yk);
%         xk = yk - grk/L;
%         zk = (1 - q*deltak)*zk + q* deltak*yk - deltak*grk/L;
%         
%         if (norm(z_old - zk) < threshold)
%             break;
%         end
%     end
%     z = zk;
% else
%     thetak = 1;
% 
%     yk = x0;
%     xk = x0;
% 
%     for i = 1:N
%         x_old = xk;
%         grk = gradf(xk);
%         yk1  =  xk - grk/L;
%         if (i == N)
%             thetak1 = 0.5*(1+sqrt(1 + 8*thetak^2));
%         else
%             thetak1 = 0.5*(1+sqrt(1 + 4*thetak^2));
%         end
%         xk = yk1 + (thetak - 1)*(yk1 - yk)/thetak1 - thetak * grk /thetak1 / L;
%         yk = yk1;
%         thetak = thetak1;
%         
%         if (norm(x_old - xk) < threshold)
%             break;
%         end
%     end
%     z = xk;
% end
end