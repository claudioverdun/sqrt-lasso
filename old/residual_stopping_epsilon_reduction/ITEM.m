function z = ITEM(gradf,mu,L,K,x0,condition)
threshold = 10^-10;

if (mu ~= 0)
    q = mu/L;

    Ak = 0;
    zk = x0;
    xk = x0;

    for k=1:K
        z_old = zk;
        Ak1 = ( (1+q)*Ak+2*(1+sqrt((1+Ak)*(1+q*Ak)) ))/(1-q)^2; 
        betak  	= Ak/(1-q)/Ak1;
        deltak 	= 0.5 * ( (1-q)^2*Ak1 -(1+q)*Ak)/(1+q+q*Ak);
        yk  = betak * xk + (1-betak) * zk;
        grk = gradf(yk);
        xk = yk - grk/L;
        zk = (1 - q*deltak)*zk + q* deltak*yk - deltak*grk/L;
        
        
        if (condition(zk))
            break;
        end
    end
    z = zk;
else
    thetak = 1;

    yk = x0;
    xk = x0;

    for k=1:K
        x_old = xk;
        grk = gradf(xk);
        yk1  =  xk - grk/L;
        thetak1 = 0.5*(1+sqrt(1 + 4*thetak^2));
        xk = yk1 + (thetak - 1)*(yk1 - yk)/thetak1 - thetak * grk /thetak1 / L;
        yk = yk1;
        thetak = thetak1;
        
        if (condition(xk))
            break;
        end
    end
    z = xk;
end
end