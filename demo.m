%-----------------------------------------------------------------
% config
%-----------------------------------------------------------------

cf        = struct;

% cf.n      = 256;
% cf.m      = 64;
% cf.s      = [2:3:.8*cf.m];

%----------------------------------------------------------------
% good setting for orthogonal arrays
% N=16;
% k=5;
% k1=k*k+2*k;
%
% cf.n      = k1*k1+2*k1;
%----------------------------------------------------------------

%----------------------------------------------------------------
% good setting for mubs
cf.n      = 1021;%1024
%----------------------------------------------------------------



cf.m      = floor(cf.n/4);
cf.s      = [3:floor(.08*cf.m):floor(.6*cf.m)];

%-----------------------------------------------------------------
% specify the random model genering the ground truth x0 on its
% sparse support (iid)
% - real vectors
% 'gaussian'
% - complex vectors
% 'cnormal' 
cf.x0model = 'gaussian';

%-----------------------------------------------------------------
% specify the models (as a cell array) for the measurement matrix
%  - real matrices
% 'gaussian', 'bernoulli','dct','oa'
% 'gaussian', 'bernoulli','dct','oa'
%  - complex matrices
% 'cnormal', 'dft','mub'

%cf.model  = {'gaussian','bernoulli','dct','oa'};
cf.model  = {'gaussian'};
%cf.model  = {'cnormal','dft','alltop'};

%-----------------------------------------------------------------
% the algorithms to test 

cf.alg                = {}

cf.alg{end+1}.name    = 'cvx_slasso';
cf.alg{end  }.legend  = 'cvx_slasso, lambda=opt';
cf.alg{end  }.lambda  = sqrt(log(cf.n/cf.m)/m);


cf.alg{end+1}.name    = 'irls_slasso';
cf.alg{end  }.legend  = 'irls_slasso, lambda=opt';
cf.alg{end  }.lambda  = sqrt(log(cf.n/cf.m)/m);
cf.alg{end  }.lambda  = sqrt(log(n)/m); % from Claudio

cf.runs   = 1000;
cf.matfile='results.mat';


cf.nmodels= numel(cf.model);
cf.nalgs  = numel(cf.alg);

%-----------------------------------------------------------------
% results
%-----------------------------------------------------------------
re   = struct;
bins=[numel(cf.s),1];

for imodel=1:cf.nmodels
    for ialg=1:numel(cf.alg)
        re.alg{imodel,ialg}.l2err.avg = zeros(bins);
        re.alg{imodel,ialg}.l2err.tot = zeros(bins);
    end
end

%-----------------------------------------------------------------
% simulation
%-----------------------------------------------------------------
ws=struct;
ws.meas = zeros(cf.nmodels,cf.m,cf.n);
for irun=1:cf.runs
    %--------------------------------------------------------------
    % generate random matrices
    %--------------------------------------------------------------
    for imodel=1:cf.nmodels
        switch(cf.model{imodel})            
          case 'cnormal',
            ws.meas(imodel,:,:) = (randn(cf.m,cf.n)+1i*randn(cf.m,cf.n))/sqrt(cf.m/2);
          case 'gaussian',
            ws.meas(imodel,:,:) = randn(cf.m,cf.n)/sqrt(cf.m);
          case 'bernoulli',       
            ws.meas(imodel,:,:) = sign(rand(cf.m,cf.n)-.5)/sqrt(cf.m);
          case 'binary'
            ws.meas(imodel,:,:) = randi([0,1],cf.m,cf.n)/sqrt(cf.m/2);
          case 'dct'
            % generate cf.m uniformly random selected rows
            rows=randperm(cf.n,cf.m);
            ws.meas(imodel,:,:)=dctsubmtx(rows-1,0:cf.n-1,cf.n)*sqrt(cf.n)/sqrt(cf.m);          
          case 'dft'
            % generate cf.m uniformly random selected rows
            rows=randperm(cf.n,cf.m);
            W=dftmtx(cf.n);
            ws.meas(imodel,:,:)=W(rows,:)/sqrt(cf.m);                      
        end
        % checking the average of the norms of the columns
        norm(ws.meas(imodel,:,1));
    end
    
        
    % loop over multiple sparsity levels
    for is=1:numel(cf.s)
        ws.s = cf.s(is);
        % generate random ground truth x0
        supp = randperm(cf.n,ws.s);        
        ws.x0 = zeros(cf.n,1);        
        switch(cf.x0model)
          case 'gaussian',
            ws.x0(supp) = randn(size(supp));
          case 'cnormal',
            ws.x0(supp) = (randn(size(supp))+1i*randn(size(supp)))/sqrt(2);
        end
        
        % take measurements
        %ws.y = ws.meas*ws.x0;
        for imodel=1:cf.nmodels; 
            switch(imodel)
% $$$               case 'dft',
% $$$                 y = fft(ws.x0)/sqrt(cf.m);
% $$$                 ws.y(imodel,:) 
              otherwise,
                ws.y(imodel,:) = squeeze(ws.meas(imodel,:,:))*ws.x0;
            end
        end;

        
        % reconstruction using different reconstruction methods        
        for ialg=1:cf.nalgs
            for imodel=1:cf.nmodels                
                switch(cf.alg{ialg}.name)               
                  case 'bp_spgl1',
                    % basis pursuit (BP) using SPGL1-toolnox (https://www.cs.ubc.ca/~mpf/spgl1)    
                    opts    = spgSetParms('verbosity',0,'bpTol',1.e-4);
                    ws.xhat = spg_bp(squeeze(ws.meas(imodel,:,:)), ws.y(imodel,:).', opts);
                  case 'cvx_slasso',
                    % sqrt-lasso via cvx-toolbox
                    A=squeeze(ws.meas(imodel,:,:));
                    b=ws.y(imodel,:).';
                    lambda=cf.alg{ialg}.lambda;
                    cvx_begin quiet
                    %cvx_precision low
                    variable x(cf.n)
                    minimize(norm(A*x - b,2)/sqrt(cf.m) + 2*lambda/cf.m*norm(x,1))
                    cvx_end
                    ws.xhat = x;

                  case 'irls_slasso',
                    
                    A=squeeze(ws.meas(imodel,:,:));
                    b=ws.y(imodel,:).';
                    lambda = cf.alg{ialg}.lambda;
                    s      = ws.s;

                    %epsilon decay rule for ||Ax-b||_2
                    eps1=10;
                    
                    %epsilon decay rule for ||x||_1
                    eps2=10;

                    % eps min in oder to avoid numerical problems
                    epsmin = 10^-10;

                    %outer iterations
                    N = 100;

                    %inner iterations
                    Nlsp=10000;

                    x_init = randn(size(A,2),1);
                    [x,x_track,eps1_track,eps2_track] = ...
                        IRLS_sqrt_LASSO(A,b,s,lambda,eps1,eps2,x_init,N,Nlsp,epsmin,'automatic','KMS');
                    ws.xhat=x;
                end

% $$$                 figure(2);
% $$$                 plot(abs(ws.x0))
% $$$                 hold on;
% $$$                 plot(abs(ws.xhat))
% $$$                 legend('orig','recovered');
% $$$                 hold off;
% $$$                 drawnow;
% $$$                 pause;
                
                % abs. and relative l2-error
                ws.l2err   = norm(ws.x0-ws.xhat,2);
                ws.l2err_r = ws.l2err/norm(ws.x0,2);                
                % histogram l2 errors
                b        = is;
                t        = re.alg{imodel,ialg}.l2err.tot(b);
                re.alg{imodel,ialg}.l2err.avg(b) = (re.alg{imodel,ialg}.l2err.avg(b)*t+ws.l2err_r)/(t+1);
                re.alg{imodel,ialg}.l2err.tot(b) =  re.alg{imodel,ialg}.l2err.tot(b)+1;
            

            end            
        end
    
    end
    
    % plot and save periodically to a mat-file
    fprintf('.');
    if ~mod(irun,5)
        if ~isempty(cf.matfile)
            save(cf.matfile,'cf','re');        
        end
        fprintf('\n');

        % show l2err hist
        figure(1);leg={};
        for imodel=1:cf.nmodels
            for ialg=1:cf.nalgs
                plot(cf.s,re.alg{imodel,ialg}.l2err.avg); hold on;
                leg{end+1}=sprintf('%s (%s)',cf.model{imodel},cf.alg{ialg}.legend);
            end
        end
        hold off;
        xlabel('sparsity');
        ylabel('NMSE');
        grid on;
        legend(leg,'Interpreter', 'none');
        title(sprintf('recovery (m=%d, n=%d)',cf.m,cf.n));
        drawnow;  
        %print -dpdf time.pdf
    end

end
