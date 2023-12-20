% load data 
rng(123);

Train = readtable('data_set_ALL_AML_train.csv','Delimiter',',','ReadRowNames',true,'ReadVariableNames', true);
Test = readtable('data_set_ALL_AML_independent.csv','Delimiter',',','ReadRowNames',true,'ReadVariableNames', true);
Labels = readtable('actual.csv','Delimiter',',','ReadVariableNames', true); 

list_train = {};

for ni = 1:length(Train.Properties.VariableNames)
    name = Train.Properties.VariableNames(ni);
    name = name{1};
    k = strfind(name,'x');
    if ~isempty(k) 
        list_train = [list_train, name];
    end
end

list_test = {};

for ni = 1:length(Test.Properties.VariableNames)
    name = Test.Properties.VariableNames(ni);
    name = name{1};
    k = strfind(name,'x');
    if ~isempty(k) 
        list_test = [list_test, name];
    end
end

m1 = length(list_train);
m2 = length(list_test);
m = m1+m2;
n = size(Train,1);

lab = strcmp(Labels.cancer,'ALL');

% prepare A and y

A = zeros(m,n);
b = zeros(m,1);

for i=1:m1
    name = list_train{i};
    A(i,:) = Train.(name);
    k = strfind(name,'x');
    num = extractAfter(name,k);
    num = round(str2double(num));
    b(i) = lab(num); 
end

for i=1:m2
    name = list_test{i};
    A(i + m1,:) = Test.(name);
    num = extractAfter(name,k);
    num = round(str2double(num));
    b(i + m1) = lab(num);
end

b = -(b-1);

for i=1:n
    A(:,n) = A(:,n) - mean(A(:,n));
    A(:,n) = A(:,n)/norm(A(:,n)); 
end

x0 = randn(n,1);
lambda_opt = sqrt( log(n)/m); %s*sqrt( log(n)/m);
lambda = lambda_opt/1.5;

eps0 = 10^-6;
eps2 = 10^-8;
eps1 = lambda*10^-8;
Lmax = norm(A,2);
L = lambda/ sqrt(eps2) + 1.0/sqrt(eps1);
mu=0;

timeout = 10;
threshold = 10^-12;
threshold_ITEM = 10^-12;
sp_tr = 0.025;

% [x_1, t_1] = ITEM_sqrtlasso_to_rie(A,b,lambda,mu,L,eps1,eps2,x0,1000000, x, threshold_ITEM, timeout);
% fprintf('ITEM: RME: %.6f TIME: %.2f Sparsity %.02f Stab. Sparsity: %.1f \n', ...
%     norm(A*x_1 - b)/norm(b), ...
%     t_1, ...
%     sparsity(x_1,sp_tr), ...
%     norm(x_1)^2/norm(x_1,'inf')^2)
% 
% [x_2, t_2] = proximal_gradient_to_rie(A,b,0.0,lambda,L,x0,1000000, x, threshold, timeout);
% fprintf('Prox. Grad.: RME: %.6f TIME: %.2f Sparsity %.02f Stab. Sparsity: %.1f \n', ...
%     norm(A*x_2 - b)/norm(b), ...
%     t_2, ...
%     sparsity(x_2,sp_tr), ...
%     norm(x_2)^2/norm(x_2,'inf')^2)

% [x_3, t_3] = proximal_newton_to_rie(A,b,0.0,lambda,x0,1000,1000, x, threshold, timeout);
% fprintf('Prox. Newton: RME: %.6f TIME: %.2f Sparsity %.02f Stab. Sparsity: %.1f \n', 
% norm(A*x_3 - b)/norm(b), 
% t_3, sparsity(x_3,sp_tr), 
% norm(x_3)^2/norm(x_3,'inf'))

[x_4, t_4] = smooth_concomitant_lasso_v2_to_rie(A, b, 10000, eps1, lambda, x0, x, threshold, timeout);
[s_4, ent_4] = sparsity(x_4,sp_tr);
A_4 = A(:,ent_4);
x_tr_4 = lsqr(A_4,b);
fprintf('Concomitant: RME: %.6f TIME: %.2f Sparsity %.02f Stab. Sparsity: %.1f RME TR: %.6f \n', ...
    norm(A*x_4 - b)/norm(b), ...
    t_4, ...
    s_4, ...
    norm(x_4)^2/norm(x_4,'inf')^2, ...
    norm(A_4*x_tr_4 - b)/norm(b))

[x_5, t_5] = IRLS_eps_decay_to_rie(A,b,lambda,eps0,x0,10000,1000,'sqrt',s,'lsqr_fun', x, threshold, timeout);
[s_5, ent_5] = sparsity(x_5,sp_tr);
A_5 = A(:,ent_5);
x_tr_5 = lsqr(A_5,b);
fprintf('IRLS(sqrt): RME: %.6f TIME: %.2f Sparsity %.02f Stab. Sparsity: %.1f RME TR: %.6f \n', ...
    norm(A*x_5 - b)/norm(b), ...
    t_5, ...
    s_5, ...
    norm(x_5)^2/norm(x_5,'inf')^2, ...
    norm(A_5*x_tr_5 - b)/norm(b))

[x_6, t_6] = IRLS_eps_decay_to_rie(A,b,lambda,0.5,x0,10000,1000,'sigma',s,'lsqr_fun', x, threshold, timeout);
[s_6, ent_6] = sparsity(x_6,sp_tr);
A_6 = A(:,ent_6);
x_tr_6 = lsqr(A_6,b);
fprintf('IRLS(sigma): RME: %.6f TIME: %.2f Sparsity %.02f Stab. Sparsity: %.1f RME TR: %.6f \n', ...
    norm(A*x_6 - b)/norm(b), ...
    t_6, ...
    s_6, ...
    norm(x_6)^2/norm(x_6,'inf')^2, ...
    norm(A_6*x_tr_6 - b)/norm(b))

[x_7, t_7] = IRLS_eps_decay_restart_to_rie(A,b,lambda,eps0,x0,1000,10,'sqrt',s,'lsqr_fun', x, threshold, timeout);
[s_7, ent_7] = sparsity(x_7,sp_tr);
A_7 = A(:,ent_7);
x_tr_7 = lsqr(A_7,b);
fprintf('IRSL+restart(10): RME: %.6f TIME: %.2f Sparsity %.02f Stab. Sparsity: %.1f RME TR: %.6f \n', ...
    norm(A*x_7 - b)/norm(b), ...
    t_7, ...
    s_7, ...
    norm(x_7)^2/norm(x_7,'inf')^2, ...
    norm(A_7*x_tr_7 - b)/norm(b))

[x_8, t_8] = IRLS_eps_decay_restart_to_rie(A,b,lambda,eps0,x0,1000,1000,'sqrt',s,'lsqr_fun', x, threshold, timeout);
[s_8, ent_8] = sparsity(x_8,sp_tr);
A_8 = A(:,ent_8);
x_tr_8 = lsqr(A_8,b);
fprintf('IRLS+restart(1000): RME: %.6f TIME: %.2f Sparsity %.02f Stab. Sparsity: %.1f RME TR: %.6f \n', ...
    norm(A*x_8 - b)/norm(b), ...
    t_8, ...
    s_8, ...
    norm(x_8)^2/norm(x_8,'inf')^2, ...
    norm(A_8*x_tr_8 - b)/norm(b))

function [s, entries] = sparsity(x, threshold)
t = max(abs(x));
entries = x > t*threshold;
s = sum(x > t*threshold);
end