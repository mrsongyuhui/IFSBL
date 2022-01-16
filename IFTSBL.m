function [X] = IFTSBL(Y, D)
% Inversion-Free Sparse Bayesian Learning for Temporally Correlated Signal Recovery
%
% ============= Author =============
%   Yuhui Song (yuhuis@mun.ca)
%
% ============= INPUT ARGUMENTS ====================
%   Y      : M*T measurement matrix
%   D      : M*N dictionary matrix

% ============= OUTPUT ARGUMENTS ===================
%   X      : N*T estimated coefficient matrix

% Data dimension
[M,N] = size(D);
[~,T] = size(Y);
K = T;
% Initialization
n_precision = 1/eps;
x_prior_precision = ones(N,1);
x_mean = zeros(N,K);
Z = zeros(N,K);
[~,~,V] = svd(Y);
C = V(:,1:K);
% Pre-calculation
singular = svd(D,'econ');
max_eigenvalue = singular(1)^2;
DD = D'*D;
diag_DD = diag(DD);
% Hyperparameters
Times = 2000;
epsilon = 1e-4;
threshold = 1e10;
keep_list = linspace(1,N,N);
t = 0;

while(1)
    t = t + 1;
    
    % Prune components
    index = find(x_prior_precision>threshold,1);
    if ~isempty(index)
        index = find(x_prior_precision<threshold);
        x_mean = x_mean(index,:);
        x_covariance = x_covariance(index,:);
        x_prior_precision = x_prior_precision(index,:);
        Z = Z(index,:);
        D = D(:,index);
        DD = D'*D;
        diag_DD = diag(DD);
        keep_list = keep_list(index);
    end
    
    % Update Lipschitz constant
    Lipschitz = 2*diag(C'*C)*max_eigenvalue + 1e-10;
    
    % Update x covariance matrix
    x_mean_old = x_mean;
    x_covariance = 1./(Lipschitz'*n_precision/2+x_prior_precision);
    
    % Update x mean
    residual_error = Y-D*Z*C';
    for k = 1:K
        x_mean(:,k) = n_precision*x_covariance(:,k).*...
            (-DD*(C(:,k)'*C(:,k))*Z(:,k)+D'*(residual_error+D*Z(:,k)*C(:,k)')*C(:,k)+Lipschitz(k)*Z(:,k)/2);
    end
    
    % Update x precision
    x_prior_precision = K./sum(x_mean.*x_mean + x_covariance,2);
    
    % Update Z
    Z = x_mean;
    
    % Update noise precision
    n_precision = M*T/(norm(Y-D*x_mean*C','fro')^2+...
        sum(diag(C'*C)'.*sum(repmat(diag_DD,1,K).*x_covariance,1)));
    
    % Update C
    C_old = C;
    for k = 1:K
        w = Y-D*(Z*C'-Z(:,k)*C(:,k)');
        top = Z(:,k)'*D'*w ;
        bottom = Z(:,k)'*DD*Z(:,k) + sum(diag(DD).*x_covariance(:,k));
        C(:,k) = top'/bottom;
    end
    
    % Stop criterion
    d_x_mean = max(max(abs(x_mean_old*C_old'-x_mean*C')));
    if d_x_mean < epsilon || t > Times % control the number of iterations
        break;
    end
    
end
% Recovery X
X = zeros(N,T);
X(keep_list,:) = x_mean*C';
end