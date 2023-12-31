function [X]=IFSBLB(Y,A)
[M,T]=size(Y);
[~,P]=size(A);
F=A;
K=T;
%% Initialization
a=10^(-10);
b=a;
maxiter=1000;
tol=1e-8;
alpha=1;
delta_inv=ones(P,1)*1;
[~,~,V]=svd(Y);
B=V(:,1:K)';
mu=zeros(P,K);
z=mu;
keep_list = [1:P]';
iter = 0;
Phi=F;
maxeig = max(eig(Phi*Phi'));
converged = false;
while ~converged
    iter = iter + 1;
    %% Calculate mu and sigma
    old_mu = mu;
    PhiTwo = Phi'*Phi;
    diagPhiTwo = diag(PhiTwo);
    for k= 1:K
        B_k=B(k,:);
        temp=zeros(length(delta_inv),1);
        for i=1:K
            if i~=k
                temp=temp+z(:,i)*B(i,:)*B_k';
            end
        end
        W1=Y*B_k'-Phi*temp;
        % Inversion free
        L=B_k*B_k'*maxeig+1e-10;
        mu(:,k)=alpha*(L*z(:,k)+Phi'*W1-B_k*B_k'*PhiTwo*z(:,k))./(L*alpha+1./delta_inv);
        Exx(:,k)= mu(:,k).*conj(mu(:,k))+1./(L*alpha+1./delta_inv);
    end
    z=mu;
    %% Update alpha
    resid=Y-Phi*mu*B;
    for j=1:K
        bkbk=B(j,:)*B(j,:)';
        L=bkbk*maxeig+1e-10;
        factorized_Sigma = 1./(L*alpha+1./delta_inv);
        traceAGA1(:,j)=sum( factorized_Sigma.*diagPhiTwo   );
        traceAGA(:,j)= traceAGA1(:,j)  * bkbk ;
    end
    b_k=b+ 0.5* ( norm(resid,'fro')^2 + sum(traceAGA,2) );
    a_k=a+ (M*T)/2;
    alpha=a_k/b_k;
    %% Update delta
    sum_temp1=sum(Exx,2);
    c_k=K/2+a;
    d_k=b+0.5*sum_temp1;
    delta_inv=d_k ./ c_k;
    %% Update B
    old_B = B;
    for kk=1:K
        SA(:,kk)=Phi*mu(:,kk);
        p1=  SA(:,kk)'*SA(:,kk) + traceAGA1(:,kk);
        temp11=zeros(length(delta_inv),1);
        for ii=1:K
            if ii~=kk
                temp11=temp11+mu(:,ii)*B(ii,:);
            end
        end
        W2=Y-Phi*temp11;
        p2=SA(:,kk)'*W2;
        B(kk,:) = p2/ p1;
    end
    %%  Prune
    erro=  max(max(abs(mu*B - old_mu*old_B)));
    if erro < tol || iter >= maxiter
        converged = true;
    end
    if min(delta_inv) < 10^(-7)
        index=find( delta_inv>10^(-7) );
        delta_inv= delta_inv(index);
        Phi=Phi(:,index);
        mu=mu(index,:);
        Exx=Exx(index,:);
        keep_list = keep_list(index);
        z=z(index,:);
    end
end
mu_est = zeros(P,K);
mu_est(keep_list,:) = mu;
X = mu_est*B;
end