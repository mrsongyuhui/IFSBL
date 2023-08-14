% clc;clear all;load('demo.mat');
% IFTSBLNEW(Y, Phi, Wgen);
%% Discription : 
  % This code is for the paper 'Fast Variational Bayesian Inference for
  % Temporally Correlated Sparse Signal Recovery'. 

  
  % The proposed algorithm can achieve higher performance and more
  % robustness with line 88;
  
  % Otherwise please normalize 'B' (line 98-100).

  %  Written by Zheng Cao

  
  function [X]=IFSBLB(Y,A,Wgen,lambda_opt)


[M,T]=size(Y);
[~,P]=size(A);
F=A;
K=T;

%% initialization
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
munew=zeros(P,K);
keep_list = [1:P]';
iter = 0;
Phi=F;


 maxeig = max(eig(Phi*Phi'));
converged = false;
while ~converged
    
    iter = iter + 1;
   
 %% calculate mu and Sigma
 old_mu = mu;

 PhiTwo = Phi'*Phi;
 diagPhiTwo = diag(PhiTwo);
 %Phi_delta = Phi *  diag(delta_inv);
 %   Exx=[]; 
 for k= 1:K
     B_k=B(k,:);
     %V_temp= 1/(alpha*(B_k*B_k'))*eye(M) + Phi_delta * Phi';
     %Sigma=diag(delta_inv) -Phi_delta' * (V_temp \Phi_delta);
     temp=zeros(length(delta_inv),1);
     for i=1:K
         if i~=k
             temp=temp+z(:,i)*B(i,:)*B_k';
         end
     end
     W1=Y*B_k'-Phi*temp;
     %mu(:,k)=alpha*Sigma*Phi'*W1;
     %Exx(:,k)= mu(:,k).*conj(mu(:,k))+ real(diag(Sigma ));
     
     
     
     % Inverse free
     L=B_k*B_k'*maxeig+1e-10;
     %mu(:,k)=alpha*inv(L*alpha*eye(P)+diag(1./delta_inv))*(L*mu(:,k)+Phi'*W1-B_k*B_k'*Phi'*Phi*mu(:,k));
     mu(:,k)=alpha*(L*z(:,k)+Phi'*W1-B_k*B_k'*PhiTwo*z(:,k))./(L*alpha+1./delta_inv);
     
     % factorization
     %factorized_Sigma = 1./(alpha*B_k*B_k'*diagPhiTwo+1./delta_inv);
     Exx(:,k)= mu(:,k).*conj(mu(:,k))+1./(L*alpha+1./delta_inv);
     
     
 end
 
 z=mu;
        
     
  %%  update alpha
  resid=Y-Phi*mu*B;
  %   traceAGA1=[];traceAGA=[]; 
  
%     for j=1:K
%       PGP=diag(Phi*diag(delta_inv)*Phi');
%       bkbk=B(j,:)*B(j,:)';
%       traceAGA1(:,j)=sum( PGP./(  1 +  alpha*bkbk *PGP  )   );
%       traceAGA(:,j)= traceAGA1(:,j)  * bkbk ;
%   end


  for j=1:K
      %PGP=diag(Phi*diag(delta_inv)*Phi'); diag(diag(delta_inv)*Phi'*Phi);
      bkbk=B(j,:)*B(j,:)';
      L=bkbk*maxeig+1e-10;
      factorized_Sigma = 1./(L*alpha+1./delta_inv);
      traceAGA1(:,j)=sum( factorized_Sigma.*diagPhiTwo   );
      traceAGA(:,j)= traceAGA1(:,j)  * bkbk ;
  end
  
  b_k=b+ 0.5* ( norm(resid,'fro')^2 + sum(traceAGA,2) );
  a_k=a+ (M*T)/2;
  alpha=a_k/b_k;
    

    
    
  %% update delta
  delta_last=delta_inv;
  sum_temp1=sum(Exx,2);
  c_k=K/2+a;
  d_k=b+0.5*sum_temp1;
  delta_inv=d_k ./ c_k;
    

  %%  update B
  old_B = B;
  %   SA=[];
  for kk=1:K
      SA(:,kk)=Phi*mu(:,kk);
      p1=  SA(:,kk)'*SA(:,kk) + traceAGA1(:,kk);            %0*(2*mu(:,kk)-z(:,kk))'*PhiTwo*z(:,kk) + 0*bkbk*maxeig*norm(mu(:,kk)-z(:,kk))^2;
      temp11=zeros(length(delta_inv),1);
      for ii=1:K
          if ii~=kk
              temp11=temp11+mu(:,ii)*B(ii,:);
          end
      end
      W2=Y-Phi*temp11;
      p2=SA(:,kk)'*W2;
      B(kk,:) = p2/ p1;
%     if norm(B(kk,:))<1
%     B(kk,:)=B(kk,:)/norm(B(kk,:));
%     end
  end
%  [~,f]=chol(B'*B); 
%  if f 
%      pause(); 
%  end
% B = B/norm(B,'fro');
% B = B+max(max(abs(B)))*eye(T);
%B=eye(T);
%  [~,f]=chol(B'*B); 
%  if f 
%      pause(); 
%  end
   
  
  
   %%  Set threshold and prune out the Values less than 10 ^ (- 3) in Delta_inv
  

    
    erro=  max(max(abs(mu*B - old_mu*old_B)));  
    %erro=  max(max(abs(delta_inv - delta_last)));  
    if erro < tol || iter >= maxiter
        converged = true;
    end
    %r1(iter)=erro;
%     r2(iter)=norm(Wgen-mu*B,'fro')^2/norm(Wgen,'fro')^2;
% %     nmse(iter) = norm(mu*B-Wgen,'F')/norm(Wgen,'F');
%     
     if iter>0 
    if min(delta_inv) < 10^(-7) % &  length(delta_inv)>= M
        index=find( delta_inv>10^(-7) );
        delta_inv= delta_inv(index);
        delta_last=delta_last(index);
%         %Sigma = Sigma(index,index,:);
        Phi=Phi(:,index);
        mu=mu(index,:);
        Exx=Exx(index,:);
        keep_list = keep_list(index);
        z=z(index,:);
    end
 end

end
% plot(r2)
% iter
mu_est = zeros(P,K);
mu_est(keep_list,:) = mu;
X=mu_est*B;%semilogy(r1);
% subplot(1,2,1);%subplot(1,2,2);plot(r2);
  end

