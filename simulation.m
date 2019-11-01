% -- tensor factorization method -- %

% -- Yun Yang -- %
% last modified: Oct, 10, 2012 -- %

clear;clc;close all;


%constants
p=200; %number of features
N=4000; %total size
n=600; %training size
ep=max(6,floor(log(n)/log(4))); %expected maximum number of predictors
d=4; %number of categories for each features
d0=1; %prior for Dirichlet Distribution
c=0.3;
pM=[1-3*c*ep/p,c*ep/p,c*ep/p,c*ep/p]; %prior probability for kj 
np=0; %number of predictors included in the model

% generate data
x=zeros(N,p);
for i=1:p
    x(:,i)=randsample(d,N,true);
end
% y=(rand(N,1)<(1./(1+(exp(2*(-2*(x(:,9)==1)+(x(:,9)==2)-0.5*(x(:,9)==5)-...
%     (x(:,11)==3)+2*(x(:,11)==5)+2*(x(:,13)==1)-...
%     1.5*(x(:,15)==4)-1.5*(x(:,17)==4)+2*(x(:,19)==3)))))));
A0=rand(4,4,4);
A=tensor(A0.^2./(A0.^2+(1-A0).^2));
y=(rand(N,1)<A(x(:,[9,11,13])));

train=randsample(N,n);
X0=x;
Y0=y;
x=x(train,:);
Y=y(train);

MS=300; %number of iterations for first stage
cM=(2.^(1:d)-2)/2;
M=ones(MS+1,p);G=ones(p,d); z=ones(n,p);log0=zeros(MS,1);
for k=1:MS     
    M00=M(k,:);
    for j=1:p
        M0=M00(j);
        if M0==1
          if np<ep
            new=binornd(1,0.5*ones(1,d-1));
            while sum(new)==0
                new=binornd(1,0.5*ones(1,d-1));
            end
            GG=G(j,:)+[0 new];
            MM=M00;
            MM0=MM;
            MM(j)=2;
            zz=z;
            zz(:,j)=GG(x(:,j));
            ind1=find(MM>1);
            ind2=find(MM0>1);
            if isempty(ind2)
                ind2=1;
            end
            logR=logml(zz(:,ind1),Y,MM(ind1),pM,p)-logml(z(:,ind2),Y,MM0(ind2),pM,p);
            logR=logR+log(0.5)+log(cM(d));
            if log(rand)<logR&&np<ep
                G(j,:)=GG;
                M00=MM;
                z=zz;
                np=np+1;
            else
                M00=MM0;
            end    
          end
          if M00(j)==1&&np>0
              ind1=find(M00>1); %switch-random select a feature to be replaced
              tempind=randsample(length(ind1),1);  
              temp=ind1(tempind);
              zz=z;       
              zz(:,temp)=ones(n,1);
              MM=M00;
              MM0=MM;
              MM(temp)=1;
              GG=G(temp,:);   
              per=randsample(d,d);
              GG0=GG;
              for s=1:d
                  GG0(s)=GG(per(s));
              end
              GG=GG0;
              MM(j)=MM0(temp);
              zz(:,j)=GG(x(:,j));
              ind1=find(MM>1);
              ind2=find(MM0>1);
              logR=logml(zz(:,ind1),Y,MM(ind1),pM,p)-logml(z(:,ind2),Y,MM0(ind2),pM,p);
              if log(rand)<logR
                G(j,:)=GG;
                G(temp,:)=ones(1,d);
                M00=MM;
                z=zz;
              end
          end
        end
        if M0>1&&M0<d
            if rand<0.5
                cnew=randsample(M00(j),2);
                lnew=max(cnew);
                snew=min(cnew);
                GG=G(j,:);
                GG(GG==lnew)=snew;
                if lnew<M00(j)
                    GG(GG==M00(j))=lnew;
                end
                zz=z;
                zz(:,j)=GG(x(:,j));
                MM=M00;
                MM0=MM;
                MM(j)=M00(j)-1;
                ind1=find(MM>1);
                ind2=find(MM0>1);
                if isempty(ind1)
                    ind1=1;
                end
                if isempty(ind2)
                    ind2=1;
                end
                logR=logml(zz(:,ind1),Y,MM(ind1),pM,p)-logml(z(:,ind2),Y,MM0(ind2),pM,p);
                if M00(j)>2
                    [z0,mm]=unique(sort(GG),'legacy');
                    gn=mm-[0 mm(1:(end-1))];
                    logR=logR-log(sum(cM(gn)))+log(M00(j)*(M00(j)-1)/2);
                else
                    logR=logR-log(cM(d))-log(0.5); 
                end
                if log(rand)<logR
                    G(j,:)=GG;
                    M00=MM;
                    z=zz;
                    if M00(j)==1
                        np=np-1;
                    end
                else
                    M00=MM0;
                end
            else
                [z0,mm]=unique(sort(G(j,:)),'legacy');
                gn=mm-[0 mm(1:(end-1))];
                pgn=cM(gn)/sum(cM(gn));
                l=sum(mnrnd(1,pgn).*(1:M00(j)));
                new=binornd(1,0.5*ones(1,gn(l)-1));
                while sum(new)==0
                    new=binornd(1,0.5*ones(1,gn(l)-1));
                end
                GG=G(j,:);
                GG(GG==l)=l+(M00(j)+1-l)*[0 new];
                zz=z;
                zz(:,j)=GG(x(:,j));
                MM=M00;
                MM0=MM;
                MM(j)=M00(j)+1;
                ind1=find(MM>1);
                ind2=find(MM0>1);
                if isempty(ind2)
                    ind2=1;
                end                
                logR=logml(zz(:,ind1),Y,MM(ind1),pM,p)-logml(z(:,ind2),Y,MM0(ind2),pM,p);
                if M00(j)<d-1
                    logR=logR-log(M00(j)*(M00(j)+1)/2)+log(sum(cM(gn)));
                else
                    logR=logR-log(d*(d-1)/2)-log(0.5);
                end
                if log(rand)<logR
                    G(j,:)=GG;
                    M00=MM;
                    z=zz;
                else
                    M00=MM0; 
                end     
            end
        end
        if M0==d
            cnew=randsample(d,2);
            lnew=max(cnew);
            snew=min(cnew);
            GG=G(j,:);
            GG(GG==lnew)=snew;
            if lnew<d
                GG(GG==M00(j))=lnew;
            end
            zz=z;
            zz(:,j)=GG(x(:,j));
            MM=M00;
            MM0=MM;
            MM(j)=d-1;
            ind1=find(MM>1);
            ind2=find(MM0>1);
            if isempty(ind2)
               ind2=1;
            end            
            logR=logml(zz(:,ind1),Y,MM(ind1),pM,p)-logml(z(:,ind2),Y,MM0(ind2),pM,p);
            logR=logR+log(0.5)+log(d*(d-1)/2);
            if log(rand)<logR
                G(j,:)=GG;
                M00=MM;
                z=zz;
            else
                M00=MM0;
            end   
        end 
        if M00(j)>1  %resplitting-change the splitting scheme
            zz=z;                  
            GG=G(j,:);   
            per=randsample(d,d);
            GG0=GG;
            for s=1:d
                GG0(s)=GG(per(s));
            end
            GG=GG0;
            zz(:,j)=GG(x(:,j));
            ind1=find(M00>1);
            ind2=find(M00>1);
            logR=logml(zz(:,ind1),Y,M00(ind1),pM,p)-logml(z(:,ind2),Y,M00(ind2),pM,p);
            if log(rand)<logR
              G(j,:)=GG;
              z=zz;
            end
        end
    end
    M(k+1,:)=M00;
    % print informations in each iteration
    ind1=find(M(k+1,:)>1);
    if isempty(ind1)
        ind1=1;
    end   
    log0(k)=logml(z(:,ind1),Y,M(k+1,ind1),pM,p);
    [aa,b]=find(M(k+1,:)-1);
    fprintf('k = %i, %i important predictors = {',k,np);
    for i=1:length(b)
        fprintf(' %i(%i)',b(i),M(k+1,b(i)));
    end
    fprintf(' }. %f \n',log0(k));        
end

K=1000; %number of iterations for second stage

aveM=mean(M(floor(MS/2):MS,:),1);
% [a,b]=max(log0);
MM0=round(aveM);
% MM0=M(b,:);
if sum(MM0)==p
    [m I]=max(mean(M,1));
    MM0(I)=2;
end
ind=find(MM0>1);
M0=MM0(ind);
p0=length(ind);
z=ones(n,p0);
M=ones(K+1,1)*M0;
for j=1:p0
    z(:,j)=randsample(M(1,j),n,true);
end
x0=x(:,ind);

pi=zeros(p0,d,d,K);
PP=zeros(K,d^6);
%Gibbs sampler
for k=1:K
    %pi 
    cp=zeros(p0,d,d);%counts for pi,first=j,second=value of x,third=value of z
    for i=1:n
        for j=1:p0
            cp(j,x0(i,j),z(i,j))=cp(j,x0(i,j),z(i,j))+1;
        end
    end
    for j=1:p0
        for s=1:d
            r = gamrnd(cp(j,s,1:M(k,j))+1/M(k,j)/d0,1);
            pi(j,s,1:M(k,j),k)=r/sum(r);
        end
        %switch label
        [qq1 qq2]=sort(sum(reshape(pi(j,:,1:M(k,j),k),d,M(k,j)),1),'descend');
        for s=1:d
            pi(j,s,1:M(k,j),k)=pi(j,s,qq2,k);
        end
        for i=1:n
            z(i,j)=find(qq2==z(i,j));
        end
    end
   
    %lambda
    clT=tensor(zeros([2,M(k,:)]),[2,M(k,:)]);
    [z0,m]=unique(sortrows([Y+1 z]),'rows','legacy');
    clT(z0)=clT(z0)+m-[0;m(1:(end-1))];
    clTdata=tenmat(clT,1);
    cl0data=clTdata(1,:);
    cl0rdim=[];
    cl0cdim=1:p0;
    cl0tsize=M(k,:);
    cl02=tenmat(cl0data,cl0rdim,cl0cdim,cl0tsize);
    cl0=tensor(cl02);
    cl1data=clTdata(2,:);
    cl1rdim=[];
    cl1cdim=1:p0;
    cl1tsize=M(k,:);
    cl12=tenmat(cl1data,cl1rdim,cl1cdim,cl1tsize);
    cl1=tensor(cl12);
    a=tenmat(cl1,[],'t');
    b=tenmat(cl0,[],'t');
    lambda=tensor(betarnd(a(1:end)+0.5,b(1:end)+0.5),M(k,:));

    %z
    for j=1:p0
        q=zeros(n,M(k+1,j));%for compute p(z=h|-)
        for h=1:M(k+1,j)
            q(:,h)=pi(j,x0(:,j),h,k).*(Y.*reshape(double(lambda([z(:,1:(j-1)),h*ones(n,1),z(:,(j+1):p0)])),n,1)+...
                (1-Y).*(1-reshape(double(lambda([z(:,1:(j-1)),h*ones(n,1),z(:,(j+1):p0)])),n,1)))';
        end
        q=bsxfun(@rdivide,q,(sum(q,2)));
        z(:,j)=sum(bsxfun(@times,mnrnd(1,q),1:M(k+1,j)),2);
    end

    % calculate conditional probability tensor PP
    ep0=6;
    id1=ind;
    id3=(p0+1)*ones(1,ep0);
    id3(1:length(id1))=1:p0;
    MM=M(k+1,:); MM(p0+1)=1; MM=MM(id3);
    pp=pi(:,:,:,k); pp(p0+1,:,:)=[ones(d,1) zeros(d,d-1)]; pp=pp(id3,:,:);
    P=tensor(ttensor(tensor(double(lambda),MM),{reshape(pp(1,:,1:MM(1)),d,MM(1)),...
        reshape(pp(2,:,1:MM(2)),d,MM(2)),reshape(pp(3,:,1:MM(3)),d,MM(3)),reshape(pp(4,:,1:MM(4)),d,MM(4)),...
        reshape(pp(5,:,1:MM(5)),d,MM(5)),reshape(pp(6,:,1:MM(6)),d,MM(6))}));
    S=tenmat(P,[]);
    PP(k,:)=S.data;
    % print informations in each iteration
    fprintf('k = %i.\n',k);
end

PP0=mean(PP(floor(K/2):K,:),1);
PPE=tensor(reshape(PP0,d,d,d,d,d,d));
Ypred=zeros(N-n,1);
indpred=setdiff(1:N,train);
indr=setdiff(1:p,ind);
if length(ind)<6
    ind=[ind indr(1:(6-length(ind)))];
end
for i=1:(N-n)
    Ypred(i)=(PPE(X0(indpred(i),ind))>0.5);
end
mean(abs(Y0(indpred)-Ypred)) %testing error by our model
model = classRF_train(x,Y+1);
Y_hat = classRF_predict(X0(indpred,:),model);
mean(Y_hat~=(Y0(indpred)+1)) %testing error by random forests


