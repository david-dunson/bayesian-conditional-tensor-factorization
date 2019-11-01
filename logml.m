function l=logml(z,Y,M,pM,p)
[z0,m]=unique(sortrows([Y+1 z]),'rows','legacy');
C=tensor(zeros([2 M]),[2 M]);
C(z0)=C(z0)+m-[0;m(1:(end-1))];
Cdata=tenmat(C,1);
l=sum(betaln(Cdata(1,:)+0.5,Cdata(2,:)+0.5)-betaln(0.5,0.5))...
    +(p-sum(M>1))*log(pM(1));
d=length(pM);
for k=2:d
    l=l+sum(M==k)*log(pM(k)/stirling(d,k));
end





