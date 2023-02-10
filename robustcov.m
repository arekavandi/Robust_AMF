function Rhat=robustcov(X,alpha,n,p)
it=5;
w=ones(n,1);
for j=1:it
    temp2=zeros(p,p);
    for i=1:n
        temp2=temp2+w(i)*((X(:,i))*(X(:,i))');
    end
    cov=(temp2)/(sum(w));
    for i=1:n
        w(i)=exp(-((1-alpha)/(1))*(((X(:,i))')*((cov)^(-1))*(X(:,i))));
    end
end
Rhat=cov;
end