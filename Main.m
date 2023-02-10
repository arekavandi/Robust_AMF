% This code was developed by Aref Miri Rekavandi and is based on the paper titled 
% "Adaptive Matched Filter using Non-Target Free Training Data" by Rekavandi, A. M., Seghouane, A. K., & Evans, R. J. 
% which was published in the ICASSP 2020 IEEE International Conference on Acoustics, Speech, and Signal Processing (pp. 1090-1094). 
% If you utilize this code in your study, kindly reference the aforementioned article.
clc
clear all
close all
%% initialization for non-Gaussian case %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha=0.9;   %Hyperparameter alpha which contrlos the robust covariance estimator
indim=10;    % Signal dimension (N in paper)
K=40;        % Number of seconadary set samples
N=4000;      % Number of test samples to compute Detection and miss rate
t=2;         % Number of colums in target subspace matrix

epsilon=0.15;% Outlier rate
ASNR=25;     % Desired SNR

x=1:indim;
H=zeros(indim,t);
freq=[0.1 0.15 0.2];
greq=[-0.1 -0.08 -0.05];
i=sqrt(-1);
for k=1:t
    for j=1:indim
        H(j,k)=1/sqrt(indim)*exp(-1*i*2*pi*freq(k)*(j-1));
    end
end
C=[H];
cov=zeros(indim,indim);
for i=1:indim
    for j=1:indim
         cov(i,j)=0.9^(abs(i-j));
    end
end
a=random('normal',0,1,indim,N+K);

noise=(cov)^(0.5)*(a);
TETA1=random('uniform',0.3,0.3000000001,t,1);
scale=sqrt(((10^(ASNR/10))/((H*TETA1)'*(cov)^(-1)*(H*TETA1))));
ideal=ones(K,1);
for i=N+1:N+K
    if rand < epsilon
        ideal(i-N)=0;
        TETA1=random('uniform',0.3,0.3000000001,t,1);
        noise(:,i)=scale*H*TETA1+noise(:,i);
    end
end

%%%%%%%%%%% Estimating covariance matrix from secondary data%%%%%%
Rr=robustcov(noise(:,N+1:N+K),alpha,K,indim);
SS=0;
for i=1:K
   SS=SS+noise(:,N+i)*noise(:,N+i)';
end
SC=SS/K;  %Sample Covariance

%%%%%%%%%%%%%%%%% Making observations%%%%%%%%%%%%%%%%%%%%%%%%%%%
mu1=[zeros(1,N/2) ones(1,N/2)];

TETA1=random('uniform',0.3,0.3000000001,t,1);
scale=sqrt(((10^(ASNR/10))/((H*TETA1)'*(cov)^(-1)*(H*TETA1))));

for k=1:N   
    X(:,k)=mu1(k)*H*TETA1;
    Y(:,k)=scale*X(:,k)+noise(:,k);
end

SNR=10*log10((scale*H*TETA1)'*(cov)^(-1)*(scale*H*TETA1))

%% Diffrent Tests with plots %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     Htilda=(SC^(-0.5))*H;    
     PHtilda=((Htilda)*(((Htilda)')*(Htilda))^(-1)*((Htilda)'));
     
     Hbar=(Rr^(-0.5))*H;    
     PHbar=((Hbar)*(((Hbar)')*(Hbar))^(-1)*((Hbar)'));
for i=1:N
 temp=(SC^(-0.5))*Y(:,i);
 
 AMF(i)=temp'*PHtilda*temp;
 
 TS1(i)=temp'*PHtilda*temp;
 TS2(i)=1+temp'*temp;
 GLRT(i)=TS1(i)/TS2(i);
 
 temp=Rr^(-0.5)*Y(:,i);
 Proposed1(i)=temp'*PHbar*temp;

end

figure
subplot(1,3,1)
 scatter(real(GLRT(1:N/2)),imag(GLRT(1:N/2)))
  title('GLRT', 'FontName', 'Times New Roman', ...
        'FontSize',10,'Color','k', 'Interpreter', 'LaTeX')
 hold on
 scatter(real(GLRT(N/2+1:N)),imag(GLRT(N/2+1:N)))
 xlabel(['Real'], 'Interpreter', 'LaTeX')
 ylabel(['Imainary'], 'Interpreter', 'LaTeX')
 subplot(1,3,2)
 scatter(real(AMF(1:N/2)),imag(AMF(1:N/2)))
  title('AMF', 'FontName', 'Times New Roman', ...
        'FontSize',10,'Color','k', 'Interpreter', 'LaTeX')
 hold on
 scatter(real(AMF(N/2+1:N)),imag(AMF(N/2+1:N)))
  xlabel(['Real'], 'Interpreter', 'LaTeX')
 ylabel(['Imainary'], 'Interpreter', 'LaTeX')
 subplot(1,3,3)
 scatter(real(Proposed1(1:N/2)),imag(Proposed1(1:N/2)))
  title('RAMF', 'FontName', 'Times New Roman', ...
        'FontSize',10,'Color','k', 'Interpreter', 'LaTeX')
 hold on
 scatter(real(Proposed1(N/2+1:N)),imag(Proposed1(N/2+1:N)))
  xlabel(['Real'], 'Interpreter', 'LaTeX')
 ylabel(['Imainary'], 'Interpreter', 'LaTeX')

GLRT1=real(GLRT);
Proposed11=real(Proposed1);
AMF1=real(AMF);
%% ROC calculation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
i=1;
last=max([max(GLRT1),max(AMF1),max(Proposed11)]);
starting=min([min(GLRT1),min(AMF1),min(Proposed11)]);
step=(last-starting)/10000;
for th=starting:step:last
    pd1(i)=(200*(sum(GLRT1(N/2+1:N)>th)))/N;
    pf1(i)=(200*(sum(GLRT1(1:N/2)>th)))/N;
    
    pd2(i)=(200*(sum(AMF1(N/2+1:N)>th)))/N;
    pf2(i)=(200*(sum(AMF1(1:N/2)>th)))/N;  
    
    pd3(i)=(200*(sum(Proposed11(N/2+1:N)>th)))/N;
    pf3(i)=(200*(sum(Proposed11(1:N/2)>th)))/N; 
    
    i=i+1;
end

%% ROC plots %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
plot(pf1,pd1,'b','LineWidth',2)
hold on
plot(pf2,pd2,'r','LineWidth',2)
hold on
plot(pf3,pd3,'g','LineWidth',2)
hold on
grid on

    legend({'GLRT','AMF','RAMF'}, ...
        'Interpreter', 'LaTeX')
    xlabel('Probability of False Alarm (\%)', 'Interpreter', 'LaTeX')
    ylabel('Probability of Detection (\%)', 'Interpreter', 'LaTeX')
    title('', 'FontName', 'Times New Roman', ...
        'FontSize',10,'Color','k', 'Interpreter', 'LaTeX')