function [stableProb,potential]=stablePotential(falpha,Aalpha)  %Çóf,gµÄº¯?
global M alpha uniParameter
global arriveRate erlangRate erlangOrder
global I e 

b=zeros(1,M);b=[b,1];   % we use the augment matrix
a=[Aalpha,e];
stableProb=b/a; % compute stationary probability
potential=inv(alpha*I-Aalpha+uniParameter*e*stableProb)*falpha;


