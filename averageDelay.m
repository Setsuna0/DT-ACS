function delayTime=averageDelay(embedP,hAlpha,policyV)
global M alpha 
global arriveRate erlangRate erlangOrder 
global I e
% the following is first to compute the average dalay time at every state
stateDelay=zeros(M,1);
stateDelay(1)=0;
stateDelay(M)=inv(arriveRate);
if (alpha~=0)
    for i=2:M-1
        replace=0;
        for j=0:erlangOrder-1
            replace=replace+(erlangOrder-j)*erlangRate^(j-1)*policyV(i)^j/factorial(j); 
            % the sum of the order items as alpha=0
        end
        hAlpha(i)=(1-exp(-arriveRate*policyV(i)))/arriveRate+exp(-(arriveRate+erlangRate)*policyV(i))*replace;
    end
end
for i=2:M-1
    stateDelay(i)=hAlpha(i)-erlangOrder*exp(-arriveRate*policyV(i))/erlangRate;
end
b=zeros(1,M);b=[b,1];   % we use the augment matrix
a=[embedP-I,e];
stableProb=b/a;      % compute stationary probability of the embedded chain
delayTime=stableProb*stateDelay;       % delay time every step
    
    
    