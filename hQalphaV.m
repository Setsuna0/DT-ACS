function [hAlpha,Qalpha]=hQalphaV(policyV)
global M alpha uniParameter
global arriveRate erlangRate erlangOrder
global I e 


hAlpha=zeros(M,1);
if (alpha==0)
    hAlpha(1)=erlangOrder/erlangRate;
    % hAlpha(M)=1/arriveRate;
else
    hAlpha(1)=(1-erlangRate^erlangOrder/(alpha+erlangRate)^erlangOrder)/alpha;
    % hAlpha(M)=1/(alpha+arriveRate);
end
hAlpha(M)=1/(alpha+arriveRate);

Qalpha=zeros(M,M);
Qalpha(1,2)=erlangRate^erlangOrder/(alpha+erlangRate)^erlangOrder;
Qalpha(M,M-1)=arriveRate/(alpha+arriveRate);

for i=2:M-1 % note that for policy, only N-1 actions to be determined,that is 1,2,....,M-2
    sumAlpha0=0;
    sumAlpha=0;
    for j=0:erlangOrder-1
        sumAlpha0=sumAlpha0+erlangRate^(erlangOrder)*policyV(i)^j/erlangRate^(erlangOrder-j)/factorial(j);
        sumAlpha=sumAlpha+erlangRate^(erlangOrder)*policyV(i)^j/(alpha+erlangRate)^(erlangOrder-j)/factorial(j);
    end
    if (alpha==0)
        replace=0;
        for j=0:erlangOrder-1
            replace=replace+(erlangOrder-j)*erlangRate^(j-1)*policyV(i)^j/factorial(j); % the sum of the order items as alpha=0
        end
    else
        replace=(sumAlpha0-sumAlpha)/alpha; % the sum of the order items as alpha!=0
    end
    hAlpha(i)=(1-exp(-(alpha+arriveRate)*policyV(i)))/(alpha+arriveRate)+exp(-(alpha+arriveRate+erlangRate)*policyV(i))*replace;
    Qalpha(i,i-1)=arriveRate*(1-exp(-(alpha+arriveRate)*policyV(i)))/(alpha+arriveRate);
    Qalpha(i,i+1)=exp(-(alpha+arriveRate)*policyV(i))*(1-exp(-erlangRate*policyV(i))*sumAlpha0)+exp(-(alpha+arriveRate+erlangRate)*policyV(i))*sumAlpha;
end
    
    



