function hAlpha0=halpha0V(policyV)
global M alpha uniParameter
global arriveRate erlangRate erlangOrder
global I e 

% this subfunction is used only for computing the cost matrix. 

hAlpha0=zeros(M,1);

for i=2:M-1 % note that for policy, only N-1 actions to be determined,that is 1,2,....,M-2
    replace=0;
    for j=0:erlangOrder-1
        replace=replace+(erlangOrder-j)*erlangRate^(j-1)*policyV(i)^j/factorial(j); % the sum of the order items as alpha=0
    end
    hAlpha0(i)=policyV(i)+exp(-erlangRate*policyV(i))*replace;
end

    
    



