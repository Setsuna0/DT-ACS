function embedP=embedTranMatrix(policyV)
global M alpha uniParameter
global arriveRate erlangRate erlangOrder
global I e 

% give the transition probabilities of the embedded Markov chain
% corresponding to policyV
embedP=zeros(M,M);
embedP(1,2)=1;
embedP(M,M-1)=1;
for i=2:M-1
    embedP(i,i+1)=exp(-arriveRate*policyV(i));
    embedP(i,i-1)=1-exp(-arriveRate*policyV(i));
end


