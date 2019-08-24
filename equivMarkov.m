function [falpha,Aalpha,delayTime]=equivMarkov(policyV)
global M alpha uniParameter
global arriveRate erlangRate erlangOrder
global I e
global k1 k2 k3 k4 k5

% suppose the service or processing is erlangOrder-order Erlang distribution with serviceRate service rate each phase
% which are setted in the main function
costf=costMatrix(policyV); % performance vector
embedP=embedTranMatrix(policyV);   % transition matrix of the embedded chain
%distrF=distributionMatrix(policyV);    % state-sojourn-time distribution
%kenel=kenelMatrix(embedP,distrF);
%Qalpha=QalphaV(policyV);

[hAlpha,Qalpha]=hQalphaV(policyV);  % in fact, the policyV is not necessary if the kenel is given and the integration is done by computer rather myself
uniParameter=inv(min(hAlpha));  % the uniformized parameter, which is used in the solution of potentials
Halpha=diag(hAlpha);
invHalpha=inv(Halpha);
Aalpha=alpha*I-invHalpha*(I-Qalpha);    % equivalent infinitesimal generator
% test=alpha*Halpha+Qalpha;

if (alpha~=0)
    Palpha=invHalpha*(embedP-Qalpha)/alpha; 
else
    averaQt=zeros(M,M);
    averaQt(1,2)=hAlpha(1);
    averaQt(M,M-1)=hAlpha(M);
    for i=2:M-1
        averaQt(i,i-1)=inv(arriveRate)-(inv(arriveRate)+policyV(i))*exp(-arriveRate*policyV(i));
        averaQt(i,i+1)=hAlpha(i)-averaQt(i,i-1);
    end
    Palpha=invHalpha*averaQt;
end
for i=1:M
    for j=1:M
        Falpha(i,j)=Palpha(i,j)*costf(i,j);
    end
end
falpha=Falpha*e;
delayTime=averageDelay(embedP,hAlpha,policyV);
