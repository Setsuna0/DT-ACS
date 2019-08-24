function y=fPlusAg(vj,j,potential,policyV)
global M alpha uniParameter
global arriveRate erlangRate erlangOrder
global I e 

policyV(j)=vj;
[falpha,Aalpha]=equivMarkov(policyV);
y=falpha(j)+Aalpha(j,:)*potential;
