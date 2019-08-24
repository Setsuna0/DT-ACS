function time=erlangValue(bound,prob) 
%% the distribution of sojourn time
global erlangRate erlangOrder
sum=0;
for k=0:erlangOrder-1
    sum=sum+(erlangRate^k*bound^k)/factorial(k);
end
time=1-prob-exp(-erlangRate*bound)*sum;