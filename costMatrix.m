function costf=costMatrix(policyV)
global M alpha belta uniParameter
global arriveRate erlangRate erlangOrder serviceRate
global I e 
global k1 k2 k3 k4 k5

costf=zeros(M,M);
aveSojournTime=halpha0V(policyV);
expSojournTime=exp(-alpha*aveSojournTime);
aveServeTime=erlangOrder/erlangRate;
expServeTime=exp(-alpha*(aveServeTime));
if alpha==0
    disSojournTime=aveSojournTime;
    disServeTime=aveServeTime;
    disWaiteTime=aveSojournTime-aveServeTime;
else
    disSojournTime=(1-expSojournTime)/alpha;
	disServeTime=(1-expServeTime)/alpha;
    disWaiteTime=(expServeTime-expSojournTime)/alpha;
end
costf(1,2)=k1*(M-2)+k2+k4*expServeTime/disServeTime;
costf(M,M-1)=k3;
for i=2:M-1 % note that for policy, only N-1 actions to be determined,that is 1,2,....,M-2
    costf(i,i+1)=k1*(M-i-1)+(k2*disServeTime+k3*disWaiteTime(i)+k4*expServeTime+k5*policyV(i))/disSojournTime(i);
    if(policyV(i)==0)
        x=0;
    else
        disSojournTimeL=inv(arriveRate)-policyV(i)*exp(-arriveRate*policyV(i))/(1-exp(-arriveRate*policyV(i)));
        if alpha~=0
            disSojournTimeL=(1-exp(-alpha*disSojournTimeL))/alpha;
        end
        x=policyV(i)/disSojournTimeL;
    end
    costf(i,i-1)=k1*(M-i)+k3+k5*x;
end
