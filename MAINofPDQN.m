clear
global M alpha 
global arriveRate erlangRate erlangOrder
global I e 
global k1 k2 k3 k4 k5
global actionSet Qfactor actionNumber
format long;
tic;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ���������ʷ��Ӳ��ɷֲ�
% վ��Թ����ļӹ�ʱ��ֲ�ΪErlang�ֲ�
arriveRate=1;           % ����������
erlangOrder=4;          % Erlang�ֲ�����=4
erlangRate=3*2/1.5;     % Erlang�ֲ���=4
serviceRate=erlangRate/erlangOrder; % �ܷ�����=1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=15;        % վ�㻺��������
M=N+1;       % ���ڻ�����ʣ������Ϊ0�����Ƕ��һ��״̬
maxLook=1;  
minLook=0;   % �������Сǰ�Ӿ���
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k1=0.1*1;    % ��λʱ���ڿ�ʹ�õĻ����ʣ��������
k2=0.5*10;   % ��λʱ���ڵķ������
k3=1/1;      % ��λʱ��ȴ����ۣ��ȴ�ʱ��Խ�̣���λʱ���ڼӹ�ʱ��Խ��
k4=-10;      % ������һ�������Ľ���ֵ
k5=0.2*1;    % ��λʱ���ڵ�ǰ�Ӵ���
I=eye(M,M);    e=ones(M,1);  
lam=0.4;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha=0.001;       % discount factor �ۿ�����
% alpha=0;
deltaLook=0.01;    % the difference between the neighbour two actions
loopStep=1000;     % the steps for policy improving
learningStep=50;   % the steps of the learning of Q factors
discountForAverage=1;
% firstEpsilon=0.2;
% secondEpsilon=0.1;
averageQstep=0.6;   
% 0.9*0.6 is very good under firstEpsilon=0.2(than 0.25),
% better than 0.8*0.6, 0.9*0.5, 0.9*0.7, 1*0.6 
QfactorStep=0.7;
firstEpsilon=0.5;       % for epsilon-greedy policy, epsilon is decreasing
secondEpsilon=0.2;      % denote the value of loopStep/2
epsilonRate=-2*log(secondEpsilon/firstEpsilon)/loopStep;
stopEpsilon=0.01;       % for another stopping criteria
maxEqualNumber=loopStep*1;  
equalNumber=0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ��ɢ���ж�����
x=minLook;    % begin to define the discrete action set
actionSet=0;  
% begin to define the discrete action set. Here possibly minLook~=0.
while x<=maxLook 
    actionSet=[actionSet,x];
    x=x+deltaLook;
end           % ����Сǰ�Ӿ���һֱ���ӵ����ǰ�Ӿ����������ж�����
% actionSet=[actionSet,maxLook];
actionSet=[actionSet,inf];  
% including the action of state M, that is the reserve is full free, 
% then we have to wait untill the unit arrives 
% ��������ʣ����ΪMʱ����ʾû�п�棬��һֱ�ȴ����൱��ǰ�Ӿ��������
actionNumber=length(actionSet);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
currentState=ceil(rand(1,1)*M);     
greedyPolicy(1)=0;      % �������ʣ����Ϊ0ʱ����ǰ�ӣ�ֱ�Ӵ�����             
greedyPolicy(M)=inf;    % �������ʣ����ΪNʱһֱǰ�ӣ��ȴ���������
if currentState==1
    actionIndex=1;             
elseif currentState==M
    actionIndex=actionNumber;
end
greedyPolicy(2:end-1)=ones(1,M-2)*maxLook; 
% ���ʣ��1����ʼ����Ϊ����1����ǰ�Ӿ���Ϊ0��
% ���ʣ��M����ʼ����Ϊ����M����ǰ�Ӿ���Ϊ����Զ��
% ���������棬��ʼ���Ծ�Ϊ���ǰ�Ӿ���1
if currentState~=1&&currentState~=M  
    actionIndex=actionNumber-1; 
end           % initialize policy
pi=[0,0.602093902620241,0.753377944704191,...
    0.893866808528892,0.999933893038648,Inf];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Qfactor=zeros(M,actionNumber);
Qfactor(1,:)=[Qfactor(1,1),inf*ones(1,actionNumber-1)];
Qfactor(M,:)=[inf*ones(1,actionNumber-1),Qfactor(M,actionNumber)];
Qfactor(:,1)=[Qfactor(1,1);inf*ones(M-1,1)];
Qfactor(:,actionNumber)=[inf*ones(M-1,1);Qfactor(M,actionNumber)];
% ���Q����ĳ�ʼ��
visitTimes=zeros(size(Qfactor));

[falpha,Aalpha,delayTime]=equivMarkov(greedyPolicy);   
[stableProb,potential]=stablePotential(falpha,Aalpha);
lastValue=falpha+Aalpha*potential;    
% stopping criteria value
disValue=[lastValue];
averageVector=stableProb*falpha;    %�洢ÿ��ѧϰ��ƽ������

AverageEsmate=averageVector;
LookCost=averageVector;     
CostReal=averageVector;

eachTransitCost=0;      
eachTransitTime=0;

%==========================================================================
totalReward=0;            % �ܽ���ֵ��ʼ��
policy=zeros(16,1);      % ���Գ�ʼ��
value=zeros(16,5);
x=zeros(16,5);
gama=0.99;

%initialize the network
net.layers=[16 32 5];
L=size(net.layers,2);
net.rl=0.1;
net=initNN(net);
net2=net;
Loss=[];
%==========================================================================

%==========================================================================
for outStep=1:loopStep
    
    %======================================================================
    % ����̰��������Epsilonֵ�Ĵ�С
    if outStep<loopStep*1 
        epsilon=firstEpsilon*exp(-2*epsilonRate*outStep);   
        % ����ѧϰ��������̽���ĸ��ʼ�С
    elseif outStep<loopStep*0.8 
        epsilon=secondEpsilon; 
    else epsilon=0; 
    end   
    %======================================================================
    
    %==================== Q Learning begins ===============================
    for innerStep=1:learningStep
        
        visitTimes(currentState,actionIndex)=visitTimes(currentState,actionIndex)+1;    
        currentAction=greedyPolicy(currentState);
        
        %%%%%%%%%%%%%%%%%%%%%%%ȷ��ǰ�Ӵ���lookCost%%%%%%%%%%%%%%%%%%%%%%%%%
        if currentState==M
            lookCost=0;  
        else
            lookCost=k5*currentAction;  
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        [flag,sojournTime,serveTime,nextState] = Transition(currentState,currentAction);
        % ��ȡ����״̬ת���Ķ���ʱ�䡢�ӹ�ʱ�䡢������״̬����һ״̬   
        eachTransitTime=discountForAverage*eachTransitTime+(sojournTime-eachTransitTime)/((outStep-1)*learningStep+innerStep)^averageQstep;
        endSojournTime=exp(-alpha*sojournTime);
        endServeTime=exp(-alpha*serveTime);
        if alpha==0
            discoutTime=sojournTime;
            disServTime=serveTime;
            disWaitTime=sojournTime-serveTime;
        else
            discoutTime=(1-endSojournTime)/alpha;
			disServTime=(1-endServeTime)/alpha;
            disWaitTime=(endServeTime-endSojournTime)/alpha;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%����ȴ�ʱ%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if flag==0    
            costReal=(k1*(M-currentState)+k3)*sojournTime+lookCost;  
            purtCost=(k1*(M-currentState)+k3)*discoutTime+lookCost;    
        %%%%%%%%%%%%%%%%%%%%%%������ʱ%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        else        
            costReal=k1*(M-currentState-1)*sojournTime+k2*serveTime+k3*(sojournTime-serveTime)+k4+lookCost;   
            purtCost=k1*(M-currentState-1)*discoutTime+k2*disServTime+k3*disWaitTime+k4*endServeTime+lookCost;      
        end
        
        eachTransitCost=discountForAverage*eachTransitCost+(costReal-eachTransitCost)/((outStep-1)*learningStep+innerStep)^averageQstep;
        averageQcost=eachTransitCost/eachTransitTime;
        AverageEsmate=[AverageEsmate,averageQcost];     %�洢averageQcost
        costDiscouted=purtCost-averageQcost*discoutTime;
        difference=costDiscouted+endSojournTime*min(Qfactor(nextState,:))-Qfactor(currentState,actionIndex);   
        % temporal difference
        Qfactor(currentState,actionIndex)=Qfactor(currentState,actionIndex)+lam^(learningStep-innerStep)*difference/visitTimes(currentState,actionIndex)^QfactorStep;  
        % learning of Q factor
        currentState=nextState;    
        
        if currentState==1  
            actionIndex=1;
        elseif currentState==M
            actionIndex=actionNumber;
        else
            if rand(1,1)<epsilon    
                actionIndex=ceil(rand(1,1)*(actionNumber-2))+1;
            else
                [minimalQvalue,actionIndex]=min(Qfactor(currentState,:));
            end
            greedyPolicy(currentState)=actionSet(actionIndex);  
            %����ǰ�Ķ����洢����ǰ״̬��Ϊ���Ų���
        end
        
    end
    % the end of the Q learning ����Qѧϰ
    %======================================================================
    
    %======================================================================
   [s,a,t,r] = initial();      % ״̬�����������ͳ�ʼ��
   % trainDQN
    e_s = round((rand(1,1)*15)+1);
    s = e_s;
    while s~=17 && s ~=18
        x = zeros(16,1);
        x(s) = 1;
        net = NN(net,x);
        % use the epsilon-greedy to balance the exploration and exploitation
        prop = rand(1,1);
        if prop < epsilon
            [pos, a] = max(net.a{L});
            % exploitation
        else
            a = round((rand(1,1)*4) + 1);
            % exploration
        end
        % state transition
        s_n = t(s,a);
        y = net.a{L};
        % if arrive at the end state, then the taget only equals to the
        % immediate reward. if not then target equals to sum of immediate
        % reward and Q-function(the output of this Q-network).
        if s_n == 17 || s_n == 18
            y(a) = r(s,a);
        else
            x2 = zeros(16,1);
            x2(s_n) = 1;
            net2 = NN(net2,x2);
            y(a) = r(s,a) + gama * net2.a{L}(a);
        end
        net = backWard(net, y);
        if mod(outStep,50) == 0
            net2 = net;
        end
        s = s_n;
%         disp(net.J)
    end
    for i =1:16
        x = zeros(16,1);
        x(i) = 1;
        net = NN(net,x);
        [pos,current_a] = max(net.a{L});
        policy(i) = current_a;
    end
    % ���²�������
    Loss = [Loss, net.J];
    pause(0.000001); 
    plot(Loss);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
end
toc
computationTime=toc;
save qlearninga