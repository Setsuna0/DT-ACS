clear
global M alpha 
global arriveRate erlangRate erlangOrder
global I e 
global k1 k2 k3 k4 k5
global actionSet Qfactor actionNumber
format long;
tic;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 工件到达率服从泊松分布
% 站点对工件的加工时间分布为Erlang分布
arriveRate=1;           % 工件到达率
erlangOrder=4;          % Erlang分布阶数=4
erlangRate=3*2/1.5;     % Erlang分布率=4
serviceRate=erlangRate/erlangOrder; % 总服务率=1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=15;        % 站点缓冲库存容量
M=N+1;       % 由于缓冲库存剩余量可为0，于是多出一个状态
maxLook=1;  
minLook=0;   % 最大与最小前视距离
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k1=0.1*1;    % 单位时间内可使用的缓冲库剩余量代价
k2=0.5*10;   % 单位时间内的服务代价
k3=1/1;      % 单位时间等待代价，等待时间越短，单位时间内加工时间越长
k4=-10;      % 处理完一个工件的奖赏值
k5=0.2*1;    % 单位时间内的前视代价
I=eye(M,M);    e=ones(M,1);  
lam=0.4;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha=0.001;       % discount factor 折扣因子
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
% 离散化行动集合
x=minLook;    % begin to define the discrete action set
actionSet=0;  
% begin to define the discrete action set. Here possibly minLook~=0.
while x<=maxLook 
    actionSet=[actionSet,x];
    x=x+deltaLook;
end           % 从最小前视距离一直叠加到最大前视距离获得完整行动集合
% actionSet=[actionSet,maxLook];
actionSet=[actionSet,inf];  
% including the action of state M, that is the reserve is full free, 
% then we have to wait untill the unit arrives 
% 当缓冲库存剩余量为M时，表示没有库存，则一直等待，相当于前视距离无穷大
actionNumber=length(actionSet);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
currentState=ceil(rand(1,1)*M);     
greedyPolicy(1)=0;      % 当缓冲库剩余量为0时不再前视，直接处理工件             
greedyPolicy(M)=inf;    % 当缓冲库剩余量为N时一直前视，等待工件到达
if currentState==1
    actionIndex=1;             
elseif currentState==M
    actionIndex=actionNumber;
end
greedyPolicy(2:end-1)=ones(1,M-2)*maxLook; 
% 库存剩余1，初始策略为动作1，即前视距离为0；
% 库存剩余M，初始策略为动作M，即前视距离为无穷远；
% 对于其余库存，初始策略均为最大前视距离1
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
% 完成Q网络的初始化
visitTimes=zeros(size(Qfactor));

[falpha,Aalpha,delayTime]=equivMarkov(greedyPolicy);   
[stableProb,potential]=stablePotential(falpha,Aalpha);
lastValue=falpha+Aalpha*potential;    
% stopping criteria value
disValue=[lastValue];
averageVector=stableProb*falpha;    %存储每次学习的平均代价

AverageEsmate=averageVector;
LookCost=averageVector;     
CostReal=averageVector;

eachTransitCost=0;      
eachTransitTime=0;

%==========================================================================
totalReward=0;            % 总奖赏值初始化
policy=zeros(16,1);      % 策略初始化
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
    % 更新贪婪策略中Epsilon值的大小
    if outStep<loopStep*1 
        epsilon=firstEpsilon*exp(-2*epsilonRate*outStep);   
        % 随着学习次数增加探索的概率减小
    elseif outStep<loopStep*0.8 
        epsilon=secondEpsilon; 
    else epsilon=0; 
    end   
    %======================================================================
    
    %==================== Q Learning begins ===============================
    for innerStep=1:learningStep
        
        visitTimes(currentState,actionIndex)=visitTimes(currentState,actionIndex)+1;    
        currentAction=greedyPolicy(currentState);
        
        %%%%%%%%%%%%%%%%%%%%%%%确定前视代价lookCost%%%%%%%%%%%%%%%%%%%%%%%%%
        if currentState==M
            lookCost=0;  
        else
            lookCost=k5*currentAction;  
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        [flag,sojournTime,serveTime,nextState] = Transition(currentState,currentAction);
        % 获取本次状态转化的逗留时间、加工时间、工序工作状态、下一状态   
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
        
        %%%%%%%%%%%%%%%%%%%%%%工序等待时%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if flag==0    
            costReal=(k1*(M-currentState)+k3)*sojournTime+lookCost;  
            purtCost=(k1*(M-currentState)+k3)*discoutTime+lookCost;    
        %%%%%%%%%%%%%%%%%%%%%%工序工作时%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        else        
            costReal=k1*(M-currentState-1)*sojournTime+k2*serveTime+k3*(sojournTime-serveTime)+k4+lookCost;   
            purtCost=k1*(M-currentState-1)*discoutTime+k2*disServTime+k3*disWaitTime+k4*endServeTime+lookCost;      
        end
        
        eachTransitCost=discountForAverage*eachTransitCost+(costReal-eachTransitCost)/((outStep-1)*learningStep+innerStep)^averageQstep;
        averageQcost=eachTransitCost/eachTransitTime;
        AverageEsmate=[AverageEsmate,averageQcost];     %存储averageQcost
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
            %将当前的动作存储给当前状态作为最优策略
        end
        
    end
    % the end of the Q learning 结束Q学习
    %======================================================================
    
    %======================================================================
   [s,a,t,r] = initial();      % 状态、动作、奖赏初始化
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
    % 更新策略序列
    Loss = [Loss, net.J];
    pause(0.000001); 
    plot(Loss);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
end
toc
computationTime=toc;
save qlearninga