% loopStep=3000;        % the steps for policy improving
% learningStep=10;      % the steps of the learning of Q factors
% discountForAverage=1;
% firstEpsilon=0.3;
% secondEpsilon=0.1;
% averageQstep=0.9;     % 0.9*0.7 is good
% QfactorStep=0.7;
% firstEpsilon=0.8;     % for epsilon-greedy policy, epsilon is decreasing
% secondEpsilon=0.5;
% epsilonRate=-log(secondEpsilon/firstEpsilon)/loopStep;
% stopEpsilon=0.1;      % for another stopping criteria
% maxEqualNumber=loopStep*1;
% equalNumber=0;
% deltaLook=0.05;       % the difference between the neighbour two actions

clear
global M alpha 
global arriveRate erlangRate erlangOrder
% global arriveTime lossNumber
global I e 
global k1 k2 k3 k4 k5

format long;
tic;

%==========================================================================
% suppose the arrive flow is Poisson distribution, 
% 工件到达率服从泊松分布
% and the service or processing is erlangOrder Erlang distribution 
% with erlangRate service rate each phase 
% 站点对工件的加工时间分布为Erlang分布
arriveRate=1;         % arrive rate of Poisson arriving flow
% 工件到达率
erlangOrder=4;        % the order of Erlang distribution 
% Erlang分布阶数=4
erlangRate=3*2/1.5;   % service rate of every phase of Erlang distribution
% Erlang分布率=4
serviceRate=erlangRate/erlangOrder; % average service rate
% 总服务率=1
%==========================================================================

%==========================================================================
N=15;     % the capacity of the reserve
% 站点缓冲库存容量
M=N+1;   % number of states
% 由于缓冲库存剩余量可为0，于是多出一个状态
maxLook=1;  %the max time of lookahead
minLook=0;  %最大与最小前视距离
%==========================================================================

%==========================================================================
k1=0.1*1;    % reserve cost per unit time per usable
% 单位时间内可使用的缓冲库剩余量代价
k2=0.5*10;   % service cost per unit time
% 单位时间内的服务代价
k3=1/1;      % waiting cost per unit time
% 单位时间等待代价，等待时间越短，单位时间内加工时间越长
k4=-10;      % reward per processed product 
% 处理完一个工件的奖赏值
k5=0.2*1;    % look ahead cost per unit time
% 单位时间内的前视代价
I=eye(M,M);    e=ones(M,1);  
%==========================================================================

%==========================================================================
%alpha=0.01;      % discount factor 折扣因子
alpha=0.001;
deltaLook=0.01;    % the difference between the neighbour two actions
loopStep=1000;    % the steps for policy improving
learningStep=50;  % the steps of the learning of Q factors
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
x=minLook;  % begin to define the discrete action set
actionSet=0;  
% begin to define the discrete action set. Here possibly minLook~=0.
while x<=maxLook 
    actionSet=[actionSet,x];
    x=x+deltaLook;
end     % 从最小前视距离一直叠加到最大前视距离获得完整行动集合
% actionSet=[actionSet,maxLook];
actionSet=[actionSet,inf];  
% including the action of state M, that is the reserve is full free, 
% then we have to wait untill the unit arrives 
% 当缓冲库存剩余量为M时，表示没有库存，则一直等待，相当于前视距离无穷大
actionNumber=length(actionSet);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialize state 初始化状态集合
currentState=ceil(rand(1,1)*M);     
% initialize policy 初始化策略
greedyPolicy(1)=0;      
% 当缓冲库剩余量为0时不再前视，直接处理工件             
greedyPolicy(M)=inf;    
% 当缓冲库剩余量为N时一直前视，等待工件到达
if currentState==1
    actionIndex=1;               % indicate the initial action
elseif currentState==M
    actionIndex=actionNumber;
end
% for i=2:M-1    
%    j=ceil(rand(1,1)*(actionNumber-2))+1;
%    greedyPolicy(i)=actionSet(j);  
%    if currentState==i 
%       actionIndex=j; 
%    end       % in order to record the action being used
% end
greedyPolicy(2:end-1)=ones(1,M-2)*maxLook;  %(maxLook-minLook)/2;
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
% 初始化Q矩阵为0
% initialize the value of Q, why multiply by 0.1 ?
% for state 1, the corresponding value is only (1,1), 
% for state M, (M,actionNumber)
Qfactor(1,:)=[Qfactor(1,1),inf*ones(1,actionNumber-1)];
Qfactor(M,:)=[inf*ones(1,actionNumber-1),Qfactor(M,actionNumber)];
Qfactor(:,1)=[Qfactor(1,1);inf*ones(M-1,1)];
Qfactor(:,actionNumber)=[inf*ones(M-1,1);Qfactor(M,actionNumber)];
visitTimes=zeros(size(Qfactor));  % the visiting times of state-action tuple

[falpha,Aalpha,delayTime]=equivMarkov(greedyPolicy);   
% uniParameter will also be return, 
% and delayTime is the average delay time of every state transition
[stableProb,potential]=stablePotential(falpha,Aalpha);
lastValue=falpha+Aalpha*potential;    
% stopping criteria value
disValue=[lastValue];
averageVector=stableProb*falpha;      
% to store the average cost for every learningStep
AverageEsmate=averageVector;
% arriveTime=0;
% totalCost=0;        % the total cost accumulated in our simulation
% totalTime=0;        % the total past time in simulation
% averageQcost=0;     % the estimate of the average cost
eachTransitCost=0;
eachTransitTime=0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for outStep=1:loopStep
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if outStep<loopStep*1 
        epsilon=firstEpsilon; 
        epsilon=firstEpsilon*exp(-2*epsilonRate*outStep);   
        % decreasing as time goes
        % 贪婪Epsilon的选择
    elseif outStep<loopStep*0.8 
        epsilon=secondEpsilon; 
    else epsilon=0; 
    end     % it can be computed by an inverse exponential function 
    % 以上为Q学习中策略选择部分的Epsilon做出选择
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%% Q Learning begins 开始进行Q学习 %%%%%%%%%%%%%%%%%%%%
    for innerStep=1:learningStep
        
        visitTimes(currentState,actionIndex)=visitTimes(currentState,actionIndex)+1;    
        % the visiting numbers of state-action tuples, in order to determine the stepsize for Q learning
        currentAction=greedyPolicy(currentState);
        % 根据贪婪策略为当前状态选择相应的动作（前视距离）
       
        %%%%%%%%%%%%%%%%%%%%%%%确定前视代价lookCost%%%%%%%%%%%%%%%%%%%%%%%%%
        if currentState==M
            lookCost=0;  % 库存为空，再怎么前视都不存在代价   
            % indexM=0;  % the cost for taking the action of looking ahead
        else
            lookCost=k5*currentAction;  % 库存不为空，则代价与前视距离成正比     
            % indexM=1;  % lookCost=k5*currentAction*indexM; 
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        [flag,sojournTime,serveTime,nextState] = Transition(currentState,currentAction);
        % 获取本次状态转化的逗留时间、加工时间、工序工作状态、下一状态
        % totalTime=discountForAverage*totalTime+sojournTime;   
        % 下面计算平均代价在discountForAverage＝1时，是等价的。
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
        if flag==0     % which means waiting
         
            costReal=(k1*(M-currentState)+k3)*sojournTime+lookCost;  
            purtCost=(k1*(M-currentState)+k3)*discoutTime+lookCost;  
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % reserve/waiting cost/the latter cost is generated immediately at current state 
            % totalCost=discountForAverage*totalCost+costReal;
            % averageQcost=totalCost/totalTime;
            % eachTransitCost=discountForAverage*eachTransitCost+(costReal-eachTransitCost)/((outStep-1)*learningStep+innerStep)^averageQstep;
            % averageQcost=eachTransitCost/eachTransitTime;
            % costDiscouted=(k1*(M-currentState)+k3-averageQcost)*discoutTime+lookCost;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
        %%%%%%%%%%%%%%%%%%%%%%工序工作时%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        else           % which means serving
          
            costReal=k1*(M-currentState-1)*sojournTime+k2*serveTime+k3*(sojournTime-serveTime)+k4+lookCost;   
            purtCost=k1*(M-currentState-1)*discoutTime+k2*disServTime+k3*disWaitTime+k4*endServeTime+lookCost;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % the latter cost is generated immediately at current state 
            % totalCost=discountForAverage*totalCost+costReal;
            % averageQcost=totalCost/totalTime;
            % eachTransitCost=discountForAverage*eachTransitCost+(costReal-eachTransitCost)/((outStep-1)*learningStep+innerStep)^averageQstep;
            % averageQcost=eachTransitCost/eachTransitTime;
            % costDiscouted=(k1*(M-currentState-1)-averageQcost)*discoutTime+k2*disServTime+k3*disWaitTime+k4*endSojournTime+lookCost;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
        end
        
        eachTransitCost=discountForAverage*eachTransitCost+(costReal-eachTransitCost)/((outStep-1)*learningStep+innerStep)^averageQstep;
        averageQcost=eachTransitCost/eachTransitTime;
        AverageEsmate=[AverageEsmate,averageQcost];
        costDiscouted=purtCost-averageQcost*discoutTime;
        difference=costDiscouted+endSojournTime*min(Qfactor(nextState,:))-Qfactor(currentState,actionIndex);   
        % temporal difference
        Qfactor(currentState,actionIndex)=Qfactor(currentState,actionIndex)+difference/visitTimes(currentState,actionIndex)^QfactorStep;  
        % learning of Q factor
        currentState=nextState;
        % s=s'
        
        if currentState==1  
            actionIndex=1;
        elseif currentState==M
            actionIndex=actionNumber;
        else
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %当随机数小于Epsilon时将选择Explore，探索未知的最优的动作
            if rand(1,1)<epsilon    
                % epsilon-greedy policy, in fact it is to generate epsilon-greedy action at next state
                actionIndex=ceil(rand(1,1)*(actionNumber-2))+1;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %当随机数大于Epsilon时将选择Expolit，即采取当前已知的最优动作
            else
                [minimalQvalue,actionIndex]=min(Qfactor(currentState,:));
            end
            greedyPolicy(currentState)=actionSet(actionIndex);  
            % generate the action of the next state which is epsilon greedy
        end
        
    end
    % the end of the Q learning 结束Q学习
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    optimalPolicy=greedyPolicy;
    
    for i=2:M-1    
        % determine the optimal policy by Q factors
        [minimalQvalue2,actionIndex2]=min(Qfactor(i,:));
        optimalPolicy(i)=actionSet(actionIndex2);
    end
    
    [falpha,Aalpha,delayTime]=equivMarkov(optimalPolicy);   
    % uniParameter will also be return, 
    % and delayTime is the average delay time of every state transition
    [stableProb,potential]=stablePotential(falpha,Aalpha);
    
    lastValue=falpha+Aalpha*potential;     disValue=[disValue,lastValue];
    % stopping criteria value
    newAverageCost=stableProb*falpha; 
    averageVector=[averageVector,newAverageCost];
    
    if newAverageCost-averageVector(end-1)<stopEpsilon
        equalNumber=equalNumber+1;
        if equalNumber==maxEqualNumber break; end
    else
        equalNumber=0;
    end
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

toc
computationTime=toc;
t=0:length(averageVector)-1;
plot(t,disValue(1,:));
save qlearninga