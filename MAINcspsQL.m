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
% ���������ʷ��Ӳ��ɷֲ�
% and the service or processing is erlangOrder Erlang distribution 
% with erlangRate service rate each phase 
% վ��Թ����ļӹ�ʱ��ֲ�ΪErlang�ֲ�
arriveRate=1;         % arrive rate of Poisson arriving flow
% ����������
erlangOrder=4;        % the order of Erlang distribution 
% Erlang�ֲ�����=4
erlangRate=3*2/1.5;   % service rate of every phase of Erlang distribution
% Erlang�ֲ���=4
serviceRate=erlangRate/erlangOrder; % average service rate
% �ܷ�����=1
%==========================================================================

%==========================================================================
N=15;     % the capacity of the reserve
% վ�㻺��������
M=N+1;   % number of states
% ���ڻ�����ʣ������Ϊ0�����Ƕ��һ��״̬
maxLook=1;  %the max time of lookahead
minLook=0;  %�������Сǰ�Ӿ���
%==========================================================================

%==========================================================================
k1=0.1*1;    % reserve cost per unit time per usable
% ��λʱ���ڿ�ʹ�õĻ����ʣ��������
k2=0.5*10;   % service cost per unit time
% ��λʱ���ڵķ������
k3=1/1;      % waiting cost per unit time
% ��λʱ��ȴ����ۣ��ȴ�ʱ��Խ�̣���λʱ���ڼӹ�ʱ��Խ��
k4=-10;      % reward per processed product 
% ������һ�������Ľ���ֵ
k5=0.2*1;    % look ahead cost per unit time
% ��λʱ���ڵ�ǰ�Ӵ���
I=eye(M,M);    e=ones(M,1);  
%==========================================================================

%==========================================================================
%alpha=0.01;      % discount factor �ۿ�����
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
% ��ɢ���ж�����
x=minLook;  % begin to define the discrete action set
actionSet=0;  
% begin to define the discrete action set. Here possibly minLook~=0.
while x<=maxLook 
    actionSet=[actionSet,x];
    x=x+deltaLook;
end     % ����Сǰ�Ӿ���һֱ���ӵ����ǰ�Ӿ����������ж�����
% actionSet=[actionSet,maxLook];
actionSet=[actionSet,inf];  
% including the action of state M, that is the reserve is full free, 
% then we have to wait untill the unit arrives 
% ��������ʣ����ΪMʱ����ʾû�п�棬��һֱ�ȴ����൱��ǰ�Ӿ��������
actionNumber=length(actionSet);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialize state ��ʼ��״̬����
currentState=ceil(rand(1,1)*M);     
% initialize policy ��ʼ������
greedyPolicy(1)=0;      
% �������ʣ����Ϊ0ʱ����ǰ�ӣ�ֱ�Ӵ�����             
greedyPolicy(M)=inf;    
% �������ʣ����ΪNʱһֱǰ�ӣ��ȴ���������
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
% ��ʼ��Q����Ϊ0
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
        % ̰��Epsilon��ѡ��
    elseif outStep<loopStep*0.8 
        epsilon=secondEpsilon; 
    else epsilon=0; 
    end     % it can be computed by an inverse exponential function 
    % ����ΪQѧϰ�в���ѡ�񲿷ֵ�Epsilon����ѡ��
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%% Q Learning begins ��ʼ����Qѧϰ %%%%%%%%%%%%%%%%%%%%
    for innerStep=1:learningStep
        
        visitTimes(currentState,actionIndex)=visitTimes(currentState,actionIndex)+1;    
        % the visiting numbers of state-action tuples, in order to determine the stepsize for Q learning
        currentAction=greedyPolicy(currentState);
        % ����̰������Ϊ��ǰ״̬ѡ����Ӧ�Ķ�����ǰ�Ӿ��룩
       
        %%%%%%%%%%%%%%%%%%%%%%%ȷ��ǰ�Ӵ���lookCost%%%%%%%%%%%%%%%%%%%%%%%%%
        if currentState==M
            lookCost=0;  % ���Ϊ�գ�����ôǰ�Ӷ������ڴ���   
            % indexM=0;  % the cost for taking the action of looking ahead
        else
            lookCost=k5*currentAction;  % ��治Ϊ�գ��������ǰ�Ӿ��������     
            % indexM=1;  % lookCost=k5*currentAction*indexM; 
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        [flag,sojournTime,serveTime,nextState] = Transition(currentState,currentAction);
        % ��ȡ����״̬ת���Ķ���ʱ�䡢�ӹ�ʱ�䡢������״̬����һ״̬
        % totalTime=discountForAverage*totalTime+sojournTime;   
        % �������ƽ��������discountForAverage��1ʱ���ǵȼ۵ġ�
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
            
        %%%%%%%%%%%%%%%%%%%%%%������ʱ%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
            %�������С��Epsilonʱ��ѡ��Explore��̽��δ֪�����ŵĶ���
            if rand(1,1)<epsilon    
                % epsilon-greedy policy, in fact it is to generate epsilon-greedy action at next state
                actionIndex=ceil(rand(1,1)*(actionNumber-2))+1;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %�����������Epsilonʱ��ѡ��Expolit������ȡ��ǰ��֪�����Ŷ���
            else
                [minimalQvalue,actionIndex]=min(Qfactor(currentState,:));
            end
            greedyPolicy(currentState)=actionSet(actionIndex);  
            % generate the action of the next state which is epsilon greedy
        end
        
    end
    % the end of the Q learning ����Qѧϰ
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