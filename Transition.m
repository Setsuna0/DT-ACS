function [flag,sojournTime,serveTime,nextState] = Transition(currentState,currentAction)
% this function is to compute the state sojourn time, the generated cost in this transition.
% Detailed explanation goes here
global arriveRate erlangRate erlangOrder


arriveTime=-log(1-rand(1,1))/arriveRate;     
% 工件到达时间按指数随机分布

if(currentAction>=arriveTime)        
    % 表示前视距离内有工件即将到达，因此需要等待
    flag=0;
    % which means waiting
    nextState=currentState-1;
    % 等待工件放入缓存区后将使得缓存区剩余量少1
    sojournTime=arriveTime;
    % 本次等待时间完全等于工件到达时间，不进行加工
    serveTime=0;
    % 不进行加工，加工时间为0
else
    % 表示前视距离内没有工件，那么就对缓存区内部的工件进行加工
    flag=1;             
    % which means serving
    nextState=currentState+1;
    % 取出缓存区工件加工后缓存区剩余容量加1
    serveTime=serveErlang(rand(1,1));
    % 加工工件的时间服从Erlang分布
    if serveTime<currentAction
        sojournTime=currentAction;
    else
        sojournTime=serveTime;
    end
end
    
