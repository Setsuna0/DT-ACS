function [flag,sojournTime,serveTime,nextState] = Transition(currentState,currentAction)
% this function is to compute the state sojourn time, the generated cost in this transition.
% Detailed explanation goes here
global arriveRate erlangRate erlangOrder


arriveTime=-log(1-rand(1,1))/arriveRate;     
% ��������ʱ�䰴ָ������ֲ�

if(currentAction>=arriveTime)        
    % ��ʾǰ�Ӿ������й���������������Ҫ�ȴ�
    flag=0;
    % which means waiting
    nextState=currentState-1;
    % �ȴ��������뻺������ʹ�û�����ʣ������1
    sojournTime=arriveTime;
    % ���εȴ�ʱ����ȫ���ڹ�������ʱ�䣬�����мӹ�
    serveTime=0;
    % �����мӹ����ӹ�ʱ��Ϊ0
else
    % ��ʾǰ�Ӿ�����û�й�������ô�ͶԻ������ڲ��Ĺ������мӹ�
    flag=1;             
    % which means serving
    nextState=currentState+1;
    % ȡ�������������ӹ��󻺴���ʣ��������1
    serveTime=serveErlang(rand(1,1));
    % �ӹ�������ʱ�����Erlang�ֲ�
    if serveTime<currentAction
        sojournTime=currentAction;
    else
        sojournTime=serveTime;
    end
end
    
