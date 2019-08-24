% clear the enviroment
clear;
clc;
close all;

% use value iteration
[s,a,t,r] = initial();      % 状态、动作、奖赏初始化
tic;
totalReward = 0;            % 总奖赏值初始化
policy =  zeros(16,1);      % 策略初始化
epsilon = 0.90;             % 贪婪策略
value = zeros(16,5);
maxiter = 1000;             % 迭代次数
x = zeros(16,5);
gama = 0.99;

%initialize the network
net.layers = [16 32 5];
L = size(net.layers,2);
net.rl = 0.1;
net = initNN(net);
net2 = net;
Loss = [];

% trainDQN
for iter = 1: maxiter
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
        
        if mod(iter,50) == 0
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
    pause(0.001); 
    plot(Loss);
%     disp(iter);       % 显示当前迭代次数
end
toc