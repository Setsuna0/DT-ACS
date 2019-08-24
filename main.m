% use value iteration
[s,a,t,r] = initial();

totalReward = 0;
policy =  zeros(16,1);
epsilon = 0.5;
value = zeros(16,5);
maxiter = 10000;
x = zeros(16,5);
gama =0.99;

for iter = 1: maxiter
    e_s = round((rand(1,1)*15)+1);
    s = e_s;
    while s ~= 17 && s ~= 18
        prop = rand(1,1);
        if prop < epsilon
            [pos,a] = max(value(s,:))z;
            s_n = t(s,a);
            x(s,a) = 1;
        else
            a = round((rand(1,1)*4) + 1);
            s_n = t(s,a);
            x(s,a) = 1;
        end
        if s_n == 18 || s_n ==17
            value(s,a) = r(s,a);
%             s,s_n
        else
            [max_value,posi] = max(value(s_n,:));
            value(s,a) = r(s,a) + gama * max_value;
%             s,s_n
%             print
        end
        s = s_n;
    end
    for i =1:16
        [pos,current_a] = max(value(i,:));
        policy(i) = current_a;
    end
%     disp(iter);
end
policy