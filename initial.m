function [s, action, transitions, reward] = initial(  )
global actionSet actionNumber
    % define the states, actions and rewards in this problem
    state = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16];
    s = reshape(state,[4,4]);
    % action = [1,2,3,4,5];
    action = actionSet;
    transitions = zeros(16,actionNumber);
    reward = zeros(16,actionNumber);
    % initialize the transitions
    j1 = 5;
    j2 = 1;
    j3 = 0;
    for i = 1:16
        if i<13
            transitions(i,1) = j1;
            j1 = j1 + 1;
        else
            if i == 15
                transitions(i,1) = 18;
            else
                transitions(i,1) = 17;
            end
        end

        if i < 5
            transitions(i,2) = 17;
        else
            transitions(i,2) = j2;
            j2 = j2 + 1;
        end

        if mod(i,4) == 1
            transitions(i,3) = 17;
        else
            transitions(i,3) = i - 1;
        end

        if mod(i,4) ==0
            transitions(i,4) = 17;
        else
            transitions(i,4) = i + 1;
        end

        transitions(i,5) = i;
    end

    % initialize the reward of each state

    for i = 1:16
        if i<13
            reward(i,1) = 0.1;
        else
            if i == 15
                reward(i,1) = 0.5;
            else
                reward(i,1) = -0.5;
            end
        end

        if i < 5
            reward(i,2) = -0.5;
        else
            reward(i,2) = -0.1;
        end

        if mod(i,4) == 1
            reward(i,3) = -0.5;
        elseif mod(i,4) == 0
            reward(i,3) = 0.1;
        else
            reward(i,3) = -0.1;
        end

        if mod(i,4) == 3
            reward(i,4) = -0.1;
        elseif mod(i,4) == 0
            reward(i,4) = -0.5;
        else
            reward(i,4) = 0.1;
        end

        reward(i,5) = -0.2;
    end
end

