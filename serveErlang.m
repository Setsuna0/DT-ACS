function serveTime=serveErlang(prob)
% to compute the serving time for erlang distribution
global erlangRate erlangOrder

serviceTime=erlangOrder/erlangRate; % average service time
infBound=0;                         % in order to compute the serving time for erlang distribution
supBound=10*serviceTime;            %  infBound,supBound gives the two terminal bounds of erlang function,
Error=1;
epsilon=0.01;       % give the minimal distance of these two bounds

infTime=erlangValue(infBound,prob);     % inftime and suptime mean the error of the distribution equation. 
                                        % Since at first infBound is zero, so that the infTime is negative (within absolute precision)
supTime=erlangValue(supBound,prob);
if infTime==0       % I do not know whether infinitesimal value will be viewed as 0, so here is done which may be redundant
    serveTime=infBound;
    Error=0;
end
while infTime*supTime>0    % which means that the probability is beyond what the supBound represents, so we have to improve the supBound
    infBound=supBound;
    infTime=supTime;
    supBound=supBound*2;
    supTime=erlangValue(supBound,prob);
end
if supTime==0
    serveTime=supBound;
    Error=0;
end
while Error>epsilon     % if it holds, infBound is less than zero, and supBound is lager than zero
    midBound=(infBound+supBound)/2;
    midTime=erlangValue(midBound,prob);
    if midTime==0
        serveTime=midBound;
        Error=0;
        break
    elseif midTime*infTime<0
        supBound=midBound;
    else
        infBound=midBound;
    end
    Error=supBound-infBound;
end
if(Error~=0)
    serveTime=(infBound+supBound)/2;
end

