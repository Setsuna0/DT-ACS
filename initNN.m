function net = initNN(net)
net.W = {}; 
    for i = 1:size(net.layers,2)-1
        net.W{i} = (rand(net.layers(i+1),net.layers(i))-0.5) * 2 * sqrt(6/(net.layers(i) + net.layers(i+1)));
    end
end

