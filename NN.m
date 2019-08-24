function net = NN( net,x )
net.a = {};
net.a{1} = x;

for l=1:size(net.layers,2)-1
    net.z{l+1} = net.W{l} * net.a{l};
    net.a{l+1} = sigmoid(net.z{l+1});
end

end

