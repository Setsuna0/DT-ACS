function net = backWard( net,y )
L = size(net.layers,2);
J = 1 / 2 * sum(sum((net.a{L} - y).^2));
net.d = {};
net.dW = {};
net.d{L} = (net.a{L} - y) .* sigmGrad(net.z{L});

for l = L-1:-1:2
    net.d{l} = (net.W{l}' * net.d{l+1}) .* sigmGrad(net.z{l});
end

for l = 1 : L-1
    net.dW{l} = net.d{l+1} * net.a{l}';
    net.W{l} = net.W{l} - net.rl * net.dW{l};
end

net.J = J;
end

