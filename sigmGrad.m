function [ x ] = sigmGrad( x )
x = exp(-x) ./ (1 + exp(-x));
end

