pkg load statistics;

% this is a 3 layers neural network

inputs = [0, 1, 1];

function [s] = g(x)
  s = 1./(1+exp(-x));
end

function [y] = forward_propagation(inputs)
  th = random('Normal', 0, 10,[3, 3, 2]);
  th_layer_2 = random('Normal', 0, 10, [3, 1]);

  a = zeros(1, 3, 2);
  a(:, :, 1) = g(inputs * th(:, :, 1));
  a(:, :, 2) = g(a(:,:,1) * th(:, :, 2));
  y = g(a(:,:,2) * th_layer_2);
end


forward_propagation(inputs)
