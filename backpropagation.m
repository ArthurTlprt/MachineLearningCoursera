pkg load statistics;
% https://github.com/Borye/machine-learning-coursera-1/blob/master/Week%205%20Assignments/Neural%20Network%20Learning/ex4.pdf
% this is a 3 layers neural network


th = random('Normal', 0, 10,[3, 3]);
last_weights = random('Normal', 0, 10,[3, 1]);
thetaVec = [th(:); last_weights(:)];

% activation function
function [s] = g(x)
  s = 1./(1+exp(-x));
end

function [s] = g_derivative(x)
  s = x.*(x-1);
end

function [y] = forward_propagation(inputs, thetaVec)
  th = reshape(thetaVec(1:9), 3, 3);
  last_weights = reshape(thetaVec(10:12), 3, 1);
  inputs = g(inputs * th);
  y = g(inputs * last_weights);
end

function [J, gradientVec] = back_propagation(thetaVec)
  th = reshape(thetaVec(1:9), 3, 3);
  last_weights = reshape(thetaVec(10:12), 3, 1);
  training_inputs = [1, 0, 1;
                    1, 1, 1;
                    1, 0, 0];
  training_outputs = [1, 1, 0];
  J = [];
  DELTA = zeros(1, 12)
  for i=1:3
    delta = [];
    a1 = training_inputs(i, :);
    a2 = g(training_inputs(i, :) * th);
    a3 = forward_propagation(training_inputs(i,:), thetaVec);
    delta(end+1) = a3 - training_outputs(i);
    delta(end+1) = last_weights.' * delta(1) .* g_derivative(a2);
    DELTA() += delta
    thetaVec += delta;
  end
  J(end+1) = cost_function(training_inputs, training_outputs, thetaVec);
end

function [J] = cost_function(thetaVec)
  training_inputs = [1, 0, 1;
                    1, 1, 1;
                    1, 0, 0];
  training_outputs = [1, 1, 0];
  J=0;
  for i=1:3
    J1 = training_outputs(i) * log(forward_propagation(training_inputs(i, :), thetaVec));
    J2 = (1 - training_outputs(i)) * log(1 - forward_propagation(training_inputs(i, :), thetaVec));
    J += -(J1 + J2)/3;
  end
  J;
end

% y = forward_propagation([1, 1, 1], thetaVec);


thetaOptimized = fminunc(@cost_function, thetaVec);

forward_propagation([1,1,0], thetaOptimized)
forward_propagation([1,0,1], thetaOptimized)
forward_propagation([1,1,1], thetaOptimized)
forward_propagation([1,0,0], thetaOptimized)
