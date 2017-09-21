pkg load statistics;
% https://github.com/Borye/machine-learning-coursera-1/blob/master/Week%205%20Assignments/Neural%20Network%20Learning/ex4.pdf
% this is a 3 layers neural network


th = random('Normal', 0, 10,[3, 2]);
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
  Theta1 = reshape(thetaVec(1:6), 3, 2);
  Theta2 = reshape(thetaVec(7:9), 3, 1);
  a1 = [1, inputs];
  a2 = [1, g(a1 * Theta1)];
  a3 = g(a2 * Theta2);
  y = a3;
end

function [J, grad] = back_propagation(thetaVec)

  lambda = 1;

  Theta1 = reshape(thetaVec(1:6), 3, 2);
  Theta2 = reshape(thetaVec(7:9), 3, 1);
  Theta1_grad=0;
  Theta2_grad=0;

  training_inputs = [0, 1;
                    1, 1;
                    0, 0];
  training_outputs = [1, 1, 0];

  J=0;
  m = 3;
  for i=1:m
    J1 = training_outputs(i) * log(forward_propagation(training_inputs(i, :), thetaVec));
    J2 = (1 - training_outputs(i)) * log(1 - forward_propagation(training_inputs(i, :), thetaVec));
    J += -(J1 + J2)/m;
  end

  for i=1:m
    a1 = [1, training_inputs(i, :)];
    a2 = [1, g(a1 * Theta1)];
    a3 = g(a2 * Theta2);

    delta_3 = a3 - training_outputs(i);
    delta_2 = ((Theta2.') * delta_3 .* g_derivative(a2));

    Theta1_grad += delta_2 * (a1.');
  	Theta2_grad += delta_3 * (a2.');
  end

  Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
  Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

  grad = [Theta1_grad(:) ; Theta2_grad(:)];

end


% y = forward_propagation([1, 1, 1], thetaVec);

thetaOptimized = fminunc(@back_propagation, thetaVec);

forward_propagation([1,0], thetaOptimized)
forward_propagation([0,1], thetaOptimized)
forward_propagation([1,1], thetaOptimized)
forward_propagation([0,0], thetaOptimized)
