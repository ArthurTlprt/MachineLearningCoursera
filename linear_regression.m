
% taille des exemples
n = 11;

% taille des variables
m = 2;

alpha = 0.1;

x = [ 1, 1;
      1, 2;
      1, 2;
      1, 3;
      1, 3;
      1, 4;
      1, 5;
      1, 6;
      1, 6;
      1, 6;
      1, 8;
      1, 10];

y = [ -890;
      -1411;
      -1560;
      -2220;
      -2091;
      -2878;
      -3537;
      -3268;
      -3920;
      -4163;
      -5471;
      -5167];

th = [-500;
      -662];


function [th, gaps] = gradient_descent(x, y, th, alpha)
  gaps = [];
  for it=1:100
    temp = zeros(2, 1);
    temp(1) = th(1) - alpha * j_derivative(x, y, th, 1);
    temp(2) = th(2) - alpha * j_derivative(x, y, th, 2);
    th = temp;
    gaps(end+1) = gap(x, y, th);
  end
end

function [gap] = j_derivative(x, y, th, j)
  gap = 0;
  for i=1:2
    gap += (linear_f(x(i), th(i)) - y(i)) * x(i, j)/3;
  end
end

function [gap] = gap(x, y, th)
  gap = 0;
  for i=1:11
    gap += (linear_f(x(i, :), th) - y(i))/11;
  end
end

function [J] = cost_function(th)
  x = [ 1, 1;
        1, 2;
        1, 2;
        1, 3;
        1, 3;
        1, 4;
        1, 5;
        1, 6;
        1, 6;
        1, 6;
        1, 8;
        1, 10];

  y = [ -890;
        -1411;
        -1560;
        -2220;
        -2091;
        -2878;
        -3537;
        -3268;
        -3920;
        -4163;
        -5471;
        -5167];
  m = 11;
  J=0;
  for i=1:m
    J1 = y(i) * log(linear_f(x(i, :), th));
    J2 = (1 - y(i)) * log(1 - linear_f(x(i, :), th));
    J += -(J1 + J2)/m;
  end
end

function [z] = logistic_f(x, th)
  disp('x');
  x(i, :)
  disp('th');
  th
  z = 1 ./ (1+exp(-th.'*x));
end

function [z] = linear_f(x, th)
  z = x * th;
end

[th, gaps] = gradient_descent(x, y, th, alpha);
figure;
plot(1:100, gaps);
thetaOptimized = fminunc(@cost_function, th)

pause;
