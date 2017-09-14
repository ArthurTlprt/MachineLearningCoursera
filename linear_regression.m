
% taille des exemples
n = 3;

% taille des variables
m = 4;

alpha = 0.1;

x = [ 1, 2, 2, 1;
      1, 3, 2, 2;
      1, 1, 3, 2];

y = [ 3;
      4;
      5];

th = [0.4;
      0.6;
      0.7];


function [th, gaps] = gradient_descent(x, y, th, alpha)
  gaps = []
  for it=1:100
    temp = zeros(3, 1);
    temp(1) = th(1) - alpha * j_derivative(x, y, th, 1);
    temp(2) = th(2) - alpha * j_derivative(x, y, th, 2);
    temp(3) = th(3) - alpha * j_derivative(x, y, th, 3);
    th = temp;
    gaps(end+1) = gap(x, y, th);
  end
end

function [gap] = j_derivative(x, y, th, j)
  gap = 0;
  for i=1:3
    gap += (linear_f(x(i), th(i)) - y(i)) * x(i, j)/3;
  end
end

function [gap] = gap(x, y, th)
  gap = 0;
  for i=1:3
    gap += (linear_f(x(i), th(i)) - y(i))/3;
  end
end

function [z] = logistic_f(x, th)
  z = 1 ./ (1+exp(-th.'*x));
end

function [z] = linear_f(x, th)
  z = th.' *x;
end

[th, gaps] = gradient_descent(x, y, th, alpha);
plot(1:100, gaps)
pause;
