input = [1, 1, 0]

function [s] = g(x)
  s = 1./(1+exp(-x));
end

disp('  and')
th = [-30, 20, 20];
g(input * th.')

disp('  or')
th = [-10, 20, 20];
g(input * th.')

disp('  nor')
th = [10, -20, -20];
g(input * th.')

disp('  xnor')
th1 = [-30, 20, 20;
      10, -20, -20];
th2 = [-10, 20, 20];
a2 = [1, 0, 0];
a2(2) = g(input * th1.')(1)
a2(3) = g(input * th1.')(2)
a3 = g(a2 * th2.')
