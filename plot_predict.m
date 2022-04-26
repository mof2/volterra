function plot_predict(test, mu, s2, input, target, noise)
% plot_predict: plots scalar prediction of a regressor for scalar test input. If
% provided, 4-sigma 95% confidence area (assuming the prediction is done
% using a Gaussian process) is shaded. Optionally, the training data is
% plotted as stars.
%
% usage: plot_predict(x_test, y_pred)
%    or: plot_predict(x_test, y_pred, s2)
%    or: plot_predict(x_test, y_pred, s2, x_train, y_train)
%    or: plot_predict(x_test, y_pred, s2, x_train, y_train, noise)
% 
% where:
%
%  test    is a nn x 1 vector of scalar targets
%  mu      is a nn x 1 vector of predicted means
%  s2      is a nn x 1 vector of predicted variances
%  input   is a n x 1 vector of training inputs
%  target  is a n x 1 vector of targets
%  noise   is a scalar noise variance
%
%
%  (C) Copyright 2005, M.O.Franz & C.E.Rasmussen

clf
axis([min(test) max(test) 1.1*min([mu; target]) 1.1*max([mu; target])]);

% plot predicted confidence interval + noise
if nargin == 6 % if provided
    s2 = reshape(s2, length(s2), 1);                    % make sure that input is a column vector
    test = reshape(test, length(test), 1);
    mu = reshape(mu, length(mu), 1);
    dev = [mu + 2*(sqrt(s2) + sqrt(noise)); flipud(mu - 2*(sqrt(s2) + ...
            sqrt(noise)))];
    x_min = test(1) + 0.005*abs(test(end) - test(1));   % filling aree must be smaller
    x_max = test(end) - 0.005*abs(test(end) - test(1)); % than axes
    a = axis;
    y_min = 0.99*a(3);                         % filling area must be smaller
    y_max = 0.99*a(4);                         % than axes
    xs = max(min([test; flipud(test)], x_max), x_min);   % x values of polygon
    ys = max(min(dev, y_max), y_min);   % y values of polygon
    fill(xs, ys, [18 18 18]/19, 'EdgeColor', [18 18 18]/19);
    hold on
end

% plot predicted confidence interval without noise
if nargin > 2 % if provided
    s2 = reshape(s2, length(s2), 1);                    % make sure that input is a column vector
    test = reshape(test, length(test), 1);
    mu = reshape(mu, length(mu), 1);
    dev = [mu + 2*sqrt(s2); flipud(mu - 2*sqrt(s2))];
    x_min = test(1) + 0.005*abs(test(end) - test(1));   % filling aree must be smaller
    x_max = test(end) - 0.005*abs(test(end) - test(1)); % than axes
    a = axis;
    y_min = 0.99*a(3);                         % filling area must be smaller
    y_max = 0.99*a(4);                         % than axes
    xs = max(min([test; flipud(test)], x_max), x_min);   % x values of polygon
    ys = max(min(dev, y_max), y_min);   % y values of polygon
    fill(xs, ys, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
    hold on
end
if nargin > 3
    plot(input, target, 'b *')                  % plot training data if provided
end
plot(test, mu, 'r')                             % plot predicted mean
axis([min(test) max(test) 1.1*min([mu; target]) 1.1*max([mu; target])]);
hold off
