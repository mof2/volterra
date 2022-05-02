% 22.04.22. Demo script for using the poly_reg package in order to obtain
% impulse response functions, linear and also higher-order, a.k.a. Volterra
% kernels: http://www.scholarpedia.org/article/Volterra_and_Wiener_series
% and predict the system response to arbitrary input/forcing.

clear
close all

% Add libraries to path if not already in there
% lib =    {'/Users/tamasbodai/Documents/MATLAB/Useful/poly_reg'; ...
%               '/Users/bodai/Documents/MATLAB/Useful/poly_reg'};
%for i1 = 1:length(lib)
%    if isempty(strfind(path, lib{i1}));
%        path(lib{i1},path);
%    end
%end

t_tr = 10; % transient time to reach the attractor
h = 0.1; % trajectory sampling increment
tspan = 0:h:t_tr;

% 1. Simulation: converge to the fixed point attractor of the autonomous
% system
u = @(t)0; % dummy for the input
myfun = @(t,x)lin_sys(t,x,u);
[t,x] = ode45(myfun,tspan,[4; 1]);

% The attractor (we know it's simply the origin (0,0))
x_a = x(end,:);

fh = figure;
plot(t,x(:,1),'-o',t,x(:,2),'-o');
xlabel('Time t');
legend('x_1','x_2')

% 2. Force the system by a step-wise input u(t) being initially situated on
% the attractor
nt = length(tspan);
u = @(t)1;
myfun = @(t,x)lin_sys(t,x,u);
[t,x] = ode45(myfun,tspan,x_a);

figure(fh); hold on
plot(t+t_tr,x(:,1),'-o',t+t_tr,x(:,2),'-o');

% The linear or first-order (1) impulse response function (IRF), a.k.a.
% Volterra kernel: http://www.scholarpedia.org/article/Volterra_and_Wiener_series
h_1 = diff(x(:,1));
figure
plot(h_1,'-o');

% explicit input
u1 = cat(2,zeros(size(tspan)), ones(size(tspan)));
step = 102;

% convert input in sliding window format for regression (see 4.)
memory = 50; % how may sampling points back in time are used as input for the Volterra operators
x0 = zeros(length(tspan), memory);
for i=1:length(tspan)
    x0(i,:) = u1(step + i - memory - 1:step + i - 2);
end
y0 = x(:,1); % associated output

% 3. Test the prediction performed by using the IRF; apply any desirable
% forcing u
om = 1;
u = @(t)sin(om*t);
myfun = @(t,x)lin_sys(t,x,u);
[t,x] = ode45(myfun,tspan,x_a);

% test input and output
u2 = u(tspan);
x1 = zeros(length(tspan) - memory, memory);
y1 = zeros(length(tspan) - memory, 1);
for i=1:length(tspan) - memory
    x1(i,:) = u2(i:memory + i - 1);
    y1(i) = x(memory + i - 1, 1);
end

% The prediction obtained by using the IRF
x_pr1 = conv(u(tspan),h_1);
fh = figure;
plot(t,x(:,1),'-o',tspan(1:end-1),x_pr1(1:nt-1),'-o');

% 4. Obtain h_1 using poly_reg
% Copy lines from sinc_test.m:
% constants
max_order = 8;		% maximum polynomial order
ptype = 'ihp';		% kernel type ('ihp' or 'ap')
method = 'gpp';		% model selection method ('llh', 'gpp', 'loo')
n_iter = 20;		% number of iterations
hp0 = [log(0.6); log(sqrt(0.001))]; % initial guess for hyperparameters

% do regression and predict on training data
erg = gpP_amsd(n_iter, hp0, x0, y0, ptype, method, 1);
mu = gpP_pred(erg, x0);

% plot prediction on training set
figure
plot(tspan, mu, '-o', tspan, y0,'-o')
title('Prediction on training set (linear)')
xlabel('time [s]')
legend('prediction','true model output')

% prediction on test set
figure
mu = gpP_pred(erg, x1);
plot(0:h:5, mu,'-o', 0:h:5, y1,'-o')
title('Prediction on test set (linear)')
xlabel('time [s]')
legend('prediction','true model output')

% compute explicit 1st-order Volterra operator
volt_deg = 1;
eta = gpP_volt(erg, volt_deg);
figure
memspan = 0:h:4.9;
resp = flip(eta); % time delay is in reverse time direction
plot(memspan, resp,'-o', memspan, h_1(1:memory), '-o')
title('First order Volterra kernel')
xlabel('Delay \tau [s]');
legend('polynomial regression','impulse response')

% random input for 2nd-order nonlinear system
x = rand(2000,1);
y = conv(x,h_1);
y = y.^2;
x2 = zeros(length(x) - memory, memory);
y2 = zeros(length(x) - memory, 1);
for i=1:length(x) - memory
    x2(i,:) = x(i:memory + i - 1);
    y2(i) = y(memory + i - 1);
end

% 5. Obtain e.g. the second-order Volterra kernel for a nonlinear operator
% do regression and predict on training data
hp0 = [log(0.6); log(sqrt(0.001))]; % initial guess for hyperparameters
erg = gpP_amsd(n_iter, hp0, x2, y2, ptype, method, 2);
[mu, s2] = gpP_pred(erg, x2);

% plot prediction on training set
figure
tspan = h*((1:length(mu)) - 1);
plot(tspan, mu,'-o', tspan, y2, '-o')
title('Prediction on training set (nonlinear)')
xlabel('time [s]')
legend('prediction','true model output')

% prediction on test set
t = h*(0:499);
x = sin(t);
y = conv(x,h_1);
y = y.^2;
x3 = zeros(length(x) - memory, memory);
y3 = zeros(length(x) - memory, 1);
for i=1:length(x) - memory
    x3(i,:) = x(i:memory + i - 1);
    y3(i) = y(memory + i - 1);
end

figure
[mu, s2] = gpP_pred(erg, x3);
tspan = h*((1:length(mu)) - 1);
plot(tspan, mu,'-o', tspan, y3, '-o')
title('Prediction on test set (nonlinear)')
xlabel('time [s]')
legend('prediction','true model output')

% 2nd-order Volterra kernel
volt_deg = 2;
eta = gpP_volt(erg, volt_deg);
eta = reshape(eta, memory, memory); % kernel is returned as 1d vector,
                                    % needs to be reshaped into 2d
eta = flipud(fliplr(eta)); % time delay is measured in reverse time direction
                          % kernel mst be flipped for standrad display
                          % convention

figure
[X,Y] = meshgrid(memspan,memspan);
surf(X,Y,eta)
title('Second order Volterra kernel')
xlabel('Delay \tau_1 [s]');
ylabel('Delay \tau_2 [s]');

% Linear system with a stable spiral fixed point attractor.
function dydt = lin_sys(t,x,u)
    dydt = [-1 -2;
             3 -1]*x + u(t);
end