function gp = preg(nit, hp, input, target, tinput, ttarget, ptype, method, degree)
% PREG: Polynomial kernel regression with automatic model selection. The 
% function returns a struct that contains all parameters of the best 
% Gaussian Process on the training data. Prints out MSE on training and 
% test set.
%
% usage: gp = preg(nit, hp, input, target, tinput, ttarget, ptype, method,
%					degree)
%
% where:
%
%   nit    is the number of iterations
%   hp     is a vector of starting values for the hyperparameters
%   input  is a n by D matrix of training inputs
%   target is a (column) vector (of size n) of targets
%   ptype  is a string containing the polynomial type
%               'ihp':      inhomogeneous polynomial kernel
%               'ap':       adaptive polynomial kernel
%   method is a string containing the model selection method
%               'llh':      log likelihood
%               'gpp':      Geissers surrogate predictive probability
%               'loo':      Leave-one-out MSE
%   degrees are the polynominal degrees to do model selection
%   over. e.g. 1:4, 7:10, 3, ...
%
%   gp	    is the returned gaussian process struct, also
%	    containing the following values
%	gp.minval  is the minimum value of the chosen evaluation criterion
%	gp.degree  is the optimal polynomial degree
%	gp.hp      is the optimal set hyperparameters
%	gp.invK	   inverse covariance matrix
%	gp.invKt   gp.invK * target
%	gp.Q	   Gram matrix
%
% All hyperparameters are collected in the vector P as follows:
%
% - all polynomial kernels contain vs and vn as the first 2 parameters:
%       P = [ log(vs); log(vn); ..
% - the adaptive polynomial kernel contains additionally the weights w_i for
%   each degree of nonlinearity:
%       P = .. w_1; w_2; ..; w_{degree+1} ]
% The hyperparameter vs is proportional to the "signal std dev" and vn is 
% the "noise std dev". 
%
% Notes
% (1) the reason why the log of the parameters are used in vs, vn is that this
% often leads to a better conditioned (and unconstrained) optimization problem
% than using the raw hyperparameters themselves.
% (2) when loo is chosen as evaluation criterion, the derivative w.r.t. vs
% is always set to 0 (although it is not 0 in reality) such that vs remains 
% constant during minimization. This ensures better convergence since the
% loo criterion is invariant under scaling of P. The resulting vs and vn
% values reflect only the proper signal-to-noise ratio, not the correct
% absolute values. This means that predicted variances in gpP_pred can only
% be determined up to a scale factor!
%
% (C) Copyright 2003 - 2006, 
% Matthias Franz (2005-05-26), Peter Vincent Gehler(2005-05-26)


gp = gpP_amsd(nit, hp, input, target, ptype, method, degree);
env = sprintf('poly_type: %s  degree: %d  model selection: %s', ptype, gp.degree, method);
mu_i = gpP_pred(gp, input); % prediction on training data
mu_t = gpP_pred(gp, tinput); % prediction on test data
disp(env);

% training and test error
err = (mu_i - target).^2;
tr_err = mean(err); % MSE on training data
err = (mu_t - ttarget).^2;
te_err = mean(err); % MSE on test data
disp(sprintf(' train: mse = %g, test: mse = %g', tr_err, te_err));
