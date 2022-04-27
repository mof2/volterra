function gp = gpP_build(type, degree, hp, method, input, target);
% buildGP: prepare a Gaussian process structure 
%
% usage: gp = buildGP(type, degree, hp, method, target)
% where:
%
%   type  is a string containing the polynomial type
%               'ihp':      inhomogeneous polynomial kernel
%               'ap':       adaptive polynomial kernel
%   degree  is the polynomial degree of the GP
%   hp      is a (column) vector of hyperparameters
%   method  is a string containing the model selection method
%               'llh':      log likelihood
%               'gpp':      Geissers surrogate predictive probability
%               'loo':      Leave-one-out MSE
%   target  is a (column) vector (of size n) of targets
%
%   gp	    is the returned gaussian process struct, it simply contains
%			the parameters provided to the function
%
% - all polynomial kernels contain vs and vn as the first 2 parameters:
%       P = [ log(vs); log(vn); ..
% - the adaptive polynomial kernel contains additionally the weights w_i for
%   each degree of nonlinearity:
%       P = .. w_1; w_2; ..; w_{degree+1} ]
%
%  (C) Copyright 2005, Peter Gehler & M.O.Franz

gp.ptype = type;
gp.degree = degree; 
gp.hp = hp; 
gp.method = method;
gp.target = target;
gp.input = input;
