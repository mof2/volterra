function [K, gp] = gpP_cov(hp, gp);
% gpP_cov: compute covariance and Gram matrix for a polynomial covariance 
% function and independent Gaussian noise model. 
%
% usage: [K gp] = gpP_cov(hp,gp) 
%
% where:
%
%   gp     is gaussian process struct (see 'buildGP.m')
%
%   K      is the returned covariance matrix
%   gp     is the returned gaussian process. For adaptive polynomial
%	   kernel 'ap' the following values are updated
%	   gp.Q		Gram matrix
%	   gp.invQ	inverse Gram matrix
%	The gram matrix is computed by 'scalarProduct'. 
% The form of the covariance function is
%       C(x_p,x_q) = vs^2 * k(x_p,x_q) + vn^2 * delta_{p,q}
% The second term with the kronecker delta is the
% noise contribution. The hyperparameter vs is the "signal std dev"
% and vn is the "noise std dev". All hyperparameters are collected 
% in the vector P as follows:
%
% - the inhomogeneous kernel contains vs and vn as the first 2 parameters:
%       P = [ log(vs); log(vn); ..
% - the adaptive polynomial kernel contains additionally the weights w_i for
%   each degree of nonlinearity:
%       P = .. w_1; w_2; ..; w_{degree+1} ]
%
% Note: the reason why the log of the parameters are used in vs, vn is that this
% often leads to a better conditioned (and unconstrained) optimization problem
% than using the raw hyperparameters themselves.
%
% (C) Copyright 2005, Matthias Franz (2005-03-01)

% constants
min_cond = 1e-12;	% maximum condition number for adding an additional
					% ridge in computing the inverse
reg = 1e-7;			% ridge size

vs = hp(1);         % signal variance
vn = hp(2);         % noise variance

% Gram matrix Q
if strcmp(gp.ptype, 'ap') % ap kernel
    dims = size(gp.Ki);
    Q = ones(dims(1)) + exp(hp(3))*gp.Ki(:,:,1);
    for i = 2:length(hp)-2
        Q = Q + exp(hp(i+2))*gp.Ki(:,:,i);
	end
	gp.Q = Q;
end

% sanity check
if ~isfield(gp,'Q')
    error('no gram matrix provided to compute the covariance matrix');
end

% compute covariance by adding noise delta
K = exp(2*vs)*gp.Q + exp(2*vn)*eye(size(gp.Q)); 

% add a small ridge if K is badly conditioned
if rcond(K) < min_cond
    [h h] = size(K);
    K = K + reg*trace(abs(K))*eye(h)/h;
end
