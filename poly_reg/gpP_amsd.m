function gp = gpP_amsd(nit, hp0, input, target, ptype, method, degrees)
% gpP_amsd: do automatic model selection for polynomial regression. The function
% returns the minimum of minus the log likelihood (LLH), Geissers surrogate predictive 
% probability (GPP) or the mean squared leave-one-out (LOO) error (depending on the 
% chosen method) and the optimal degree and hyperparameters. 
%
% usage: gp = gpP_amsd(nit, hp0, input, target, ptype, method, degrees)
%               for standard GP regression / KRR
%
% where:
%
%   nit     is the number of iterations
%   hp0     is a vector of starting values for the hyperparameters
%   input  is a n by D matrix of training inputs
%   target is a (column) vector (of size n) of targets
%   ptype  is a string containing the polynomial type
%               'ihp':      inhomogeneous polynomial kernel
%               'ap':       adaptive polynomial kernel
%               'tvg':      translation variant gaussian (exp(-<x,y>))
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
%	gp.invQ	   inverse Gram matrix
%
% The hyperparameter vs is 
% proportional to the "signal std dev" and vn is the "noise std dev". All 
% hyperparameters are collected in the vector P as follows:
%
% - all polynomial kernels contain vs and vn as the first 2 parameters:
%       P = [ log(vs); log(vn); ..
% - the adaptive polynomial kernel contains additionally the weights w_i for
%   each degree of nonlinearity:
%       P = .. w_1; w_2; ..; w_{degree+1} ]
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
% (C) Copyright 2003 - 2005, 
% Matthias Franz (2005-05-26), Peter Vincent Gehler(2005-05-26)


% sanity check of the parameters
if strcmp(ptype,'tvg') && max(degrees) > 1 % no degrees for tvg
    warning(sprintf('kernel type %s has always degree 1',ptype));
    degrees = 1;
end
if length(hp0) > 2 % delete additional hps
	warning('Only two hyperparameter init values allowed, deleting the rest...');
    hp0(3:end) = [];
end

min_e = 1/eps; % init minimum performance
for deg = degrees
    
    % init hyperparams to start value for each degree
    hp1 = hp0;
	if strcmp(ptype, 'ap') % add ones as init values for ap
		hp1 = [hp0; ones(deg, 1)];
	end
	
	% build gaussian process struct for each degree
    gp = gpP_build(ptype, deg, hp1, method, input, target);

    % compute Gram matrix (needed only once during minimization)
    switch ptype
		case 'ap' % partial Gram matrices for ap
			gp.Ki = gpP_gram(deg, ptype, hp1, input);
		otherwise
			gp.Q = gpP_gram(deg, ptype, hp1, input);
	end
	
	% find best hyperparams
    [gp.hp, fP, i] = minimize(hp1, 'gpP_eval_wrapper', nit, gp); 
    
	% record performance and compute covariance
    [gp, e] = gpP_eval(gp.hp, gp); % performance on training set
    fprintf('degree %d, used %d iterations, minval %.04g\n', deg, i, e);

    % retain information if the current performance is best. 
    if e < min_e
        min_e = e;
        gp.minval = e;
        tgp = gp;
    end
end

% return Gaussian process with best performance
gp = tgp;

% remove all data which is not needed for prediction
if isfield(gp,'Ki'), gp = rmfield(gp,'Ki'); end

