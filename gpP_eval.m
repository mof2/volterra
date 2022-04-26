function [gp, fP, dfP, e] = gpP_eval(P, gp)
% gpP_eval: evaluate Gaussian process regression with polynomial
% covariance function and independent Gaussian noise model. The function
% returns minus the log likelihood (LLH), Geissers surrogate predictive 
% probability (GPP) or the mean squared leave-one-out (LOO) error (depending on the 
% chosen method) and its partial derivatives with respect to the 
% hyperparameters; this function is used to fit the hyperparameters. 
%
% usage: [gp fP dfP e] = gpP_eval(P, gp)
%
% where:
%
%   P      is a (column) vector of hyperparameters
%   gp	   is a gaussian process struct (see buildGP for details),
%	   the returned gp will have filled precomputed matrices
%	   which are the used for prediction
%	gp.K		covariance matrix
%	gp.invK		inverse covariance matrix
%	gp.invKt	gp.invK * targets
%
%   fP     is the returned value of LLH, GPP or LOO 
%   dfP    is a (column) vector (of size D+2) of partial derivatives
%            of fP wrt each of the hyperparameters
%
% The hyperparameter vs is proportional to the "signal std dev" and
% vn is the "noise std dev". All hyperparameters are collected in
% the vector P as follows:
%
% - all kernels contain vs and vn as the first 2 parameters:
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
% This function can conveniently be used with the "minimize" function to train
% a Gaussian process:
%
% [P, fP, i] = minimize(P, 'gpP_eval', length, Ki, target, method)
%
% (C) Copyright 2003 - 2005, Matthias Franz (2005-02-09), after GP code for 
% Gaussian kernels of Carl Rasmussen

input = gp.input;				% inputs
target = gp.target;				% targets
vs = P(1);                      % signal variance
vn = P(2);                      % noise variance
e = 0;                          % set error to 0 by default


% sanity check for parameters, otherwise indicate divergence
if max(P) > 50 || ((length(P) > 3) && (max(exp(P(3:end))) < 0.01 || max(exp(P(3:end))) > 100000))
    fP = 100000000000000;     % e^50 makes no sense in polynomials when input is
    dfP = zeros(size(P));      % sensibly normalized.
    e = 1;                      % set error flag => divergence
    return
end
if sum(isnan(gp.hp))
    error('gpP_eval: hyperparameters reached NaN!');
end

% center training data
n = length(target);
mu = mean(target);
target = target - mu;

% compute Covariance matrix 
[K, gp] = gpP_cov(P,gp);

% dC/dvs
if ~strcmp(gp.method, 'loo') % d/dvs not needed for loo
    dC(:,:,1) = gp.Q;             
end

% dC/dvn
dC(:,:,2) = eye(size(gp.Q));

% dC/d_zeta for ap akernel
if strcmp(gp.ptype, 'ap')
	for i = 1:length(P)-2
        dC(:,:,i+2) = exp(2*vs)*gp.Ki(:,:,i);
    end
end

% inner derivatives
d_in(1) = 2*exp(2*vs);
d_in(2) = 2*exp(2*vn);
for i = 3:length(P)
    d_in(i) = exp(P(i));
end

dfP = zeros(length(P), 1);
switch gp.method % model selection method
    
    case 'llh', % log likelihood
        [invK logdetK]= invertCovariance(K,gp.ptype); % fast computation in C
        invKt = invK*target;
        loglikelihood = -0.5*logdetK - 0.5*target'*invKt - 0.5*n*log(2*pi);
        fP = -loglikelihood;

        % ... and its partial derivatives
        for i=1:length(P)
            H1 = invK*dC(:,:,i);
            H2 = invKt'*(dC(:,:,i)*invKt);
            dfP(i) = d_in(i)/2*(trace(H1) - H2);
        end

    case 'gpp', % Geissers surrogate predictive probability
        invK = invertCovariance(K,gp.ptype); 
        invKt = invK*target;
        diaginvK = diag(invK);
        gspp = 0.5*(mean(log(diaginvK) - invKt.^2 ./ diaginvK) - log(2*pi));
        fP = -gspp;

        % ... and its negative partial derivatives
        invKtodiaginvK = invKt ./ diaginvK;
        h = 0.5*(1./diaginvK + invKtodiaginvK.^2)';
        for i=1:length(P)
            H = invK*(dC(:,:,i)*invK);
            dfP(i) = -d_in(i)/n*(invKtodiaginvK'*(H*target) - h*diag(H));
        end

    case 'loo', % Leave-one-out MSE
        invK = invertCovariance(K,gp.ptype); 
        invKt = invK*target;
        diaginvK = diag(invK);
        loo = -mean((invKt./diaginvK).^2);
        fP = -loo;

        % ... and its partial derivatives
        h = ((invKt.^2) ./ (diaginvK.^3))';
        invKtodiaginvK = (invKt./diaginvK.^2)';
        dfP(1) = 0; % vs is kept constant (loo is invariant w.r.t. scaling)
        for i=2:length(P)
            H = invK*dC(:,:,i)*invK;
            dfP(i) = -2*d_in(i)/n*(invKtodiaginvK*(H*target) - h*diag(H));
        end

    otherwise,
        error('gpP_eval',...
            sprintf('Model evaluation method not supported : "%s"', gp.method));
    
end % switch method

% retain (inverse) covariance and expensive K\t if applicable
gp.K = K;
gp.invK = invK;
gp.invKt = invKt;
