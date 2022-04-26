function [pr_mean, pr_var] = gpP_pred(gp, test);
% gpP_pred: Gaussian process prediction with polynomial
% covariance function and independent Gaussian noise model. The
% gaussian process struct 'gp' must contain the precomputed inverse
% variance 'gp.invK'. This is computed with 'gpP_eval'. If
% gpP_ams(d) is used for model selection all the necessary values
% are filled in. (Note that invKt must be computed with centered target data, but
% target must be given uncentered to get consistent predictions. On output,
% mean and (noise free) variance are returned.)
%
% usage: [m var] = gpP_pred(gp, input, target, test)
%
% where:
%
%   gp     is a Gaussian Process struct with precomputed variables
%   test   is a nn by D matrix of test inputs, D as for 'input'
%
%   pr_mean is a (column) vector (of size nn) of prediced means
%   pr_var is a (column) vector (of size nn) of predicted variances
%
% Note that predicted variances should be only computed when
% hyperparameters were chosen by either 'llh' or 'gpp'. Optimizing "loo"
% determines variances only up to a scale factor!
%
%
% (C) Copyright 2005, Matthias Franz (2005-03-01)


% constants
max_length = 10000;             % maximal length of testset before switching
                                % to a different test set evaluation method

vs = gp.hp(1);                  % signal variance
vn = gp.hp(2);                  % noise variance
input = gp.input;				% training input
target = gp.target;				% training output
[nn, D] = size(test);           % number of test cases

% mean of training outputs
mu = mean(target);				

% compute the necessary entities if not precomputed. The model
% selection gpP_ams(d) will take care of this and the inverse
% covariance together will be already computed. 
% this is a bit nasty but ensures that gpP_pred will work as a
% stand alone method2
if isfield(gp,'invK'), invK = gp.invK; 
else % inverse covariance is missing

	% sanity check for parameters
	if gp.degree < 1
		error('regression for degree < 1 not possible.');
	end

	% Gram matrix
	if strcmp(gp.ptype,'ap')
		gp.Ki = gpP_gram(gp.degree, gp.ptype, gp.hp, input);
		dims = size(gp.Ki);
		Q = ones(dims(1)) + exp(gp.hp(3))*gp.Ki(:,:,1);
		for i = 2:length(gp.hp)-2
			Q = Q + exp(gp.hp(i+2))*gp.Ki(:,:,i);
		end
		gp.Q = Q;
	else
		gp.Q = gpP_gram(gp.degree, gp.ptype, gp.hp, input);
	end
	gp.K = gpP_cov(gp.hp,gp); % covariance
    gp.invK = invertCovariance(gp.K, gp.ptype); % inverse covariance
end % of computing the inverse covariance

% compute inv(Covariance) * targets 
if isfield(gp,'invKt')
	invKt = gp.invKt;
else % field is missing
	target = target - mu; % center training data
	invKt = gp.invK * target; 
end

% prediction
if nn < max_length % small test set -> compute all at once
    
    % cross-covariance
    Qt = scalarProduct(input, test, gp.ptype, gp.degree, gp.hp);
    Qt = exp(2*vs)*Qt;
    
    % ... write out the desired terms
    pr_mean = Qt' * invKt + mu;

    % variances also required ...
    if nargout > 1 
        % ... target covariance
        tt = scalarProduct(test, test, gp.ptype, gp.degree, gp.hp);
	    tt = exp(2*vs)*diag(tt);
        pr_var = tt - sum(Qt.*(gp.invK*Qt), 1)';
    end

else % large test set => do it one by one without storing full
    
    pr_mean = zeros(nn, 1);
    pr_var = zeros(nn, 1);
    
    for j = 1:max_length:nn
		indices = j:min(j+max_length,length(test));
        tst = test(indices,:);
        Qt = scalarProduct(input, tst, gp.ptype, gp.degree, gp.hp);
        Qt = exp(2*vs)*Qt;
        
		% ... write out the desired terms
        pr_mean(indices) = Qt'*invKt + mu;  % predicted means

        if nargout == 2
			tt = scalarProduct(tst, tst, gp.ptype, gp.degree, gp.hp);
			tt = exp(2*vs)*diag(tt);
			pr_var(indices ) = tt - sum(Qt'*(gp.invK*Qt),1)';
        end
    end
end
