function eta = gpP_volt(gp, volt_deg)
% gpP_volt: computes the nth-degree explicit Volterra operator from an 
% implicit Volterra series given in the Gaussian process struct gp. The
% Gaussian process struct 'gp' must contain the precomputed inverse
% variance 'gp.invK'. This is computed with 'gpP_eval'. If gpP_amsd is 
% used for model selection all the necessary values are filled in. 
%
% usage: eta = gpP_volt(gp, volt_deg)
%
% where:
%
%   gp			is a Gaussian Process struct with precomputed variables
%   volt_deg	is the degree of the Volterra operator of interest
%
%   eta			are the coefficients of the monomials in the nth-order
%				Volterra operator. The order of the coefficients follows
%				the order of the monomials computed by alltupels.m, e.g.,
%				for 2D input and the 2nd-order operator, we have 
%
%					indices			monomials		coefficient		result
%					1     1	=>		x1^2			h_11			eta(1)
%					2     1	=>		x2*x1			h_21			eta(2)
%					1     2	=>		x1*x2			h_21			eta(3)	
%					2     2	=>		X2^2			h_22			eta(4)
%
%				In general, starting with 1, the nth index is only changed
%				when the (n-1)th index has undergone all possible
%				permutations.
%
%
% (C) Copyright 2005, Matthias Franz (2005-03-01)


% computes explicit coefficients of Volterra operator of given order for
% dual coefficient vector alpha and data matrix X (data points in rows) 

vs = gp.hp(1);                  % signal variance
vn = gp.hp(2);                  % noise variance
input = gp.input;				% training input
target = gp.target;				% training output

% mean of training outputs
mu = mean(target);				

% compute the necessary entities if not precomputed. The model
% selection gpP_amsd will take care of this and the inverse
% covariance together will be already computed. 
% this is a bit nasty but ensures that gpP_volt will work as a
% stand alone method
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

% compute Phi
dims = size(input);
N = dims(1); % number of data points
len = dims(2); % dimensionality of input

% special case: constant term
if volt_deg == 0
    phi = ones(1, N);
    
% special case: linear term
elseif volt_deg == 1
    phi = input';

% nonlinear terms
else

    % compute indices of all tupels of given order
    ind = alltupels(len, volt_deg); % get all possible tupels of indices
    M = len^volt_deg; % number of monomials of given order

    % fill design matrix
    phi = zeros(M, N);
    for i=1:N
        for j=1:M
            phi(j,i) = input(i, ind(j,1));
            for k=2:volt_deg
                phi(j,i) = phi(j,i) * input(i, ind(j,k));
            end
        end
    end
end

% regression weights
w = exp(2*vs)*invKt;

% correction factor for different kernel types
switch gp.ptype 
    case 'ihp',
        a = factorial(gp.degree)/factorial(volt_deg)/factorial(gp.degree - volt_deg);
    case 'ap'
        a = exp(gp.hp(volt_deg + 2));
   otherwise,
        error('gpP_volt', sprintf('Polynomial type not supported : "%s"', gp.ptype));
end

% compute Volterra coefficients
eta = a * phi * w;
