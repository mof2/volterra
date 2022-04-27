function Ki = gpP_gram(degree, ptype, hp, input)

% gpP_gram: compute either Gram matrix for ihp, or the partial Gram 
% matrices for each degree of nonlinearity for the adaptive polynomial 
% kernel. 
%
% usage: 
%   Ki = gpP_gram(degree,ptype,hp,input) for standard GP regression / KRR
%
% where:
%
%   degree is the degree of the polynomial kernel
%   ptype  is the type of the kernel, see 'help buildGP'
%   hp	   are the hyperparameters of the kernel
%   input  is a n by D matrix of training inputs
% where D is the dimension of the input.
%
%   Ki     are either the Gram matrix for the inhomogeneous and the trans-
%          lation-variantkernel  or the single Gram matrices for the different
%          orders of nonlinearity for the adaptive polynomial
%          kernel. In that case the Gram matrix is computed in 'gpP_cov.m'
%
% See 'help scalarProduct' for the different types of supported kernels.
%
% (C) Copyright 2006, Matthias Franz (2006-05-22)

[n, D] = size(input);           % number of examples and dimension of input space

% sanity check for parameters
if degree < 1, error('regression for degree < 1 not possible.'); end

% compute Gram matrix
switch ptype
	case 'ap', % partial gram matrices for adaptive kernels
		Ki = zeros(n, n, degree);
		Ki(:,:,1) = input*input';
		for i=2:degree
			Ki(:,:,i) = Ki(:,:,1).^i;
		end
	otherwise % Gram matrix
		Ki = scalarProduct(input, input, ptype, degree);
end

