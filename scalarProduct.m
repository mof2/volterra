function kxy = scalarProduct(x, y, ptype, degree, hp) 
% scalarProduct: Compute the scalar product between x and y
% depending on the selected kernel
%
% usage: kxy = scalarProduct(x,y,kernel_type,degree);
%     or kxy = scalarProduct(x,y,kernel_type,degree,hp);
%
% where
%
%	x,y	are matrices containg the data samples [nSamples x dim]
%	xy	is the standard scalar product of x and y (x*y')
%		only for kernels depending on the inner Product
%	kernel_type	a string containing the kernel type
%		'ihp'	inhomogenous polynomial kernel
%		'ap'	adaptive polynomial kernel
%	degree:	the degree of the kernel
%	hp	Hyperparameters for ap kernel
%
%	kxy	the returned scalar Product k(x,y)
%	
% - 'ihp' the inhomogeneous polynomial kernel: 
%       K_pq = ( x_p . x_q + 1 )^degree
% - 'ap' the adaptive polynomial kernel:
%       K_pq = sum_i {w_i (x_p . x_q)^i }
%
% (C) Copyright 2006, Matthias Franz & Peter Gehler

% precompute inner product
xy = x * y';
[n,nn] = size(xy);

% choose k and compute k(x,y)
switch ptype
	case 'ihp', % inhomogeneous polynomial kernel
		if degree == 1 % linear kernel
			kxy = xy + ones(n, nn);
		else % nonlinear kernel
			kxy = (1+xy).^degree;
		end
	case 'ap',   % adaptive polynomial kernel
		QQ = xy;
		if nargin < 5,
			hp(3:degree+2) = 0;
		end
		kxy = ones(n, nn) + exp(hp(3))*QQ;
		for i = 2:degree
			kxy = kxy + exp(hp(i+2))*QQ.^i;
		end
	otherwise
		error(sprintf('kernel type %s not supported', ptype));
end
