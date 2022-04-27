function [invK,logdetK] = invertCovariance(K,ptype);
% invertCovariance: Inversion of the Covariance matrix
% This function is the only place where the covariance matrix is
% inverted. 
%
% usage: invK = invertCovariance(K,ptype)
%    or: [invK,logdetK] = invertCovariance(K,ptype)
%
% where: 
%
%   K		is the covariance matrix
%   ptype	is a string conainting the polynomial type
%
%   invK	the returned inverse of K
%   logdetK	the returned log determinat of the covariance matrix
%
%   if 'rcond(K)' < min_cond then the pseudo-inverse is
%		computed, otherwise the inverse is taken, the
%		default value is 1e-12;
%
% (C) Copyright 2006, Matthias Franz & Peter Gehler (2005-07-13)

% constants
min_cond = 1e-12; % minimum condition no. for computing inv instead of pinv


if nargout == 1 % only inverse required
	if rcond(K) < min_cond	% take pseudoinverse if condition number is bad
		invK = pinv(K);
	else
		invK = inv(K);		% otherwise take standard inverse
	end
else % both inverse and logdet required
	[U,S,V] = svd(K);
	invK = V * diag(1./diag(S)) * U';
	logdetK = sum(log(diag(S)));
end
