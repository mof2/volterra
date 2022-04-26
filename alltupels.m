function R = alltupels(n, k, Rold)
% ALLTUPELS: computes all k-tupels of the numbers 1..n, repetitions 
% inside the tupel are allowed. 
%
% usage: R = alltupels(n, k)
% 
% where:
%
%	n	is the maximum entry in the tupel (minimum is always 1),
%		correpsonds to the dimensionality of the input.
%
%	k	is the desired tupel size.
%
%
%	R	is a n^k x k matrix where the rows contain the tupels in a fixed
%		order, e.g., for n = k = 2, we have 
%
%		R =		1     1
%				2     1
%				1     2
%				2     2
%
%		In general, starting with 1, the nth index is only changed when 
%		the (n-1)th index has undergone all possible permutations.
%
%
% (C) Copyright 2005, Matthias Franz (2005-03-01)


if k == 0 || n == 0
    R = []
    return
end

if nargin == 2
    Rold = [1:n]';
end

if k == 1
    R = Rold;
    return
end

R = [];
for i=1:n
    app = [Rold, i*ones(size(Rold,1),1)];
    R = [R; app];
end
if size(R,2) ~= k
    R = alltupels(n, k, R);
end
