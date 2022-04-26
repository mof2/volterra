function [XtrnN, XtstN] = norm_data(Xtrn, Xtst)
% NORM_DATA: normalizes data such that every component remains in the
% interval [0,1]. Should be always used before polynomial regression since
% polynomials are very sensitive to the scaling of the data.
%
% usage: [XtrnN, XtstN, XstatN] = norm_data(Xtrn, Xtst, Xstat)
% 
% where:
%
%	Xtrn	is a n x d matrix containing n training data points of
%			dimension d
%
%	Xtst	is a m x d matrix containing m test data points of the same
%			dimension f
%
%
%	XtrnN	are the normalised training data
%
%	XtstN	are the normalised test data
%
% Note that only the training data are guaranteed to be in [0, 1]^d, the 
% test data	are normalised using the extrema of the training data. 
%
%
% (C) Copyright 2005, Matthias Franz (2005-03-01)

[n_trn, d] = size(Xtrn);
[n_tst, d] = size(Xtst);
xmax = max(Xtrn); 
xmin = min(Xtrn);
XtrnN = (Xtrn - repmat(xmin, n_trn, 1)) ./ repmat(xmax - xmin, n_trn, 1);
XtstN = (Xtst - repmat(xmin, n_tst, 1)) ./ repmat(xmax - xmin, n_tst, 1);
