function [fP, dfP, e] = gpP_eval_wrapper(P, gp);
% gpP_eval_wrapper - wrapper for gpp_eval
%   In order to use minimize one has to have a function with three outputs
%   [fx, dfx, e] = func(something); To retain the variables computed from
%   gpP_eval the first output is the gp struct. For compability with
%   minimize this wrapper is needed. Here's the complete file: 
%
%   function [fP, dfP, e] = gpP_eval_wrapper(P,gp)
%       [gp,fP,dfP,e] = gpP_eval(P,gp);
%
%
%  (C) Copyright 2005, Peter Gehler & M.O.Franz

gp.hp = P;
[gp,fP,dfP,e] = gpP_eval(P,gp);
