# volterra
Estimate Volterra series via Gaussian process regression using polynomial kernels

We provide the Matlab package poly_reg and the Python package preg for the same purpose. The Python package is currently under development and not yet publicly available.

## The poly_reg package
Poly_reg is a Matlab package for doing Gaussian process regression [1] using a polynomial covariance function. It can be used to find a Volterra or Wiener series expansion of an unknown system where only pairs of vector-valued inputs and scalar outputs are given [2]. The package provides the following functionalities:

1 .Automatic model selection according to one of three possible criteria: a. the log-likelihood of the observed data [1]; b. Geisser’s surrogate predictive probability [3]; c. the analytically computed leave-one-out error on the training set [4].
2. Computation of the predictive distribution (i.e., predictive mean and variance) of the output for a new, previously unseen input.
3. Computation of the explicit nth-order Volterra operator from the implicitly given polynomial expansion (see [2])

The available polynomial covariance functions of order p are

1.       the inhomogeneous polynomial kernel:  k(x, y) = (x’*y + 1)^p

2.       the adaptive polynomial kernel: k(x,y) = sum_i^p (w_i x’y)^I where each degree of nonlinearity receives an individual weight w_i that is found during model selection.

The package consists of the following routines:

* alltupels.m: computes all k-tupels of the numbers 1..n, repetitions inside the tupel are allowed. Needed for computing explicit Volterra operators in gpP_volt.
* gpP_amsd.m: automatic model selection for polynomial regression
* gpP_build.m: prepare a Gaussian Process structure (initialization)
* gpP_cov.m: compute covariance and Gram matrix for a polynomial covariance function and independent Gaussian noise model.
* gpP_eval.m: evaluate Gaussian process regression with polynomial covariance function and independent Gaussian noise model. This function is used to fit the hyperparameters.
* gpP_eval_wrapper: wrapper function needed for interfacing to Carl’s minimize function.
* gpP_gram.m: compute either Gram matrix for ihp, or the partial Gram matrices for each degree of nonlinearity for the adaptive polynomial kernel.
* gpP_pred.m: Gaussian process prediction with polynomial covariance function and independent Gaussian noise model.
* gpP_volt.m: computes the nth-degree explicit Volterra operator from an implicit Volterra series.
* invertCovariance.m: Inversion of the Covariance matrix. This function is the only place where the covariance matrix is inverted.
* norm_data.m: normalizes data such that every component remains in the interval [0,1]. Should be always used before polynomial regression since polynomials are very sensitive to the scaling of the data.  
* preg.m: Full Polynomial kernel regression with automatic model selection. Prints out MSE on training and test set.
* scalarProduct.m: computes scalar product depending on the selected kernel.

For installing the package, please download all files in this directory. In addition, you need the function minimize.m from Carl Rasmussen for doing model selection (see documentation) in the same directory. A simple 1D toy example showing the basic regression functionality is given in the programming example sinc_test.m (together with the plotting routine plot_predict.m). 

## Automatic model selection

The function gpP_amsd.m selects the optimal hyperparameters of the Gaussian process using either the minimum of minus the log likelihood (LLH) [1], Geissers surrogate predictive probability (GPP) [3] or the mean squared leave-one-out (LOO) error [4].

Usage: gp = gpP_amsd(nit, hp0, input, target, ptype, method, degrees)

where:

* nit          is the number of iterations
* hp0       is a vector of starting values for the hyperparameters
* input     is a n by D matrix of training inputs
* target    is a (column) vector (of size n) of targets
* ptype    is a string containing the polynomial type ('ihp':inhomogeneous polynomial kernel, 'ap': adaptive polynomial kernel)
* method is a string containing the model selection method ('llh':log likelihood, 'gpp': Geissers surrogate predictive probability, 'loo': Leave-one-out MSE)
* degrees are the polynominal degrees to do model selection over. e.g. 1:4, 7:10, 3, ...
 
gp is the returned gaussian process struct, containing the following values

* gp.minval           is the minimum value of the chosen evaluation criterion
* gp.degree          is the optimal polynomial degree
* gp.hp                is the optimal set hyperparameters
* gp.invK              inverse covariance matrix
* gp.invKt             gp.invK * target
* gp.Q                 Gram matrix
* gp.invQ             inverse Gram matrix
 
The hyperparameter vs is proportional to the "signal std dev" and vn is the "noise std dev". All hyperparameters are collected in the vector P as follows: (1) all polynomial kernels contain vs and vn as the first 2 parameters: P = [ log(vs); log(vn); (2) the adaptive polynomial kernel contains additionally the weights w_i for each degree of nonlinearity:  P = .. w_1; w_2; ..; w_{degree+1} ]

 
Notes

(1) the reason why the log of the parameters are used in vs, vn is that this often leads to a better conditioned (and unconstrained) optimization problem
than using the raw hyperparameters themselves.

(2) when loo is chosen as evaluation criterion, the derivative w.r.t. vs is always set to 0 (although it is not 0 in reality) such that vs remains constant during minimization. This ensures better convergence since the loo criterion is invariant under scaling of P. The resulting vs and vn values reflect only the proper signal-to-noise ratio, not the correct absolute values. This means that predicted variances in gpP_pred can only be determined up to a scale factor!


## Computation of the predictive distribution

Computation of the predictive distribution (i.e., predictive mean and variance) of the output for a new, previously unseen input is done by the gpP_pred.m.

Usage: [m var] = gpP_pred(gp, input, target, test)

where: 

* gp     is a Gaussian Process struct with precomputed variables
* test   is a nn by D matrix of test inputs, D as for 'input'
* pr_mean is a (column) vector (of size nn) of prediced means
* pr_var is a (column) vector (of size nn) of predicted variances

Note that predicted variances should be only computed when hyperparameters were chosen by either 'llh' or 'gpp'. Optimizing "loo" determines variances only up to a scale factor! The gaussian process struct 'gp' must contain the precomputed inverse variance 'gp.invK'. This is computed with 'gpP_eval'. If gpP_ams(d) is used for model selection all the necessary values are filled in. (Note that invKt must be computed with centered target data, but target must be given uncentered to get consistent predictions. On output, mean and (noise free) variance are returned.) If no model selection was done before, the ‘gp’ struct can be filled by the function gpP_build.m.

Usage: gp = buildGP(type, degree, hp, method, target)

where:

* type      is a string containing the polynomial type: ('ihp': inhomogeneous polynomial kernel, 'ap': adaptive polynomial kernel)
* degree   is the polynomial degree of the GP
* hp          is a (column) vector of hyperparameters
* method  is a string containing the model selection method ('llh':log likelihood, 'gpp': Geissers surrogate predictive probability, 'loo': Leave-one-out MSE)
* target    is a (column) vector (of size n) of targets
* gp         is the returned gaussian process struct, it simply contains the parameters provided to the function

 
## Computation of the explicit nth-order Volterra operator

The computation of the explicit nth-order Volterra operator from the implicitly given polynomial expansion is done by the function gpP_volt.m.

Usage: eta = gpP_volt(gp, volt_deg)

where:

* gp            is a Gaussian Process struct with precomputed variables
* volt_deg is the degree of the Volterra operator of interest
*  eta           are the coefficients of the monomials in the nth-order Volterra operator. The order of the coefficients follows the order of the monomials computed by alltupels.m, e.g., for 2D input and the 2nd-order operator, we have

indices         monomials            coefficient          result

1     1 =>      x1^2                                 h_11                 eta(1)

2     1 =>      x2*x1                               h_21                 eta(2)

1     2 =>      x1*x2                               h_21                 eta(3) 

2     2 =>      X2^2                                 h_22                 eta(4)

In general, starting with 1, the nth index is only changed when the (n-1)th index has undergone all possible permutations.

As in gpP_pred, the Gaussian process struct 'gp' must contain the precomputed inverse variance 'gp.invK'. This is computed with 'gpP_eval'. If gpP_amsd is used for model selection all the necessary values are filled in.


References
[1] Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes for machine learning. Cambridge, MA: MIT Press.
[2] Franz, M. O. and B. Schölkopf  (2006) A unifying view of Wiener and Volterra theory and polynomial kernel regression. Neural Computation (in press) [PDF].
[3] S. Sundararajan, S.S. Keerthi (2001). Predictive approaches for choosing hyperparameters in Gaussian processes. Neural Computation 13, 1103-1118.
[4] V. Vapnik (1982). Estimation of dependences based on empirical data. New York: Springer.
 
