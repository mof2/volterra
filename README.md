# volterra
Estimate Volterra series via Gaussian process regression using polynomial kernels

We provide the Matlab package *poly_reg* and the Python package *preg* for the same purpose. The Python package is currently under development and not yet publicly available.

## The poly_reg package
Authors: Matthias O. Franz and Peter V. Gehler

Poly_reg is a Matlab package for doing Gaussian process regression [1] using a polynomial covariance function. It can be used to find a Volterra or Wiener series expansion of an unknown system where only pairs of vector-valued inputs and scalar outputs are given [2]. The package provides the following functionalities:

1. Automatic model selection according to one of three possible criteria: a. the log-likelihood of the observed data [1]; b. Geisser’s surrogate predictive probability [3]; c. the analytically computed leave-one-out error on the training set [4].
2. Computation of the predictive distribution (i.e., predictive mean and variance) of the output for a new, previously unseen input.
3. Computation of the explicit nth-order Volterra operator from the implicitly given polynomial expansion (see [2])

The available polynomial covariance functions of order p are

1. the inhomogeneous polynomial kernel:
```
         k(x, y) = (x’*y + 1)^p
```
2. the adaptive polynomial kernel: 
```
         k(x,y) = sum_i^p (w_i x’y)^I 
```
where each degree of nonlinearity receives an individual weight w_i that is found during model selection.

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
* minimize.m: Carl Rasmussen's minimize, provided here for convenience, see [http://learning.eng.cam.ac.uk/carl/code/minimize/](http://learning.eng.cam.ac.uk/carl/code/minimize/).

For installing the package, please download all files in the directory *poly_reg*. A simple 1D toy example showing the basic regression functionality is given in the programming example sinc_test.m (together with the plotting routine plot_predict.m). 

Further documentation on *poly_reg* can be found [here.](https://github.com/mof2/volterra/wiki)


## References

[1] Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes for machine learning. Cambridge, MA: MIT Press.

[2] Franz, M. O. and B. Schölkopf (2006). A unifying view of Wiener and Volterra theory and polynomial kernel regression. Neural Computation. 18, 3097 – 3118.

[3] S. Sundararajan, S.S. Keerthi (2001). Predictive approaches for choosing hyperparameters in Gaussian processes. Neural Computation 13, 1103-1118.

[4] V. Vapnik (1982). Estimation of dependences based on empirical data. New York: Springer.
 
