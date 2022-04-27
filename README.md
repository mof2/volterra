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

For installing the package, please download all files in the directory *poly_reg*. A simple 1D toy example showing the basic regression functionality is given in the programming example sinc_test.m (together with the plotting routine plot_predict.m). 

Further documentation on *poly_reg* can be found [here.](https://github.com/mof2/volterra/wiki)


## References

[1] Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes for machine learning. Cambridge, MA: MIT Press.

[2] Franz, M. O. and B. Schölkopf (2006). A unifying view of Wiener and Volterra theory and polynomial kernel regression. Neural Computation. 18, 3097 – 3118.

[3] S. Sundararajan, S.S. Keerthi (2001). Predictive approaches for choosing hyperparameters in Gaussian processes. Neural Computation 13, 1103-1118.

[4] V. Vapnik (1982). Estimation of dependences based on empirical data. New York: Springer.
 
