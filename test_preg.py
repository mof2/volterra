from __future__ import print_function
import logging
from math import log, sqrt
import numpy as np
from preg import Preg
from numpy import linalg as LA


# get a logger instance
lg = logging.getLogger('test')

# test standard case of positive definite covariance
def test_invert_cov_posdef():

    gp = Preg(lg, 'ihp', 3, [log(0.6), log(sqrt(0.001))])
    x = np.array([[0.89153353, 0.02574414], [0.32448397, 0.90612746],
        [0.31792538, 0.35136833]])
    gp.X_ = x
    gp.cov(gp.hyp)
    gp.invert_cov('llh')
    print('Inverse covariance:')
    print(gp.invK_)
    print('K * invK:')
    print(np.dot(gp.K_, gp.invK_))
    d = (np.dot(gp.K_, gp.invK_) - np.eye(3)).max()
    print('Maximal difference between K*invK and 1: {:e}'.format(d))
    assert d < 1e-12


# test case of non-positive-definite covariance
def test_invert_cov_nonposdef():

    gp = Preg(lg, 'ihp', 3, [log(0.6), log(sqrt(0.001))])
    x = np.array([[0.89153353e9, 0.02574414e-9], [0.32448397, 0.90612746e9],
        [0.31792538e-9, 0.35136833e9]])
    gp.X_ = x
    gp.cov(gp.hyp)
    gp.invert_cov('llh')
    print('Using standard inverse instead of Cholesky inverse...')
    print('Covariance:')
    print(gp.K_)
    print('Condition number of covariance: {:e}'.format(np.linalg.cond(gp.K_, np.inf)))
    print('Inverse covariance:')
    print(gp.invK_)
    print('K * invK:')
    print(np.dot(gp.K_, gp.invK_))
    d = np.abs((np.dot(gp.K_, gp.invK_) - np.eye(3))).max()
    print('Maximal difference between K*invK and 1: {:e}'.format(d))
    assert d < 0.26


# check gradient
def check_gradient(gp, y, modsel):

    eps = 0.00001 # small gradient step
    hyp = gp.hyp.copy()
    (f, df, e) = gp.objective(y, hyp, modsel) # get partial derivatives in df
    dy = np.zeros(np.shape(df)) # finite difference estimates of partial derivatives
    for i in np.arange(0, np.shape(hyp)[0]): # perturb each dimension
        hyp[i] = hyp[i] + eps # positive step
        (y0, h, e) = gp.objective(y, hyp, modsel)
        hyp[i] = hyp[i] - 2.0*eps # negative step
        (y1, h, e) = gp.objective(y, hyp, modsel)
        hyp[i] = hyp[i] + eps # go back
        dy[i] = (y0 - y1)/(2.0*eps) # finite difference estimate of partial derivative
    if modsel == 'loo':
        dy[0] = 0 # gradient in vs direction undefined for loo
    print('Analytically computed gradient:')
    print(df)
    print('Numerically computed gradient:')
    print(dy)
    assert np.allclose(df, dy, 1e-5)


# test optimization function evaluation
def test_optimization_function_evaluation():

    gp = Preg(lg, 'ihp', 3, np.asarray([log(0.6), log(sqrt(0.001)), -log(2.0*0.7**2.0)]))
    x = np.array([[0.89153353, 0.02574414], [0.32448397, 0.90612746],
        [0.31792538, 0.35136833]])
    gp.X_ = x
    y = np.array([0.447567, 1.567576, 0.99334])
    gp.mu_ = y.mean()

    # check gradient
    print('Log likelihood:')
    check_gradient(gp, y, 'llh') # log likelihood
    print('Geissers surrogate predictive probability:')
    check_gradient(gp, y, 'gpp') # GPP
    print('Leave-one-out error:')
    check_gradient(gp, y, 'loo') # LOO


# test automatic model selection
def test_automatic_model_selection():

    gp = Preg(lg, 'ihp', 3, [log(0.6), log(sqrt(0.001))])

    # noisy sinc training data
    x = np.array([-0.5208, -0.3077, -0.5008, -0.2259, -0.1579, 0.2802, 0.5751, -0.4600,
        0.6880, 0.4809])
    y = np.array([-1.9036, 2.2076, -1.4193, 4.0189, 6.1554, 2.9018, -1.9928, -1.1309,
        -0.6635, -1.4592])

    # run automatic model selection
    gp.amsd(x, y, range(1,9), 'llh', 20)
    print('Optimal GP (deg = {:d}):'.format(gp.degree))
    print(np.exp(gp.hyp))
    assert gp.degree == 8
    assert np.allclose(np.exp(gp.hyp), [6.86656682, 0.34434424], 1e-5)


# test prediction
def test_prediction():

    # noisy sinc training data
    x0 = np.array([0.6970, -0.6793, -0.6843, 0.0173, 0.2066, -0.6772, 0.2709, 0.6879,
        0.5645, -0.4709])
    y0 = np.array([-1.1876, -1.0000, -0.9789, 7.7148, 5.0757, -1.1393, 3.0856, -0.9981,
        -1.8604, -1.6527])

    # noisy sinc test data
    x1 = np.array([-0.3706, -0.6336, -0.1051, -0.3466, -0.4404, 0.8635, -0.2006, -0.2412,
        0.1857, -0.8630])
    y1 = np.array([0.4453, -1.8290, 7.0541, 1.0854, -0.8009, 0.3676, 5.0637, 3.6125,
        5.4220, 0.8088])

    # GP
    gp = Preg(lg, 'ihp', 3, [log(0.6), log(sqrt(0.001))])
    tr_err, te_err = gp.preg(x0, y0, x1, y1, range(1,9), 'llh', 20)
    print('Best hyperparameters:')
    print(np.exp(gp.hyp), gp.degree)
    mu_tr = gp.predict(x0)
    mu_te, v = gp.pred_meanvar(x1)
    print('Predicted training targets:')
    print(mu_tr)
    print('Predicted test targets:')
    print(mu_te)
    print('Predicted test variances:')
    print(v)

    # checks
    assert np.allclose(tr_err, 0.0206787, 1e-2)
    assert np.allclose(te_err, 39.8932, 1e-2)
    assert np.allclose(mu_tr, [-1.0165, -1.0909, -0.8590, 7.6926, 4.8472, -1.1825, 3.3478,
        -1.1698, -1.8937, -1.6223], 1e-3)
    assert np.allclose(mu_te, [1.2631, -2.5829, 7.4557, 2.0071, -0.8229, 4.1459, 5.9684,
        5.0304, 5.2994, 20.2913], 1e-3)
    assert np.allclose(v, [1.32478620e-01, 9.04166543e-02, 1.30338202e-01, 1.61805959e-01,
        6.00532614e-02, 9.91696460e+00, 2.08243705e-01, 2.22876669e-01, 2.29153548e-02,
        3.84916601e+01], 1e-2)


# test adaptive polynomial kernel
def test_adaptive_polynomial_kernel():

    # noisy sinc training data
    x0 = np.array([0.6970, -0.6793, -0.6843, 0.0173, 0.2066, -0.6772, 0.2709, 0.6879,
        0.5645, -0.4709])
    y0 = np.array([-1.1876, -1.0000, -0.9789, 7.7148, 5.0757, -1.1393, 3.0856, -0.9981,
        -1.8604, -1.6527])

    # check that ap = sp for coefficients e^hyp = 1
    gp = Preg(lg, 'ap', 4, np.asarray([log(0.6), log(sqrt(0.001)), 0, 0, 0, 0]))
    gp.X_ = x0
    gp.mu_ = y0.mean()
    gp.gram(gp.hyp)
    print('Gram matrix ap:')
    print(gp.G_)
    G_old = gp.G_
    gp.covtype = 'sp'
    gp.gram(gp.hyp)
    print('Gram matrix sp:')
    print(gp.G_)
    assert np.allclose(gp.G_, G_old, 1e-5)

    # check gradient for ap
    print('Log likelihood gradient for ap:')
    gp.covtype = 'ap'
    check_gradient(gp, y0, 'llh') # log likelihood


# test fit
def test_fit():

    # noisy sinc training data
    x0 = np.array([0.6970, -0.6793, -0.6843, 0.0173, 0.2066, -0.6772, 0.2709, 0.6879,
        0.5645, -0.4709])
    y0 = np.array([-1.1876, -1.0000, -0.9789, 7.7148, 5.0757, -1.1393, 3.0856, -0.9981,
        -1.8604, -1.6527])

    # noisy sinc test data
    x1 = np.array([-0.3706, -0.6336, -0.1051, -0.3466, -0.4404, 0.8635, -0.2006, -0.2412,
        0.1857, -0.8630])
    y1 = np.array([0.4453, -1.8290, 7.0541, 1.0854, -0.8009, 0.3676, 5.0637, 3.6125,
        5.4220, 0.8088])

    mu_te = Preg(lg, 'ihp', 8, np.log([7.70041964, 0.22707334])).fit(x0, y0).predict(x1)
    assert np.allclose(mu_te, [1.2631, -2.5829, 7.4557, 2.0071, -0.8229, 4.1459, 5.9684,
        5.0304, 5.2994, 20.2913], 1e-3)


# test automatic model selection and copy function
def test_ams_and_copy():

    gp = Preg(lg, 'sp', 2, [log(0.6), log(sqrt(0.001))])
    gp2 = Preg(lg, 'ihp', 3, [0, 0])
    gp3 = Preg(lg, 'ihp', 5, [1, 1])
    gp3.set_params(**gp.get_params())
    print('New parameters of gp3:')
    print(gp3.get_params())

    # noisy sinc training data
    x = np.array([-0.5208, -0.3077, -0.5008, -0.2259, -0.1579, 0.2802, 0.5751, -0.4600,
        0.6880, 0.4809])
    y = np.array([-1.9036, 2.2076, -1.4193, 4.0189, 6.1554, 2.9018, -1.9928, -1.1309,
        -0.6635, -1.4592])

    # run automatic model selection
    gp3.ams(x, y, 'llh', 20)
    gp2.copy(gp3)
    print('Optimal GP:')
    print(np.exp(gp2.hyp))
    assert np.allclose(np.exp(gp2.hyp), [1.53960048e-04, 2.76200228e+00], 1e-5)


# test Volterra operator
def test_volt():

    # init gp
    gp = Preg(lg, 'sp', 3, [0, log(sqrt(0.001))])
    gp.X_ = np.resize(np.arange(0, 6, dtype=np.float64), (3,2))
    gp.K_ = np.eye(3, dtype=np.float64)
    gp.invKt_ = np.ones(3, dtype=np.float64)

    print('Data matrix:')
    print(gp.X_)
    print('Weights:')
    print(gp.invKt_)

    print('O-order:')
    eta0 = gp.volt(0)
    print(eta0)
    assert eta0 == 3.0

    print('1st-order:')
    eta1 = gp.volt(1)
    print(eta1)
    assert np.allclose(eta1, [6., 9.], 0, 1e-8)

    print('2nd-order:')
    eta2 = gp.volt(2)
    p = np.zeros((2,2))
    for i in np.arange(0,3):
        datapt = gp.X_[i,:]
        p += np.outer(datapt, datapt)
    print(eta2)
    assert np.allclose(eta2, p, 0, 1e-8)

    print('3rd-order:')
    eta3 = gp.volt(3)
    p = np.zeros((2,2,2))
    for i in np.arange(0,3):
        datapt = gp.X_[i,:]
        p += np.tensordot(datapt, np.tensordot(datapt, datapt, axes=0), axes=0)
    print(eta3)
    assert np.allclose(eta3, p, 0, 1e-8)


