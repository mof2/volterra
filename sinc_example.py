#! /usr/bin/env python
# -*- coding: utf-8 -*-

# imports
from __future__ import print_function
import logging
import numpy as np
from preg import Preg
from math import sin, sqrt, log, exp
import matplotlib.pyplot as plt

# constants
max_order = 8           # maximum polynomial order
covtype = 'ihp'         # kernel type ('ihp' or 'ap')
modsel = 'gpp'          # model selection method ('llh', 'gpp', 'loo')
n_iter = 20             # number of line searches in minimization
n_data = 90             # number of data points (training + test data)
ntr = 20              # number of training data
sigma_0 = 0.05          # simulated noise variance

# generate data from sinc function
x = 2.0*np.random.rand(n_data) - 1.0
y = np.zeros(n_data)
for i in range(0,n_data):
    if x[i] != 0:
        y[i] = sin(8.0*x[i])/x[i]
    else:
        y[i] = 8.0
y = y + sqrt(sigma_0)*np.random.randn(n_data)  # add noise

#  divide into training set ...
x0 = x[0:ntr]
y0 = y[0:ntr]

#  ... and test set
x1 = x[ntr:]
y1 = y[ntr:]

# true function (for plotting purposes)
xt = np.linspace(-1.0, 1.0, 101)
yt = np.zeros(101)
for i in range(0,101):
    if xt[i] != 0:
        yt[i] = sin(8.0*xt[i])/xt[i]
    else:
        yt[i] = 8.0

# get a logger instance
lg = logging.getLogger('sinc_example')
lg.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
lg.addHandler(ch)

# regression
if covtype in ['ihp', 'sp']:  # initial guess for hyperparams for ihp, sp kernel
    hp0 = [log(0.6), log(sqrt(0.001))]
elif covtype == 'ap': # initial guess for hyperparams for ap kernel
    hp0 = [log(0.6), log(sqrt(0.001)), 0]
gp = Preg(lg, covtype, 1, hp0) # init GP struct
gp.preg(x0, y0, x1, y1, range(1,max_order + 1), modsel, n_iter)     # do regression
print('Condition number of covariance matrix: {:e}'.format(np.linalg.cond(gp.K_, np.inf)))
mu, s2 = gp.pred_meanvar(xt)                            # predict on interval [-1,1]
print('best hyperparameters:')
print(np.exp(gp.hyp))

# stop console logging
ch = lg.handlers[0]
lg.removeHandler(ch)

# plot prediction
noise_est = exp(2.0*gp.hyp[1])                          # noise estimate
fig, ax = plt.subplots(1)
ax.fill_between(xt, mu+2.0*(np.sqrt(s2) + sqrt(noise_est)), mu-2.0*(np.sqrt(s2) +
    sqrt(noise_est)), facecolor='blue', alpha=0.2)      # show 2*(std + noise_std)
ax.fill_between(xt, mu+2.0*np.sqrt(s2), mu-2.0*np.sqrt(s2), facecolor='blue', alpha=0.5)
ax.grid()                                               # show 2*std darker
plt.plot(xt, yt, 'y')                                   # show true function
plt.plot(xt, mu, 'r')                                   # show estimated function
plt.plot(x0, y0, 'b *')                                 # show training data
plt.ylim([-5,12])
plt.show()