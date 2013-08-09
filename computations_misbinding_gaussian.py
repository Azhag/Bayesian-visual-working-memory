#!/usr/bin/env python
# encoding: utf-8
"""
computations_misbinding_gaussian.py

Created by Loic Matthey on 2013-07-29.
Copyright (c) 2013 Gatsby Unit. All rights reserved.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as scsp
import scipy.optimize as scopt

import pypr.clustering.gmm as pygmm

from utils import *
# from statisticsmeasurer import *
# from randomfactorialnetwork import *
# from datagenerator import *
# from slicesampler import *

from dataio import *
import progress

if __name__ == '__main__':

    # Generate data from two gaussians

    alpha = 0.05
    epsilon = 0.0
    mu = 0.0
    mu1 = mu - epsilon
    mu2 = mu + epsilon
    sigma1 = 0.05
    sigma2 = 0.05

    N = 1000

    # Sample data from two gaussians
    X = pygmm.sample_gaussian_mixture([np.array([mu1]), np.array([mu2])], [[[sigma1]], [[sigma2]]], [alpha, 1.-alpha], samples=N)[:, 0]

    dx = 0.2
    deltaX = dx*np.random.randn(N)

    # Gradient epsilon
    f1 = lambda x, epsilon: 1./(np.sqrt(2*np.pi)*sigma1)*np.exp(-0.5*(x - mu + epsilon)**2./sigma1**2.)
    f2 = lambda x, epsilon: 1./(np.sqrt(2*np.pi)*sigma2)*np.exp(-0.5*(x - mu - epsilon)**2./sigma2**2.)
    h = lambda x, epsilon: alpha*f1(x, epsilon) + (1.-alpha)*f2(x, epsilon)
    g = lambda x, epsilon: (1.-alpha)*f2(x, epsilon)*(x - mu - epsilon)/sigma2**2. - alpha*f1(x, epsilon)*(x - mu + epsilon)/sigma1**2.
    par_g_eps = lambda x, epsilon: alpha*f1(x, epsilon)*((x - mu + epsilon)**2. - sigma1**2.)/sigma1**4. + (1. - alpha)*f2(x, epsilon)*((x - mu - epsilon)**2. - sigma2**2.)/sigma2**4.
    par_g_x = lambda x, epsilon: alpha*f1(x, epsilon)*(x-mu+epsilon)**2./sigma1**4.-alpha*f1(x, epsilon)/sigma1**2. + (1. - alpha)*f2(x, epsilon)/sigma2**2. - (1.-alpha)*f2(x, epsilon)*(x - mu - epsilon)**2./sigma2**4.
    par_h_x = lambda x, epsilon: -(1.-alpha)*f2(x, epsilon)*(x - mu - epsilon)/sigma2**2. - alpha*f1(x, epsilon)*(x - mu + epsilon)/sigma1**2.

    par_ll_eps_fct = lambda epsilon, x: g(x, epsilon)/h(x, epsilon)
    par_ll_eps_sum_fct = lambda epsilon, x: np.sum(np.ma.masked_invalid(par_ll_eps_fct(epsilon, x)))

    par_2_ll_eps_fct = lambda epsilon, x: (par_g_eps(x, epsilon)*h(x, epsilon) - g(x, epsilon)**2.)/h(x, epsilon)**2.
    par_2_ll_eps_sum_fct = lambda epsilon, x: np.sum(np.ma.masked_invalid(par_2_ll_eps_fct(epsilon, x)))

    par_ll_eps_x_fct = lambda epsilon, x: (par_g_x(x, epsilon)*h(x, epsilon) - g(x, epsilon)*par_h_x(x, epsilon))/h(x, epsilon)**2.
    par_ll_eps_x_sum_fct = lambda epsilon, x: np.sum(np.ma.masked_invalid(par_ll_eps_x_fct(epsilon, x)))

    epsilon_space = np.linspace(-1.0, 1.0, 3000)
    par_ll_eps = np.zeros((epsilon_space.size, N))
    par_ll_eps_mean_bis = np.zeros(epsilon_space.size)

    par_2_ll_eps_mean = np.zeros(epsilon_space.size)
    par_ll_eps_x_mean = np.zeros(epsilon_space.size)

    for epsilon_i, epsilon_ in enumerate(epsilon_space):
        par_ll_eps[epsilon_i] = par_ll_eps_fct(epsilon_, X+deltaX)
        par_ll_eps_mean_bis[epsilon_i] = par_ll_eps_sum_fct(epsilon_, X)

        par_2_ll_eps_mean[epsilon_i] = par_2_ll_eps_sum_fct(epsilon_, X)        

        par_ll_eps_x_mean[epsilon_i] = par_ll_eps_x_sum_fct(epsilon_, X)

    par_ll_eps_mean = np.nansum(par_ll_eps, axis=-1)

    print "Optimal epsilon", scopt.brentq(par_ll_eps_sum_fct, -0.5, 2.0, args=(X,))
    print "Optimal epsilon, x + dx", scopt.brentq(par_ll_eps_sum_fct, -0.5, 3.0, args=(X+deltaX,))
    
    plt.figure()
    plt.hist(X, bins=100)
    plt.figure()
    plt.hist(X+deltaX, bins=100)

    # plt.figure()
    # plt.hist(par_ll_eps_mean, bins=100)

    plt.figure()
    plt.plot(epsilon_space, par_ll_eps_mean_bis)
    plt.plot(epsilon_space, par_ll_eps_mean)
    plt.plot(epsilon_space, np.zeros_like(epsilon_space), ':')
    plt.title('der ll psi')

    plt.figure()
    plt.plot(epsilon_space, par_2_ll_eps_mean)
    plt.title('der^2 ll psi')

    plt.figure()
    plt.plot(epsilon_space, par_ll_eps_x_mean)
    plt.title('der der ll / psi x')

    plt.show()



