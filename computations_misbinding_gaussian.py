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

    alpha = 1.0
    epsilon = 1.0
    mu1 = 0.0
    mu2 = mu1 + epsilon
    sigma1 = 0.05
    sigma2 = 0.05

    N = 1000

    # Sample data from two gaussians
    X = pygmm.sample_gaussian_mixture([np.array([mu1]), np.array([mu2])], [[[sigma1]], [[sigma2]]], [alpha, 1.-alpha], samples=N)[:, 0]

    dx = 0.5
    deltaX = dx*np.random.rand(N)

    # Gradient epsilon
    par_ll_eps = (X - mu1 - epsilon)/sigma2**2.

    print np.sum(par_ll_eps)

    plt.figure()
    plt.hist(X, bins=100)

    plt.figure()
    plt.hist(par_ll_eps, bins=100)


    plt.show()



