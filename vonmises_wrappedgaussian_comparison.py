#!/usr/bin/env python
# encoding: utf-8
"""
vonmises_wrappedgaussian_comparison.py


Created by Loic Matthey on 2013-05-16
Copyright (c) 2013 . All rights reserved.
"""


import numpy as np
import matplotlib.pyplot as plt
import utils
import statsmodels.nonparametric.kde as stmokde

# Sample from wrapped gaussian

num_samples = 1000

std_target = 1.5
samples = np.random.normal(0.0, std_target, size=num_samples)

samples_w = utils.wrap_angles(samples)

x = np.linspace(-np.pi, np.pi, 10000)

# KDE
samples_kde = stmokde.KDEUnivariate(samples)
samples_kde.fit()
samples_w_kde = stmokde.KDEUnivariate(samples_w)
samples_w_kde.fit()

# Von Mises
samples_vonmises = utils.fit_vonmises_samples(samples, num_points=300, return_fitted_data=True, should_plot=False)
samples_w_vonmises = utils.fit_vonmises_samples(samples_w, num_points=300, return_fitted_data=True, should_plot=False)

plt.figure()
plt.hist(samples, bins=100, normed=True)
plt.plot(samples_vonmises['support'], samples_vonmises['fitted_data'], 'r', linewidth=3)
plt.plot(samples_kde.support, samples_kde.density, 'g', linewidth=3)


plt.figure()
plt.hist(samples_w, bins=100, normed=True)
plt.plot(samples_vonmises['support'], samples_vonmises['fitted_data'], 'r', linewidth=3)
plt.plot(samples_w_kde.support, samples_w_kde.density, 'g', linewidth=3)

print 'Target std: %.2f, fitted kappa: %.3f, corresponding std: %.3f' % (std_target, samples_w_vonmises['parameters'][1], utils.kappa_to_stddev(samples_w_vonmises['parameters'][1]))

plt.show()

