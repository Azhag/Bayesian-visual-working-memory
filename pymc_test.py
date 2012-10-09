#!/usr/bin/env python
# encoding: utf-8

import pymc
# import pymc_mymodel
import pylab as plt
import pymc_model_fullcollapsed_1D

# S = pymc.MCMC(pymc_mymodel, db='pickle')

# S.sample(iter=10000, burn=5000, thin=2)

# pymc.Matplot.plot(S)


# # Checking for convergence
# scores = pymc.geweke(S, intervals=20)
# pymc.Matplot.geweke_plot(scores)

# plt.show()

S = pymc.MCMC(pymc_model_fullcollapsed_1D, db='pickle')

S.sample(iter=20000, burn=5000, thin=5)

pymc.Matplot.plot(S)

# Checking for convergence
# scores = pymc.geweke(S, intervals=20)
# pymc.Matplot.geweke_plot(scores)

# S.theta.summary()

plt.show()
