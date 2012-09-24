#!/usr/bin/env python
# encoding: utf-8


import pymc
import numpy as np
import scipy.special as scsp

## Try with 1D model
def population_code_response(theta, N=100, kappa=0.1, amplitude=1.0):

    pref_angles = np.linspace(-np.pi, np.pi, N, endpoint=False)

    return amplitude*np.exp(kappa*np.cos(theta - pref_angles))/(2.*np.pi*scsp.i0(kappa))

N     = 50
kappa = 4.0
sigma = 0.1
amplitude = 1.0

# Put a prior on all the theta?
with_prior = True

## Generate dataset
M = 1
stimuli_used = 1.2*np.ones(M)
# stimuli_used = np.random.rand(M)*np.pi/2.

dataset = np.zeros((M, N))
for i, stim in enumerate(stimuli_used):
    dataset[i] = population_code_response(stim, N=N, kappa=kappa, amplitude=amplitude) + sigma*np.random.randn(N)


## Now build the model

if with_prior:
    mu_theta = pymc.CircVonMises('mu', mu=0.0, kappa=0.001)
    kappa_theta = pymc.Uniform('kappa', lower=0.0, upper=100.0)

all_theta = np.empty(M, dtype=object)
all_memory = np.empty(M, dtype=object)
all_mu_popresp = np.empty(M, dtype=object)

for i in np.arange(M):
    if with_prior:
        all_theta[i] = pymc.CircVonMises('theta_%d' % i, mu=mu_theta, kappa=kappa_theta)    
    else:
        all_theta[i] = pymc.CircVonMises('theta_%d' % i, mu=0.0, kappa=0.001)

    
    def deterministic_popresponse(theta, N=N, kappa=kappa):
        """
            Get an angle, returns the population response.

            Dimension N
        """
        # Maybe enforce that theta \in [0, 2pi]...
        pref_angles = np.linspace(-np.pi, np.pi, N, endpoint=False)
        
        output = np.exp(kappa*np.cos(theta - pref_angles))/(2.*np.pi*scsp.i0(kappa))

        return output


    all_mu_popresp[i] = pymc.Deterministic(eval=deterministic_popresponse, name="mu_popresp_%d" % i, parents={'theta': all_theta[i]}, doc='Population response for data %d' % i)

    all_memory[i] = pymc.MvNormal('m_%d' % i, mu=all_mu_popresp[i], tau=1./sigma**0.5*np.eye(N), value=dataset[i], observed=True)
    

