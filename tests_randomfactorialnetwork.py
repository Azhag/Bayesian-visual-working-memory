#!/usr/bin/env python
# encoding: utf-8
"""
tests_randomfactorialnetwork.py

Created by Loic Matthey on 2013-04-26.
Copyright (c) 2013 Gatsby Unit. All rights reserved.
"""

import pylab as plt
import numpy as np
import scipy as sp

from scipy.spatial.distance import pdist

from nose import with_setup

from utils import *
from randomfactorialnetwork import *


def setup():
    N_sqrt = 30.
    N = int(N_sqrt**2.)
    rc_scale = 5.0

    global rn
    rn = RandomFactorialNetwork.create_full_conjunctive(N, R=2, scale_moments=(rc_scale, 0.0000001), ratio_moments=(1.0, 0.0001), response_type='bivariate_fisher')

    global theta_space
    theta_space = np.linspace(-np.pi, np.pi, 1000)
    global neuron0_resp
    neuron0_resp = np.array([rn.get_neuron_response(0, stimulus_input=(x, 0.0)) for x in theta_space])


@with_setup(setup)
def test_compute_network_response_statistics_theoretical_marginalise_theta2_mean():
    '''
        Verifying that the theoretical and empirical mean estimates are close
    '''
    
    stats = rn.compute_network_response_statistics_theoretical()
    mean_theoretical = stats['mean']
    
    kappa1, kappa2 = rn.rc_scale
    
    mean0_theo = mean_theoretical[0]
    mean0_empirical = np.trapz(neuron0_resp.flatten()/(2.*np.pi), theta_space)
    mean0_theo_bis =  sp.integrate.quad(lambda x: 1./(2.*np.pi)*(rn.get_neuron_response(0, stimulus_input=(x, 0.0))), -np.pi, np.pi)[0]
    mean0_theo_ter = sp.integrate.quad(lambda x: 1./(2.*np.pi)*1./(4*np.pi**2.*scsp.i0(kappa1)*scsp.i0(kappa2))*np.exp(kappa1*np.cos(x- rn.neurons_preferred_stimulus[0,0]) + kappa2*np.cos(0.0 - rn.neurons_preferred_stimulus[0, 1])), -np.pi, np.pi)[0]

    print mean0_theo
    print mean0_empirical
    print mean0_theo_bis
    print mean0_theo_ter

    assert np.allclose(mean0_theo, mean0_empirical, atol=1e-3), 'Mean theo close to empirical'
    assert np.allclose(mean0_theo, mean0_theo_bis, atol=1e-3), 'Mean theo close to integral of rn.get_neuron_response'
    assert np.allclose(mean0_theo, mean0_theo_ter, atol=1e-3), 'Mean theo close to integral of receptive field function'


@with_setup(setup)
def test_compute_network_response_statistics_theoretical_marginalise_theta2_cov():
    '''
        Verifying that the theoretical and empirical covariance estimates are close
    '''
    
    stats = rn.compute_network_response_statistics_theoretical()
    cov_theoretical = stats['cov']

    kappa1, kappa2 = rn.rc_scale
    theta2 = 0.0

    cov_theo00 = cov_theoretical[0,0]
    cov_emp00 =  np.cov(neuron0_resp.T)
    secmom_emp00 = np.trapz(neuron0_resp.flatten()**2./(2.*np.pi), theta_space)
    cov_emp00_bis = np.trapz((neuron0_resp.flatten() - np.trapz(neuron0_resp.flatten()/(2.*np.pi), theta_space))**2./(2.*np.pi), theta_space)

    mu_i = lambda x: (1./(4*np.pi**2.*scsp.i0(kappa1)*scsp.i0(kappa2))*np.exp(kappa1*np.cos(x- rn.neurons_preferred_stimulus[0,0]) + kappa2*np.cos(0.0 - rn.neurons_preferred_stimulus[0, 1])))
    mu_i_bis = lambda x: (1./(16*np.pi**4.*scsp.i0(kappa1)**2.*scsp.i0(kappa2)**2.)*np.exp(2.*kappa1*np.cos(x- (rn.neurons_preferred_stimulus[0,0] + rn.neurons_preferred_stimulus[0,0])/2.)*np.cos((rn.neurons_preferred_stimulus[0,0] - rn.neurons_preferred_stimulus[0,0])/2.) + 2.*kappa2*np.cos(theta2 - (rn.neurons_preferred_stimulus[0,1] + rn.neurons_preferred_stimulus[0,1])/2.)*np.cos((rn.neurons_preferred_stimulus[0,1] - rn.neurons_preferred_stimulus[0,1])/2.)))
    mmu_i = sp.integrate.quad(mu_i, -np.pi, np.pi)[0]
    
    # Those next two are wrong. Doesn't take p(theta) into account
    cov_int00_nonorm     = sp.integrate.quad(lambda x: (mu_i(x) - mmu_i)**2., -np.pi, np.pi)
    cov_int00_nonorm_bis =  sp.integrate.quad(lambda x: mu_i(x)**2., -np.pi, np.pi)[0] - mmu_i**2.
    
    mmu_i = sp.integrate.quad(mu_i, -np.pi, np.pi)[0]/(2.*np.pi)
    cov_int00 =  sp.integrate.quad(lambda x: 1./(2*np.pi)*(mu_i(x) - mmu_i)**2., -np.pi, np.pi)[0]
    cov_int00_bis =  sp.integrate.quad(lambda x: 1./(2*np.pi)*mu_i(x)**2., -np.pi, np.pi)[0] - mmu_i**2.

    secmom_int00 = sp.integrate.quad(lambda x: 1./(2*np.pi)*mu_i_bis(x), -np.pi, np.pi)[0]
    cov_theo00_explicit = scsp.i0(2*kappa1*np.cos((rn.neurons_preferred_stimulus[0,0] - rn.neurons_preferred_stimulus[0,0])/2.))/(16.*np.pi**4.*scsp.i0(kappa1)**2.*scsp.i0(kappa2)**2.)*np.exp(2.*kappa2*np.cos(theta2 - (rn.neurons_preferred_stimulus[0,1]+rn.neurons_preferred_stimulus[0,1])/2.)*np.cos((rn.neurons_preferred_stimulus[0,1]-rn.neurons_preferred_stimulus[0,1])/2.)) - mmu_i**2
    cov_int00_theo00 = sp.integrate.quad(lambda x: 1./(2*np.pi)*mu_i_bis(x), -np.pi, np.pi)[0] - mmu_i**2

    # Checking covariances
    assert np.all(pdist(np.array([cov_theo00, cov_emp00, cov_emp00_bis, cov_int00, cov_int00_bis, cov_theo00_explicit, cov_int00_theo00])[:, np.newaxis]) < 1e-3)

    # checking second moments
    assert np.all(pdist(np.array([secmom_emp00, secmom_int00])[:, np.newaxis]) < 1e-3)



