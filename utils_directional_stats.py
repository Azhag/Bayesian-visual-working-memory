#!/usr/bin/env python
# encoding: utf-8
"""
utils_directional_stats.py

Created by Loic Matthey on 2013-09-08.
Copyright (c) 2013 Gatsby Unit. All rights reserved.
"""

import numpy as np

import scipy.special as spsp
import scipy.optimize as spopt
import scipy.stats as spst

import matplotlib.pyplot as plt

import statsmodels.distributions as stmodsdist

############################## DIRECTIONAL STATISTICS ################################

def sample_angle(size=1):
    return np.random.random(size=size)*2.*np.pi - np.pi


def mean_angles(angles):
    '''
        Returns the mean angle out of a set of angles
    '''

    return np.angle(np.mean(np.exp(1j*angles), axis=0))


def wrap_angles(angles, bound=np.pi):
    '''
        Wrap angles in a [-bound, bound] space.

        For us: get the smallest angle between two responses
    '''

    # if np.isscalar(angles):
    #     while angles < -bound:
    #         angles += 2.*bound
    #     while angles > bound:
    #         angles -= 2.*bound
    # else:
    #     while np.any(angles < -bound):
    #         angles[angles < -bound] += 2.*bound
    #     while np.any(angles > bound):
    #         angles[angles > bound] -= 2.*bound

    angles = np.mod(angles + bound, 2*bound) - bound

    return angles

def kappa_to_stddev(kappa):
    '''
        Convert kappa to wrapped gaussian std dev

        std = 1 - I_1(kappa)/I_0(kappa)
    '''
    # return 1.0 - spsp.i1(kappa)/spsp.i0(kappa)
    return np.sqrt(-2.*np.log(spsp.i1e(kappa)/spsp.i0e(kappa)))


def stddev_to_kappa_single(stddev):
    '''
        Converts stddev to kappa

        No closed-form, does a line optimisation
    '''

    errfunc = lambda kappa, stddev: (np.exp(-0.5*stddev**2.) - spsp.i1e(kappa)/spsp.i0e(kappa))**2.
    kappa_init = 1.0
    kappa_opt = spopt.fmin(errfunc, kappa_init, args=(stddev, ), disp=False)

    return kappa_opt[0]

stddev_to_kappa=np.vectorize(stddev_to_kappa_single, doc='''
        Converts stddev to kappa

        vectorized stddev_to_kappa_single
    ''')

def test_stability_stddevtokappa(target_kappa=2.):
    '''
        Small test, shows how stable the inverse relationship between stddev and kappa is
    '''

    nb_iterations = 1000
    kappa_evolution = np.empty(nb_iterations)
    for i in xrange(nb_iterations):
        if i == 0:
            kappa_evolution[i] = stddev_to_kappa(kappa_to_stddev(target_kappa))
        else:
            kappa_evolution[i] = stddev_to_kappa(kappa_to_stddev(kappa_evolution[i-1]))


    print kappa_evolution[-1]

    plt.figure()
    plt.plot(kappa_evolution)
    plt.show()


def angle_population_vector(angles):
    '''
        Compute the complex population mean vector from a set of angles

        Mean over Axis 0
    '''

    return np.mean(np.exp(1j*angles), axis=0)


def angle_population_vector_weighted(angles, weights):
    '''
        Compute the weighted mean of the population vector of a set of angles
    '''
    return np.nansum(np.exp(1j*angles)*weights, axis=0)/np.nansum(weights)


def angle_population_mean(angles=None, angle_population_vec=None):
    '''
        Compute the mean of the angle population complex vector.

        If no angle_population_vec given, computes it from angles (be clever)
    '''
    if angle_population_vec is None:
        angle_population_vec = angle_population_vector(angles)

    return np.angle(angle_population_vec)


def angle_population_R(angles=None, angle_population_vec=None, weights=None):
    '''
        Compute R, the length of the angle population complex vector.

        Used to compute Standard deviation and diverse tests.

        If weights is provided, computes a weighted population mean vector instead.
    '''

    if angle_population_vec is None:
        if weights is None:
            angle_population_vec = angle_population_vector(angles)
        else:
            angle_population_vec = angle_population_vector_weighted(angles, weights)

    return np.abs(angle_population_vec)


def angle_circular_std_dev(angles=None, angle_population_vec=None):
    '''
        Compute the circular standard deviation from an angle population complex vector.

        If no angle_population_vec given, computes it from angles (be clever)
    '''
    if angle_population_vec is None:
        angle_population_vec = angle_population_vector(angles)

    return np.sqrt(-2.*np.log(np.abs(angle_population_vec)))


def compute_mean_std_circular_data(angles):
    '''
        Compute the mean vector and the std deviation according to the Circular Statistics formula
        Assumes a NxTxR matrix, averaging over N
    '''

    # Angle population vector
    angle_mean_vector = angle_population_vector(angles)

    # Population mean
    angle_mean_error = angle_population_mean(angle_population_vec=angle_mean_vector)

    # Circular standard deviation estimate
    angle_std_dev_error = angle_circular_std_dev(angle_population_vec=angle_mean_vector)

    # Mean bias
    angle_bias = np.mean(np.abs(angles), axis=0)


    return dict(mean=angle_mean_error, std=angle_std_dev_error, population_vector=angle_mean_vector, bias=angle_bias)


def compute_precision_samples(samples, square_precision=True, remove_chance_level=False):
    '''
        Compute precision from a set of samples.

        Can either use the squared precision definition (~ FI) or the one used in circular statistics (1/circ_std_dev)
        Can remove the chance level (Bays)
    '''

    precision = compute_angle_precision_from_std(angle_circular_std_dev(samples), square_precision=square_precision)

    if remove_chance_level:
        # Remove the chance level
        precision -= compute_precision_chance(samples.size)

    return precision



def compute_angle_precision_from_std(circular_std_dev, square_precision=True):
    '''
        Computes the precision from the circular std dev

        precision = 1/std**2  (square_precision = True)
    '''

    return 1./circular_std_dev**(2.**square_precision)


def compute_precision_chance(N):
    '''
        Compute the chance precision obtained under an uniform distribution
        policy for angular variable recall
    '''

    # Expected precision under uniform distribution
    x = np.logspace(-2, 2, 100)

    return np.trapz(N/(np.sqrt(x)*np.exp(x+N*np.exp(-x))), x)


def enforce_distance(theta1, theta2, min_distance=0.1):
    return np.abs(wrap_angles(theta1 - theta2)) > min_distance


def enforce_distance_set(new_item, other_items, min_distance=0.001):
    return all(enforce_distance(new_item[0], other_item[0], min_distance=min_distance) and enforce_distance(new_item[1], other_item[1], min_distance=min_distance) for other_item in other_items)


def rayleigh_test(angles):
    '''
        Performs Rayleigh Test for non-uniformity of circular data.

        Compares against Null hypothesis of uniform distribution around circle

        Assume one mode and data sampled from Von Mises.

        Use other tests for different assumptions.

        Uses implementation close to CircStats Matlab toolbox, maths from [Biostatistical Analysis, Zar].
    '''

    if angles.ndim > 1:
        angles = angles.flatten()

    N = angles.size

    # Compute Rayleigh's R
    R = N*angle_population_R(angles)

    # Compute Rayleight's z
    z = R**2. / N

    # Compute pvalue (Zar, Eq 27.4)
    pvalue = np.exp(np.sqrt(1. + 4*N + 4*(N**2. - R**2)) - 1. - 2.*N)

    return dict(pvalue=pvalue, z=z)


def V_test(angles, mean_apriori=0.0):
    '''
        Performs a V Test for non-uniformity of circular data

        The V test assumes a known mean angle a-priori
    '''

    if angles.ndim > 1:
        print "V Test on multiple dimensions: ", angles.shape


    # Get some statistics
    mu = mean_angles(angles)
    N = angles.shape[0]

    # Compute Rayleigh's R
    R = N*angle_population_R(angles)

    # Compute V statistics (Zar, Eq 26.5)
    V = R*np.cos(mu - mean_apriori)

    # Compute u statistic (Zar, Eq 26.6)
    u = V*np.sqrt(2./N)

    # p-value, compared to Univariate gaussian, correct for large sample size (couldn't find min N)
    pvalue = 1. - spst.norm.cdf(u)

    return dict(pvalue=pvalue, u=u, V=V, R=R)


def bootstrap_v_test(angles, mean_apriori=0.0, nb_samples=5000):
    '''
        Performs a bootstrap evaluation of the v-test score.

        Uses nb_samples reruns
    '''

    # Compute many bootstrap estimations of the V score
    angles_bootstrap = sample_angle((angles.size, nb_samples))
    v_scores_bootstrap = V_test(angles_bootstrap, mean_apriori=mean_apriori)['V']

    # Estimate Empirical CDF
    v_scores_ecdf = stmodsdist.empirical_distribution.ECDF(v_scores_bootstrap)

    # Compute current V score
    v_test = V_test(angles, mean_apriori=mean_apriori)

    # Check its CDF to compute the bootstrapped p-value
    p_value_bootstrap = 1. - v_scores_ecdf(v_test['V'])

    return dict(p_value_bootstrap=p_value_bootstrap, v_scores_ecdf=v_scores_ecdf, v_test=v_test)



