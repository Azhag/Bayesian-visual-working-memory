#!/usr/bin/env python
# encoding: utf-8
"""
utils_directional_stats.py

Created by Loic Matthey on 2013-09-08.
Copyright (c) 2013 Gatsby Unit. All rights reserved.
"""

import numpy as np

import scipy.special as scsp
import scipy.optimize as spopt

import matplotlib.pyplot as plt


############################## DIRECTIONAL STATISTICS ################################

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
    # return 1.0 - scsp.i1(kappa)/scsp.i0(kappa)
    return np.sqrt(-2.*np.log(scsp.i1e(kappa)/scsp.i0e(kappa)))


def stddev_to_kappa(stddev):
    '''
        Converts stddev to kappa

        No closed-form, does a line optimisation
    '''

    errfunc = lambda kappa, stddev: (np.exp(-0.5*stddev**2.) - scsp.i1e(kappa)/scsp.i0e(kappa))**2.
    kappa_init = 1.0
    kappa_opt = spopt.fmin(errfunc, kappa_init, args=(stddev, ), disp=False)

    return kappa_opt[0]


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


def angle_population_mean(angles=None, angle_population_vec=None):
    '''
        Compute the mean of the angle population complex vector.

        If no angle_population_vec given, computes it from angles (be clever)
    '''
    if angle_population_vec is None:
        angle_population_vec = angle_population_vector(angles)

    return np.angle(angle_population_vec)


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


def compute_angle_precision_from_std(circular_std_dev, square_precision=True):
    '''
        Computes the precision from the circular std dev

        precision = 1/std**2  (square_precision = True)
    '''

    return 1./circular_std_dev**(2.**square_precision)

