#!/usr/bin/env python
# encoding: utf-8
"""
utils_math.py

Created by Loic Matthey on 2013-09-08.
Copyright (c) 2013 Gatsby Unit. All rights reserved.
"""

import numpy as np

from utils_fitting import fit_gaussian_mixture

############################ MATH AND STATS ##################################

def argmin_indices(array):
    return np.unravel_index(np.nanargmin(array), array.shape)

def argmax_indices(array):
    return np.unravel_index(np.nanargmax(array), array.shape)

def nanmean(array, axis=None):
    if not axis is None:
        return np.ma.masked_invalid(array).mean(axis=axis)
    else:
        return np.ma.masked_invalid(array).mean()

def nanmedian(array, axis=None):
    if not axis is None:
        return np.ma.extras.median(np.ma.masked_invalid(array), axis=axis)
    else:
        return np.ma.extras.median(np.ma.masked_invalid(array))


def nanstd(array, axis=None):
    if not axis is None:
        return np.ma.masked_invalid(array).std(axis=axis)
    else:
        return np.ma.masked_invalid(array).std()

def dropnan(array):
    '''
        Take an array, put it in a MaskedArray with ~np.isfinite masking.
        Returns the compressed() 1D array.
    '''
    return np.ma.masked_invalid(array).compressed()

def sample_invgamma(alpha, beta):
        '''
            Sample from an inverse gamma. numpy uses the shape/scale, not alpha/beta...
        '''
        return 1./np.random.gamma(alpha, 1./beta)

def sample_log_bernoulli(lp1, lp0):
    '''
        Sample a bernoulli from log-transformed probabilities
    '''
    #print lp0-lp1
    if (lp0-lp1) < -500:
        p1 = 1.
    elif (lp0-lp1) > 500:
        p1 = 0.
    else:
        p1 = 1./(1+np.exp(lp0-lp1))

    return np.random.rand() < p1

def sample_discrete_logp(log_prob):
    '''
        Use the logistic link function to get back to probabilities (thanks Sam Roweis)
        Also put a constant in it to avoid underflows
    '''

    b = - np.log(self.K) - np.max(log_prob)

    prob = np.exp(log_prob+b)/np.sum(np.exp(log_prob+b))
    cum_prob = np.cumsum(prob)

    return np.where(np.random.rand() < cum_prob)[0][0]  # Slightly faster than np.find



def tril_set(array, vector_input, check_sizes=False):
    '''
        Sets the lower triangular part of array with vector_input

        Hope all sizes work...
    '''

    if check_sizes:
        num_elements = np.sum(np.fromfunction(lambda i,j: i>j, array.shape))
        assert vector_input.size == num_elements, "Wrong number of inputs, need %d" % num_elements

    array[np.fromfunction(lambda i,j: i>j, array.shape)] = vector_input


def triu_set(array, vector_input, check_sizes=False):
    '''
        Sets the upper triangular part of array with vector_input

        Hope all sizes work...
    '''

    if check_sizes:
        num_elements = np.sum(np.fromfunction(lambda i,j: i<j, array.shape))
        assert vector_input.size == num_elements, "Wrong number of inputs, need %d" % num_elements

    array[np.fromfunction(lambda i,j: i<j, array.shape)] = vector_input


def triu_2_tril(array):
    '''
        Copy the upper triangular part of an array into its lower triangular part
    '''

    array[np.fromfunction(lambda i,j: i>j, array.shape)] = array[np.fromfunction(lambda i,j: i<j, array.shape)]


def powerlaw(x, amp, index):
    '''
        Computes the powerlaw:
        y = amp * (x**index)
    '''

    return amp * (x**index)


def mean_std_distrib(xdata, ydata):
    '''
        Compute the mean and standard deviation of a distribution
    '''
    mean_fit = np.sum(ydata*xdata)/np.sum(ydata)
    std_fit = np.sqrt(np.abs(np.sum((xdata - mean_fit)**2*ydata)/np.sum(ydata)))

    return mean_fit, std_fit


def skewness_distrib(xdata, ydata):
    '''
        Compute the skewness of a distribution
    '''

    distrib = ydata / np.trapz(ydata, xdata)

    mu, sigma = mean_std_distrib(xdata, distrib)

    skewness = np.trapz(distrib*(xdata - mu)**3./sigma**3., xdata)

    return skewness


def kurtosis_distrib(xdata, ydata):
    '''
        Compute the kurtosis for a distribution
    '''

    distrib = ydata / np.trapz(ydata, xdata)

    mu, _ = mean_std_distrib(xdata, distrib)

    kurtosis = np.trapz(distrib*(xdata - mu)**4., xdata)/np.trapz(distrib*(xdata - mu)**2., xdata)**2.

    return kurtosis

def bimodality_coefficient(xdata, ydata):
    '''
        Compute Sarle's bimodality coefficient

        beta = \skew^2 +1 / kurtosis

        beta is:
          1.0  for Bernoulli
          0.33 for Gaussian
          0.0  for heavy-tailed
    '''

    skewness = skewness_distrib(xdata, ydata)
    kurtosis = kurtosis_distrib(xdata, ydata)

    return (skewness**2. + 1.)/kurtosis


def ashman_d(xdata, ydata):
    '''
        Compute Ashman D coefficient for bimodality

        Basically compares the positions of two modes

        D = 2^0.5 |mu_1 - mu_2|/sqrt(sigma_1^2 + sigma_2^2)

        For a mixture, D > 2 required for clean separation of distributions.
    '''

    mixture_params = fit_gaussian_mixture(xdata, ydata, should_plot=False, return_fitted_data=False)

    D = 2**0.5 * np.abs(mixture_params[1] - mixture_params[3])/np.sqrt(mixture_params[2]**2. + mixture_params[4]**2.)

    return D


def aic(K, LL):
    '''
        Compute Akaike Information Criterion

        K: number of parameters
        LL: loglikelihood of model
    '''

    return 2*K - 2.*LL

def bic(K, LL, N):
    '''
        Compute Bayesian Information Criterion

        K: number of parameters
        LL: loglikelihood of model
    '''

    return -LL + K/2.*np.log(N)

