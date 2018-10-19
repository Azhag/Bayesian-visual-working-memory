#!/usr/bin/env python
# encoding: utf-8
"""
EM_circularmixture.py

Created by Loic Matthey on 2013-09-20.
Copyright (c) 2013 Gatsby Unit. All rights reserved.
"""

import numpy as np
import scipy.special as spsp
import scipy.stats as spst
from sklearn.model_selection import KFold
import statsmodels.distributions as stmodsdist

import utils

import progress

def fit(responses, target_angle, nontarget_angles=np.array([[]]), initialisation_method='mixed', nb_initialisations=5, debug=False):
    '''
        Return maximum likelihood values for a mixture model, with:
            - 1 target Von Mises component
            - 1 circular uniform random component
            - 1 nontarget Von Mises components
        Inputs in radian, in the -pi:pi range.
            - responses: Nx1
            - target_angle: Nx1
            - nontarget_angles NxK

        Adapted from Matlab code by P. Bays
        Ref: Bays PM, Catalao RFG & Husain M. The precision of visual working
            memory is set by allocation of a shared resource. Journal of Vision 9(10): 7, 1-11 (2009)
    '''

    # Clean inputs
    if nontarget_angles.size > 0:
        nontarget_angles = nontarget_angles[:, ~np.all(np.isnan(nontarget_angles), axis=0)]

    N = float(np.sum(~np.isnan(responses)))
    K = float(nontarget_angles.shape[1])
    max_iter = 1000
    epsilon = 1e-5

    # Initial parameters
    initial_parameters_list = initialise_parameters(responses.size, K, initialisation_method, nb_initialisations)
    overall_LL = -np.inf
    LL = np.nan
    initial_i = 0
    best_kappa, best_mixt_target, best_mixt_random, best_mixt_nontargets = (np.nan, np.nan, np.nan, np.nan)

    for (kappa, mixt_target, mixt_random, mixt_nontargets, resp_ik) in initial_parameters_list:

        if debug:
            print "New initialisation point: ", (kappa, mixt_target, mixt_random, mixt_nontargets)

        old_LL = -np.inf

        i = 0
        converged = False

        # Precompute some matrices
        error_to_target = wrap(target_angle - responses)
        error_to_nontargets = wrap(nontarget_angles - responses[:, np.newaxis])

        # EM loop
        while i < max_iter and not converged:

            # E-step
            if debug:
                print "E", i, LL, kappa, mixt_target, mixt_nontargets, mixt_random
            resp_ik[:, 0] = mixt_target * vonmisespdf(error_to_target, 0.0, kappa)
            resp_r = mixt_random/(2.*np.pi)
            if K > 0:
                resp_ik[:, 1:] = mixt_nontargets/K * vonmisespdf(error_to_nontargets, 0.0, kappa)
            W = np.sum(resp_ik, axis=1) + resp_r


            # Compute likelihood
            LL = np.nansum(np.log(W))
            dLL = LL - old_LL
            old_LL = LL

            if (np.abs(dLL) < epsilon):
                converged = True
                break

            # M-step
            mixt_target = np.nansum(resp_ik[:, 0]/W)/N
            mixt_nontargets = np.nansum(resp_ik[:, 1:]/W[:, np.newaxis])/N
            mixt_random = np.nansum(resp_r/W)/N

            # Update kappa, a bit harder. Could be done in complex angular coordinates I think.
            rw = resp_ik/W[:, np.newaxis]
            S = np.c_[np.sin(error_to_target), np.sin(error_to_nontargets)]
            C = np.c_[np.cos(error_to_target), np.cos(error_to_nontargets)]
            r1 = np.nansum(S*rw)
            r2 = np.nansum(C*rw)

            if np.abs(np.nansum(rw)) < 1e-10:
                if debug:
                    print "Kappa diverged:", kappa, r1, r2, np.nansum(rw)
                kappa = 0
            else:
                R = (r1**2 + r2**2)**0.5/np.nansum(rw)
                kappa = A1inv(R)

            if debug:
                print "M", i, LL, kappa, mixt_target, mixt_nontargets, mixt_random

            # Weird correction...
            # if N <= 15:
            #     if kappa < 2:
            #         kappa = np.max([kappa - 2./(N*kappa), 0])
            #     else:
            #         kappa = kappa*(N-1)**3/(N**3 + N)

            i += 1

        # if not converged:
        #     if debug:
        #         print "Warning, Em_circularmixture.fit() did not converge before ", max_iter, "iterations"
        #     kappa = np.nan
        #     mixt_target = np.nan
        #     mixt_nontargets = np.nan
        #     mixt_random = np.nan
        #     rw = np.nan

        if LL > overall_LL and np.isfinite(LL):
            if debug:
                print "New best!", initial_i, overall_LL, LL
            overall_LL = LL
            (best_kappa, best_mixt_target, best_mixt_nontargets, best_mixt_random) = (kappa, mixt_target, mixt_nontargets, mixt_random)

        initial_i += 1

    result_dict = dict(kappa=best_kappa, mixt_target=best_mixt_target, mixt_nontargets=best_mixt_nontargets, mixt_random=best_mixt_random, train_LL=overall_LL, K=K)

    # Compute BIC and AIC scores
    result_dict['bic'] = bic(result_dict, N)
    result_dict['aic'] = aic(result_dict)

    return result_dict


def compute_loglikelihood(responses, target_angle, nontarget_angles, parameters):
    '''
        Compute the loglikelihood of the provided dataset, under the actual parameters

        parameters: (kappa, mixt_target, mixt_nontargets, mixt_random)
    '''

    resp_out = compute_responsibilities(responses, target_angle, nontarget_angles, parameters)

    # Compute likelihood
    return np.nansum(np.log(resp_out['W']))


def compute_responsibilities(responses, target_angle, nontarget_angles, parameters):
    '''
        Compute the responsibilities per datapoint.
        Actually provides the likelihood as well, returned as 'W'
    '''

    (kappa, mixt_target, mixt_nontargets, mixt_random) = (parameters['kappa'], parameters['mixt_target'], parameters['mixt_nontargets'], parameters['mixt_random'])

    if nontarget_angles.size > 0:
        nontarget_angles = nontarget_angles[:, ~np.all(np.isnan(nontarget_angles), axis=0)]

    K = float(nontarget_angles.shape[1])

    error_to_target = wrap(target_angle - responses)
    error_to_nontargets = wrap(nontarget_angles - responses[:, np.newaxis])

    resp_target = mixt_target * vonmisespdf(error_to_target, 0.0, kappa)
    resp_random = mixt_random/(2.*np.pi)
    if K > 0.:
        resp_nontargets = mixt_nontargets/K  * vonmisespdf(error_to_nontargets, 0.0, kappa)
    else:
        resp_nontargets = np.empty((responses.size, 0))

    W = resp_target + np.sum(resp_nontargets, axis=1) + resp_random

    resp_target /= W
    resp_nontargets /= W[:, np.newaxis]
    resp_random /= W

    return dict(target=resp_target, nontargets=resp_nontargets, random=resp_random, W=W)


def cross_validation_kfold(responses, target_angle, nontarget_angles, K=2, shuffle=False, initialisation_method='fixed', nb_initialisations=5, debug=False):
    '''
        Perform a k-fold cross validation fit.

        Report the loglikelihood on holdout data as validation metric
    '''

    # Build the kfold iterator. Sklearn is too cool.
    kf = KFold(responses.size, n_folds=K, shuffle=shuffle)

    if debug:
        print "%d-fold cross validation. %d in training, %d in testing. ..." % (K, (K-1.)/K*responses.size, responses.size/float(K))

    # Store test loglikelihoods
    test_LL = np.zeros(K)
    k_i = 0
    fits_all = []


    best_fit = None

    for train, test in kf:
        # Fit the model to the training subset
        curr_fit = fit(responses[train], target_angle[train], nontarget_angles[train], initialisation_method, nb_initialisations, debug=debug)

        # Compute the testing loglikelihood
        test_LL[k_i] = compute_loglikelihood(responses[test], target_angle[test], nontarget_angles[test], curr_fit)

        # Store all parameter fits
        fits_all.append(curr_fit)

        k_i += 1

    # Store best parameters. Choose the median of train/test LL
    median_index = np.argmin(np.abs(test_LL - np.median(test_LL)))
    best_test_LL = test_LL[median_index]
    best_fit = fits_all[median_index]

    # Do some unzipping
    fitted_params_names = curr_fit.keys()
    for param_name in fitted_params_names:
        exec(param_name + "_all = [fit['" + param_name + "'] for fit in fits_all]")

    return dict(test_LL=test_LL, train_LL=np.array(train_LL_all), fits_all=fits_all, best_fit=best_fit, best_test_LL=best_test_LL, kappa_all=np.array(kappa_all), mixt_target_all=np.array(mixt_target_all), mixt_nontargets_all=np.array(mixt_nontargets_all), mixt_random_all=np.array(mixt_random_all))




def initialise_parameters(N, K, method='fixed', nb_initialisations=10):
    '''
        Initialises all parameters:
         - Von mises concentration
         - Mixture proportions, for Target, Nontarget and random
         - Responsabilities, per datapoint

        Do like Paul and try multiple initial conditions
    '''

    if method == 'fixed':
        return initialise_parameters_fixed(N, K)
    elif method == 'random':
        return initialise_parameters_random(N, K, nb_initialisations)
    elif method == 'mixed':
        all_params = initialise_parameters_fixed(N, K)
        all_params.extend(initialise_parameters_random(N, K, nb_initialisations))
        return all_params


def initialise_parameters_random(N, K, nb_initialisations=10):
    '''
        Initialise parameters, with random values.
         - Von mises concentration
         - Mixture proportions, for Target, Nontarget and random
         - Responsabilities, per datapoint

        Provides nb_initialisations possible values
    '''

    all_params = []
    resp_ik = np.empty((N, int(K+1)))

    for i in xrange(nb_initialisations):
        kappa = np.random.rand()*300.

        mixt_target = np.random.rand()
        mixt_nontargets = np.random.rand()
        mixt_random = np.random.rand()

        mixt_sum = mixt_target + mixt_nontargets + mixt_random

        mixt_target /= mixt_sum
        mixt_nontargets /= mixt_sum
        mixt_random /= mixt_sum

        all_params.append((kappa, mixt_target, mixt_random, mixt_nontargets, resp_ik))

    return all_params


def initialise_parameters_fixed(N, K):
    '''
        Initialises all parameters:
         - Von mises concentration
         - Mixture proportions, for Target, Nontarget and random
         - Responsabilities, per datapoint

        Do like Paul and try multiple initial conditions
    '''

    kappa = [1., 10, 100, 300, 4000, 20, 0.3]
    mixt_nontargets = [0.01, 0.1, 0.4, 0.01, 0.01, 0.01, 0.05]
    mixt_random = [0.01, 0.1, 0.4, 0.1, 0.01, 0.5, 0.05]

    mixt_target = [1. - mixt_nontargets[i] - mixt_random[i] for i in xrange(len(kappa))]

    # resp_ik = []
    # for k in kappa:
    #     resp_ik.append(np.empty((N, K+1)))

    resp_ik = [np.empty((N, int(K+1))), ]*len(kappa)

    return zip(kappa, mixt_target, mixt_random, mixt_nontargets, resp_ik)


def wrap(angles):
    '''
        Wrap angles in a -max_angle:max_angle space
    '''

    max_angle = np.pi

    angles = np.mod(angles + max_angle, 2*max_angle) - max_angle

    return angles


def vonmisespdf(x, mu, K):
    '''
        Von Mises PDF (switch to Normal if high kappa)
    '''
    if K > 700.:
        return np.sqrt(K)/(np.sqrt(2*np.pi))*np.exp(-0.5*(x - mu)**2.*K)
    else:
        return np.exp(K*np.cos(x-mu)) / (2.*np.pi * spsp.i0(K))


def A1inv(R):
    '''
        Invert A1() function
    '''

    if R >= 0.0 and R < 0.53:
        return 2*R + R**3 + (5.*R**5)/6.
    elif R < 0.85:
        return -0.4 + 1.39*R + 0.43/(1. - R)
    else:
        return 1./(R**3 - 4*R**2 + 3*R)


def bootstrap_nontarget_stat(responses,
                             target,
                             nontargets=np.array([[]]),
                             nontarget_bootstrap_ecdf=None,
                             nb_bootstrap_samples=100,
                             resample_responses=False,
                             resample_targets=False):
    '''
        Performs a bootstrap evaluation of the nontarget mixture proportion distribution.

        Use that to construct a test for existence of misbinding errors
    '''
    nontarget_bootstrap_samples = None
    bootstrap_results = []

    if nontarget_bootstrap_ecdf is None:
        # Get samples
        if resample_responses:
            bootstrap_responses = utils.sample_angle((responses.size, nb_bootstrap_samples))
        if resample_targets:
            bootstrap_targets = utils.sample_angle((responses.size, nb_bootstrap_samples))
        bootstrap_nontargets = utils.sample_angle((nontargets.shape[0], nontargets.shape[1], nb_bootstrap_samples))

        for i in progress.ProgressDisplay(np.arange(nb_bootstrap_samples), display=progress.SINGLE_LINE):

            if resample_responses and resample_targets:
                em_fit = fit(bootstrap_responses[..., i], bootstrap_targets[..., i], bootstrap_nontargets[..., i])
            elif resample_responses and not resample_targets:
                em_fit = fit(bootstrap_responses[..., i], target, bootstrap_nontargets[..., i])
            elif not resample_responses and resample_targets:
                em_fit = fit(responses, bootstrap_targets[..., i], bootstrap_nontargets[..., i])
            elif not resample_responses and not resample_targets:
                em_fit = fit(responses, target, bootstrap_nontargets[..., i])
            else:
                raise ValueError('Weird! %d %d' % (resample_responses, resample_targets))

            bootstrap_results.append(em_fit)

        nontarget_bootstrap_samples = np.array([bootstr_res['mixt_nontargets'] for bootstr_res in bootstrap_results])

        # Estimate CDF
        nontarget_bootstrap_ecdf = stmodsdist.empirical_distribution.ECDF(nontarget_bootstrap_samples)

    # Compute the p-value for the provided
    em_fit = fit(responses, target, nontargets)
    p_value_bootstrap = 1. - nontarget_bootstrap_ecdf(em_fit['mixt_nontargets'])

    return dict(p_value=p_value_bootstrap, nontarget_ecdf=nontarget_bootstrap_ecdf, em_fit=em_fit, nontarget_bootstrap_samples=nontarget_bootstrap_samples, bootstrap_results_all=bootstrap_results)


def aic(em_fit_result_dict):
    '''
        Compute Akaike Information Criterion.
    '''
    # Number of parameters:
    # - mixt_target: 1
    # - mixt_random: 1
    # - mixt_nontarget: 1
    # - kappa: 1
    K = 4

    return utils.aic(K, em_fit_result_dict['train_LL'])


def bic(em_fit_result_dict, N):
    '''
        Compute the Bayesian Information Criterion score
    '''

    # Number of parameters:
    # - mixt_target: 1
    # - mixt_random: 1
    # - mixt_nontarget: 1
    # - kappa: 1
    K = 4

    return utils.bic(K, em_fit_result_dict['train_LL'], N)


def sample_from_fit(em_fit_result_dict, targets, nontargets):
    '''
        Get N samples from the Mixture model defined by em_fit_result_dict
    '''

    N = targets.size
    K = nontargets.shape[1]

    # Pre-sample items on target
    responses = spst.vonmises.rvs(em_fit_result_dict['kappa'], size=(N))

    # Randomly flip some to nontargets or random component, depending on a random coin toss (classical cumulative prob trick)
    samples_rand_N = np.random.random((N, 1))

    probs_components = np.r_[np.array([em_fit_result_dict['mixt_target']]), np.array([em_fit_result_dict['mixt_nontargets']]*K)/K, em_fit_result_dict['mixt_random']]
    cumprobs_components = np.cumsum(probs_components)

    samples_components = samples_rand_N < cumprobs_components

    # Move the targets
    responses += samples_components[:, 0]*targets
    samples_components *= ~samples_components[:, 0][:, np.newaxis]

    # Move the nontargets
    for k in xrange(K):
        responses += samples_components[:, k+1]*nontargets[:, k]
        samples_components *= ~samples_components[:, k+1][:, np.newaxis]

    # Resample randomly the random ones
    responses[samples_components[:, -1]] = utils.sample_angle(size=np.sum(samples_components[:, -1]))

    return responses


def test():
    '''
        Does a Unit test, samples data from a mixture of one Von mises and random perturbations. Then fits the model and check if everything works.
    '''

    N = 5000
    N_rnd = N/5
    target = np.zeros(N)

    # kappa_space = np.array([1.0, 5.0, 20., 100, 3000, 5000., 0.5])
    kappa_space = np.array([5.0])

    kappa_fitted = np.zeros(kappa_space.size)

    for kappa_i, kappa in enumerate(kappa_space):
        print kappa

        responses = spst.vonmises.rvs(kappa, size=(N))
        responses[np.random.randint(N, size=N_rnd)] = utils.sample_angle(N_rnd)

        em_fit = fit(responses, target, debug=False)

        kappa_fitted[kappa_i] = em_fit['kappa']

        print em_fit

    # Check if estimated kappa is within 20% of target one
    print kappa_fitted, kappa_fitted/kappa_space
    assert np.all(np.abs(kappa_fitted/kappa_space - 1.0) < 0.20)


def test_bootstrap_nontargets():
    '''
        Check how the bootstrapped test for misbinding errors behaves
    '''

    N = 300
    nb_nontargets = 1
    kappa = 5.0

    target = np.zeros(N)
    nontargets = utils.wrap_angles(np.linspace(0.0, 2*np.pi, nb_nontargets + 1, endpoint=False)[1:])*np.ones((N, nb_nontargets))

    responses = spst.vonmises.rvs(kappa, size=(N))
    responses[np.random.randint(N, size=N/3)] = utils.sample_angle(N/3)

    # em_fit = fit(responses, target, nontargets)

    bootstrap_results = bootstrap_nontarget_stat(responses, target, nontargets)

    print bootstrap_results

    assert bootstrap_results['p_value'] > 0.05, "No misbinding here, should not reject H0"





if __name__ == '__main__':
    test()
    # pass


