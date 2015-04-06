#!/usr/bin/env python
# encoding: utf-8
"""
EM_circularmixture_allitems_uniquekappa.py

Created by Loic Matthey on 2013-11-21.
Copyright (c) 2013 Gatsby Unit. All rights reserved.
"""

import numpy as np
import scipy.special as spsp
import scipy.optimize as spopt
import scipy.stats as spst
from sklearn.cross_validation import KFold
import statsmodels.distributions as stmodsdist

import matplotlib.pyplot as plt

import utils

import progress

def fit(T_space, responses, targets_angle, nontargets_angles=np.array([[]]), initialisation_method='random', nb_initialisations=5, debug=False, force_random_less_than=None):
    '''
        Modified mixture model where we fit a parametric power law to kappa as a function of number of items.
        Assumes that we gather samples across samples N and times T.

        Return maximum likelihood values for a different mixture model, with:
            - 1 probability of target
            - 1 probability of nontarget
            - 1 probability of random circular
            - alpha/beta parameters of kappa = alpha t**beta, where t=number of items
        Inputs in radian, in the -pi:pi range.
            - responses: TxN
            - targets_angle: TxN
            - nontargets_angles TxNx(T-1)  <- TODO wrong

        Modified from Bays et al 2009
    '''

    # Clean inputs
    # not_nan_indices = ~np.isnan(responses)
    # responses = responses[not_nan_indices]
    # targets_angle = targets_angle[not_nan_indices]

    # if nontargets_angles.size > 0:
    #     nontargets_angles = nontargets_angles[:, ~np.all(np.isnan(nontargets_angles), axis=0)]

    # nontargets_angles = nontargets_angles[not_nan_indices]

    Tnum = T_space.size
    Tmax = T_space.max()
    N = float(np.sum(~np.isnan(responses)))/float(Tnum)
    max_iter = 1000
    epsilon = 1e-4
    dLL = np.nan

    # Initial parameters
    initial_parameters_list = initialise_parameters(N, T_space, initialisation_method, nb_initialisations)
    overall_LL = -np.inf
    LL = np.nan
    initial_i = 0
    best_alpha, best_beta, best_mixt_target, best_mixt_random, best_mixt_nontargets = (np.nan, np.nan, np.nan, np.nan, np.nan)

    for (alpha, beta, mixt_target, mixt_random, mixt_nontargets, resp_nik) in initial_parameters_list:

        if debug:
            print "New initialisation point: ", (alpha, beta, mixt_target, mixt_random, mixt_nontargets)

        old_LL = -np.inf

        i = 0
        converged = False

        # Precompute some matrices
        error_to_target = wrap(targets_angle - responses)
        error_to_nontargets = wrap(nontargets_angles - responses[:, :, np.newaxis])
        errors_all = np.c_[error_to_target[:, :, np.newaxis], error_to_nontargets]


        # EM loop
        while i < max_iter and not converged:

            # E-step
            if debug:
                print "E", i, LL, dLL, alpha, beta, mixt_target, mixt_nontargets, mixt_random
            for T_i, T in enumerate(T_space):
                resp_nik[T_i, :, 0] = mixt_target[T_i]*vonmisespdf(error_to_target[T_i], 0.0, compute_kappa(T, alpha, beta))

                resp_nik[T_i, :, 1:T] = mixt_nontargets[T_i, :(T-1)]*vonmisespdf(error_to_nontargets[T_i, :, :(T-1)], 0.0, compute_kappa(T, alpha, beta))
            resp_random = mixt_random/(2.*np.pi)

            W = np.nansum(resp_nik, axis=-1) + resp_random[:, np.newaxis]

            resp_nik /= W[:, :, np.newaxis]
            resp_r = resp_random[:, np.newaxis] / W

            # Compute likelihood
            LL = np.nansum(np.log(W))
            dLL = LL - old_LL
            old_LL = LL

            if (np.abs(dLL) < epsilon):
                converged = True
                break

            # M-step
            mixt_target = np.nansum(resp_nik[..., 0], axis=1)/N
            mixt_nontargets = np.nansum(resp_nik[..., 1:], axis=1)/N
            mixt_random = np.nansum(resp_r, axis=-1)/N

            # Update kappa
            if np.abs(np.nanmean(resp_nik)) < 1e-10 or np.all(np.isnan(resp_nik)):
                if debug:
                    print "Kappas diverged:", alpha, beta, np.nanmean(resp_nik)
                alpha=0
                beta=0
                break
            else:
                # Estimate alpha and beta with a numerical M-step, 2D optimisaiton over the loglikelihood.
                # Combine all samples and times.
                alpha, beta = numerical_M_step(T_space, resp_nik, errors_all, alpha, beta)

                # R = utils.angle_population_R(np.r_[error_to_target, error_to_nontargets.reshape(int(N*K))], weights=np.r_[rw[:, 0], rw[:, 1:].reshape(int(N*K))])
                # kappa = A1inv(R)

                # Clamp kappa to avoid overfitting
                # if kappa > 1000:
                #     kappa = 1000

            if debug:
                print "M", i, LL, dLL, alpha, beta, mixt_target, mixt_nontargets, mixt_random

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
            (best_alpha, best_beta, best_mixt_target, best_mixt_nontargets, best_mixt_random) = (alpha, beta, mixt_target, mixt_nontargets, mixt_random)

        initial_i += 1

    result_dict = dict(alpha=best_alpha, beta=best_beta, kappa=compute_kappa(T_space, best_alpha, best_beta), mixt_target=best_mixt_target, mixt_nontargets=best_mixt_nontargets, mixt_nontargets_sum=np.sum(best_mixt_nontargets, axis=-1), mixt_random=best_mixt_random, train_LL=overall_LL, T_space=T_space)

    # Compute BIC and AIC scores
    result_dict['bic'] = bic(result_dict, N)
    result_dict['aic'] = aic(result_dict)

    return result_dict


def compute_kappa(t, alpha, beta):
    '''
        Compute kappa given alpha and beta
    '''
    return alpha*t**beta


def numerical_M_step(T_space, resp_nik, errors_all, alpha, beta):
    '''
        Perform a numerical M-step, optimizing the loglikelihood over both alpha and beta.

    '''

    def loglikelihood_closure(params, args):
        '''
            params: (alpha, beta)
            args: (T_space, resp_nik, errors_all)
        '''
        LL_tot = 0
        for T_i, T in enumerate(args['T_space']):
            LL_tot += np.nansum(args['resp_nik'][T_i]*compute_kappa(T, params[0], params[1])*np.cos(args['errors_all'][T_i])) \
                -np.nansum(args['resp_nik'][T_i])*np.log(spsp.i0(compute_kappa(T, params[0], params[1])))

        if np.isnan(LL_tot):
            LL_tot = np.inf

        return -LL_tot

    args = dict(T_space=T_space, resp_nik=resp_nik, errors_all=errors_all)

    res = spopt.minimize(loglikelihood_closure, (alpha, beta), args=args, bounds=((0, 100), (-1.0, 0.0)), options=dict(disp=False))
    # print res['x']

    # alpha_space = np.linspace(0, 100, 100)
    # beta_space = np.linspace(-0.1, -3.0, 101)
    # fit = np.array([[loglikelihood_closure((alpha_, beta_), args) for alpha_ in alpha_space] for beta_ in beta_space]).T

    # utils.pcolor_2d_data(fit, alpha_space, beta_space)
    # plt.show()

    return res['x'][0], res['x'][1]


def compute_loglikelihood(responses, targets_angle, nontargets_angles, parameters):
    '''
        Compute the loglikelihood of the provided dataset, under the actual parameters

        parameters: (kappa, mixt_target, mixt_nontargets, mixt_random)
    '''

    resp_out = compute_responsibilities(responses, targets_angle, nontargets_angles, parameters)

    # Compute likelihood
    return np.nansum(np.log(resp_out['W']))


def compute_responsibilities(responses, targets_angle, nontargets_angles, parameters):
    '''
        Compute the responsibilities per datapoint.
        Responses: TxN
        Targets: TxN
        Nontargets: TxNx(T-1)

        Actually provides the likelihood as well, returned as 'W'
    '''

    (alpha, beta, mixt_target, mixt_nontargets, mixt_random) = (parameters['alpha'], parameters['beta'], parameters['mixt_target'], parameters['mixt_nontargets'], parameters['mixt_random'])

    # if nontargets_angles.size > 0:
        # nontargets_angles = nontargets_angles[:, ~np.all(np.isnan(nontargets_angles), axis=0)]

    T = int(responses.shape[0])

    error_to_target = wrap(targets_angle - responses)
    error_to_nontargets = wrap(nontargets_angles - responses[:, :, np.newaxis])

    resp_target = np.empty(responses.shape)
    resp_random = np.ones(responses.shape)*mixt_random/(2.*np.pi)
    resp_nontargets = np.nan*np.empty(nontargets_angles.shape)

    for t in xrange(T):
        resp_target[t] = mixt_target*vonmisespdf(error_to_target[t], 0.0, compute_kappa(t+1, alpha, beta))

        resp_nontargets[t, :, :t] = mixt_nontargets*vonmisespdf(error_to_nontargets[t, :, :t], 0.0, compute_kappa(t+1, alpha, beta))

    W = resp_target + np.nansum(resp_nontargets, axis=-1) + resp_random

    resp_target /= W
    resp_nontargets /= W[:, :, np.newaxis]
    resp_random /= W

    return dict(target=resp_target, nontargets=resp_nontargets, random=resp_random, W=W)


def cross_validation_kfold(responses, targets_angle, nontargets_angles, K=2, shuffle=False, initialisation_method='fixed', nb_initialisations=5, debug=False, force_random_less_than=None):
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
        curr_fit = fit(responses[train], targets_angle[train], nontargets_angles[train], initialisation_method, nb_initialisations, debug=debug, force_random_less_than=force_random_less_than)

        # Compute the testing loglikelihood
        test_LL[k_i] = compute_loglikelihood(responses[test], targets_angle[test], nontargets_angles[test], curr_fit)

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




def initialise_parameters(N, T_space, method='random', nb_initialisations=10):
    '''
        Initialises all parameters:
         - Von mises concentration
         - Mixture proportions, for Target, Nontarget and random
         - Responsabilities, per datapoint

        Do like Paul and try multiple initial conditions
    '''

    if method == 'fixed':
        raise NotImplementedError('not yet')
        return initialise_parameters_fixed(N, T_space)
    elif method == 'random':
        return initialise_parameters_random(N, T_space, nb_initialisations)
    elif method == 'mixed':
        raise NotImplementedError('not yet')
        all_params = initialise_parameters_fixed(N, T_space)
        all_params.extend(initialise_parameters_random(N, T_space, nb_initialisations))
        return all_params


def initialise_parameters_random(N, T_space, nb_initialisations=10):
    '''
        Initialise parameters, with random values.
         - Von mises concentration
         - Mixture proportions, for Target, Nontarget and random
         - Responsabilities, per datapoint

        Provides nb_initialisations possible values
    '''

    Tnum = T_space.size
    Tmax = T_space.max()

    all_params = []
    resp_nik = np.nan*np.empty((Tnum, int(N), Tmax))

    for i in xrange(nb_initialisations):
        alpha = np.random.rand()*30.
        beta = -np.random.rand()*1

        # Force a strong on-target prior...
        mixt_target = (np.random.rand(Tnum)*0.5 + 0.5)*1.5
        mixt_nontargets = np.random.rand(Tnum, Tmax-1)*(0.5/T_space[:, np.newaxis])

        for T_i, T in enumerate(T_space):
            mixt_nontargets[T_i, (T-1):] = np.nan

        mixt_random = np.random.rand(Tnum)*0.2

        mixt_sum = mixt_target + np.nansum(mixt_nontargets, axis=-1) + mixt_random

        mixt_target /= mixt_sum
        mixt_nontargets /= mixt_sum[:, np.newaxis]
        mixt_random /= mixt_sum

        all_params.append((alpha, beta, mixt_target, mixt_random, mixt_nontargets, resp_nik))

    return all_params


def initialise_parameters_fixed(N, K):
    '''
        Initialises all parameters:
         - Von mises concentration
         - Mixture proportions, for Target, Nontarget and random
         - Responsabilities, per datapoint

        Do like Paul and try multiple initial conditions
    '''

    kappa_fixed = np.array([1., 10, 100, 300, 4000, 20, 0.3])
    mixt_nontargets_fixed = ([0.1, 0.1, 0.4, 0.01, 0.01, 0.05, 0.1]*np.ones(K)[:, np.newaxis]).T/K
    mixt_random_fixed = [0.01, 0.1, 0.4, 0.1, 0.01, 0.05, 0.1]

    mixt_target_fixed = [1. - np.sum(mixt_nontargets_fixed[i]) - mixt_random_fixed[i] for i in xrange(len(mixt_random_fixed))]

    resp_ik_fixed = [np.empty((int(N), int(K+1))), ]*kappa_fixed.shape[0]

    return zip(kappa_fixed, mixt_target_fixed, mixt_random_fixed, mixt_nontargets_fixed, resp_ik_fixed)


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
        return np.sqrt(K)/(np.sqrt(2*np.pi))*np.exp(-0.5*(x -mu)**2.*K)
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


def bootstrap_nontarget_stat(responses, target, nontargets=np.array([[]]), sumnontargets_bootstrap_ecdf=None, allnontargets_bootstrap_ecdf=None, nb_bootstrap_samples=100, resample_responses=False, resample_targets=False):
    '''
        Performs a bootstrap evaluation of the nontarget mixture proportion distribution.

        Use that to construct a test for existence of misbinding errors
    '''

    if sumnontargets_bootstrap_ecdf is None and allnontargets_bootstrap_ecdf is None:
        # Get samples
        if resample_responses:
            bootstrap_responses = utils.sample_angle((responses.size, nb_bootstrap_samples))
        if resample_targets:
            bootstrap_targets = utils.sample_angle((responses.size, nb_bootstrap_samples))
        bootstrap_nontargets = utils.sample_angle((nontargets.shape[0], nontargets.shape[1], nb_bootstrap_samples))

        bootstrap_results = []

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

        if resample_targets:
            if nontargets.shape[1] > 0:
                sumnontargets_bootstrap_samples = np.array([np.nansum(bootstr_res['mixt_nontargets']) for bootstr_res in bootstrap_results] + [bootstr_res['mixt_target'] for bootstr_res in bootstrap_results])
            else:
                sumnontargets_bootstrap_samples = np.array([bootstr_res['mixt_target'] for bootstr_res in bootstrap_results])
        else:
            sumnontargets_bootstrap_samples = np.array([np.sum(bootstr_res['mixt_nontargets']) for bootstr_res in bootstrap_results])
            allnontargets_bootstrap_samples = np.array([bootstr_res['mixt_nontargets'] for bootstr_res in bootstrap_results]).flatten()

        # Estimate CDF
        sumnontargets_bootstrap_ecdf = stmodsdist.empirical_distribution.ECDF(sumnontargets_bootstrap_samples)
        allnontargets_bootstrap_ecdf = stmodsdist.empirical_distribution.ECDF(allnontargets_bootstrap_samples)
    else:
        allnontargets_bootstrap_samples = None
        sumnontargets_bootstrap_samples = None
        bootstrap_results = None

    # Compute the p-value for the current em_fit under the empirical CDF
    p_value_sum_bootstrap = np.nan
    p_value_all_bootstrap = np.nan

    em_fit = fit(responses, target, nontargets)
    if sumnontargets_bootstrap_ecdf is not None:
        p_value_sum_bootstrap = 1. - sumnontargets_bootstrap_ecdf(np.sum(em_fit['mixt_nontargets']))
    if allnontargets_bootstrap_ecdf is not None:
        p_value_all_bootstrap = 1. - allnontargets_bootstrap_ecdf(em_fit['mixt_nontargets'])

    return dict(p_value=p_value_sum_bootstrap, nontarget_ecdf=sumnontargets_bootstrap_ecdf, em_fit=em_fit, nontarget_bootstrap_samples=sumnontargets_bootstrap_samples, bootstrap_results_all=bootstrap_results, allnontarget_bootstrap_samples=allnontargets_bootstrap_samples, allnontarget_ecdf=allnontargets_bootstrap_ecdf, allnontarget_p_value=p_value_all_bootstrap)


def aic(em_fit_result_dict):
    '''
        Compute Akaike Information Criterion.
    '''
    # Number of parameters:
    # - mixt_target: Tnum
    # - mixt_random: Tnum
    # - mixt_nontarget: sum(T_space - 1)
    # - alpha: 1
    # - beta: 1
    K = em_fit_result_dict['mixt_target'].size + em_fit_result_dict['mixt_random'].size + np.sum(em_fit_result_dict['T_space'] - 1) + 2

    return utils.aic(K, em_fit_result_dict['train_LL'])


def bic(em_fit_result_dict, N):
    '''
        Compute the Bayesian Information Criterion score
    '''

    # Number of parameters:
    # - mixt_target: Tnum
    # - mixt_random: Tnum
    # - mixt_nontarget: sum(T_space - 1)
    # - alpha: 1
    # - beta: 1
    K = em_fit_result_dict['mixt_target'].size + em_fit_result_dict['mixt_random'].size + np.sum(em_fit_result_dict['T_space'] - 1) + 2

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



def test_simple():
    '''
        Does a Unit test, samples data from a mixture of one Von mises and random perturbations. Then fits the model and check if everything works.
    '''

    N = 1000
    Tmax = 5
    T_space = np.arange(1, Tmax+1)

    alpha = 9.8
    beta = -0.58

    angles_nontargets = utils.sample_angle((Tmax, Tmax-1))
    targets = np.zeros((Tmax, N))
    nontargets = np.ones((Tmax, N, Tmax-1))*angles_nontargets[:, np.newaxis, :]
    responses = np.zeros((Tmax, N))

    # Correct nontargets just to be sure
    for K_i, K in enumerate(T_space):
        nontargets[K_i, :, (K-1):] = np.nan

    for K in xrange(Tmax):
        kappa_target = alpha*(K+1.0)**beta

        em_fit_target = dict(kappa=kappa_target, alpha=alpha, beta=beta, mixt_target=0.7, mixt_nontargets=0.2, mixt_random=0.1)

        import em_circularmixture

        # Sample from Von Mises
        responses[K] = sample_from_fit(em_fit_target, targets[K], nontargets[K, :, :K])


        em_fit = em_circularmixture.fit(responses[K], targets[K], nontargets[K, :, :K])

        print "True: kappa={kappa:.5}, pt={mixt_target:.3}, pnt={mixt_nontargets:.3}, pr={mixt_random:.3}".format(**em_fit_target)
        print "Fitted: kappa={kappa:.5}, pt={mixt_target:.3}, pnt={mixt_nontargets:.3}, pr={mixt_random:.3}".format(**em_fit)

    # Now try full fit with alpha/beta
    # em_fit = fit(responses, targets, nontargets)

    return T_space, responses, targets, nontargets
    # Check if estimated kappa is within 20% of target one

def test_bays09like():
    '''
        Uses kappa and prob mixtures from Bays09
    '''

    N = 2000

    T_space = np.array([1, 2, 4, 6])
    Tnum = T_space.size
    Tmax = T_space.max()
    kappa_space = np.array([ 19.76349326,  11.2619971 ,   9.22001848,   8.30524648])
    probtarget_space = np.array([ 0.98688956,  0.92068596,  0.71474023,  0.5596124 ])
    probnontarget_space = np.array([ 0.        ,  0.02853913,  0.10499085,  0.28098455])
    probrandom_space = np.array([ 0.01311044,  0.05077492,  0.18026892,  0.15940305])

    beta, alpha = utils.fit_powerlaw(T_space, kappa_space)


    angles_nontargets = utils.sample_angle((Tnum, Tmax-1))
    targets = np.zeros((Tnum, N))

    nontargets = np.ones((Tnum, N, Tmax-1))*angles_nontargets[:, np.newaxis, :]
    responses = np.zeros((Tnum, N))

    for K_i, K in enumerate(T_space):
        nontargets[K_i, :, (K-1):] = np.nan


    for T_i, T in enumerate(T_space):
        kappa_target = alpha*(T)**beta

        em_fit_target = dict(kappa=kappa_target, alpha=alpha, beta=beta, mixt_target=probtarget_space[T_i], mixt_nontargets=probnontarget_space[T_i], mixt_random=probrandom_space[T_i])

        import em_circularmixture

        # Sample from Von Mises
        responses[T_i] = sample_from_fit(em_fit_target, targets[T_i], nontargets[T_i, :, :T_i])


        em_fit = em_circularmixture.fit(responses[T_i], targets[T_i], nontargets[T_i, :, :(T-1)])

        print "True: kappa={kappa:.5}, pt={mixt_target:.3}, pnt={mixt_nontargets:.3}, pr={mixt_random:.3}".format(**em_fit_target)
        print "Fitted: kappa={kappa:.5}, pt={mixt_target:.3}, pnt={mixt_nontargets:.3}, pr={mixt_random:.3}".format(**em_fit)

    # Now try full fit with alpha/beta
    # em_fit = fit(responses, targets, nontargets)

    return T_space, responses, targets, nontargets





if __name__ == '__main__':
    # T_space, responses, targets, nontargets = test_simple()
    T_space, responses, targets, nontargets = test_bays09like()

    result_dict = fit(T_space, responses, targets, nontargets, debug=True)
    # test_bays09like()
    # pass


