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

def fit(T_space, responses, targets_angle, nontargets_angles=np.array([[]]), initialisation_method='random', nb_initialisations=10, debug=False, force_random_less_than=None):
    '''
        Modified mixture model where we fit a parametric power law to kappa as a function of number of items.
        Assumes that we gather samples across samples N, times T and time of recall trecall.

        This uses two powerlaws:
        kappa = theta_0 t**theta_1 tr**theta_2

        Return maximum likelihood values for a mixture model, with:
            - 1 probability of target
            - 1 probability of nontarget
            - 1 probability of random circular
            - kappa_theta parameters of theta_0 t**theta_1 tr**theta_2, where t=number of items, tr = time of recall
        Inputs in radian, in the -pi:pi range.
            - responses: TxTxN
            - targets_angle: TxTxN
            - nontargets_angles TxTxNx(T-1)

        This supports only a unique N for all conditions.
        Could be extended to support [T][tr][N_T_tr], but the flattening out is a bit tough

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
    N = responses.shape[-1]
    max_iter = 1000
    epsilon = 1e-3
    dLL = np.nan

    # Initial parameters
    initial_parameters_list = initialise_parameters(N, T_space, initialisation_method, nb_initialisations)
    overall_LL = -np.inf
    LL = np.nan
    initial_i = 0
    best_kappa_theta, best_mixt_target, best_mixt_random_tr, best_mixt_nontargets_tr = (np.nan, np.nan, np.nan, np.nan)

    for (kappa_theta, mixt_target_tr, mixt_random_tr, mixt_nontargets_tr, resp_trnk) in progress.ProgressDisplay(initial_parameters_list):
        # mixt_target_tr: t, r
        # mixt_nontargets_tr: t, r
        # mixt_random_tr: t, r

        if debug:
            print "New initialisation point: ", (kappa_theta, mixt_target_tr, mixt_random_tr, mixt_nontargets_tr)

        old_LL = -np.inf

        i = 0
        converged = False

        # Precompute some matrices
        error_to_target_trn = wrap(targets_angle - responses)
        error_to_nontargets_trnk = wrap(nontargets_angles - responses[:, :, :, np.newaxis])
        errors_all_trnk = np.c_[error_to_target_trn[:, :, :, np.newaxis], error_to_nontargets_trnk]


        # EM loop
        while i < max_iter and not converged:

            # E-step
            if debug:
                print "E", i, LL, dLL, kappa_theta, mixt_target_tr, mixt_nontargets_tr, mixt_random_tr
            for T_i, T in enumerate(T_space):
                for trecall_i, trecall in enumerate(T_space):
                    if trecall <= T:
                        resp_trnk[T_i, trecall_i, :, 0] = mixt_target_tr[T_i, trecall_i]*vonmisespdf(error_to_target_trn[T_i, trecall_i], 0.0, compute_kappa(T, trecall, kappa_theta))

                        resp_trnk[T_i, trecall_i, :, 1:T] = mixt_nontargets_tr[T_i, trecall_i]/(T - 1.0)*vonmisespdf(error_to_nontargets_trnk[T_i, trecall_i, :, :(T-1)], 0.0, compute_kappa(T, trecall, kappa_theta))
            resp_random_tr1 = mixt_random_tr[:, :, np.newaxis]/(2.*np.pi)

            W_trn = np.nansum(resp_trnk, axis=-1) + resp_random_tr1

            resp_trnk /= W_trn[:, :, :, np.newaxis]
            resp_random_trn = resp_random_tr1/W_trn

            # Compute likelihood
            LL = np.nansum(np.log(W_trn[np.tril_indices(Tnum)]))
            dLL = LL - old_LL
            old_LL = LL

            if (np.abs(dLL) < epsilon):
                converged = True
                break

            # M-step
            mixt_target_tr = np.nansum(resp_trnk[..., 0], axis=-1)/N
            mixt_nontargets_tr = np.nansum(np.nansum(resp_trnk[..., 1:], axis=-1), axis=-1)/N
            mixt_random_tr = np.nansum(resp_random_trn, axis=-1)/N

            # Update kappa
            if np.abs(np.nanmean(resp_trnk)) < 1e-10 or np.all(np.isnan(resp_trnk)):
                if debug:
                    print "Kappas diverged:", kappa_theta, np.nanmean(resp_trnk)
                kappa_theta[:] = 0
                break
            else:
                # Estimate kappa_theta with a numerical M-step, 3D optimisaiton over the loglikelihood.
                # Combine all samples, nitems and recall times.
                kappa_theta = numerical_M_step(T_space, resp_trnk, errors_all_trnk, kappa_theta)

            # BIC
            result_dict = dict(
                kappa_theta=kappa_theta,
                kappa=compute_kappa_all(T_space, kappa_theta),
                mixt_target_tr=mixt_target_tr,
                mixt_nontargets_tr=mixt_nontargets_tr,
                mixt_random_tr=mixt_random_tr,
                train_LL=LL,
                T_space=T_space)
            bic_curr = bic(result_dict, np.log(W_trn))

            if debug:
                print "M", i, LL, dLL, kappa_theta, mixt_target_tr, mixt_nontargets_tr, mixt_random_tr, bic_curr

            i += 1


        # if not converged:
        #     if debug:
        #         print "Warning, Em_circularmixture.fit() did not converge before ", max_iter, "iterations"
        #     kappa = np.nan
        #     mixt_target_tr = np.nan
        #     mixt_nontargets_tr = np.nan
        #     mixt_random_tr = np.nan
        #     rw = np.nan

        if LL > overall_LL and np.isfinite(LL):
            if debug:
                print "New best!", initial_i, overall_LL, LL
            overall_LL = LL
            (best_kappa_theta, best_mixt_target_tr, best_mixt_nontargets_tr, best_mixt_random_tr) = (kappa_theta, mixt_target_tr, mixt_nontargets_tr, mixt_random_tr)

        initial_i += 1


    # Compute BIC and AIC scores
    result_dict['bic'] = bic(result_dict, np.log(W_trn))
    result_dict['aic'] = aic(result_dict)

    return result_dict


def compute_kappa(t, r, kappa_theta):
    '''
        Compute kappa given kappa_theta

        kappa = theta[0] * t**theta[1] * r**theta[2]
    '''
    return kappa_theta[0]*t**kappa_theta[1]*r**kappa_theta[2]


def compute_kappa_all(T_space, kappa_theta):
    '''
        Compute kappa for all (T, trecall)
    '''
    kappa_all = np.nan*np.empty((T_space.size, T_space.size))
    for T_i, T in enumerate(T_space):
        for trecall_i, trecall in enumerate(T_space):
            if trecall <= T:
                kappa_all[T_i, trecall_i] = compute_kappa(T, trecall, kappa_theta)

    return kappa_all


def numerical_M_step(T_space, resp_trnk, errors_all_trnk, kappa_theta):
    '''
        Perform a numerical M-step, optimizing the loglikelihood over kappa_theta

    '''

    def loglikelihood_closure(params, args):
        '''
            params: kappa_theta = (alpha, beta, gamma)
            args: (T_space, resp_trnk, errors_all_trnk)
        '''
        LL_tot = 0
        for T_i, T in enumerate(args['T_space']):
            for trecall_i, trecall in enumerate(args['T_space']):
                if trecall <= T:
                    LL_tot += np.nansum(args['resp_trnk'][T_i, trecall_i]*compute_kappa(T, trecall, params)*np.cos(args['errors_all_trnk'][T_i, trecall_i])) \
                                    -np.nansum(args['resp_trnk'][T_i, trecall_i])*np.log(spsp.i0(compute_kappa(T, trecall, params)))

        if np.isnan(LL_tot):
            LL_tot = np.inf

        return -LL_tot

    args = dict(T_space=T_space, resp_trnk=resp_trnk, errors_all_trnk=errors_all_trnk)

    res = spopt.minimize(loglikelihood_closure, kappa_theta, args=args, bounds=((0, 100), (-2.0, 0.0), (-2.0, 0.0)), options=dict(disp=False))
    # print res['x']

    #  Plots to check optimisation surface
    # # alpha_space = np.linspace(0, 100, 100)
    # alpha = 10
    # beta_space = np.linspace(-2.0, 0.1, 101)
    # gamma_space = beta_space
    # # gamma = -0.8
    # # fit = np.array([[loglikelihood_closure((alpha_, beta_, gamma), args) for alpha_ in alpha_space] for beta_ in beta_space]).T
    # fit = np.array([[loglikelihood_closure((alpha, beta_, gamma_), args) for gamma_ in gamma_space] for beta_ in beta_space]).T
    # utils.pcolor_2d_data(fit, gamma_space, beta_space)
    # plt.show()

    return res['x']



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
    resp_trnk = np.nan*np.empty((Tnum, Tnum, int(N), Tmax))

    for i in xrange(nb_initialisations):
        kappa_theta = np.empty(3)
        kappa_theta[0] = np.random.rand()*30.
        kappa_theta[1] = -np.random.rand()*1
        kappa_theta[2] = -np.random.rand()*1

        # Force a strong on-target prior...
        mixt_target_tr = (np.random.rand(Tnum, Tnum)*0.5 + 0.5)*1.5
        mixt_nontargets_tr = np.random.rand(Tnum, Tnum)*0.2
        mixt_random_tr = np.random.rand(Tnum, Tnum)*0.2

        for T_i, T in enumerate(T_space):
            for trecall_i, trecall in enumerate(T_space):
                if trecall > T:
                    mixt_nontargets_tr[T_i, trecall_i] = np.nan
                    mixt_random_tr[T_i, trecall_i] = np.nan
                    mixt_target_tr[T_i, trecall_i] = np.nan


        mixt_sum_tr = mixt_target_tr + mixt_nontargets_tr + mixt_random_tr

        mixt_target_tr /= mixt_sum_tr
        mixt_nontargets_tr /= mixt_sum_tr
        mixt_random_tr /= mixt_sum_tr

        all_params.append((kappa_theta, mixt_target_tr, mixt_random_tr, mixt_nontargets_tr, resp_trnk))

    return all_params



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


def aic(em_fit_result_dict):
    '''
        Compute Akaike Information Criterion.
    '''
    # Number of parameters:
    # - mixt_target_tr: 1
    # - mixt_random_tr: 1
    # - mixt_nontarget: 1
    # - alpha: 1
    # - beta: 1
    K = em_fit_result_dict['mixt_target_tr'].size + em_fit_result_dict['mixt_random_tr'].size + em_fit_result_dict['mixt_nontargets_tr'].size + 2

    return utils.aic(K, em_fit_result_dict['train_LL'])


def bic(em_fit_result_dict, LL_all):
    '''
        Compute the Bayesian Information Criterion score

        Split it, associating the parameters to the number of datapoint they really take care of.
    '''

    # Number of parameters:
    # - mixt_target_tr: 1
    # - mixt_random_tr: 1
    # - mixt_nontarget_trk: 1
    # - alpha: 1
    # - beta: 1
    # - gamma: 1


    # First count the Loglikelihood
    bic_tot = -2. * np.nansum(LL_all[np.tril_indices(LL_all.shape[0])])

    # Then count alpha, beta and gamma, for all datapoints appropriately
    K = 3
    bic_tot += K * np.log(np.nansum(np.isfinite(LL_all)))

    # Now do the mixture proportions per condition
    for nitems_i, nitems in enumerate(em_fit_result_dict['T_space']):
        for trecall_i, trecall in enumerate(em_fit_result_dict['T_space']):
            if trecall <= nitems:
                K = 3
                bic_tot += K * np.log(np.nansum(np.isfinite(LL_all[nitems_i, trecall_i])))

    return bic_tot




######################################################################
######################################################################
######################################################################



def test_simple():
    '''
        Does a Unit test, samples data from a mixture of one Von mises and random perturbations. Then fits the model and check if everything works.
    '''
    import em_circularmixture

    show_checks = False

    N = 200
    Tmax = 5
    T_space = np.arange(1, Tmax+1)

    kappa_theta = np.array([10., -0.7, -0.3])
    angles_nontargets = utils.sample_angle((Tmax, Tmax, Tmax-1))

    targets = np.zeros((Tmax, Tmax, N))
    nontargets = np.ones((Tmax, Tmax, N, Tmax-1))*angles_nontargets[:, :, np.newaxis, :]
    responses = np.zeros((Tmax, Tmax, N))

    # Filter impossible trecalls
    ind_filt = np.triu_indices(Tmax, 1)
    targets[ind_filt] = np.nan
    nontargets[ind_filt] = np.nan
    responses[ind_filt] = np.nan

    # Correct nontargets just to be sure
    for T_i, T in enumerate(T_space):
        for trecall_i, trecall in enumerate(T_space):
            nontargets[T_i, trecall_i, :, (T-1):] = np.nan

    for T_i, T in enumerate(T_space):
        for trecall_i, trecall in enumerate(T_space):
            if trecall <= T:
                kappa_target = compute_kappa(T, trecall, kappa_theta)
                em_fit_target = dict(kappa=kappa_target, mixt_target=0.75, mixt_nontargets=0.15, mixt_random=0.1)

                # Sample from Von Mises
                responses[T_i, trecall_i] = em_circularmixture.sample_from_fit(em_fit_target, targets[T_i, trecall_i], nontargets[T_i, trecall_i, :, :(T - 1)])

                print "T: {T:d}, trecall: {trecall:d}".format(T=T, trecall=trecall)

                if show_checks:
                    em_fit = em_circularmixture.fit(responses[T_i, trecall_i], targets[T_i, trecall_i], nontargets[T_i, trecall_i, :, :(T - 1)])
                    print "True: kappa={kappa:.5}, pt={mixt_target:.3}, pnt={mixt_nontargets:.3}, pr={mixt_random:.3}".format(**em_fit_target)
                    print "Fitted: kappa={kappa:.5}, pt={mixt_target:.3}, pnt={mixt_nontargets:.3}, pr={mixt_random:.3}".format(**em_fit)

    # Now try full fit with alpha/beta/gamma
    # result_dict = fit(T_space, responses, targets, nontargets, debug=True)

    return T_space, responses, targets, nontargets
    # Check if estimated kappa is within 20% of target one



if __name__ == '__main__':
    T_space, responses, targets, nontargets = test_simple()
    # T_space, responses, targets, nontargets = test_bays09like()

    result_dict = fit(T_space, responses, targets, nontargets, debug=False)

    print result_dict
    # result_dict = fit(T_space, responses, targets, nontargets, debug=True)
    # test_bays09like()
    # pass


