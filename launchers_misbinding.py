#!/usr/bin/env python
# encoding: utf-8
"""
launchers_misbinding.py


Created by Loic Matthey on 2013-06-18
Copyright (c) 2013 . All rights reserved.
"""

import matplotlib.pyplot as plt
import numpy as np

from randomfactorialnetwork import *
from datagenerator import *
from statisticsmeasurer import *
from slicesampler import *
from utils import *
from dataio import *
from gibbs_sampler_continuous_fullcollapsed_randomfactorialnetwork import *

import launchers


###### PBS runners #####

def launcher_do_average_posterior(args):
    '''
        Compute average posterior for a fixed set of stimuli.


        Show graphically how it looks like.
    '''

    # TODO Could analyse it theoretically at some point, e.g. probability of answering nontarget?


    print "Doing a piece of work for launcher_do_average_posterior"

    try:
        # Convert Argparse.Namespace to dict
        all_parameters = vars(args)
    except TypeError:
        # Assume it's already done
        assert type(args) is dict, "args is neither Namespace nor dict, WHY?"
        all_parameters = args


    # Create DataIO
    #  (complete label with current variable state)
    dataio = DataIO(output_folder=all_parameters['output_directory'], label=all_parameters['label'].format(**all_parameters))

    # Fix some parameters
    all_parameters['stimuli_generation'] = lambda T: np.linspace(-np.pi*0.6, np.pi*0.6, T)
    all_parameters['stimuli_generation_recall'] = 'random'
    all_parameters['enforce_first_stimulus'] = False
    num_points = 500

    if 'do_precision' in all_parameters:
        do_precision = all_parameters['do_precision']
    else:
        do_precision = True

    result_all_log_posterior = np.nan*np.ones((all_parameters['N'], num_points))
    result_all_thetas = np.zeros(all_parameters['N'])

    search_progress = progress.Progress(all_parameters['N'])
    save_every = 10
    print_every = 10
    run_counter = 0
    ax_handle = None

    plt.ion()

    # all_parameters['rc_scale']  = rc_scale

    (random_network, data_gen, stat_meas, sampler) = launchers.init_everything(all_parameters)

    ### WORK WORK WORK work? ###
    all_angles = np.linspace(-np.pi, np.pi, num_points)

    if do_precision:
        print 'Precision...'
        sampler.sample_theta(num_samples=all_parameters['num_samples'], burn_samples=100, selection_method=all_parameters['selection_method'], selection_num_samples=all_parameters['selection_num_samples'], integrate_tc_out=False, debug=True)

        result_all_thetas, targets, nontargets = sampler.collect_responses()


    print "Average posterior..."

    for n in xrange(all_parameters['N']):
        if run_counter % print_every == 0:
            print "%.2f%% %s/%s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

        result_all_log_posterior[n] = sampler.compute_likelihood_fullspace(n=n, all_angles=all_angles, num_points=num_points, should_exponentiate=False, remove_mean=True)[:, -1].T

        ### /Work ###

        search_progress.increment()
        if run_counter % save_every == 0 or search_progress.done():
            dataio.save_variables_default(locals())

            # Plots
            # plt.figure(1)
            # plt.plot(result_all_log_posterior.T, hold=False)

            # ax_handle = plot_mean_std_area(all_angles, nanmean(result_all_log_posterior[ axis=0), nanstd(result_all_log_posterior[ axis=0), ax_handle=ax_handle)
            # ax_handle.hold(False)

            # dataio.save_current_figure('FI_compare_theo_finite-precisionvstheo-{label}_{unique_id}.pdf')


        run_counter += 1

    #### Plots ###
    plot_mean_std_area(all_angles, nanmean(result_all_log_posterior, axis=0), nanstd(result_all_log_posterior, axis=0), ax_handle=ax_handle)
    dataio.save_current_figure('avg_posterior-posterior-{label}_{unique_id}.pdf')
    if do_precision:
        sampler.plot_histogram_errors(bins=50, nice_xticks=True)
        dataio.save_current_figure('avg_posterior-hist_errors-{label}_{unique_id}.pdf')

        sampler.plot_histogram_responses(bins=50, show_angles=True, nice_xticks=True)
        dataio.save_current_figure('avg_posterior-hist_responses-{label}_{unique_id}.pdf')

    print "All finished"

    plt.show()

    return locals()


def launcher_do_variability_mixture(args):
    '''
        Compute posterior in no noise case, to see the effect of ratio on the mixture probability

    '''

    print "Doing a piece of work for launcher_do_variability_mixture"

    try:
        # Convert Argparse.Namespace to dict
        all_parameters = vars(args)
    except TypeError:
        # Assume it's already done
        assert type(args) is dict, "args is neither Namespace nor dict, WHY?"
        all_parameters = args

    # Create DataIO
    #  (complete label with current variable state)
    dataio = DataIO(output_folder=all_parameters['output_directory'], label=all_parameters['label'].format(**all_parameters))


    # Do it for multiple ratios and multiple sigmas
    plt.ion()

    # Fix some parameters
    all_parameters['stimuli_generation'] = 'separated'
    all_parameters['stimuli_generation_recall'] = 'random'
    all_parameters['enforce_first_stimulus'] = False
    all_parameters['num_samples'] = 500
    all_parameters['selection_method'] = 'last'
    all_parameters['code_type'] = 'mixed'
    all_parameters['autoset_parameters'] = True
    all_parameters['M'] = 100
    all_parameters['N'] = 100
    all_parameters['inference_method'] = 'none'
    all_parameters['T']  = 2
    all_parameters['sigmay'] = 0.0000001

    # ratio_space = np.array([0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9])
    ratio_space = np.linspace(0.01, 0.7, 20.)

    all_parameters['sigmax'] = 0.05

    num_points = 500

    result_all_posterior = np.nan*np.ones((ratio_space.size, all_parameters['N'], num_points))
    result_all_mixture_params = np.nan*np.ones((ratio_space.size, all_parameters['N'], 3))
    result_all_bimodality_tests = np.nan*np.ones((ratio_space.size, all_parameters['N'], 2))

    search_progress = progress.Progress(ratio_space.size*all_parameters['N'])
    save_every = 10
    print_every = 10
    run_counter = 0
    ax_handle = None


    all_angles = np.linspace(-np.pi, np.pi, num_points)

    for ratio_i, ratio_conj in enumerate(ratio_space):

        all_parameters['ratio_conj'] = ratio_conj
        (random_network, data_gen, stat_meas, sampler) = launchers.init_everything(all_parameters)

        ### WORK WORK WORK work? ###

        print "Average posterior..."

        for n in xrange(all_parameters['N']):
            if run_counter % print_every == 0:
                print "%.2f%% %s/%s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

            result_all_posterior[ratio_i, n] = sampler.compute_likelihood_fullspace(n=n, all_angles=all_angles, num_points=num_points, should_exponentiate=True, remove_mean=True)[:, -1].T

            result_all_mixture_params[ratio_i, n] = fit_gaussian_mixture_fixedmeans(all_angles, result_all_posterior[ratio_i, n], fixed_means=data_gen.stimuli_correct[n, :, -1], normalise=True, return_fitted_data=False, should_plot=False)

            result_all_bimodality_tests[ratio_i, n] = (bimodality_coefficient(all_angles, result_all_posterior[ratio_i, n]),
                                                       ashman_d(all_angles, result_all_posterior[ratio_i, n])
                                                       )

            ### /Work ###
            search_progress.increment()
            # if run_counter % save_every == 0 or search_progress.done():
                # dataio.save_variables_default(locals())


        print result_all_bimodality_tests

    plt.figure()
    plt.plot(ratio_space, np.mean(result_all_bimodality_tests[..., 0], axis=1))
    plt.title('Bimodality coefficient')

    plt.figure()
    plt.plot(ratio_space, np.mean(result_all_bimodality_tests[..., 1], axis=1))
    plt.title('Ashman D')

    plt.figure()
    plt.plot(ratio_space, np.mean(result_all_mixture_params[..., 0], axis=1))
    plt.title('Mixture proportion')

    # plt.figure()
    # plt.plot(np.mean(np.abs(result_all_mixture_params[..., 1] - result_all_mixture_params[..., 3]), axis=1))
    # plt.title('|mu_1 - mu_2|')

    plt.figure()
    plt.plot(ratio_space, np.mean(result_all_mixture_params[..., 0]*data_gen.stimuli_correct[:, 0, -1] + (1.-result_all_mixture_params[..., 0])*data_gen.stimuli_correct[:, 1, -1], axis=1))
    plt.title('alpha mu_1 + alpha mu_2')

    return locals()









