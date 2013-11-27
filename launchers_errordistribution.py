#!/usr/bin/env python
# encoding: utf-8
"""
launchers_errordistribution.py


Created by Loic Matthey on 2013-11-12
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
import em_circularmixture

import launchers

def launcher_do_error_distributions(args):
    '''
        Collect responses for error distribution plots (used in generator/reloader_error_distribution_*.py)

        Do it for T items.

        Looks like the Bays 2009, used in paper.
    '''

    print "Doing a piece of work for launcher_do_error_distributions"

    try:
        # Convert Argparse.Namespace to dict
        all_parameters = vars(args)
    except TypeError:
        # Assume it's already done
        assert type(args) is dict, "args is neither Namespace nor dict, WHY?"
        all_parameters = args

    print all_parameters

    # Create DataIO
    #  (complete label with current variable state)
    dataio = DataIO(output_folder=all_parameters['output_directory'], label=all_parameters['label'].format(**all_parameters))
    save_every = 1
    run_counter = 0

    # Result arrays
    result_responses = np.nan*np.ones((all_parameters['N'], all_parameters['num_repetitions']))
    result_target = np.nan*np.ones((all_parameters['N'], all_parameters['num_repetitions']))
    result_nontargets = np.nan*np.ones((all_parameters['N'], all_parameters['T']-1, all_parameters['num_repetitions']))
    result_em_fits = np.nan*np.ones((5, all_parameters['num_repetitions']))  # kappa, mixt_target, mixt_nontarget, mixt_random, ll


    search_progress = progress.Progress(all_parameters['num_repetitions'])

    for repet_i in xrange(all_parameters['num_repetitions']):
        print "%.2f%%, %s left - %s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

        print "Fit for T=%d, %d/%d" % (all_parameters['T'], repet_i+1, all_parameters['num_repetitions'])

        # Update parameter

        ### WORK WORK WORK work? ###

        # Instantiate
        (_, _, _, sampler) = launchers.init_everything(all_parameters)

        # Sample
        sampler.run_inference(all_parameters)

        # Collect and store responses
        (responses, target, nontarget) = sampler.collect_responses()
        result_responses[:, repet_i] = responses
        result_target[:, repet_i] = target
        result_nontargets[..., repet_i] = nontarget

        # Fit mixture model
        curr_params_fit = em_circularmixture.fit(*sampler.collect_responses())
        result_em_fits[..., repet_i] = [curr_params_fit[key] for key in ('kappa', 'mixt_target', 'mixt_nontargets', 'mixt_random', 'train_LL')]

        ### /Work ###

        search_progress.increment()
        if run_counter % save_every == 0 or search_progress.done():
            dataio.save_variables_default(locals())
        run_counter += 1

    # Finished
    dataio.save_variables_default(locals())

    print "All finished"
    return locals()


def launcher_do_error_distributions_allT(args):
    '''
        Compute histograms of errors distributions. Also get histogram of bias to nontargets.

        Do it for t=1...T items.

        Looks like the Bays 2009, used in paper.
    '''

    print "Doing a piece of work for launcher_do_error_distributions_allT"

    try:
        # Convert Argparse.Namespace to dict
        all_parameters = vars(args)
    except TypeError:
        # Assume it's already done
        assert type(args) is dict, "args is neither Namespace nor dict, WHY?"
        all_parameters = args

    print all_parameters

    # Create DataIO
    #  (complete label with current variable state)
    dataio = DataIO(output_folder=all_parameters['output_directory'], label=all_parameters['label'].format(**all_parameters))
    save_every = 1
    run_counter = 0
    bins = 51

    # Parameters to vary
    T_all = all_parameters['T']
    T_space = np.arange(1, T_all+1)

    # Result arrays
    result_responses = np.nan*np.ones((T_space.size, all_parameters['N'], all_parameters['num_repetitions']))
    result_target = np.nan*np.ones((T_space.size, all_parameters['N'], all_parameters['num_repetitions']))
    result_nontargets = np.nan*np.ones((T_space.size, all_parameters['N'], T_all-1, all_parameters['num_repetitions']))
    result_em_fits = np.nan*np.ones((T_space.size, 5, all_parameters['num_repetitions']))  # kappa, mixt_target, mixt_nontarget, mixt_random, ll


    search_progress = progress.Progress(T_space.size*all_parameters['num_repetitions'])

    for repet_i in xrange(all_parameters['num_repetitions']):
        for T_i, T in enumerate(T_space):
            print "%.2f%%, %s left - %s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

            print "Fit for T=%d, %d/%d" % (T, repet_i+1, all_parameters['num_repetitions'])

            # Update parameter
            all_parameters['T'] = T

            ### WORK WORK WORK work? ###

            # Instantiate
            (_, _, _, sampler) = launchers.init_everything(all_parameters)

            # Sample
            sampler.run_inference(all_parameters)

            # Collect and store responses
            (responses, target, nontarget) = sampler.collect_responses()
            result_responses[T_i, :, repet_i] = responses
            result_target[T_i, :, repet_i] = target
            result_nontargets[T_i, :, :T_i, repet_i] = nontarget[:, :T_i]

            # Fit mixture model
            curr_params_fit = em_circularmixture.fit(*sampler.collect_responses())
            result_em_fits[T_i, :, repet_i] = [curr_params_fit[key] for key in ('kappa', 'mixt_target', 'mixt_nontargets', 'mixt_random', 'train_LL')]


            # Do plots
            sampler.plot_histogram_errors(bins=bins)
            dataio.save_current_figure('papertheo_histogram_errorsM%dsigmax%.2fT%d_{label}_{unique_id}.pdf' % tuple([all_parameters[key] for key in ('M', 'sigmax', 'T')]))

            if T > 1:
                sampler.plot_histogram_bias_nontarget(dataio=dataio)

            ### /Work ###

            search_progress.increment()
            if run_counter % save_every == 0 or search_progress.done():
                dataio.save_variables_default(locals())
            run_counter += 1

    # Finished
    dataio.save_variables_default(locals())

    print "All finished"
    return locals()


