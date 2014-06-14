#!/usr/bin/env python
# encoding: utf-8
"""
launchers_higher_dimensions.py

Created by Loic Matthey on 2014-05-20
Copyright (c) 2014 . All rights reserved.
"""

import matplotlib.pyplot as plt
import numpy as np

import traceback

import utils
import dataio as DataIO
import progress

import launchers


def launcher_do_check_scaling_ratio_with_M(args):
    '''
        Reviewer 3 asked to see if the proportion of conjunctive units varies with M when a given precision is to be achieved.

        Check it.
    '''

    print "Doing a piece of work for launcher_do_check_scaling_ratio_with_M"

    all_parameters = utils.argparse_2_dict(args)
    print all_parameters

    if all_parameters['burn_samples'] + all_parameters['num_samples'] < 200:
        print "WARNING> you do not have enough samples I think!", all_parameters['burn_samples'] + all_parameters['num_samples']

    if 'plots_during_simulation_callback' in all_parameters:
        plots_during_simulation_callback = all_parameters['plots_during_simulation_callback']
        del all_parameters['plots_during_simulation_callback']
    else:
        plots_during_simulation_callback = None

    # Create DataIO
    #  (complete label with current variable state)
    dataio = DataIO.DataIO(output_folder=all_parameters['output_directory'], label=all_parameters['label'].format(**all_parameters))
    save_every = 1
    run_counter = 0

    # Fix some parameters
    all_parameters['autoset_parameters'] = True

    # Parameters to vary
    nb_M_space = 10
    M_max = 800
    M_min = 20
    M_space = np.arange(M_min, M_max, np.ceil((M_max - M_min)/float(nb_M_space)), dtype=int)
    nb_ratio_space = 10
    ratio_space = np.linspace(0.0001, 1.0, nb_ratio_space)

    # Result arrays
    result_all_precisions = np.nan*np.ones((M_space.size, ratio_space.size, all_parameters['num_repetitions']))
    result_em_fits = np.nan*np.ones((M_space.size, ratio_space.size, 5, all_parameters['num_repetitions']))  # kappa, mixt_target, mixt_nontarget, mixt_random, ll

    search_progress = progress.Progress(M_space.size*ratio_space.size*all_parameters['num_repetitions'])

    for repet_i in xrange(all_parameters['num_repetitions']):
        for M_i, M in enumerate(M_space):
            for ratio_i, ratio in enumerate(ratio_space):
                print "%.2f%%, %s left - %s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

                print "Fit for M=%d, ratio=%.3f  %d/%d" % (M, ratio, repet_i+1, all_parameters['num_repetitions'])

                # Update parameter
                all_parameters['M'] = M
                all_parameters['ratio_conj'] = ratio

                ### WORK WORK WORK work? ###


                try:
                    # Instantiate
                    (_, _, _, sampler) = launchers.init_everything(all_parameters)

                    # Sample
                    sampler.run_inference(all_parameters)

                    # Compute precision
                    print "get precision..."
                    result_all_precisions[M_i, ratio_i, repet_i] = sampler.get_precision()

                    # Fit mixture model
                    print "fit mixture model..."
                    curr_params_fit = sampler.fit_mixture_model(use_all_targets=False)
                    result_em_fits[M_i, ratio_i, :, repet_i] = [curr_params_fit[key] for key in ('kappa', 'mixt_target', 'mixt_nontargets_sum', 'mixt_random', 'train_LL')]

                except Exception:
                    # oh well...
                    print "something failed here, so sad"

                print result_all_precisions[M_i, ratio_i, repet_i], curr_params_fit

                ## Run callback function if exists
                if plots_during_simulation_callback:
                    print "Doing plots..."
                    try:
                        # Best super safe, if this fails then the simulation must continue!
                        plots_during_simulation_callback['function'](locals(), plots_during_simulation_callback['parameters'])
                        print "plots done."
                    except Exception as e:
                        print "error during plotting callback function", plots_during_simulation_callback['function'], plots_during_simulation_callback['parameters']
                        print e
                        traceback.print_exc()

                ### /Work ###
                search_progress.increment()
                if run_counter % save_every == 0 or search_progress.done():
                    dataio.save_variables_default(locals())
                run_counter += 1

    # Finished
    dataio.save_variables_default(locals())

    print "All finished"
    return locals()


def launcher_do_check_scaling_ratio_with_M_single(args):
    '''
        Reviewer 3 asked to see if the proportion of conjunctive units varies with M when a given precision is to be achieved.

        Check it.

        Single M, ratio_conj pair. Called from a generator.
    '''

    print "Doing a piece of work for launcher_do_check_scaling_ratio_with_M_single"

    all_parameters = utils.argparse_2_dict(args)
    print all_parameters

    if all_parameters['burn_samples'] + all_parameters['num_samples'] < 200:
        print "WARNING> you do not have enough samples I think!", all_parameters['burn_samples'] + all_parameters['num_samples']

    if 'plots_during_simulation_callback' in all_parameters:
        plots_during_simulation_callback = all_parameters['plots_during_simulation_callback']
        del all_parameters['plots_during_simulation_callback']
    else:
        plots_during_simulation_callback = None

    # Create DataIO
    #  (complete label with current variable state)
    dataio = DataIO.DataIO(output_folder=all_parameters['output_directory'], label=all_parameters['label'].format(**all_parameters))
    save_every = 1
    run_counter = 0

    # Fix some parameters
    all_parameters['autoset_parameters'] = True

    # Parameters to vary

    # Result arrays
    result_all_precisions = np.nan*np.ones((all_parameters['num_repetitions']))
    result_em_fits = np.nan*np.ones((5, all_parameters['num_repetitions']))  # kappa, mixt_target, mixt_nontarget, mixt_random, ll

    search_progress = progress.Progress(all_parameters['num_repetitions'])

    for repet_i in xrange(all_parameters['num_repetitions']):
        print "%.2f%%, %s left - %s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

        print "Fit for M=%d, ratio=%.3f  %d/%d" % (all_parameters['M'], all_parameters['ratio_conj'], repet_i+1, all_parameters['num_repetitions'])

        # Update parameter

        ### WORK WORK WORK work? ###

        # Instantiate
        (_, _, _, sampler) = launchers.init_everything(all_parameters)

        # Sample
        sampler.run_inference(all_parameters)

        # Compute precision
        print "get precision..."
        result_all_precisions[repet_i] = sampler.get_precision()

        # Fit mixture model
        print "fit mixture model..."
        curr_params_fit = sampler.fit_mixture_model(use_all_targets=False)
        result_em_fits[:, repet_i] = [curr_params_fit[key] for key in ('kappa', 'mixt_target', 'mixt_nontargets_sum', 'mixt_random', 'train_LL')]

        print result_all_precisions[repet_i], curr_params_fit

        ## Run callback function if exists
        if plots_during_simulation_callback:
            print "Doing plots..."
            try:
                # Best super safe, if this fails then the simulation must continue!
                plots_during_simulation_callback['function'](locals(), plots_during_simulation_callback['parameters'])
                print "plots done."
            except Exception as e:
                print "error during plotting callback function", plots_during_simulation_callback['function'], plots_during_simulation_callback['parameters']
                print e
                traceback.print_exc()

        ### /Work ###
        search_progress.increment()
        if run_counter % save_every == 0 or search_progress.done():
            dataio.save_variables_default(locals())
        run_counter += 1

    # Finished
    dataio.save_variables_default(locals())

    print "All finished"
    return locals()

