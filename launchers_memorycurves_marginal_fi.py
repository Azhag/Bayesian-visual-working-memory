#!/usr/bin/env python
# encoding: utf-8
"""
launchers_memorycurves_marginal_fi.py


Created by Loic Matthey on 2013-10-20
Copyright (c) 2013 . All rights reserved.
"""

# import matplotlib.pyplot as plt
import numpy as np

import utils
import dataio as DataIO
import progress

import launchers


def launcher_do_memory_curve_marginal_fi(args):
    '''
        Run the model for 1..T items, computing:
        - Precision of samples
        - EM mixture model fits
        - Marginal Inverse Fisher Information
    '''

    print "Doing a piece of work for launcher_do_memory_curve_marginal_fi"

    all_parameters = utils.argparse_2_dict(args)
    print all_parameters

    if all_parameters['burn_samples'] + all_parameters['num_samples'] < 200:
        print "WARNING> you do not have enough samples I think!", all_parameters['burn_samples'] + all_parameters['num_samples']


    # Create DataIO
    #  (complete label with current variable state)
    dataio = DataIO.DataIO(output_folder=all_parameters['output_directory'], label=all_parameters['label'].format(**all_parameters))
    save_every = 1
    run_counter = 0

    # Parameters to vary
    T_all = all_parameters['T']
    T_space = np.arange(1, T_all+1)

    # Result arrays
    result_all_precisions = np.nan*np.ones((T_space.size, all_parameters['num_repetitions']))
    result_marginal_inv_fi = np.nan*np.ones((T_space.size, 4, all_parameters['num_repetitions']))  # inv_FI, inv_FI_std, FI, FI_std
    result_em_fits = np.nan*np.ones((T_space.size, 5, all_parameters['num_repetitions']))  # kappa, mixt_target, mixt_nontarget, mixt_random, ll

    # If desired, will automatically save all Model responses.
    if all_parameters['subaction'] == 'collect_responses':
        result_responses = np.nan*np.ones((T_space.size, all_parameters['N'], all_parameters['num_repetitions']))
        result_target = np.nan*np.ones((T_space.size, all_parameters['N'], all_parameters['num_repetitions']))
        result_nontargets = np.nan*np.ones((T_space.size, all_parameters['N'], all_parameters['T']-1, all_parameters['num_repetitions']))

    search_progress = progress.Progress(T_space.size*all_parameters['num_repetitions'])

    for repet_i in xrange(all_parameters['num_repetitions']):
        for T_i, T in enumerate(T_space):
            print "%.2f%%, %s left - %s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

            print "Fit for T=%d, %d/%d" % (T, repet_i+1, all_parameters['num_repetitions'])

            # Update parameter
            all_parameters['T'] = T

            ### WORK WORK WORK work? ###

            # Fix some parameters
            all_parameters['stimuli_generation'] = 'separated'
            # all_parameters['slice_width'] = np.pi/64.

            # Instantiate
            (_, _, _, sampler) = launchers.init_everything(all_parameters)

            # Sample
            sampler.run_inference(all_parameters)

            # Compute precision
            print "get precision..."
            result_all_precisions[T_i, repet_i] = sampler.get_precision()

            # Fit mixture model
            print "fit mixture model..."
            curr_params_fit = sampler.fit_mixture_model(use_all_targets=True)
            curr_params_fit['mixt_nontargets_sum'] = np.sum(curr_params_fit['mixt_nontargets'])
            result_em_fits[T_i, :, repet_i] = [curr_params_fit[key] for key in ('kappa', 'mixt_target', 'mixt_nontargets_sum', 'mixt_random', 'train_LL')]

            # Compute marginal inverse fisher info
            print "compute marginal inverse fisher info"
            marginal_fi_dict = sampler.estimate_marginal_inverse_fisher_info_montecarlo()
            result_marginal_inv_fi[T_i, :, repet_i] = [marginal_fi_dict[key] for key in ('inv_FI', 'inv_FI_std', 'FI', 'FI_std')]

            # If needed, store responses
            if all_parameters['subaction'] == 'collect_responses':
                (responses, target, nontarget) = sampler.collect_responses()
                result_responses[T_i, :, repet_i] = responses
                result_target[T_i, :, repet_i] = target
                result_nontargets[T_i, :, :T_i, repet_i] = nontarget

                print "collected responses"


            print result_all_precisions[T_i, repet_i], curr_params_fit, marginal_fi_dict
            ### /Work ###

            search_progress.increment()
            if run_counter % save_every == 0 or search_progress.done():
                dataio.save_variables_default(locals())
            run_counter += 1

    # Finished
    dataio.save_variables_default(locals())

    print "All finished"
    return locals()


