#!/usr/bin/env python
# encoding: utf-8
"""
launchers_fit_mixturemodels.py

Used to get Memory curves outputs, fit the EM Mixture model on
them and get a distance to specified datasets.

Created by Loic Matthey on 2013-10-20
Copyright (c) 2013 . All rights reserved.
"""

# import matplotlib.pyplot as plt
import numpy as np

import utils
import dataio as DataIO
import progress

import launchers

import load_experimental_data


def launcher_do_fit_mixturemodels(args):
    '''
        Run the model for 1..T items, computing:
        - Precision of samples
        - EM mixture model fits
        - Theoretical Fisher Information
        - EM Mixture model distances to set of currently working datasets.
    '''

    print "Doing a piece of work for launcher_do_fit_mixturemodels"

    all_parameters = utils.argparse_2_dict(args)
    print all_parameters

    if all_parameters['burn_samples'] + all_parameters['num_samples'] < 200:
        print "WARNING> you do not have enough samples I think!", all_parameters['burn_samples'] + all_parameters['num_samples']


    # Create DataIO
    #  (complete label with current variable state)
    dataio = DataIO.DataIO(output_folder=all_parameters['output_directory'], label=all_parameters['label'].format(**all_parameters))
    save_every = 1
    run_counter = 0


    # Load datasets to compare against
    data_bays2009 = load_experimental_data.load_data_bays09(data_dir=all_parameters['experiment_data_dir'], fit_mixture_model=True)
    bays09_experimental_mixtures_mean = data_bays2009['em_fits_nitems_arrays']['mean']
    # Assume that T_space >= max(T_space_bays09)
    bays09_T_space = np.unique(data_bays2009['n_items'])

    data_gorgo11 = load_experimental_data.load_data_simult(data_dir=all_parameters['experiment_data_dir'], fit_mixture_model=True)
    gorgo11_experimental_emfits_mean = data_gorgo11['em_fits_nitems_arrays']['mean']
    gorgo11_T_space = np.unique(data_gorgo11['n_items'])

    # Parameters to vary
    T_max = all_parameters['T']
    T_space = np.arange(1, T_max+1)
    repetitions_axis = -1

    # Result arrays
    result_all_precisions = np.nan*np.empty((T_space.size, all_parameters['num_repetitions']))
    result_fi_theo = np.nan*np.empty((T_space.size, all_parameters['num_repetitions']))
    result_fi_theocov = np.nan*np.empty((T_space.size, all_parameters['num_repetitions']))
    result_em_fits = np.nan*np.empty((T_space.size, 6, all_parameters['num_repetitions']))  # kappa, mixt_target, mixt_nontarget, mixt_random, ll, bic
    result_em_fits_allnontargets = np.nan*np.empty((T_space.size, 5+(T_max-1), all_parameters['num_repetitions']))  # kappa, mixt_target, mixt_nontarget (T-1), mixt_random, ll, bic
    result_dist_bays09 = np.nan*np.empty((T_space.size, 4, all_parameters['num_repetitions']))  # kappa, mixt_target, mixt_nontarget, mixt_random
    result_dist_gorgo11 = np.nan*np.empty((T_space.size, 4, all_parameters['num_repetitions']))  # kappa, mixt_target, mixt_nontarget, mixt_random
    result_dist_bays09_emmixt_KL = np.nan*np.empty((T_space.size, all_parameters['num_repetitions']))
    result_dist_gorgo11_emmixt_KL = np.nan*np.empty((T_space.size, all_parameters['num_repetitions']))

    # If desired, will automatically save all Model responses.
    if all_parameters['collect_responses']:
        print "--- Collecting all responses..."
        result_responses = np.nan*np.ones((T_space.size, all_parameters['N'], all_parameters['num_repetitions']))
        result_target = np.nan*np.ones((T_space.size, all_parameters['N'], all_parameters['num_repetitions']))
        result_nontargets = np.nan*np.ones((T_space.size, all_parameters['N'], T_max-1, all_parameters['num_repetitions']))

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

            # Compute precision
            print "get precision..."
            result_all_precisions[T_i, repet_i] = sampler.get_precision()

            # Fit mixture model
            print "fit mixture model..."
            curr_params_fit = sampler.fit_mixture_model(use_all_targets=False)
            # curr_params_fit['mixt_nontargets_sum'] = np.sum(curr_params_fit['mixt_nontargets'])
            result_em_fits[T_i, :, repet_i] = [curr_params_fit[key] for key in ['kappa', 'mixt_target', 'mixt_nontargets_sum', 'mixt_random', 'train_LL', 'bic']]
            result_em_fits_allnontargets[T_i, :2, repet_i] = [curr_params_fit['kappa'], curr_params_fit['mixt_target']]
            result_em_fits_allnontargets[T_i, 2:(2+T-1), repet_i] = curr_params_fit['mixt_nontargets']
            result_em_fits_allnontargets[T_i, -3:, repet_i] = [curr_params_fit[key] for key in ('mixt_random', 'train_LL', 'bic')]

            # Compute fisher info
            print "compute fisher info"
            result_fi_theo[T_i, repet_i] = sampler.estimate_fisher_info_theocov(use_theoretical_cov=False)
            result_fi_theocov[T_i, repet_i] = sampler.estimate_fisher_info_theocov(use_theoretical_cov=True)

            # Compute distances to datasets
            if T in bays09_T_space:
                result_dist_bays09[T_i, :, repet_i] = (bays09_experimental_mixtures_mean[:, bays09_T_space==T].flatten() - result_em_fits[T_i, :4, repet_i])**2.

                result_dist_bays09_emmixt_KL[T_i, repet_i] = utils.KL_div(result_em_fits[T_i, 1:4, repet_i], bays09_experimental_mixtures_mean[1:, bays09_T_space==T].flatten())

            if T in gorgo11_T_space:
                result_dist_gorgo11[T_i, :, repet_i] = (gorgo11_experimental_emfits_mean[:, gorgo11_T_space == T].flatten() - result_em_fits[T_i, :4, repet_i])**2.

                result_dist_gorgo11_emmixt_KL[T_i, repet_i] = utils.KL_div(result_em_fits[T_i, 1:4, repet_i], gorgo11_experimental_emfits_mean[1:, gorgo11_T_space==T].flatten())


            # If needed, store responses
            if all_parameters['collect_responses']:
                (responses, target, nontarget) = sampler.collect_responses()
                result_responses[T_i, :, repet_i] = responses
                result_target[T_i, :, repet_i] = target
                result_nontargets[T_i, :, :T_i, repet_i] = nontarget

                print "collected responses"


            print "CURRENT RESULTS:\n", result_all_precisions[T_i, repet_i], curr_params_fit, result_fi_theo[T_i, repet_i], result_fi_theocov[T_i, repet_i], np.sum(result_dist_bays09[T_i, :, repet_i]), np.sum(result_dist_gorgo11[T_i, :, repet_i]), "\n"
            ### /Work ###

            search_progress.increment()
            if run_counter % save_every == 0 or search_progress.done():
                dataio.save_variables_default(locals())
            run_counter += 1

    # Finished
    dataio.save_variables_default(locals())

    print "All finished"
    return locals()


