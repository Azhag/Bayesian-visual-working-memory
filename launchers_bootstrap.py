#!/usr/bin/env python
# encoding: utf-8
"""
launchers_bootstrap.py


Created by Loic Matthey on 2013-11-12
Copyright (c) 2013 . All rights reserved.
"""

import numpy as np

import utils
import dataio as DataIO
import em_circularmixture
import em_circularmixture_allitems
import em_circularmixture_allitems_uniquekappa

import launchers

import progress

def launcher_do_nontarget_bootstrap(args):
    '''
        Compute a bootstrap estimate, using outputs from the model run earlier
    '''

    print "Doing a piece of work for launcher_do_nontarget_bootstrap"

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
    dataio = DataIO.DataIO(output_folder=all_parameters['output_directory'], label=all_parameters['label'].format(**all_parameters))
    save_every = 1
    run_counter = 0

    # Load the data
    model_outputs = utils.load_npy( '/nfs/home2/lmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/code/git-bayesian-visual-working-memory/Experiments/error_distribution/error_distribution_conj_M100T6repetitions5_121113_outputs/global_plots_errors_distribution-plots_errors_distribution-cc1a49b0-f5f0-4e82-9f0f-5a16a2bfd4e8.npy')
    # model_outputs = utils.load_npy( '/Users/loicmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/code/git-bayesian-visual-working-memory/Experiments/bootstrap_nontargets/global_plots_errors_distribution-plots_errors_distribution-cc1a49b0-f5f0-4e82-9f0f-5a16a2bfd4e8.npy')

    data_responses_all = model_outputs['result_responses_all'][..., 0]
    data_target_all = model_outputs['result_target_all'][..., 0]
    data_nontargets_all = model_outputs['result_nontargets_all'][..., 0]
    T_space = model_outputs['T_space']
    sigmax_space = model_outputs['sigmax_space']

    K = data_nontargets_all.shape[-1]

    # Result arrays
    result_bootstrap_samples_allitems = np.nan*np.ones((sigmax_space.size, T_space.size, all_parameters['num_repetitions']))
    result_bootstrap_samples = np.nan*np.ones((sigmax_space.size, T_space.size, all_parameters['num_repetitions']))
    result_bootstrap_samples_allitems_uniquekappa_sumnontarget = np.nan*np.ones((sigmax_space.size, T_space.size, all_parameters['num_repetitions']))
    result_bootstrap_samples_allitems_uniquekappa_allnontarget = np.nan*np.ones((sigmax_space.size, T_space.size, K*all_parameters['num_repetitions']))

    search_progress = progress.Progress(sigmax_space.size*(T_space.size-1))

    for sigmax_i, sigmax in enumerate(sigmax_space):
        for T_i, T in enumerate(T_space[1:]):
            T_i += 1

            print "%.2f%%, %s left - %s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

            print "Bootstrap for T=%d, sigmax=%.2f, %d bootstrap samples" % (T, sigmax, all_parameters['num_repetitions'])

            # Update parameter

            ### WORK WORK WORK work? ###

            # Get some bootstrap samples
            bootstrap_allitems_nontargets_allitems_uniquekappa = em_circularmixture_allitems_uniquekappa.bootstrap_nontarget_stat(
                    data_responses_all[sigmax_i, T_i],
                    data_target_all[sigmax_i, T_i],
                    data_nontargets_all[sigmax_i, T_i, :, :T_i],
                    nb_bootstrap_samples=all_parameters['num_repetitions'],
                    resample_targets=False)
            # bootstrap_allitems_nontargets_allitems = em_circularmixture_allitems.bootstrap_nontarget_stat(
            #         data_responses_all[sigmax_i, T_i],
            #         data_target_all[sigmax_i, T_i],
            #         data_nontargets_all[sigmax_i, T_i, :, :T_i],
            #         nb_bootstrap_samples=all_parameters['num_repetitions'],
            #         resample_targets=False)
            bootstrap_allitems_nontargets = em_circularmixture.bootstrap_nontarget_stat(
                    data_responses_all[sigmax_i, T_i],
                    data_target_all[sigmax_i, T_i],
                    data_nontargets_all[sigmax_i, T_i, :, :T_i],
                    nb_bootstrap_samples=all_parameters['num_repetitions'],
                    resample_targets=False)

            # Collect and store responses
            # result_bootstrap_samples_allitems[sigmax_i, T_i] = bootstrap_allitems_nontargets_allitems['nontarget_bootstrap_samples']
            result_bootstrap_samples[sigmax_i, T_i] = bootstrap_allitems_nontargets['nontarget_bootstrap_samples']

            result_bootstrap_samples_allitems_uniquekappa_sumnontarget[sigmax_i, T_i] = bootstrap_allitems_nontargets_allitems_uniquekappa['nontarget_bootstrap_samples']
            result_bootstrap_samples_allitems_uniquekappa_allnontarget[sigmax_i, T_i, :all_parameters['num_repetitions']*T_i] = bootstrap_allitems_nontargets_allitems_uniquekappa['allnontarget_bootstrap_samples']

            print result_bootstrap_samples_allitems[sigmax_i, T_i]
            print result_bootstrap_samples[sigmax_i, T_i]
            print result_bootstrap_samples_allitems_uniquekappa_sumnontarget[sigmax_i, T_i]
            print result_bootstrap_samples_allitems_uniquekappa_allnontarget[sigmax_i, T_i]

            ### /Work ###

            search_progress.increment()
            if run_counter % save_every == 0 or search_progress.done():
                dataio.save_variables_default(locals())
            run_counter += 1

    # Finished
    dataio.save_variables_default(locals())

    print "All finished"
    return locals()




