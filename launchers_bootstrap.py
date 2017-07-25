#!/usr/bin/env python
# encoding: utf-8
"""
launchers_bootstrap.py


Created by Loic Matthey on 2013-11-12
Copyright (c) 2013 . All rights reserved.
"""

import numpy as np
import os

import utils
import dataio as DataIO
import em_circularmixture
import em_circularmixture_allitems
import em_circularmixture_allitems_uniquekappa
import load_experimental_data


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
    if all_parameters['subaction'] == 'mixed':
        # Mixed runs
        model_outputs = utils.load_npy( os.path.join(os.getenv("WORKDIR_DROP", None), 'Experiments', 'bootstrap_nontargets', 'global_plots_errors_distribution-plots_errors_distribution-d977e237-cfce-473b-a292-00695e725259.npy'))
    else:
        # Conjunctive runs
        model_outputs = utils.load_npy( os.path.join(os.getenv("WORKDIR_DROP", None), 'Experiments', 'bootstrap_nontargets', 'global_plots_errors_distribution-plots_errors_distribution-cc1a49b0-f5f0-4e82-9f0f-5a16a2bfd4e8.npy'))

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


def launcher_do_nontarget_bootstrap_misbindingruns(args):
    '''
        Compute a bootstrap estimate, using outputs from a Misbinding generator run.
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
    if all_parameters['subaction'] == 'mixed' or all_parameters['subaction'] == '':
        # Mixed runs
        model_outputs = utils.load_npy( os.path.join(os.getenv("WORKDIR_DROP", None), 'Experiments', 'bootstrap_nontargets', 'SAVE_global_plots_misbinding_logposterior-plots_misbinding_logposterior-36eb41e9-6370-453e-995e-3876d5105388.npy'))

    data_responses_all = model_outputs['result_all_thetas']
    data_target = model_outputs['target_angle']
    data_nontargets = model_outputs['nontarget_angles']
    ratio_space = model_outputs['ratio_space']

    # Result arrays
    result_bootstrap_samples_allitems = np.nan*np.ones((ratio_space.size, all_parameters['num_repetitions']))
    result_bootstrap_samples = np.nan*np.ones((ratio_space.size, all_parameters['num_repetitions']))
    result_bootstrap_samples_allitems_uniquekappa_sumnontarget = np.nan*np.ones((ratio_space.size, all_parameters['num_repetitions']))
    result_bootstrap_samples_allitems_uniquekappa_allnontarget = np.nan*np.ones((ratio_space.size, all_parameters['num_repetitions']))

    search_progress = progress.Progress(ratio_space.size)

    for ratio_conj_i, ratio_conj in enumerate(ratio_space):
        print "%.2f%%, %s left - %s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

        print "Bootstrap for ratio=%.2f, %d bootstrap samples" % (ratio_conj, all_parameters['num_repetitions'])

        ### WORK WORK WORK work? ###

        # Get some bootstrap samples
        bootstrap_allitems_nontargets_allitems_uniquekappa = em_circularmixture_allitems_uniquekappa.bootstrap_nontarget_stat(
                data_responses_all[ratio_conj_i],
                data_target,
                data_nontargets,
                nb_bootstrap_samples=all_parameters['num_repetitions'],
                resample_targets=False)

        bootstrap_allitems_nontargets = em_circularmixture.bootstrap_nontarget_stat(
                data_responses_all[ratio_conj_i],
                data_target,
                data_nontargets,
                nb_bootstrap_samples=all_parameters['num_repetitions'],
                resample_targets=False)

        # Collect and store responses
        result_bootstrap_samples[ratio_conj_i] = bootstrap_allitems_nontargets['nontarget_bootstrap_samples']

        result_bootstrap_samples_allitems_uniquekappa_sumnontarget[ratio_conj_i] = bootstrap_allitems_nontargets_allitems_uniquekappa['nontarget_bootstrap_samples']
        result_bootstrap_samples_allitems_uniquekappa_allnontarget[ratio_conj_i] = bootstrap_allitems_nontargets_allitems_uniquekappa['allnontarget_bootstrap_samples']

        print result_bootstrap_samples_allitems[ratio_conj_i]
        print result_bootstrap_samples[ratio_conj_i]
        print result_bootstrap_samples_allitems_uniquekappa_sumnontarget[ratio_conj_i]
        print result_bootstrap_samples_allitems_uniquekappa_allnontarget[ratio_conj_i]

        ### /Work ###

        search_progress.increment()
        if run_counter % save_every == 0 or search_progress.done():
            dataio.save_variables_default(locals())
        run_counter += 1

    # Finished
    dataio.save_variables_default(locals())

    print "All finished"
    return locals()


def launcher_do_bootstrap_experimental(args):
    '''
        Compute a bootstrap estimate, using outputs from the experimental data
    '''

    print "Doing a piece of work for launcher_do_bootstrap_experimental"

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
    dataset = load_experimental_data.load_data(
        experiment_id=all_parameters.get('experiment_id', 'bays09'),
        fit_mixture_model=True)

    # Result arrays
    result_bootstrap_nitems_samples = np.nan*np.empty((dataset['n_items_size'], all_parameters['num_repetitions']))
    result_bootstrap_subject_nitems_samples = np.nan*np.empty((dataset['subject_size'], dataset['n_items_size'], all_parameters['num_repetitions']))

    search_progress = progress.Progress(dataset['subject_size']*(dataset['n_items_size']-1))

    for n_items_i, n_items in enumerate(np.unique(dataset['n_items'])):
        if n_items > 1:
            print "Nitems %d, all subjects" % (n_items)

            # Data collapsed accross subjects
            ids_filtered = (dataset['n_items'] == n_items).flatten()

            bootstrap = em_circularmixture.bootstrap_nontarget_stat(
                dataset['response'][ids_filtered, 0],
                dataset['item_angle'][ids_filtered, 0],
                dataset['item_angle'][ids_filtered, 1:n_items],
                nb_bootstrap_samples=all_parameters['num_repetitions'],
                resample_targets=False)

            result_bootstrap_nitems_samples[n_items_i] = bootstrap['nontarget_bootstrap_samples']

            print result_bootstrap_nitems_samples

            for subject_i, subject in enumerate(np.unique(dataset['subject'])):
                print "Nitems %d, subject %d" % (n_items, subject)

                # Bootstrap per subject and nitems
                ids_filtered = (dataset['subject'] == subject).flatten() & (dataset['n_items'] == n_items).flatten()

                # Compute bootstrap if required

                bootstrap = em_circularmixture.bootstrap_nontarget_stat(
                    dataset['response'][ids_filtered, 0],
                    dataset['item_angle'][ids_filtered, 0],
                    dataset['item_angle'][ids_filtered, 1:n_items],
                    nb_bootstrap_samples=all_parameters['num_repetitions'],
                    resample_targets=False)
                result_bootstrap_subject_nitems_samples[subject_i, n_items_i] = bootstrap['nontarget_bootstrap_samples']

                print result_bootstrap_subject_nitems_samples[:, n_items_i]

                search_progress.increment()
                if run_counter % save_every == 0 or search_progress.done():
                    dataio.save_variables_default(locals())
                run_counter += 1


    # Finished
    dataio.save_variables_default(locals())

    print "All finished"
    return locals()


def launcher_do_bootstrap_experimental_sequential(args):
    '''
        Compute a bootstrap estimate, using outputs from the experimental data
    '''

    print "Doing a piece of work for launcher_do_bootstrap_experimental"

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
    dataset = load_experimental_data.load_data(
        experiment_id='gorgo11_sequential',
        fit_mixture_model=True)

    # Result arrays
    result_nontarget_bootstrap_nitems_trecall = np.nan*np.empty((
        dataset['n_items_size'], dataset['n_items_size'],
        all_parameters['num_repetitions']))
    result_nontarget_bootstrap_subject_nitems_trecall = np.nan*np.empty((
        dataset['subject_size'], dataset['n_items_size'],
        dataset['n_items_size'], all_parameters['num_repetitions']))

    search_progress = progress.Progress(dataset['subject_size']*(dataset['n_items_size']-1)*dataset['n_items_size'])

    for n_items_i, n_items in enumerate(np.unique(dataset['n_items'])):
        if n_items > 1:
            for trecall_i, trecall in enumerate(np.unique(dataset['n_items'])):
                print "Nitems %d, trecall %d, all subjects" % (n_items, trecall)
                # Data collapsed accross subjects
                ids_filtered = (
                    (dataset['n_items'] == n_items) &
                    (dataset['probe'] == trecall) &
                    (~dataset['masked'])).flatten()

                bootstrap = em_circularmixture.bootstrap_nontarget_stat(
                    dataset['response'][ids_filtered, 0],
                    dataset['item_angle'][ids_filtered, 0],
                    dataset['item_angle'][ids_filtered, 1:n_items],
                    nb_bootstrap_samples=all_parameters['num_repetitions'],
                    resample_targets=False)

                result_nontarget_bootstrap_nitems_trecall[n_items_i, trecall_i] = bootstrap['nontarget_bootstrap_samples']

                print result_nontarget_bootstrap_nitems_trecall

                for subject_i, subject in enumerate(np.unique(dataset['subject'])):
                    print "Nitems %d, trecall %d, subject %d" % (
                        n_items, trecall, subject)

                    # Bootstrap per subject and nitems
                    ids_filtered = (
                        (dataset['n_items'] == n_items) &
                        (dataset['probe'] == trecall) &
                        (dataset['subject'] == subject) &
                        (~dataset['masked'])).flatten()

                    # Compute bootstrap if required
                    bootstrap = em_circularmixture.bootstrap_nontarget_stat(
                        dataset['response'][ids_filtered, 0],
                        dataset['item_angle'][ids_filtered, 0],
                        dataset['item_angle'][ids_filtered, 1:n_items],
                        nb_bootstrap_samples=all_parameters['num_repetitions'],
                        resample_targets=False)
                    result_nontarget_bootstrap_subject_nitems_trecall[
                        subject_i, n_items_i, trecall_i] = bootstrap['nontarget_bootstrap_samples']

                    print result_nontarget_bootstrap_subject_nitems_trecall[:, n_items_i, trecall_i]

                    search_progress.increment()
                    if run_counter % save_every == 0 or search_progress.done():
                        dataio.save_variables_default(locals())
                    run_counter += 1


    # Finished
    dataio.save_variables_default(locals())

    print "All finished"
    return locals()
