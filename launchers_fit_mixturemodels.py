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
import em_circularmixture_parametrickappa_doublepowerlaw


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
    # result_fi_theo = np.nan*np.empty((T_space.size, all_parameters['num_repetitions']))
    # result_fi_theocov = np.nan*np.empty((T_space.size, all_parameters['num_repetitions']))
    result_em_fits = np.nan*np.empty((T_space.size, 6, all_parameters['num_repetitions']))  # kappa, mixt_target, mixt_nontarget, mixt_random, ll, bic
    # result_em_fits_allnontargets = np.nan*np.empty((T_space.size, 5+(T_max-1), all_parameters['num_repetitions']))  # kappa, mixt_target, mixt_nontarget (T-1), mixt_random, ll, bic
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
            # result_em_fits_allnontargets[T_i, :2, repet_i] = [curr_params_fit['kappa'], curr_params_fit['mixt_target']]
            # result_em_fits_allnontargets[T_i, 2:(2+T-1), repet_i] = curr_params_fit['mixt_nontargets']
            # result_em_fits_allnontargets[T_i, -3:, repet_i] = [curr_params_fit[key] for key in ('mixt_random', 'train_LL', 'bic')]

            # Compute fisher info
            # print "compute fisher info"
            # result_fi_theo[T_i, repet_i] = sampler.estimate_fisher_info_theocov(use_theoretical_cov=False)
            # result_fi_theocov[T_i, repet_i] = sampler.estimate_fisher_info_theocov(use_theoretical_cov=True)

            # Compute distances to datasets
            if T in bays09_T_space:
                result_dist_bays09[T_i, :, repet_i] = (bays09_experimental_mixtures_mean[:, bays09_T_space == T].flatten() - result_em_fits[T_i, :4, repet_i])**2.

                result_dist_bays09_emmixt_KL[T_i, repet_i] = utils.KL_div(result_em_fits[T_i, 1:4, repet_i], bays09_experimental_mixtures_mean[1:, bays09_T_space == T].flatten())

            if T in gorgo11_T_space:
                result_dist_gorgo11[T_i, :, repet_i] = (gorgo11_experimental_emfits_mean[:, gorgo11_T_space == T].flatten() - result_em_fits[T_i, :4, repet_i])**2.

                result_dist_gorgo11_emmixt_KL[T_i, repet_i] = utils.KL_div(result_em_fits[T_i, 1:4, repet_i], gorgo11_experimental_emfits_mean[1:, gorgo11_T_space == T].flatten())


            # If needed, store responses
            if all_parameters['collect_responses']:
                (responses, target, nontarget) = sampler.collect_responses()
                result_responses[T_i, :, repet_i] = responses
                result_target[T_i, :, repet_i] = target
                result_nontargets[T_i, :, :T_i, repet_i] = nontarget

                print "collected responses"


            print "CURRENT RESULTS:\n", result_all_precisions[T_i, repet_i], curr_params_fit, np.sum(result_dist_bays09[T_i, :, repet_i]), np.sum(result_dist_gorgo11[T_i, :, repet_i]), "\n"
            ### /Work ###

            search_progress.increment()
            if run_counter % save_every == 0 or search_progress.done():
                dataio.save_variables_default(locals())
            run_counter += 1

    # Finished
    dataio.save_variables_default(locals())

    print "All finished"
    return locals()


def launcher_do_fit_mixturemodels_sequential_fixedtrecall(args):
    '''
        Run the model for 1..T items sequentially, using the t_recall provided, computing:
        - Precision of samples
        - EM mixture model fits
        - Theoretical Fisher Information
        - EM Mixture model distances to set of currently working datasets.
    '''

    raise ValueError("You shouldn't run this, it's not really a useful thing..")



def launcher_do_fit_mixturemodels_sequential_alltrecall(args):
    '''
        Run the model for 1..T items sequentially, for all possible trecall/T.
        Compute:
        - Precision of samples
        - EM mixture model fits. Both independent and collapsed model.
        - Theoretical Fisher Information
        - EM Mixture model distances to set of currently working datasets.
    '''

    print "Doing a piece of work for launcher_do_fit_mixturemodels_sequential_alltrecall"

    all_parameters = utils.argparse_2_dict(args)
    print all_parameters

    if all_parameters['burn_samples'] + all_parameters['num_samples'] < 200:
        print "WARNING> you do not have enough samples I think!", all_parameters['burn_samples'] + all_parameters['num_samples']

    # Create DataIO
    #  (complete label with current variable state)
    dataio = DataIO.DataIO(output_folder=all_parameters['output_directory'], label=all_parameters['label'].format(**all_parameters))
    save_every = 1
    run_counter = 0

    # Load dataset to compare against
    data_gorgo11_sequ = load_experimental_data.load_data_gorgo11_sequential(data_dir=all_parameters['experiment_data_dir'], fit_mixture_model=True)
    gorgo11_sequ_T_space = np.unique(data_gorgo11_sequ['n_items'])


    # Parameters to vary
    T_max = all_parameters['T']
    T_space = np.arange(1, T_max+1)
    repetitions_axis = -1

    # Result arrays
    result_all_precisions = np.nan*np.empty((T_space.size, T_space.size, all_parameters['num_repetitions']))
    result_fi_theo = np.nan*np.empty((T_space.size, T_space.size, all_parameters['num_repetitions']))
    result_fi_theocov = np.nan*np.empty((T_space.size, T_space.size, all_parameters['num_repetitions']))
    result_em_fits = np.nan*np.empty((T_space.size, T_space.size, 6, all_parameters['num_repetitions']))  # kappa, mixt_target, mixt_nontarget, mixt_random, ll, bic
    result_em_fits_collapsed_tr = np.nan*np.empty((T_space.size, T_space.size, 4, all_parameters['num_repetitions']))  # kappa_tr, mixt_target_tr, mixt_nontarget_tr, mixt_random_tr
    result_em_fits_collapsed_summary = np.nan*np.empty((5, all_parameters['num_repetitions'])) # bic, ll, kappa_theta

    result_dist_gorgo11_sequ = np.nan*np.empty((T_space.size, T_space.size, 4, all_parameters['num_repetitions']))  # kappa, mixt_target, mixt_nontarget, mixt_random
    result_dist_gorgo11_sequ_emmixt_KL = np.nan*np.empty((T_space.size, T_space.size, all_parameters['num_repetitions']))

    result_dist_gorgo11_sequ_collapsed = np.nan*np.empty((T_space.size, T_space.size, 4, all_parameters['num_repetitions']))
    result_dist_gorgo11_sequ_collapsed_emmixt_KL = np.nan*np.empty((T_space.size, T_space.size, all_parameters['num_repetitions']))

    gorgo11_sequ_collapsed_mixtmod_mean = data_gorgo11_sequ['collapsed_em_fits_doublepowerlaw_array']


    # If desired, will automatically save all Model responses.
    if all_parameters['collect_responses']:
        print "--- Collecting all responses..."
        result_responses = np.nan*np.empty((T_space.size, T_space.size, all_parameters['N'], all_parameters['num_repetitions']))
        result_target = np.nan*np.empty((T_space.size, T_space.size, all_parameters['N'], all_parameters['num_repetitions']))
        result_nontargets = np.nan*np.empty((T_space.size, T_space.size, all_parameters['N'], T_max-1, all_parameters['num_repetitions']))

    search_progress = progress.Progress(T_space.size*(T_space.size + 1)/2.*all_parameters['num_repetitions'])

    for repet_i in xrange(all_parameters['num_repetitions']):
        for T_i, T in enumerate(T_space):
            for trecall_i, trecall in enumerate(np.arange(T, 0, -1)):
                # Inverting indexing of trecall, to be consistent. trecall_i 0 == last item.
                # But trecall still means the actual time of recall!
                print "%.2f%%, %s left - %s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())
                print "Fit for T=%d, tr=%d, %d/%d" % (T, trecall, repet_i+1, all_parameters['num_repetitions'])

                # Update parameter
                all_parameters['T'] = T
                all_parameters['fixed_cued_feature_time'] = trecall - 1

                ### WORK WORK WORK work? ###
                # Instantiate
                (_, _, _, sampler) = launchers.init_everything(all_parameters)

                # Sample
                sampler.run_inference(all_parameters)

                # Compute precision
                print "get precision..."
                result_all_precisions[T_i, trecall_i, repet_i] = sampler.get_precision()

                # Fit mixture model, independent
                print "fit mixture model..."
                curr_params_fit = sampler.fit_mixture_model(use_all_targets=False)
                result_em_fits[T_i, trecall_i, :, repet_i] = [curr_params_fit[key] for key in ['kappa', 'mixt_target', 'mixt_nontargets_sum', 'mixt_random', 'train_LL', 'bic']]

                # Compute fisher info
                print "compute fisher info"
                result_fi_theo[T_i, trecall_i, repet_i] = sampler.estimate_fisher_info_theocov(use_theoretical_cov=False)
                result_fi_theocov[T_i, trecall_i, repet_i] = sampler.estimate_fisher_info_theocov(use_theoretical_cov=True)

                # Compute distances to datasets (this is for the non-collapsed stuff, not the best)
                if T in gorgo11_sequ_T_space:
                    gorgo11_sequ_mixtures_mean = data_gorgo11_sequ['em_fits_nitems_trecall_arrays'][gorgo11_sequ_T_space==T, trecall_i, :4].flatten()

                    result_dist_gorgo11_sequ[T_i, trecall_i, :, repet_i] = (gorgo11_sequ_mixtures_mean - result_em_fits[T_i, trecall_i, :4, repet_i])**2.
                    result_dist_gorgo11_sequ_emmixt_KL[T_i, trecall_i, repet_i] = utils.KL_div(result_em_fits[T_i, trecall_i, 1:4, repet_i], gorgo11_sequ_mixtures_mean[1:])


                # If needed, store responses
                if all_parameters['collect_responses']:
                    print "collect responses"
                    (responses, target, nontarget) = sampler.collect_responses()
                    result_responses[T_i, trecall_i, :, repet_i] = responses
                    result_target[T_i, trecall_i, :, repet_i] = target
                    result_nontargets[T_i, trecall_i, :, :T_i, repet_i] = nontarget


                print "CURRENT RESULTS:\n", result_all_precisions[T_i, trecall_i, repet_i], curr_params_fit, result_fi_theo[T_i, trecall_i, repet_i], result_fi_theocov[T_i, trecall_i, repet_i], np.sum(result_dist_gorgo11_sequ[T_i, trecall_i, :, repet_i]), np.sum(result_dist_gorgo11_sequ_emmixt_KL[T_i, trecall_i, repet_i]), "\n"
                ### /Work ###

                search_progress.increment()
                if run_counter % save_every == 0 or search_progress.done():
                    dataio.save_variables_default(locals())
                run_counter += 1

        # Fit Collapsed mixture model
        # TODO check dimensionality...
        print 'Fitting Collapsed double powerlaw mixture model...'
        params_fit = em_circularmixture_parametrickappa_doublepowerlaw.fit(T_space, result_responses[..., repet_i], result_target[..., repet_i], result_nontargets[..., repet_i], debug=False)

        # First store the parameters that depend on T/trecall
        for i, key in enumerate(['kappa', 'mixt_target_tr', 'mixt_nontargets_tr', 'mixt_random_tr']):
            result_em_fits_collapsed_tr[..., i, repet_i] =  params_fit[key]

        # Then the ones that do not, only one per full collapsed fit.
        result_em_fits_collapsed_summary[0, repet_i] = params_fit['bic']
        # result_em_fits_collapsed_summary[1, repet_i] = params_fit['train_LL']
        result_em_fits_collapsed_summary[2:, repet_i] = params_fit['kappa_theta']

        # Compute distances to dataset for collapsed model
        result_dist_gorgo11_sequ_collapsed[..., repet_i] = (gorgo11_sequ_collapsed_mixtmod_mean - result_em_fits_collapsed_tr[..., repet_i])**2.
        result_dist_gorgo11_sequ_collapsed_emmixt_KL[..., repet_i] = utils.KL_div(result_em_fits_collapsed_tr[..., 1:4, repet_i], gorgo11_sequ_collapsed_mixtmod_mean[..., 1:], axis=-1)


    # Finished
    dataio.save_variables_default(locals())

    print "All finished"
    return locals()


def launcher_do_fit_mixturemodel_dualrecall(args):
    '''
        Run the model for T items, trying to fit
        the DualRecall dataset, which has two conditions.

        Get:
        - Precision
        - EM mixture model fits
        - Theoretical Fisher Information
        - EM Mixture model distances
    '''

    print "Doing a piece of work for launcher_do_fit_mixturemodel_dualrecall"

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
    data_dualrecall = load_experimental_data.load_data_dualrecall(data_dir=all_parameters['experiment_data_dir'], fit_mixture_model=True)
    dualrecall_T_space = data_dualrecall['data_to_fit']['n_items']

    dualrecall_experimental_angle_emfits_mean = data_dualrecall['em_fits_angle_nitems_arrays']['mean']
    dualrecall_experimental_colour_emfits_mean = data_dualrecall['em_fits_colour_nitems_arrays']['mean']

    # Parameters to vary
    repetitions_axis = -1

    # Result arrays
    result_all_precisions = np.nan*np.empty((all_parameters['num_repetitions']))
    result_fi_theo = np.nan*np.empty((all_parameters['num_repetitions']))
    result_fi_theocov = np.nan*np.empty((all_parameters['num_repetitions']))
    result_em_fits = np.nan*np.empty((6, all_parameters['num_repetitions']))  # kappa, mixt_target, mixt_nontarget, mixt_random, ll, bic
    result_dist_dualrecall_angle = np.nan*np.empty((4, all_parameters['num_repetitions']))  # kappa, mixt_target, mixt_nontarget, mixt_random
    result_dist_dualrecall_angle_emmixt_KL = np.nan*np.empty((all_parameters['num_repetitions']))
    result_dist_dualrecall_colour = np.nan*np.empty((4, all_parameters['num_repetitions']))  # kappa, mixt_target, mixt_nontarget, mixt_random
    result_dist_dualrecall_colour_emmixt_KL = np.nan*np.empty((all_parameters['num_repetitions']))

    # If desired, will automatically save all Model responses.
    if all_parameters['collect_responses']:
        print "--- Collecting all responses..."
        result_responses = np.nan*np.ones((all_parameters['N'], all_parameters['num_repetitions']))
        result_target = np.nan*np.ones((all_parameters['N'], all_parameters['num_repetitions']))
        result_nontargets = np.nan*np.ones((all_parameters['N'], all_parameters['T'] - 1, all_parameters['num_repetitions']))

    search_progress = progress.Progress(all_parameters['num_repetitions'])

    for repet_i in xrange(all_parameters['num_repetitions']):
        print "%.2f%%, %s left - %s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

        print "Fit for T=%d, %d/%d" % (all_parameters['T'], repet_i+1, all_parameters['num_repetitions'])

        ## Update parameter

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
        # curr_params_fit['mixt_nontargets_sum'] = np.sum(curr_params_fit['mixt_nontargets'])
        result_em_fits[:, repet_i] = [curr_params_fit[key] for key in ['kappa', 'mixt_target', 'mixt_nontargets_sum', 'mixt_random', 'train_LL', 'bic']]

        # Compute fisher info
        print "compute fisher info"
        result_fi_theo[repet_i] = sampler.estimate_fisher_info_theocov(use_theoretical_cov=False)
        result_fi_theocov[repet_i] = sampler.estimate_fisher_info_theocov(use_theoretical_cov=True)

        # Compute distances to datasets
        if all_parameters['T'] in dualrecall_T_space:
            # Angle trials
            result_dist_dualrecall_angle[:, repet_i] = (dualrecall_experimental_angle_emfits_mean[:, dualrecall_T_space == all_parameters['T']].flatten() - result_em_fits[:4, repet_i])**2.
            result_dist_dualrecall_angle_emmixt_KL[repet_i] = utils.KL_div(result_em_fits[1:4, repet_i], dualrecall_experimental_angle_emfits_mean[1:, dualrecall_T_space==all_parameters['T']].flatten())

            # Colour trials
            result_dist_dualrecall_colour[:, repet_i] = (dualrecall_experimental_colour_emfits_mean[:, dualrecall_T_space == all_parameters['T']].flatten() - result_em_fits[:4, repet_i])**2.
            result_dist_dualrecall_colour_emmixt_KL[repet_i] = utils.KL_div(result_em_fits[1:4, repet_i], dualrecall_experimental_colour_emfits_mean[1:, dualrecall_T_space==all_parameters['T']].flatten())

        # If needed, store responses
        if all_parameters['collect_responses']:
            (responses, target, nontarget) = sampler.collect_responses()
            result_responses[:, repet_i] = responses
            result_target[:, repet_i] = target
            result_nontargets[..., repet_i] = nontarget

            print "collected responses"


        print "CURRENT RESULTS:\n", result_all_precisions[repet_i], curr_params_fit, result_fi_theo[repet_i], result_fi_theocov[repet_i], np.sum(result_dist_dualrecall_angle[:, repet_i]), np.sum(result_dist_dualrecall_colour[:, repet_i]), "\n"
        ### /Work ###

        search_progress.increment()
        if run_counter % save_every == 0 or search_progress.done():
            dataio.save_variables_default(locals())
        run_counter += 1

    # Finished
    dataio.save_variables_default(locals())

    print "All finished"
    return locals()


