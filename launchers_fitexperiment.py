#!/usr/bin/env python
# encoding: utf-8
"""
launchers_fitexperiment.py


Created by Loic Matthey on 2013-09-30
Copyright (c) 2013 . All rights reserved.
"""

import matplotlib.pyplot as plt
import numpy as np

import launchers

from utils import *
from dataio import *
from fitexperiment import *
import progress



def launcher_do_fitexperiment(args):
    '''
        Perform a simple estimation of the loglikelihood of the data, under a model with provided parameters

        If inference_method is not none, also fits a EM mixture model, get the precision and the fisher information
    '''

    print "Doing a piece of work for launcher_do_fitexperiment"


    all_parameters = argparse_2_dict(args)
    print all_parameters

    if all_parameters['burn_samples'] + all_parameters['num_samples'] < 200:
        print "WARNING> you do not have enough samples I think!", all_parameters['burn_samples'] + all_parameters['num_samples']

    # Force some parameters
    all_parameters.setdefault('experiment_ids', ['gorgo11', 'bays09', 'dualrecall'])
    if 'fitexperiment_parameters' not in all_parameters:
        fitexperiment_parameters = dict(experiment_ids=all_parameters['experiment_ids'], fit_mixture_model=True)

    print "\n T={:d}, experiment_ids {}\n".format(all_parameters['T'], all_parameters['experiment_ids'])

    # Create DataIO
    #  (complete label with current variable state)
    dataio = DataIO(output_folder=all_parameters['output_directory'], label=all_parameters['label'].format(**all_parameters))
    save_every = 1
    run_counter = 0

    # Result arrays
    result_fitexperiments = np.nan*np.empty((3, all_parameters['num_repetitions']))  # BIC total, LL, LL90
    result_fitexperiments_all = np.nan*np.empty((3, len(all_parameters['experiment_ids']), all_parameters['num_repetitions']))  # BIC, LL, LL90; per experiments,
    if all_parameters['inference_method'] != 'none':
        result_all_precisions = np.nan*np.empty((all_parameters['num_repetitions']))
        result_em_fits = np.nan*np.empty((6, all_parameters['num_repetitions']))   # kappa, mixt_target, mixt_nontarget, mixt_random, ll, bic
        result_fi_theo = np.nan*np.empty((all_parameters['num_repetitions']))
        result_fi_theocov = np.nan*np.empty((all_parameters['num_repetitions']))

    if all_parameters['sigma_output'] > 0.0:
        # We asked for the additional noise convolved, need to take it into account.
        result_fitexperiments_noiseconv = np.nan*np.empty((3, all_parameters['num_repetitions']))  # bic (K+1), LL conv, LL90 conv
        result_fitexperiments_noiseconv_all = np.nan*np.empty((3, len(all_parameters['experiment_ids']), all_parameters['num_repetitions']))  # bic, LL conv, LL90 conv


    search_progress = progress.Progress(all_parameters['num_repetitions'])
    for repet_i in xrange(all_parameters['num_repetitions']):

        print "%d/%d | %.2f%%, %s left - %s" % (repet_i+1, all_parameters['num_repetitions'], search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

        ### WORK WORK WORK work? ###
        # Instantiate
        (_, _, _, sampler) = launchers.init_everything(all_parameters)

        ### Do the actual FitExperiment computations
        fit_exp = FitExperiment(sampler, fitexperiment_parameters)

        ## Compute and store the BIC and LL
        if all_parameters['code_type'] == 'mixed':
            K_nb_params = 3
        else:
            K_nb_params = 2

        bic_loglik_dict = fit_exp.compute_bic_loglik_all_datasets(K=K_nb_params)

        for exper_i, exper in enumerate(all_parameters['experiment_ids']):
            try:
                result_fitexperiments_all[0, exper_i, repet_i] = bic_loglik_dict[exper]['bic']
                result_fitexperiments_all[1, exper_i, repet_i] = bic_loglik_dict[exper]['LL']
                result_fitexperiments_all[2, exper_i, repet_i] = bic_loglik_dict[exper]['LL90']
            except TypeError:
                pass

        result_fitexperiments[:, repet_i] = np.nansum(result_fitexperiments_all[..., repet_i], axis=1)

        if all_parameters['sigma_output'] > 0.0:
            # Compute the loglikelihoods with the convolved posterior. Slowish.

            ## Compute and store the BIC and LL
            bic_loglik_noise_convolved_dict = fit_exp.compute_bic_loglik_noise_convolved_all_datasets(precision=150)

            for exper_i, exper in enumerate(all_parameters['experiment_ids']):
                try:
                    result_fitexperiments_noiseconv_all[0, exper_i, repet_i] = bic_loglik_noise_convolved_dict[exper]['bic']
                    result_fitexperiments_noiseconv_all[1, exper_i, repet_i] = bic_loglik_noise_convolved_dict[exper]['LL']
                    result_fitexperiments_noiseconv_all[2, exper_i, repet_i] = bic_loglik_noise_convolved_dict[exper]['LL90']
                except TypeError:
                    pass

            result_fitexperiments_noiseconv[:, repet_i] = np.nansum(result_fitexperiments_noiseconv_all[:, :, repet_i], axis=1)

        # If sampling_method is not none, try to get em_fits and others. EXTRA SLOW.
        if not all_parameters['inference_method'] == 'none':
            parameters = dict([[key, eval(key)] for key in ['all_parameters', 'repet_i', 'result_all_precisions', 'result_em_fits', 'result_fi_theo', 'result_fi_theocov']])

            def additional_computations(sampler, parameters):
                for key, val in parameters.iteritems():
                    locals()[key] = val

                # Sample
                print "sampling..."
                sampler.run_inference(all_parameters)

                # Compute precision
                print "get precision..."
                result_all_precisions[repet_i] = sampler.get_precision()

                # Fit mixture model
                print "fit mixture model..."
                curr_params_fit = sampler.fit_mixture_model(use_all_targets=True)
                result_em_fits[:, repet_i] = [curr_params_fit[key] for key in ['kappa', 'mixt_target', 'mixt_nontargets_sum', 'mixt_random', 'train_LL', 'bic']]

                # Compute fisher info
                print "compute fisher info"
                result_fi_theo[repet_i] = sampler.estimate_fisher_info_theocov(use_theoretical_cov=False)
                result_fi_theocov[repet_i] = sampler.estimate_fisher_info_theocov(use_theoretical_cov=True)

            # Apply that on each dataset!
            fct_infos = dict(fct=additional_computations, parameters=parameters)
            fit_exp.apply_fct_all_datasets(fct_infos)


        print "CURRENT RESULTS:"
        if all_parameters['inference_method'] != 'none':
            print result_all_precisions[repet_i], result_em_fits[:, repet_i], result_fi_theo[repet_i], result_fi_theocov[repet_i]
        print "Fits LL no noise:", bic_loglik_dict

        if all_parameters['sigma_output'] > 0.0:
            print "Fits LL output noise %.2f: %s" %  (all_parameters['sigma_output'], bic_loglik_noise_convolved_dict)

        ### /Work ###

        search_progress.increment()
        if run_counter % save_every == 0 or search_progress.done():
            dataio.save_variables_default(locals())
        run_counter += 1

    ### /Work ###

    additional_variables = ['fitexperiment_parameters']
    dataio.save_variables_default(locals(), additional_variables)

    #### Plots ###

    print "All finished"

    return locals()


def launcher_do_fitexperiment_allT(args):
    '''
        Perform a simple estimation of the loglikelihood of the data, under a model with provided parameters.

        Will run for all T = 1..all_parameters['T']

        If inference_method is not none, also fits a EM mixture model, get the precision and the fisher information
    '''

    print "Doing a piece of work for launcher_do_fitexperiment_allT"


    all_parameters = argparse_2_dict(args)
    print all_parameters

    if all_parameters['burn_samples'] + all_parameters['num_samples'] < 200:
        print "WARNING> you do not have enough samples I think!", all_parameters['burn_samples'] + all_parameters['num_samples']

    # Force some parameters
    all_parameters.setdefault('experiment_ids', ['gorgo11', 'bays09', 'dualrecall'])
    if 'fitexperiment_parameters' not in all_parameters:
        fitexperiment_parameters = dict(experiment_ids=all_parameters['experiment_ids'], fit_mixture_model=True)

    print "\n T_max={:d}, experiment_ids {}\n".format(all_parameters['T'], all_parameters['experiment_ids'])

    # Create DataIO
    #  (complete label with current variable state)
    dataio = DataIO(output_folder=all_parameters['output_directory'], label=all_parameters['label'].format(**all_parameters))
    save_every = 1
    run_counter = 0

    # Parameter arrays
    T_max = all_parameters['T']
    T_min = all_parameters.get('T_min', 1)
    T_space = np.arange(T_min, T_max+1)

    ## Result arrays
    result_fitexperiments = np.nan*np.empty((T_space.size, 3, all_parameters['num_repetitions']))  # BIC total, LL, LL90
    result_fitexperiments_all = np.nan*np.empty((T_space.size, 3, len(all_parameters['experiment_ids']), all_parameters['num_repetitions']))  # BIC, LL, LL90; per experiments

    if all_parameters['inference_method'] != 'none':
        result_all_precisions = np.nan*np.empty((T_space.size, all_parameters['num_repetitions']))
        result_em_fits = np.nan*np.empty((T_space.size, 6, all_parameters['num_repetitions']))   # kappa, mixt_target, mixt_nontarget, mixt_random, ll, bic
        result_fi_theo = np.nan*np.empty((T_space.size, all_parameters['num_repetitions']))
        result_fi_theocov = np.nan*np.empty((T_space.size, all_parameters['num_repetitions']))

    if all_parameters['sigma_output'] > 0.0:
        # We asked for the additional noise convolved, need to take it into account.
        result_fitexperiments_noiseconv = np.nan*np.empty((T_space.size, 3, all_parameters['num_repetitions']))  # bic (K+1), LL conv, LL90 conv
        result_fitexperiments_noiseconv_all = np.nan*np.empty((T_space.size, 3, len(all_parameters['experiment_ids']), all_parameters['num_repetitions']))  # bic, LL conv, LL90 conv

    search_progress = progress.Progress(T_space.size*all_parameters['num_repetitions'])
    for repet_i in xrange(all_parameters['num_repetitions']):
        for T_i, T in enumerate(T_space):
            print "\nT=%d, %d/%d | %.2f%%, %s left - %s" % (T, repet_i+1, all_parameters['num_repetitions'], search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

            # Update parameter
            all_parameters['T'] = T

            ### WORK WORK WORK work? ###
            # Instantiate
            (_, _, _, sampler) = launchers.init_everything(all_parameters)

            ### Do the actual FitExperiment computations
            fit_exp = FitExperiment(sampler, fitexperiment_parameters)

            ## Compute and store the BIC and LL
            print ">> Computing BIC and LL..."
            ## Compute and store the BIC and LL
            if all_parameters['code_type'] == 'mixed':
                K_nb_params = 3
            else:
                K_nb_params = 2
            bic_loglik_dict = fit_exp.compute_bic_loglik_all_datasets(K=K_nb_params)

            for exper_i, exper in enumerate(all_parameters['experiment_ids']):
                try:
                    result_fitexperiments_all[T_i, 0, exper_i, repet_i] = bic_loglik_dict[exper]['bic']
                    result_fitexperiments_all[T_i, 1, exper_i, repet_i] = bic_loglik_dict[exper]['LL']
                    result_fitexperiments_all[T_i, 2, exper_i, repet_i] = bic_loglik_dict[exper]['LL90']
                except TypeError:
                    pass

            result_fitexperiments[T_i, :, repet_i] = np.nansum(result_fitexperiments_all[T_i, ..., repet_i], axis=1)

            if all_parameters['sigma_output'] > 0.0:
                # Compute the loglikelihoods with the convolved posterior. Slowish.
                print ">> Computing BIC and LL for convolved posterior..."

                ## Compute and store the BIC and LL
                bic_loglik_noise_convolved_dict = fit_exp.compute_bic_loglik_noise_convolved_all_datasets(precision=150)

                for exper_i, exper in enumerate(all_parameters['experiment_ids']):
                    try:
                        result_fitexperiments_noiseconv_all[T_i, 0, exper_i, repet_i] = bic_loglik_noise_convolved_dict[exper]['bic']
                        result_fitexperiments_noiseconv_all[T_i, 1, exper_i, repet_i] = bic_loglik_noise_convolved_dict[exper]['LL']
                        result_fitexperiments_noiseconv_all[T_i, 2, exper_i, repet_i] = bic_loglik_noise_convolved_dict[exper]['LL90']
                    except TypeError:
                        pass

                result_fitexperiments_noiseconv[T_i, :, repet_i] = np.nansum(result_fitexperiments_noiseconv_all[T_i, :, :, repet_i], axis=1)

            # If sampling_method is not none, try to get em_fits and others
            if not all_parameters['inference_method'] == 'none':
                parameters = dict([[key, eval(key)] for key in ['all_parameters', 'repet_i', 'result_all_precisions', 'result_em_fits', 'result_fi_theo', 'result_fi_theocov', 'T_i']])
                # if all_parameters['collect_responses']:
                #     parameters.update(dict(result_responses=result_responses, result_target=result_target, result_nontargets=result_nontargets))

                print ">> Sampling and fitting mixt model / precision / FI ..."
                def additional_computations(sampler, parameters):
                    for key, val in parameters.iteritems():
                        locals()[key] = val

                    # Sample
                    print "sampling..."
                    sampler.run_inference(all_parameters)

                    # Compute precision
                    print "get precision..."
                    result_all_precisions[T_i, repet_i] = sampler.get_precision()

                    # Fit mixture model
                    print "fit mixture model..."
                    curr_params_fit = sampler.fit_mixture_model(use_all_targets=True)
                    result_em_fits[T_i, :, repet_i] = [curr_params_fit[key] for key in ['kappa', 'mixt_target', 'mixt_nontargets_sum', 'mixt_random', 'train_LL', 'bic']]

                    # Compute fisher info
                    print "compute fisher info"
                    result_fi_theo[T_i, repet_i] = sampler.estimate_fisher_info_theocov(use_theoretical_cov=False)
                    result_fi_theocov[T_i, repet_i] = sampler.estimate_fisher_info_theocov(use_theoretical_cov=True)

                # Apply that on each dataset!
                fct_infos = dict(fct=additional_computations, parameters=parameters)
                fit_exp.apply_fct_all_datasets(fct_infos)


            print "CURRENT RESULTS:"
            if all_parameters['inference_method'] != 'none':
                print result_all_precisions[T_i, repet_i], result_em_fits[T_i, :, repet_i], result_fi_theo[T_i, repet_i], result_fi_theocov[T_i, repet_i]

            print "Fits LL no noise:", bic_loglik_dict

            if all_parameters['sigma_output'] > 0.0:
                print "Fits LL output noise %.2f: %s" %  (all_parameters['sigma_output'], bic_loglik_noise_convolved_dict)

            ### /Work ###

            search_progress.increment()
            if run_counter % save_every == 0 or search_progress.done():
                dataio.save_variables_default(locals())
            run_counter += 1

    ### /Work ###

    additional_variables = ['fitexperiment_parameters']
    dataio.save_variables_default(locals(), additional_variables)

    #### Plots ###

    print "All finished"

    return locals()


def launcher_do_fitexperiment_mixed_tworcscale(args):
    '''
        Estimate of the loglikelihood of the data, under a model with provided parameters
        Vary rcscale and rcscale2, to use with a mixed network

    '''

    print "Doing a piece of work for launcher_do_fitexperiment_mixed_tworcscale"

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

    # Initialize FitExperiment
    fit_exp = FitExperiment(all_parameters)

    # Parameters to vary
    param1_space = np.linspace(0.01, 10, 15)    # kappa conj
    param2_space = np.linspace(0.01, 30., 17)   # kappa feat

    # Result arrays
    result_log_posterior_mean = np.nan*np.ones((param1_space.size, param2_space.size))
    result_log_posterior_std = np.nan*np.ones((param1_space.size, param2_space.size))

    search_progress = progress.Progress(param1_space.size*param2_space.size)

    for i, rc_scale in enumerate(param1_space):
        for j, rc_scale2 in enumerate(param2_space):
            print "%.2f%%, %s left - %s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

            print "Fit for rc_scale=%.2f, rc_scale2=%.2f" % (rc_scale, rc_scale2)

            # Update parameter
            fit_exp.parameters['rc_scale'] = rc_scale
            fit_exp.parameters['rc_scale2'] = rc_scale2

            ### WORK WORK WORK work? ###

            # Compute the loglikelihood
            result_log_posterior_mean[i, j], result_log_posterior_std[i, j] = fit_exp.estimate_likelihood_multiple_models(num_models=all_parameters['num_repetitions'])

            ### /Work ###

            search_progress.increment()
            if run_counter % save_every == 0 or search_progress.done():
                dataio.save_variables_default(locals())
            run_counter += 1

    # Finished
    dataio.save_variables_default(locals())

    # Plot the result
    # plot_mean_std_area(M_space, self.llh_fullspace_mean, self.llh_fullspace_std)
    # dataio.save_variables(variables_to_save, locals())
    pcolor_2d_data(result_log_posterior_mean, param1_space, param2_space, "rc_scale", "rc_scale2")
    dataio.save_current_figure("fitexperiment_mixed_tworcscale_ratio{ratio_conj}_sigmax{sigmax}".format(**all_parameters) + "_{unique_id}.pdf")

    print "All finished"
    return locals()


