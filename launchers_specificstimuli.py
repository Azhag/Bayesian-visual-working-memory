#!/usr/bin/env python
# encoding: utf-8
"""
launchers_specificstimuli.py


Created by Loic Matthey on 2013-10-18
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
import em_circularmixture

import launchers

def launcher_do_mixed_special_stimuli(args):
    '''
        Fit mixed model, varying the ratio_conj
        See how the precision of recall and mixture model parameters evolve
    '''

    print "Doing a piece of work for launcher_do_mixed_special_stimuli"

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

    # Parameters to vary
    ratio_space = (np.arange(0, all_parameters['M']**0.5)**2.)/all_parameters['M']

    # Result arrays
    result_all_precisions = np.nan*np.ones((ratio_space.size, all_parameters['num_repetitions']))
    result_em_fits = np.nan*np.ones((ratio_space.size, 5, all_parameters['num_repetitions']))  # kappa, mixt_target, mixt_nontarget, mixt_random, ll
    result_em_resp = np.nan*np.ones((ratio_space.size, 1+all_parameters['T'], all_parameters['N'], all_parameters['num_repetitions']))

    # If desired, will automatically save all Model responses.
    if all_parameters['subaction'] == 'collect_responses':
        result_responses = np.nan*np.ones((ratio_space.size, all_parameters['N'], all_parameters['num_repetitions']))
        result_target = np.nan*np.ones((ratio_space.size, all_parameters['N'], all_parameters['num_repetitions']))
        result_nontargets = np.nan*np.ones((ratio_space.size, all_parameters['N'], all_parameters['T']-1, all_parameters['num_repetitions']))


    search_progress = progress.Progress(ratio_space.size*all_parameters['num_repetitions'])

    for repet_i in xrange(all_parameters['num_repetitions']):
        for ratio_i, ratio_conj in enumerate(ratio_space):
            print "%.2f%%, %s left - %s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

            print "Fit for ratio_conj=%.2f, %d/%d" % (ratio_conj, repet_i+1, all_parameters['num_repetitions'])

            # Update parameter
            all_parameters['ratio_conj'] = ratio_conj

            ### WORK WORK WORK work? ###

            # Generate specific stimuli
            all_parameters['stimuli_generation'] = 'specific_stimuli'

            # Instantiate
            (random_network, data_gen, stat_meas, sampler) = launchers.init_everything(all_parameters)

            # Sample
            sampler.run_inference(all_parameters)

            # Compute precision
            result_all_precisions[ratio_i, repet_i] = sampler.get_precision()

            # Fit mixture model
            curr_params_fit = em_circularmixture.fit(*sampler.collect_responses())
            curr_resp = em_circularmixture.compute_responsibilities(*(sampler.collect_responses() + (curr_params_fit,) ))

            result_em_fits[ratio_i, :, repet_i] = [curr_params_fit[key] for key in ('kappa', 'mixt_target', 'mixt_nontargets', 'mixt_random', 'train_LL')]
            result_em_resp[ratio_i, 0, :, repet_i] = curr_resp['target']
            result_em_resp[ratio_i, 1:-1, :, repet_i] = curr_resp['nontargets'].T
            result_em_resp[ratio_i, -1, :, repet_i] = curr_resp['random']

            print result_all_precisions[ratio_i, repet_i], curr_params_fit

            # If needed, store responses
            if all_parameters['subaction'] == 'collect_responses':
                (responses, target, nontarget) = sampler.collect_responses()
                result_responses[ratio_i, :, repet_i] = responses
                result_target[ratio_i, :, repet_i] = target
                result_nontargets[ratio_i, ..., repet_i] = nontarget

                print "collected responses"

            ### /Work ###

            search_progress.increment()
            if run_counter % save_every == 0 or search_progress.done():
                dataio.save_variables_default(locals())
            run_counter += 1

    # Finished
    dataio.save_variables_default(locals())

    print "All finished"
    return locals()


def launcher_do_mixed_special_stimuli_fixratio(args):
    '''
        Fit mixed model, varying the ratio_conj
        See how the precision of recall and mixture model parameters evolve
    '''

    print "Doing a piece of work for launcher_do_mixed_special_stimuli"

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
    result_all_precisions = np.nan*np.ones(all_parameters['num_repetitions'])
    result_em_fits = np.nan*np.ones((5, all_parameters['num_repetitions']))  # kappa, mixt_target, mixt_nontarget, mixt_random, ll
    result_em_resp = np.nan*np.ones((all_parameters['num_repetitions'], 1+all_parameters['T'], all_parameters['N']))
    # If desired, will automatically save all Model responses.
    if all_parameters['subaction'] == 'collect_responses':
        result_responses = np.nan*np.ones((all_parameters['N'], all_parameters['num_repetitions']))
        result_target = np.nan*np.ones((all_parameters['N'], all_parameters['num_repetitions']))
        result_nontargets = np.nan*np.ones((all_parameters['N'], all_parameters['T']-1, all_parameters['num_repetitions']))


    search_progress = progress.Progress(all_parameters['num_repetitions'])

    for repet_i in xrange(all_parameters['num_repetitions']):
        print "%.2f%%, %s left - %s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

        print "Fit for ratio_conj=%.2f, %d/%d" % (all_parameters['ratio_conj'], repet_i+1, all_parameters['num_repetitions'])

        ### WORK WORK WORK work? ###

        # Generate specific stimuli
        all_parameters['stimuli_generation'] = 'specific_stimuli'

        # Instantiate
        (random_network, data_gen, stat_meas, sampler) = launchers.init_everything(all_parameters)

        # Sample
        sampler.run_inference(all_parameters)

        # Compute precision
        result_all_precisions[repet_i] = sampler.get_precision()

        # Fit mixture model
        curr_params_fit = em_circularmixture.fit(*sampler.collect_responses())
        curr_resp = em_circularmixture.compute_responsibilities(*(sampler.collect_responses() + (curr_params_fit,) ))

        result_em_fits[:, repet_i] = [curr_params_fit[key] for key in ('kappa', 'mixt_target', 'mixt_nontargets', 'mixt_random', 'train_LL')]
        result_em_resp[repet_i, 0] = curr_resp['target']
        result_em_resp[repet_i, 1:-1] = curr_resp['nontargets'].T
        result_em_resp[repet_i, -1] = curr_resp['random']

        # If needed, store responses
        if all_parameters['subaction'] == 'collect_responses':
            (responses, target, nontarget) = sampler.collect_responses()
            result_responses[:, repet_i] = responses
            result_target[:, repet_i] = target
            result_nontargets[..., repet_i] = nontarget

            print "collected responses"


        ### /Work ###

        search_progress.increment()
        if run_counter % save_every == 0 or search_progress.done():
            dataio.save_variables_default(locals())
        run_counter += 1

    # Finished
    dataio.save_variables_default(locals())

    print "All finished"
    return locals()




##############################################################################
################### HIERARCHICAL CODE
##############################################################################


def launcher_do_hierarchical_special_stimuli_varyMMlower(args):
    '''
        Fit Hierarchical model, varying the ratio of M to Mlower
        See how the precision of recall and mixture model parameters evolve
    '''

    print "Doing a piece of work for launcher_do_mixed_special_stimuli"

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

    # Parameters to vary
    M_space = np.arange(1, all_parameters['M']+1)
    M_lower_space = np.arange(2, all_parameters['M']+1, 2)
    MMlower_all = np.array(cross(M_space, M_lower_space))
    MMlower_valid_space = MMlower_all[np.nonzero(np.sum(MMlower_all, axis=1) == all_parameters['M'])[0]]

    # limit space, not too big...
    MMlower_valid_space = MMlower_valid_space[::5]
    print "MMlower size", MMlower_valid_space.shape[0]

    # Result arrays
    result_all_precisions = np.nan*np.ones((MMlower_valid_space.shape[0], all_parameters['num_repetitions']))
    result_em_fits = np.nan*np.ones((MMlower_valid_space.shape[0], 5, all_parameters['num_repetitions']))  # kappa, mixt_target, mixt_nontarget, mixt_random, ll
    result_em_resp = np.nan*np.ones((MMlower_valid_space.shape[0], 1+all_parameters['T'], all_parameters['N'], all_parameters['num_repetitions']))

    # If desired, will automatically save all Model responses.
    if all_parameters['subaction'] == 'collect_responses':
        result_responses = np.nan*np.ones((MMlower_valid_space.shape[0], all_parameters['N'], all_parameters['num_repetitions']))
        result_target = np.nan*np.ones((MMlower_valid_space.shape[0], all_parameters['N'], all_parameters['num_repetitions']))
        result_nontargets = np.nan*np.ones((MMlower_valid_space.shape[0], all_parameters['N'], all_parameters['T']-1, all_parameters['num_repetitions']))


    search_progress = progress.Progress(MMlower_valid_space.shape[0]*all_parameters['num_repetitions'])

    for repet_i in xrange(all_parameters['num_repetitions']):
        for MMlower_i, MMlower in enumerate(MMlower_valid_space):
            print "%.2f%%, %s left - %s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

            print "Fit for M=%d, Mlower=%d, %d/%d" % (MMlower[0], MMlower[1], repet_i+1, all_parameters['num_repetitions'])

            # Update parameter
            all_parameters['M']             = MMlower[0]
            all_parameters['M_layer_one']   = MMlower[1]

            ### WORK WORK WORK work? ###

            # Generate specific stimuli
            all_parameters['stimuli_generation'] = 'specific_stimuli'
            all_parameters['code_type'] = 'hierarchical'

            # Instantiate
            (random_network, data_gen, stat_meas, sampler) = launchers.init_everything(all_parameters)

            # Sample
            sampler.run_inference(all_parameters)

            # Compute precision
            result_all_precisions[MMlower_i, repet_i] = sampler.get_precision()

            # Fit mixture model
            curr_params_fit = em_circularmixture.fit(*sampler.collect_responses())
            curr_resp = em_circularmixture.compute_responsibilities(*(sampler.collect_responses() + (curr_params_fit,) ))

            print curr_params_fit

            result_em_fits[MMlower_i, :, repet_i] = [curr_params_fit[key] for key in ('kappa', 'mixt_target', 'mixt_nontargets', 'mixt_random', 'train_LL')]
            result_em_resp[MMlower_i, 0, :, repet_i] = curr_resp['target']
            result_em_resp[MMlower_i, 1:-1, :, repet_i] = curr_resp['nontargets'].T
            result_em_resp[MMlower_i, -1, :, repet_i] = curr_resp['random']

            # If needed, store responses
            if all_parameters['subaction'] == 'collect_responses':
                (responses, target, nontarget) = sampler.collect_responses()
                result_responses[MMlower_i, :, repet_i] = responses
                result_target[MMlower_i, :, repet_i] = target
                result_nontargets[MMlower_i, ..., repet_i] = nontarget

                print "collected responses"

            ### /Work ###

            search_progress.increment()
            if run_counter % save_every == 0 or search_progress.done():
                dataio.save_variables_default(locals())
            run_counter += 1

    # Finished
    dataio.save_variables_default(locals())

    print "All finished"
    return locals()

