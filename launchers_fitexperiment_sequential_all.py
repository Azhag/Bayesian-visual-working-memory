#!/usr/bin/env python
# encoding: utf-8
"""
launchers_fitexperiment_allt.py


Created by Loic Matthey on 2015-10-09
Copyright (c) 2015 . All rights reserved.
"""

import numpy as np

import utils
from dataio import *
import progress
import collections

import load_experimental_data
from fitexperiment_sequential import FitExperimentSequentialAll
from fitexperiment_sequential import FitExperimentSequentialSubjectAll


def _fitexperiment_work_unit(fit_exp, all_parameters, outputs_data=None):
    """ Performs a piece of FitExperimentSequential work
    """
    if outputs_data is None:
        outputs_data = collections.defaultdict(list)

    # Setup and evaluate some statistics
    def compute_everything(self, parameters):
        results = dict()

        print ">> Computing LL all N..."
        results['result_ll_n'] = self.sampler.compute_loglikelihood_N().flatten()

        print ">> Computing LL sum..."
        results['result_ll_sum'] = np.nansum(results['result_ll_n'])
        print results['result_ll_sum']

        print ">> Computing BIC..."
        results['result_bic'] = self.sampler.compute_bic(
            K=parameters['bic_K'], LL=results['result_ll_sum'])

        print ">> Computing LL median..."
        results['result_ll_median'] = np.nanmedian(results['result_ll_n'])
        print results['result_ll_median']

        print ">> Computing LL90/92/95/97..."
        results['result_ll90_sum'] = (
            self.sampler.compute_loglikelihood_top90percent(
                all_loglikelihoods=results['result_ll_n']))
        results['result_ll92_sum'] = (
            self.sampler.compute_loglikelihood_top_p_percent(
                0.92, all_loglikelihoods=results['result_ll_n']))
        results['result_ll95_sum'] = (
            self.sampler.compute_loglikelihood_top_p_percent(
                0.95, all_loglikelihoods=results['result_ll_n']))
        results['result_ll97_sum'] = (
            self.sampler.compute_loglikelihood_top_p_percent(
                0.97, all_loglikelihoods=results['result_ll_n']))

        return results

    res_listdicts = fit_exp.apply_fct_datasets_all(
        dict(fct=compute_everything, parameters=all_parameters))

    # Put everything back together.
    # This actually flattens everything
    for key in res_listdicts[0]:
        outputs_data[key].append(
            np.array([res[key] for res in res_listdicts]).flatten())

    if not all_parameters['inference_method'] == 'none':
        # Fit mixture model
        print " fit mixture model..."
        model_fits = fit_exp.get_model_em_fits()

        emfits_distances = fit_exp.compute_dist_experimental_em_fits(model_fits)
        results = {
            'result_emfit_mse': emfits_distances['all_mse'],
            'result_emfit_mixt_kl': emfits_distances['mixt_kl']
        }
        for key, res in results.iteritems():
            outputs_data[key].append(np.nansum(res))


    return outputs_data


def launcher_do_fitexperiment_sequential_subjects_allmetrics(args):
    '''
        Given a single experiment_id, will run the model on all (T, trecall) in the experimental data, per subject (summed metric).

        Most likely you want gorgo11_seq.
        Computes several metrics (LL, BIC) and can additionally sample from the model and check the Mixture model
        summary statistics fits.

        If inference_method is not none, also fits a EM mixture model, get the precision and the fisher information
    '''

    print "Doing a piece of work for launcher_do_fitexperiment_sequential_subjects_allmetrics"


    all_parameters = utils.argparse_2_dict(args)
    print all_parameters

    if all_parameters['burn_samples'] + all_parameters['num_samples'] < 200:
        print "WARNING> you do not have enough samples I think!", all_parameters['burn_samples'] + all_parameters['num_samples']

    # Create DataIO
    #  (complete label with current variable state)
    dataio = DataIO(output_folder=all_parameters['output_directory'],
                    label=all_parameters['label'].format(**all_parameters),
                    calling_function='')
    save_every = 1
    run_counter = 0

    # Result arrays
    # sizes depend on the experiment.
    T_space = None
    all_outputs_data = collections.defaultdict(list)

    experimental_dataset = load_experimental_data.load_data(
        experiment_id=all_parameters['experiment_id'])
    subject_space = experimental_dataset['data_subject_split']['subjects_space']

    search_progress = progress.Progress(
        all_parameters['num_repetitions']*subject_space.size)

    for repet_i in xrange(all_parameters['num_repetitions']):
        curr_outputs_data = collections.defaultdict(list)
        for subject in subject_space:
            all_parameters['experiment_subject'] = subject

            ### WORK WORK WORK work? ###
            print "\n\nSubject %d, rep %d/%d | %.2f%%, %s left - %s" % (
                subject,
                repet_i+1,
                all_parameters['num_repetitions'],
                search_progress.percentage(),
                search_progress.time_remaining_str(),
                search_progress.eta_str())

            fit_exp = FitExperimentSequentialSubjectAll(all_parameters)
            _fitexperiment_work_unit(
                fit_exp, all_parameters, curr_outputs_data)

            print curr_outputs_data

            ### /Work ###

            T_space = fit_exp.T_space

            search_progress.increment()
            if run_counter % save_every == 0 or search_progress.done():
                data_to_save = locals()
                data_to_save.update(curr_outputs_data)
                dataio.save_variables_default(data_to_save)
            run_counter += 1

        # Summing across subjects, will still have num_repetitions to average over.
        for key in curr_outputs_data:
            all_outputs_data[key].append(
                np.nansum(curr_outputs_data[key]))

    # Convert results to arrays
    # Put the repetition axis at the last dimension, it's kinda my convention...
    for key in all_outputs_data:
        all_outputs_data[key] = np.array(all_outputs_data[key])
        all_outputs_data[key] = all_outputs_data[key].transpose(
            np.roll(np.arange(all_outputs_data[key].ndim), -1))

    ### /Work ###

    data_to_save = locals()
    data_to_save.update(all_outputs_data)
    dataio.save_variables_default(data_to_save)
    dataio.save_variables_default(locals())

    #### Plots ###

    print "All finished"
    return locals()


def launcher_do_fitexperiment_sequential_subject_allmetrics(args):
    '''
        Given a single experiment_id, will run the model on all (T, trecall) in the experimental data, for a fixed subject.

        Most likely you want gorgo11_seq.
        Computes several metrics (LL, BIC) and can additionally sample from the model and check the Mixture model
        summary statistics fits.

        If inference_method is not none, also fits a EM mixture model, get the precision and the fisher information
    '''

    print "Doing a piece of work for launcher_do_fitexperiment_sequential_subject_allmetrics"


    all_parameters = utils.argparse_2_dict(args)
    print all_parameters

    if all_parameters['burn_samples'] + all_parameters['num_samples'] < 200:
        print "WARNING> you do not have enough samples I think!", all_parameters['burn_samples'] + all_parameters['num_samples']

    # Create DataIO
    #  (complete label with current variable state)
    dataio = DataIO(output_folder=all_parameters['output_directory'],
                    label=all_parameters['label'].format(**all_parameters),
                    calling_function='')
    save_every = 1
    run_counter = 0

    # Result arrays
    # sizes depend on the experiment.
    T_space = None
    all_outputs_data = collections.defaultdict(list)

    experimental_dataset = load_experimental_data.load_data(
        experiment_id=all_parameters['experiment_id'])
    subject = all_parameters['experiment_subject']

    search_progress = progress.Progress(
        all_parameters['num_repetitions'])

    for repet_i in xrange(all_parameters['num_repetitions']):
        ### WORK WORK WORK work? ###
        print "\n\nSubject %d, rep %d/%d | %.2f%%, %s left - %s" % (
                subject,
                repet_i+1,
                all_parameters['num_repetitions'],
                search_progress.percentage(),
                search_progress.time_remaining_str(),
                search_progress.eta_str())

        fit_exp = FitExperimentSequentialSubjectAll(all_parameters)
        _fitexperiment_work_unit(
            fit_exp, all_parameters, all_outputs_data)
        print all_outputs_data
        ### /Work ###

        T_space = fit_exp.T_space

        search_progress.increment()
        if run_counter % save_every == 0 or search_progress.done():
            data_to_save = locals()
            data_to_save.update(all_outputs_data)
            dataio.save_variables_default(data_to_save)
        run_counter += 1


    # Convert results to arrays
    # Put the repetition axis at the last dimension, it's kinda my convention...
    for key in all_outputs_data:
        all_outputs_data[key] = np.array(all_outputs_data[key])
        all_outputs_data[key] = all_outputs_data[key].transpose(np.roll(np.arange(all_outputs_data[key].ndim), -1))

    ### /Work ###

    data_to_save = locals()
    data_to_save.update(all_outputs_data)
    dataio.save_variables_default(data_to_save)
    dataio.save_variables_default(locals())

    #### Plots ###

    print "All finished"
    return locals()


def launcher_do_fitexperiment_sequential_allmetrics(args):
    '''
        Given a single experiment_id, will run the model on all (T, trecall) in the experimental data (collapse across subjects).

        Most likely you want gorgo11_seq.
        Computes several metrics (LL, BIC) and can additionally sample from the model and check the Mixture model
        summary statistics fits.

        If inference_method is not none, also fits a EM mixture model, get the precision and the fisher information
    '''

    print "Doing a piece of work for launcher_do_fitexperiment_sequential_allmetrics"


    all_parameters = utils.argparse_2_dict(args)
    print all_parameters

    if all_parameters['burn_samples'] + all_parameters['num_samples'] < 200:
        print "WARNING> you do not have enough samples I think!", all_parameters['burn_samples'] + all_parameters['num_samples']

    # Create DataIO
    #  (complete label with current variable state)
    dataio = DataIO(output_folder=all_parameters['output_directory'],
                    label=all_parameters['label'].format(**all_parameters),
                    calling_function='')
    save_every = 1
    run_counter = 0

    # Result arrays
    # sizes depend on the experiment.
    T_space = None
    all_outputs_data = collections.defaultdict(list)

    experimental_dataset = load_experimental_data.load_data(
        experiment_id=all_parameters['experiment_id'])

    search_progress = progress.Progress(
        all_parameters['num_repetitions'])

    for repet_i in xrange(all_parameters['num_repetitions']):
        ### WORK WORK WORK work? ###
        print "\n\n\n\nrep %d/%d | %.2f%%, %s left - %s" % (
                repet_i+1,
                all_parameters['num_repetitions'],
                search_progress.percentage(),
                search_progress.time_remaining_str(),
                search_progress.eta_str())

        fit_exp = FitExperimentSequentialAll(all_parameters)
        _fitexperiment_work_unit(
            fit_exp, all_parameters, all_outputs_data)
        print all_outputs_data
        ### /Work ###

        T_space = fit_exp.T_space

        search_progress.increment()
        if run_counter % save_every == 0 or search_progress.done():
            data_to_save = locals()
            data_to_save.update(all_outputs_data)
            dataio.save_variables_default(data_to_save)
        run_counter += 1


    # Convert results to arrays
    # Put the repetition axis at the last dimension, it's kinda my convention...
    for key in all_outputs_data:
        all_outputs_data[key] = np.array(all_outputs_data[key])
        all_outputs_data[key] = all_outputs_data[key].transpose(np.roll(np.arange(all_outputs_data[key].ndim), -1))

    ### /Work ###

    data_to_save = locals()
    data_to_save.update(all_outputs_data)
    dataio.save_variables_default(data_to_save)
    dataio.save_variables_default(locals())

    #### Plots ###

    print "All finished"
    return locals()
