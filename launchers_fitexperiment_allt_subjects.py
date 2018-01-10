#!/usr/bin/env python
# encoding: utf-8
"""
launchers_fitexperiment_allt_subjects.py


Created by Loic Matthey on 2016-08-16
Copyright (c) 2016 . All rights reserved.
"""

import numpy as np

import utils
from dataio import *
from fitexperiment_allt_subjects import *
import progress


def launcher_do_fitexperiment_subject_allmetrics(args):
    '''
        Given a single experiment_id and a experiment_subject, will run the model on all T in the experimental data, for a particular subset of the data corresponding to that subject.

        Computes several metrics (LL, BIC) and can additionally sample from the model and check the Mixture model
        summary statistics fits.

        If inference_method is not none, also fits a EM mixture model, get the precision and the fisher information
    '''

    print "Doing a piece of work for launcher_do_fitexperiment_subject_allmetrics"

    all_parameters = utils.argparse_2_dict(args)
    print all_parameters

    if all_parameters['burn_samples'] + all_parameters['num_samples'] < 200:
        print "WARNING> you do not have enough samples I think!", all_parameters['burn_samples'] + all_parameters['num_samples']

    # Create DataIO
    #  (complete label with current variable state)
    dataio = DataIO(
        output_folder=all_parameters['output_directory'],
        label=all_parameters['label'].format(**all_parameters))
    save_every = 1
    run_counter = 0

    # Result arrays
    # sizes depend on the experiment.
    all_outputs_data = dict()
    T_space = None

    search_progress = progress.Progress(all_parameters['num_repetitions'])
    for repet_i in xrange(all_parameters['num_repetitions']):
        print "\n\n%d/%d | %.2f%%, %s left - %s" % (
            repet_i + 1, all_parameters['num_repetitions'],
            search_progress.percentage(), search_progress.time_remaining_str(),
            search_progress.eta_str())

        ### WORK WORK WORK work? ###

        # Let's build a FitExperimentAllT
        fit_exp = FitExperimentAllTSubject(all_parameters)

        # Setup and evaluate some statistics
        def compute_everything(self, parameters):
            results = dict()

            print ">> Computing LL all N..."
            results['result_ll_n'] = self.sampler.compute_loglikelihood_N()

            print ">> Computing LL sum..."
            results['result_ll_sum'] = np.nansum(results['result_ll_n'])
            print results['result_ll_sum']

            print ">> Computing LL median..."
            results['result_ll_median'] = np.nanmedian(results['result_ll_n'])
            print results['result_ll_median']

            print ">> Computing BIC..."
            results['result_bic'] = self.sampler.compute_bic(
                K=parameters['bic_K'], LL=results['result_ll_sum'])

            print ">> Computing LL90/95/97..."
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

            # If sampling_method is not none, try to get em_fits and others
            if not parameters['inference_method'] == 'none':
                print ">> Sampling and fitting mixt model / precision / FI ..."

                # Sample
                print " sampling..."
                self.sampler.run_inference(parameters)
                self.store_responses('samples')

                # Compute precision
                print " get precision..."
                results['result_precision'] = self.sampler.get_precision()

                # Fit mixture model
                print " fit mixture model..."
                curr_params_fit = self.sampler.fit_mixture_model(
                    use_all_targets=False)
                results['result_em_fits'] = np.array([
                    curr_params_fit[key]
                    for key in [
                        'kappa', 'mixt_target', 'mixt_nontargets_sum',
                        'mixt_random', 'train_LL', 'bic'
                    ]
                ])

                # Compute distances to data mixture model
                emfits_distances = (
                    self.compute_dist_experimental_em_fits_currentT(
                        results['result_em_fits']))
                results['result_emfit_mse'] = emfits_distances['all_mse']
                results['result_emfit_mixt_kl'] = emfits_distances['mixt_kl']

                # Compute fisher info
                print " compute fisher info"
                results['result_fi_theo'] = (
                    self.sampler.estimate_fisher_info_theocov(
                        use_theoretical_cov=False))
                results['result_fi_theocov'] = (
                    self.sampler.estimate_fisher_info_theocov(
                        use_theoretical_cov=True))

            return results

        res_listdicts = fit_exp.apply_fct_datasets_allT(
            dict(fct=compute_everything, parameters=all_parameters))

        # Put everything back together (yeah advanced python muck)
        for key in res_listdicts[0]:
            all_outputs_data.setdefault(key, []).append(
                np.array([res[key] for res in res_listdicts]))

        # print "CURRENT RESULTS:"
        # print res_listdicts

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
    for key in res_listdicts[0]:
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
