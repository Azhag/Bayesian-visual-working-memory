#!/usr/bin/env python
# encoding: utf-8
"""
launchers_fitexperiment_allt.py


Created by Loic Matthey on 2015-10-09
Copyright (c) 2015 . All rights reserved.
"""

import matplotlib.pyplot as plt
import numpy as np

import launchers

from utils import *
from dataio import *
from fitexperiment_allt import *
import progress



def launcher_do_fitexperiment_allmetrics(args):
    '''
        Given a single experiment_id, will run the model on all T in the experimental data.
        Computes several metrics (LL, BIC) and can additionally sample from the model and check the Mixture model
        summary statistics fits.

        If inference_method is not none, also fits a EM mixture model, get the precision and the fisher information
    '''

    print "Doing a piece of work for launcher_do_fitexperimentsinglet"


    all_parameters = argparse_2_dict(args)
    print all_parameters

    if all_parameters['burn_samples'] + all_parameters['num_samples'] < 200:
        print "WARNING> you do not have enough samples I think!", all_parameters['burn_samples'] + all_parameters['num_samples']

    # Create DataIO
    #  (complete label with current variable state)
    dataio = DataIO(output_folder=all_parameters['output_directory'], label=all_parameters['label'].format(**all_parameters))
    save_every = 1
    run_counter = 0

    # Result arrays
    # sizes depend on the experiment.
    result_ll_n = []
    result_ll_sum = []
    result_bic = []
    result_ll90_sum = []

    if all_parameters['inference_method'] != 'none':
        result_precision = []
        result_em_fits = []
        result_fi_theo = []
        result_fi_theocov = []

    search_progress = progress.Progress(all_parameters['num_repetitions'])
    for repet_i in xrange(all_parameters['num_repetitions']):
        print "\n\n%d/%d | %.2f%%, %s left - %s" % (repet_i+1, all_parameters['num_repetitions'], search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

        ### WORK WORK WORK work? ###

        # Let's build a FitExperimentAllT
        fit_exp = FitExperimentAllT(all_parameters)

        # Setup and evaluate some statistics
        def compute_everything(self, parameters):
            results = dict()

            print ">> Computing LL all N..."
            results['result_ll_n'] = self.sampler.compute_loglikelihood_N()

            print ">> Computing LL sum..."
            results['result_ll_sum'] = self.sampler.compute_loglikelihood()

            print ">> Computing BIC..."
            results['result_bic'] = self.sampler.compute_bic(K=parameters['bic_K'])

            print ">> Computing LL90..."
            results['result_ll90_sum'] = self.sampler.compute_loglikelihood_top90percent()

            # If sampling_method is not none, try to get em_fits and others
            if not parameters['inference_method'] == 'none':
                print ">> Sampling and fitting mixt model / precision / FI ..."

                # Sample
                print " sampling..."
                self.sampler.run_inference(parameters)

                # Compute precision
                print " get precision..."
                results['result_precision'] = self.sampler.get_precision()

                # Fit mixture model
                print " fit mixture model..."
                curr_params_fit = self.sampler.fit_mixture_model(use_all_targets=False)
                results['result_em_fits'] = np.array([curr_params_fit[key] for key in ['kappa', 'mixt_target', 'mixt_nontargets_sum', 'mixt_random', 'train_LL', 'bic']])

                # Compute fisher info
                print " compute fisher info"
                results['result_fi_theo'] = self.sampler.estimate_fisher_info_theocov(use_theoretical_cov=False)
                results['result_fi_theocov'] = self.sampler.estimate_fisher_info_theocov(use_theoretical_cov=True)

            return results

        res_listdicts = fit_exp.apply_fct_datasets_allT(dict(fct=compute_everything, parameters=all_parameters))

        # Put everything back together
        for key in res_listdicts[0]:
            curr_result = eval(key)
            curr_result.append(np.array([res[key] for res in res_listdicts]))

        print "CURRENT RESULTS:"
        print res_listdicts

        ### /Work ###

        search_progress.increment()
        if run_counter % save_every == 0 or search_progress.done():
            dataio.save_variables_default(locals())
        run_counter += 1

    # Convert results to arrays
    # Put the repetition axis at the last dimension, it's kinda my convention...
    result_ll_n = np.array(result_ll_n)
    result_ll_n = result_ll_n.transpose(np.roll(np.arange(result_ll_n.ndim), -1))
    result_ll_sum = np.array(result_ll_sum)
    result_ll_sum = result_ll_sum.transpose(np.roll(np.arange(result_ll_sum.ndim), -1))
    result_bic = np.array(result_bic)
    result_bic = result_bic.transpose(np.roll(np.arange(result_bic.ndim), -1))
    result_ll90_sum = np.array(result_ll90_sum)
    result_ll90_sum = result_ll90_sum.transpose(np.roll(np.arange(result_ll90_sum.ndim), -1))

    if all_parameters['inference_method'] != 'none':
        result_precision = np.array(result_precision)
        result_precision = result_precision.transpose(np.roll(np.arange(result_precision.ndim), -1))
        result_em_fits = np.array(result_em_fits)
        result_em_fits = result_em_fits.transpose(np.roll(np.arange(result_em_fits.ndim), -1))
        result_fi_theo = np.array(result_fi_theo)
        result_fi_theo = result_fi_theo.transpose(np.roll(np.arange(result_fi_theo.ndim), -1))
        result_fi_theocov = np.array(result_fi_theocov)
        result_fi_theocov = result_fi_theocov.transpose(np.roll(np.arange(result_fi_theocov.ndim), -1))

    ### /Work ###

    dataio.save_variables_default(locals())

    #### Plots ###

    print "All finished"
    return locals()

