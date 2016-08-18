#!/usr/bin/env python
# encoding: utf-8
"""
fitexperimentAllTSubject.py

Created by Loic Matthey on 2013-09-26.
Copyright (c) 2013 Gatsby Unit. All rights reserved.
"""

import os

import numpy as np

import experimentlauncher
import load_experimental_data
import utils

from fitexperiment_allt import FitExperimentAllT

class FitExperimentAllTSubject(FitExperimentAllT):
    '''
        Overloaded FitExperimentAllT to use a subject-subset dataset.

        Loads experimental data, set up DataGenerator and associated RFN, Sampler to optimize parameters.

        This is meant to support *per subject* fits.
        Expects a unique Subject id, will restrain the data to that one.
    '''

    def __init__(self, parameters={}, debug=True):
        '''
            FitExperimentAllTSubject takes a parameters dict, same as a full launcher_

            Will then instantiate a Sampler and force a specific DataGenerator with constrained data from human experimental data.

            Requires experiment_id to be set.
        '''
        self.subject = parameters.get('experiment_subject', 0)

        assert parameters['experiment_id'] == 'bays09', "Check me for other datasets first!"

        if debug:
            print "FitExperimentAllTSubject: subject %d" % (self.subject)

        super(self.__class__, self).__init__(parameters, debug)


    def load_dataset(self):
        '''
            Load and select dataset given the parameters.
        '''
        self.experimental_dataset = load_experimental_data.load_data(
            experiment_id=self.experiment_id,
            data_dir=self.data_dir,
            fit_mixture_model=True
        )
        self.subject_space = self.experimental_dataset['data_subject_split']['subjects_space']
        assert self.subject in self.subject_space, "Subject id not found in dataset!"

        # This is a subset of the full dataset, for this particular subject!
        self.experiment_data_to_fit = self.experimental_dataset['data_subject_split']['data_subject_nitems'][self.subject]
        self.T_space = self.experimental_dataset['data_subject_split']['nitems_space']
        self.num_datapoints = int(self.experimental_dataset['data_subject_split']['subject_smallestN'][self.subject])


    def get_em_fits_arrays(self):
        '''
            Provide the EM fits as numpy arrays

            Returns:
            * dict(mean=np.array, std=np.array)
        '''

        if self.em_fits_arrays is None:
            subject_i = np.nonzero(self.subject_space ==
                                   self.subject)[0][0]

            self.em_fits_arrays = dict()
            self.em_fits_arrays['mean'] = self.experimental_dataset['em_fits_subjects_nitems_arrays'][subject_i].T
            self.em_fits_arrays['std'] = np.zeros_like(self.em_fits_arrays['mean'])

        return self.em_fits_arrays


    def compute_dist_experimental_em_fits_currentT(self, model_fits):
        '''
            Given provided model_fits array, compute the distance to
            the loaded Experimental Data em fits.

            Inputs:
            * model_fits:   [array 6x1] ['kappa', 'mixt_target', 'mixt_nontargets_sum', 'mixt_random', 'train_LL', 'bic']

            Returns dict:
            * kappa MSE
            * mixture prop MSE
            * summed MSE
            * mixture prop KL divergence
        '''

        distances = dict()

        T_i = np.nonzero(self.T_space == self.enforced_T)[0][0]

        data_mixture_subject = self.get_em_fits_arrays()['mean']

        distances['all_mse'] = (data_mixture_subject[:4, T_i] - model_fits[:4])**2.
        distances['mixt_kl'] = utils.KL_div(data_mixture_subject[1:4, T_i], model_fits[1:4])

        return distances


###########################################################################


def test_fitexperiment_allt_subjects():

    # Set some parameters and let the others default
    experiment_parameters = dict(action_to_do='launcher_do_simple_run',
                                 inference_method='none',
                                 experiment_id='bays09',
                                 experiment_subject=2,
                                 M=100,
                                 selection_method='last',
                                 sigmax=0.1,
                                 renormalize_sigmax=None,
                                 sigmay=0.0001,
                                 num_samples=100,
                                 code_type='mixed',
                                 slice_width=0.07,
                                 burn_samples=100,
                                 ratio_conj=0.7,
                                 stimuli_generation_recall='random',
                                 autoset_parameters=None,
                                 label='test_fit_experimentallt'
                                 )
    experiment_launcher = experimentlauncher.ExperimentLauncher(run=True, arguments_dict=experiment_parameters)
    experiment_parameters_full = experiment_launcher.args_dict

    # Now let's build a FitExperimentAllTSubject
    fit_exp = FitExperimentAllTSubject(experiment_parameters_full)

    # Now compute some metrics
    def compute_metrics(self, parameters):
        results = dict()

        # Sample
        print " sampling..."
        self.sampler.force_sampling_round()

        # Loglike
        results['ll_n'] = self.sampler.compute_loglikelihood_N()

        # Mixture model
        curr_params_fit = self.sampler.fit_mixture_model(use_all_targets=False)
        results['em_fits'] = np.array([curr_params_fit[key]
            for key in ['kappa', 'mixt_target', 'mixt_nontargets_sum',
                        'mixt_random', 'train_LL', 'bic']])

        emfits_distances = self.compute_dist_experimental_em_fits_currentT(results['em_fits'])
        results['emfit_mse'] = emfits_distances['all_mse']
        results['emfit_mixt_kl'] = emfits_distances['mixt_kl']

        return results

    results = fit_exp.apply_fct_datasets_allT(
        dict(fct=compute_metrics,
             parameters=dict()
             )
    )

    print results

    return locals()



if __name__ == '__main__':
    if True:
        all_vars = test_fitexperiment_allt_subjects()

    for key, val in all_vars.iteritems():
        locals()[key] = val

